"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import pointops
from uuid import uuid4

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu, clustering_voting

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class ClsEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                label,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "allAcc"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(self, enable_voting=False, vote_k=25):
        super().__init__()
        self.enable_voting = enable_voting
        self.vote_k = vote_k

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    # def before_step(self): # for debugging only 
    #     if self.trainer.cfg.evaluate:
    #         self.eval()
    #         self.trainer.model.train()


    def _neighbor_voting(self, coords, initial_labels, vote_k):
        """neighbor voting on the initial predictions."""
        from scipy.spatial import cKDTree
        import numpy as np

        if len(coords) == 0 or vote_k < 1:
            return initial_labels

        kd_tree = cKDTree(coords)
        try:
            _, nn_indices = kd_tree.query(coords, k=vote_k)
        except Exception as e:
            print(f"Error during KDTree query: {e}")
            return initial_labels

        if vote_k == 1:
            nn_indices = nn_indices[:, np.newaxis]

        neighbor_labels = initial_labels[nn_indices]
        voted_labels = np.zeros_like(initial_labels)
        for i in range(neighbor_labels.shape[0]):
            labels = neighbor_labels[i]
            counts = np.bincount(labels, minlength=self.trainer.cfg.data.num_classes)
            voted_labels[i] = np.argmax(counts)
        return voted_labels

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start SemSegEvaluator >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        if self.enable_voting:
            print(f"Neighbor voting enabled with k={self.vote_k}")
        print(f"Length of val_loader: {len(self.trainer.val_loader)}")
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]
            else:
                coords = input_dict["coord"]  # Use current coordinates for voting
            if self.enable_voting:
                coords_np = coords.cpu().numpy()
                pred_np = pred.cpu().numpy()
                voted_pred_np = self._neighbor_voting(coords_np, pred_np, self.vote_k)
                pred = torch.from_numpy(voted_pred_np).to(pred.device)
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] Loss {loss:.4f}".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(info)
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )




@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, segment_ignore_index=(-1,), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = []
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert (
                len(input_dict["offset"]) == 1
            )  # currently only support bs 1 for each GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]

            segment = input_dict["segment"]
            instance = input_dict["instance"]
            # map to origin
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.cpu().flatten().long()
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes.append(dict(gt=gt_instances, pred=pred_instance))

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        self.trainer.logger.info(
            "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                all_ap, all_ap_50, all_ap_25
            )
        )
        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
            self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
            self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
        self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver



@HOOKS.register_module()
class  LangPretrainZeroShotSemSegEval(HookBase):
    def __init__(self, class_names, 
                text_embeddings, 
                excluded_classes=None, 
                ignore_index=-1, 
                confidence_threshold=0.1,
                vote_k=25,
                enbale_voting=False):
        """
        Args:
            class_names (list): path to a txt of class names ordered by class index
            text_embeddings (Tensor): path to text embeddings (num_classes, feat_dim)
            excluded_classes (list): Class names to exclude from final metrics
            ignore_index (int): Index to ignore in GT labels
            confidence_threshold (float): Minimum confidence to consider prediction valid
        """
        super().__init__()
        with open(class_names, "r") as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        self.num_classes = len(self.class_names)
        self.device = torch.device("cuda")

        # load text embeddings from the path
        print(f"Loading text embeddings from {text_embeddings}")
        text_embeddings = torch.load(text_embeddings)
        self.text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        print(f"Text embeddings for ZeroShotSemSegEval with shape: {self.text_embeddings.shape}")
        self.excluded_classes = excluded_classes
        self.excluded_indices = [i for i, name in enumerate(self.class_names) if name in (excluded_classes or [])]
        self.ignore_index = ignore_index
        print(f"Excluded classes for ZeroShotSemSegEval: {self.excluded_classes}")
        self.enable_voting = enbale_voting
        self.vote_k = vote_k
        self.top_k = 3 # Number of top predictions to consider
        self.confidence_threshold = confidence_threshold
        
        # State variables
        self.confusion = None
        self.fn_ignore = None
        self._reset_metrics()

    def _reset_metrics(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.fn_ignore = np.zeros(self.num_classes, dtype=np.int64)

    # def before_step(self): # for debug
    #     if self.trainer.cfg.evaluate:
    #         self.eval()
    #         self.trainer.model.train()

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def _neighbor_voting(self, coords, initial_labels, valid_mask):
        """Efficient neighbor voting using cKDTree and valid_feat_mask filtering"""
        from scipy.spatial import cKDTree
        import numpy as np

        
        valid_coords = coords[valid_mask]
        valid_labels = initial_labels[valid_mask]
        
        if len(valid_coords) == 0:
            # Create empty topk array with ignore_index
            topk_labels = np.full((len(coords), self.top_k), self.ignore_index)
            return topk_labels
        
        kd_tree = cKDTree(valid_coords)
        _, nn_indices = kd_tree.query(coords, k=self.vote_k)
        if self.vote_k == 1:
            nn_indices = nn_indices[:, None]
        
        neighbor_labels = valid_labels[nn_indices]
        # Initialize array for top-k labels (N, top_k)
        topk_labels = np.full((len(coords), self.top_k), self.ignore_index, dtype=np.int64)
        
        # Vectorized voting using numpy's unique and argsort
        for i in range(len(coords)):
            labels = neighbor_labels[i]
            
            # Count all labels including ignore_index
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) == 0:
                continue
            
            # Sort by frequency (descending) then by label (ascending)
            sorted_indices = np.lexsort((unique, -counts))
            sorted_labels = unique[sorted_indices]
            
            # Store top-k labels
            k = min(self.top_k, len(sorted_labels))
            topk_labels[i, :k] = sorted_labels[:k]
        
        return topk_labels

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Zero-Shot SemSeg Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        self._reset_metrics()

        if self.vote_k > 1 and self.enable_voting:
            print(f"Neighbor voting enabled with k={self.vote_k}")

        text_embeddings = self.text_embeddings.to(self.device)
        with torch.no_grad():
            print(f"Length of val_loader: {len(self.trainer.val_loader)}")
            for i, input_dict in enumerate(self.trainer.val_loader):
                # Move data to GPU
                input_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in input_dict.items()}
                
                # Forward pass
                output_dict = self.trainer.model(input_dict, chunk_size=600000)
                point_feat = output_dict['point_feat']['feat'] # normalized
                
                # Get ground truth labels
                segment = input_dict.get("segment", torch.full((point_feat.size(0),), self.ignore_index, device=point_feat.device))
                valid_mask = segment != self.ignore_index
                valid_segment = segment[valid_mask]
                
                if valid_segment.numel() == 0:
                    continue

                # Compute predictions
                logits = torch.mm(point_feat, text_embeddings.t())
                probs = torch.sigmoid(logits)
                max_probs, pred_labels = torch.max(probs, dim=1)
                
                # Apply confidence threshold
                pred_labels = pred_labels.cpu().numpy()
                max_probs = max_probs.cpu().numpy()
                pred_labels[max_probs < self.confidence_threshold] = self.ignore_index

                # Neighbor voting if enabled
                if self.vote_k > 1 and self.enable_voting:
                    coords = input_dict['coord'].cpu().numpy()
                    topk_labels = self._neighbor_voting(
                        coords=coords,
                        initial_labels=pred_labels,
                        valid_mask=input_dict['valid_feat_mask'].cpu().numpy()
                    )
                    pred_labels = topk_labels[:, 0]
                    if 'instance' in input_dict: # clustering voting
                        pred_labels = clustering_voting(pred_labels, input_dict['instance'].cpu().numpy(), self.ignore_index)

                # Update confusion matrix
                valid_pred = pred_labels[valid_mask.cpu().numpy()]
                valid_gt = valid_segment.cpu().numpy()
                
                for gt, pred in zip(valid_gt, valid_pred):
                    if pred == self.ignore_index:
                        self.fn_ignore[gt] += 1
                    else:
                        self.confusion[gt, pred] += 1

                # Log progress
                if (i + 1) % 10 == 0:
                    self.trainer.logger.info(
                        f"Processed {i+1}/{len(self.trainer.val_loader)} batches"
                    )
                # if i >= 10:
                #     break # temp for debug

        # Synchronize across GPUs in distributed training
        if dist.is_initialized():
            confusion_tensor = torch.tensor(self.confusion).to(self.device)
            fn_ignore_tensor = torch.tensor(self.fn_ignore).to(self.device)
            dist.all_reduce(confusion_tensor)
            dist.all_reduce(fn_ignore_tensor)
            self.confusion = confusion_tensor.cpu().numpy()
            self.fn_ignore = fn_ignore_tensor.cpu().numpy()

        # Compute and log metrics
        self._log_metrics()
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    def _log_metrics(self):
        # Calculate per-class IoU
        ious = []
        present_mask = (self.confusion.sum(axis=1) + self.fn_ignore) > 0 # note we didn't include ignore_index in the confusion matrix
        present_classes = [c for c in range(self.num_classes) if present_mask[c]]
        missing_classes = [self.class_names[c] for c in range(self.num_classes) if not present_mask[c]]
        
        for c in range(self.num_classes):
            tp = self.confusion[c, c]
            fp = self.confusion[:, c].sum() - tp
            fn = self.confusion[c, :].sum() - tp + self.fn_ignore[c]
            denom = tp + fp + fn
            ious.append(tp / denom if denom > 0 else 0.0)

        # Calculate metrics
        metrics = {
            'mIoU': np.mean([ious[c] for c in present_classes]),
            'fIoU': self._calculate_fiou(present_classes, ious),
            'global_acc': np.diag(self.confusion).sum() / self.confusion.sum(),
            'mean_class_acc': self._calculate_mean_class_acc(present_classes),
            'fw_mean_class_acc': self._calculate_fw_mean_class_acc(present_classes)
        }

        # Calculate excluded metrics
        included_classes = [c for c in present_classes if c not in self.excluded_indices]
        metrics.update({
            'mIoU_excl': np.mean([ious[c] for c in included_classes]) if included_classes else 0,
            'fIoU_excl': self._calculate_fiou(included_classes, ious) if included_classes else 0,
            'mean_class_acc_excl': self._calculate_mean_class_acc(included_classes) if included_classes else 0,
            'fw_mean_class_acc_excl': self._calculate_fw_mean_class_acc(included_classes) if included_classes else 0
        })

        # Log class-wise results
        self.trainer.logger.info("\nMissing classes: %s", missing_classes)
        self.trainer.logger.info("\n--- Per-class IoU (all classes) ---")
        for c in present_classes:
            self.trainer.logger.info(f"{self.class_names[c]:20s}: {ious[c]:.4f}")

        # main metrics
        self.trainer.logger.info("\n--- Metrics (ALL present classes) ---")
        self.trainer.logger.info(f"Mean IoU (mIoU)          : {metrics['mIoU']:.4f}")
        self.trainer.logger.info(f"Frequency-weighted IoU   : {metrics['fIoU']:.4f}")
        self.trainer.logger.info(f"Global Accuracy          : {metrics['global_acc']:.4f}")
        self.trainer.logger.info(f"Mean Class Accuracy      : {metrics['mean_class_acc']:.4f}")
        self.trainer.logger.info(f"Freq.-weighted Mean Acc  : {metrics['fw_mean_class_acc']:.4f}")

        # print excluded metrics
        self.trainer.logger.info(f"\n--- Metrics (EXCLUDED {self.excluded_classes}) ---")
        self.trainer.logger.info(f"Mean IoU (mIoU)          : {metrics['mIoU_excl']:.4f}")
        self.trainer.logger.info(f"Frequency-weighted IoU   : {metrics['fIoU_excl']:.4f}")
        self.trainer.logger.info(f"Mean Class Accuracy      : {metrics['mean_class_acc_excl']:.4f}")
        self.trainer.logger.info(f"Freq.-weighted Mean Acc  : {metrics['fw_mean_class_acc_excl']:.4f}")

        # Log to TensorBoard
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar('val/mIoU', metrics['mIoU'], current_epoch)
            self.trainer.writer.add_scalar('val/fIoU', metrics['fIoU'], current_epoch)
            self.trainer.writer.add_scalar('val/global_acc', metrics['global_acc'], current_epoch)
            self.trainer.writer.add_scalar('val/mean_class_acc', metrics['mean_class_acc'], current_epoch)

        # Save primary metric for checkpointing
        self.trainer.comm_info["current_metric_value"] = metrics['mIoU']
        self.trainer.comm_info["current_metric_name"] = "mIoU"

    def _calculate_fiou(self, classes, ious):
        """Calculate frequency-weighted IoU for specified classes"""
        total_gt = sum(self.confusion[c].sum() + self.fn_ignore[c] for c in classes)
        if total_gt == 0:
            return 0.0
        return sum((self.confusion[c].sum() + self.fn_ignore[c]) / total_gt * ious[c] for c in classes)

    def _calculate_mean_class_acc(self, classes):
        """Calculate mean class accuracy for specified classes"""
        accs = []
        for c in classes:
            correct = self.confusion[c, c]
            total = self.confusion[c].sum()
            if total > 0:
                accs.append(correct / total)
        return np.mean(accs) if accs else 0.0
    
    def _calculate_fw_mean_class_acc(self, classes):
        """ frequency-weighted mean class accuracy """
        total_gt = sum(self.confusion[c].sum() + self.fn_ignore[c] for c in classes)
        if total_gt == 0:
            return 0.0
        fw_acc = 0.0
        for c in classes:
            total = self.confusion[c].sum()
            if total > 0:
                class_acc = self.confusion[c, c] / total
                fw_acc += (total / total_gt) * class_acc
        return fw_acc


    def after_train(self):
        # self.enable_voting = True
        # self.eval()
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )
