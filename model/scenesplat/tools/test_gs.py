import numpy as np 
import os 
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)
import argparse


def arg_parser():
    # /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/exp/scannetppgs_default/semseg-gs-v3m1-0-base-no-normal_debug_save/result
    parser = argparse.ArgumentParser(description="Pointcept Testing on Gaussian Splats")
    parser.add_argument(
        "--gs_result_dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--gs_data_dir", type=str, required=True, help="Directory to load data" # get original shift
    )
    parser.add_argument(
        "--pc_data_dir", type=str, default='/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/Pointcept/data/scannetpp/val', help="Directory for val data"
    )
    args = parser.parse_args()
    return args


def main():
    args = arg_parser()
    gs_result_dir = args.gs_result_dir
    pc_data_dir = args.pc_data_dir
    gs_data_dir = args.gs_data_dir
    # find all .npy in gs_result_dir
    pred_npy_files = [f for f in os.listdir(gs_result_dir) if f.endswith('pred.npy')]
    # coord_npy_files = [f for f in os.listdir(gs_result_dir) if f.endswith('coord.npy')]
    # npy_files = sorted(npy_files)
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    record = {}

    for pred_i in pred_npy_files:
        # read pred data from the gs_result_dir
        pred_label_i = np.load(os.path.join(gs_result_dir, pred_i))
        # print(pred.shape)
        name_i = pred_i.split('_')[0]
        coord_i_before_shift = np.load(os.path.join(gs_data_dir, name_i, 'coord.npy'))
        segment_i_before_shift = np.load(os.path.join(gs_data_dir, name_i, 'segment.npy'))

        print("pred_label_i", pred_label_i.shape, "segment_i_before_shift", segment_i_before_shift.shape)
        raise ValueError("stop here")
        x_min, y_min, z_min = coord_i_before_shift.min(axis=0)
        x_max, y_max, _ = coord_i_before_shift.max(axis=0)
        shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]

        coord_i = np.load(os.path.join(gs_result_dir, name_i+'_pred_coord.npy'))
        # read gt data from the pc data dir
        print("name_i", name_i)
        gt_label_i = np.load(os.path.join(pc_data_dir, name_i, 'segment.npy'))
        gt_coord_i = np.load(os.path.join(pc_data_dir, name_i, 'coord.npy'))
        # print("gt_label_i", gt_label_i.shape)
        gt_label_i = gt_label_i[:,0].astype(np.int32)
        # x_min, y_min, z_min = gt_coord_i.min(axis=0)
        # x_max, y_max, _ = gt_coord_i.max(axis=0)
        # shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
        gt_coord_i -= shift

        print("gt_coord_i", gt_coord_i.shape, gt_coord_i.min(), gt_coord_i.max())
            
        # using scipy kd tree to find nearest neighbor
        # from scipy.spatial import KDTree
        # tree = KDTree(coord_i, leafsize=coord_i.shape[0]+1)

        # from sklearn.neighbors import BallTree
        from sklearn.neighbors import KDTree
        # tree = BallTree(gt_coord_i, leaf_size=40, metric='euclidean')
        tree = KDTree(coord_i, leaf_size=40)

        # from pykdtree.kdtree import KDTree
        # kd_tree = KDTree(coord_i)

        pred_label_i = pred_label_i[:,0]
        dist, idx = tree.query(gt_coord_i, k=1)
        idx = idx.reshape(-1) # (N, 1) -> (N,), TODO change to multi-nearest neighbor
        propogate_label_i = pred_label_i[idx]
        # 

        # open3d visualization
        if True:
            from matplotlib import pyplot as plt
            import open3d as o3d
            # gs_semantic_color = propogate_label_i
            # gs_semantic_color = (gs_semantic_color - 0) / (gs_semantic_color.max() - 0)
            # colormap = plt.cm.hsv
            # gs_semantic_color = colormap(gs_semantic_color)[:,:3]
            # gs_semantic_color = (gs_semantic_color * 255).astype(np.uint8)
            # gs_semantic_color = gs_semantic_color / 255.
            # gs_o3d = o3d.geometry.PointCloud()
            # gs_o3d.points = o3d.utility.Vector3dVector(gt_coord_i)
            # gs_o3d.colors = o3d.utility.Vector3dVector(gs_semantic_color)
            # o3d.io.write_point_cloud( "./gs_semantic.ply", gs_o3d)

            gs_semantic_color = pred_label_i
            gs_semantic_color = (gs_semantic_color - 0) / (gs_semantic_color.max() - 0)
            colormap = plt.cm.hsv
            gs_semantic_color = colormap(gs_semantic_color)[:,:3]
            gs_semantic_color = (gs_semantic_color * 255).astype(np.uint8)
            gs_semantic_color = gs_semantic_color / 255.
            gs_o3d = o3d.geometry.PointCloud()
            gs_o3d.points = o3d.utility.Vector3dVector(coord_i)
            gs_o3d.colors = o3d.utility.Vector3dVector(gs_semantic_color)
            o3d.io.write_point_cloud( "./gs_semantic_org.ply", gs_o3d)

            # reload_o3d = o3d.io.read_point_cloud(str(save_path / "gs_semantic.ply"))
            # reload_coord = np.asarray(reload_o3d.points)
            # assert len(reload_coord) == len(coord), f"reload coord not match {len(reload_coord)} and {len(coord)}"
            # print("gs_semantic_color int", gs_semantic_color.shape)
            # gs_semantic_color = np.stack([gs_semantic_color, gs_semantic_color, gs_semantic_color], axis=-1)
            # print("gs_semantic_color stack", gs_semantic_color.shape)
            # colormap the gs_semantic_color
            # mesh = trimesh.Trimesh(vertices=coord, vertex_colors=gs_semantic_color)
            # mesh.export(save_path / "gs_semantic_prune.ply")

            # reload_mesh = trimesh.load(save_path / "gs_semantic_prune.ply")
            # length_reload = len(reload_mesh.vertices)
            # assert length_reload == len(coord), f"reload mesh vertices not match {length_reload} and {len(coord)}"

            # pcd.colors = o3d.utility.Vector3dVector(gs_semantic_color / 255 )
            # o3d.io.write_point_cloud(save_path / "gs_semantic.ply", pcd)
            pc_semantic_color = gt_label_i
            pc_semantic_color = (pc_semantic_color - 0) / (pc_semantic_color.max() - 0)
            pc_semantic_color = pc_semantic_color.clip(0, 1)
            pc_semantic_color = colormap(pc_semantic_color)[:,:3]

            pc_semantic_color = (pc_semantic_color * 255).astype(np.uint8)
            pc_semantic_color = pc_semantic_color / 255.
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(gt_coord_i)
            pc_o3d.colors = o3d.utility.Vector3dVector(pc_semantic_color)
            o3d.io.write_point_cloud( "./pc_semantic.ply", pc_o3d)

            raise ValueError("stop here")


    #     assert propogate_label_i.shape == gt_label_i.shape, f"propogate_label_i.shape != gt_label_i.shape, {propogate_label_i.shape} != {gt_label_i.shape}"
    #     # calculat mIoU
    #     print("propogate_label_i", propogate_label_i.min(), propogate_label_i.max(), propogate_label_i.shape)
    #     print("gt_label_i", gt_label_i.min(), gt_label_i.max(), gt_label_i.shape)

    #     intersection, union, target = intersection_and_union(
    #         propogate_label_i, gt_label_i, 100, -1
    #     )
    #     intersection_meter.update(intersection)
    #     union_meter.update(union)
    #     target_meter.update(target)
    #     record[name_i] = dict(
    #         intersection=intersection, union=union, target=target
    #     )

    #     mask = union != 0
    #     iou_class = intersection / (union + 1e-10)
    #     iou = np.mean(iou_class[mask])
    #     acc = sum(intersection) / (sum(target) + 1e-10)

    #     m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
    #     m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

    #     print("name_i: {}, mIoU: {:.2f}, mAcc: {:.2f}, iou: {:.2f}, acc: {:.2f}".format(name_i, m_iou , m_acc , iou , acc ))


    # intersection = np.sum(
    #     [meters["intersection"] for _, meters in record.items()], axis=0
    # )
    # union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
    # target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

    # iou_class = intersection / (union + 1e-10)
    # accuracy_class = intersection / (target + 1e-10)
    # mIoU = np.mean(iou_class)
    # mAcc = np.mean(accuracy_class)
    # allAcc = sum(intersection) / (sum(target) + 1e-10)

    # print("mIoU: {:.2f}".format(mIoU * 100))
    # print("mAcc: {:.2f}".format(mAcc * 100))
    # print("allAcc: {:.2f}".format(allAcc * 100))



if __name__ == "__main__":
    main()