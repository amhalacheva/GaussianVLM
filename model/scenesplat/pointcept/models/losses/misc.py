"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES

@LOSSES.register_module()
class L1Loss(nn.Module):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction="mean",
        loss_weight=1.0,
    ):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.loss = nn.L1Loss(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight
    

@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss
    
@LOSSES.register_module()
class CosineSimilarity(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(CosineSimilarity, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, valid_feat_mask, **kwargs):
        # Compute cosine similarity along the feature dimension (assumed to be dim=1)
        cos = nn.CosineSimilarity(dim=1)
        loss = 1 - cos(pred[valid_feat_mask], target[valid_feat_mask])
        
        if self.reduction == "mean":
            # Compute the mean only over valid samples.
            valid_count = valid_feat_mask.sum()
            if valid_count > 0:
                loss = loss.sum() / valid_count
            else:
                loss = loss.sum()  
        elif self.reduction == "sum":
            loss = loss.sum()
        
        # print("cosine loss:", self.loss_weight * loss.item())
        return self.loss_weight * loss
    
@LOSSES.register_module()
class L2Loss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(L2Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, valid_feat_mask, **kwargs):
        loss = ((pred[valid_feat_mask] - target[valid_feat_mask]) ** 2).sum(dim=1)
        
        if self.reduction == "mean":
            # Average the loss over only the valid samples
            valid_count = valid_feat_mask.sum()
            if valid_count > 0:
                loss = loss.sum() / valid_count
            else:
                loss = loss.sum()  # fallback if there are no valid samples
        elif self.reduction == "sum":
            # Sum the loss over all valid samples
            loss = loss.sum()

        # print("l2 loss:", self.loss_weight * loss.item())
        return self.loss_weight * loss
    

@LOSSES.register_module()
class AggregatedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2, reduction="mean", loss_weight=1.0, schedule='all'):
        """
        Args:
            temperature (float): Temperature scaling factor.
            reduction (str): 'mean' or 'sum' to average or sum the loss over classes.
            loss_weight (float): A multiplicative factor for the loss.
        """
        super(AggregatedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.schedule = schedule
        if 'last_' in self.schedule:
            self.last_percent = float(self.schedule.split('_')[-1])/100
            print("Contrastive loss apply in last {}% of training.".format(self.schedule.split('_')[-1]))
        elif self.schedule == 'all':
            print("Contrastive loss is applied in all epochs.")
        elif self.schedule == 'skip':
            print("Contrastive loss is skipped.")

    
    def forward(self, pred, target, valid_feat_mask, segment, epoch_progress=None, **kwargs):
        """
        Args:
            pred (Tensor): Predicted language features of shape [N, D].
            target (Tensor): Ground truth language features (unused in contrastive loss).
            valid_feat_mask (Tensor): Binary mask of shape [N] (1 for valid features).
            segment (Tensor): Semantic segmentation labels of shape [N] (with -1 for ignore index).
        
        Returns:
            Tensor: The computed aggregated contrastive loss.
        """
        device = pred.device
        if 'last_' in self.schedule and epoch_progress is not None:
            if epoch_progress <= (1 - self.last_percent):
                return torch.tensor(0.0, device=device)
        elif self.schedule == 'skip':
            return torch.tensor(0.0, device=device)
        
        # If segmentation is not provided, return 0 loss.
        if segment is None:
            return torch.tensor(0.0, device=device)
        
        # Select only valid indices (mask > 0 and segment != -1)
        valid_idx = (valid_feat_mask > 0) & (segment != -1)
        if valid_idx.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        features = pred[valid_idx]   # [M, D]
        labels = segment[valid_idx]  # [M]
        
        # Find unique semseg labels
        unique_labels = torch.unique(labels)
        
        aggregated_a = []
        aggregated_b = []
        used_labels = []
        for lab in unique_labels:
            # iterate over each unique class
            indices = (labels == lab).nonzero(as_tuple=True)[0]
            if indices.numel() < 100:
                continue  # insufficient samples
            
            # Shuffle the indices randomly
            perm = indices[torch.randperm(indices.size(0))]
            split = perm.size(0) // 2
            # Ensure we have at least one element in each group
            if split == 0 or (perm.size(0) - split) == 0:
                continue
            
            group_a = features[perm[:split]]
            group_b = features[perm[split:]]
            
            # Use average pooling to get a representative vector
            agg_a = group_a.mean(dim=0)
            agg_b = group_b.mean(dim=0)
            aggregated_a.append(agg_a)
            aggregated_b.append(agg_b)
            used_labels.append(lab)
        
        if len(aggregated_a) == 0:
            return torch.tensor(0.0, device=device)
        
        # Stack aggregated features into tensors of shape [C, D] where C is the number of semseg classes.
        aggregated_a = torch.stack(aggregated_a, dim=0)
        aggregated_b = torch.stack(aggregated_b, dim=0)
        
        # Normalize the aggregated features
        aggregated_a = F.normalize(aggregated_a, p=2, dim=1)
        aggregated_b = F.normalize(aggregated_b, p=2, dim=1)
        
        # Compute cosine similarity matrix between the two sets and scale by temperature.
        # logits[i, j] = cosine_similarity(aggregated_a[i], aggregated_b[j]) / temperature
        logits = torch.matmul(aggregated_a, aggregated_b.T) / self.temperature
        
        # The diagonal elements are the positive pairs.
        targets = torch.arange(logits.size(0), device=device)
        
        # Compute cross-entropy loss in both directions.
        loss_a = F.cross_entropy(logits, targets)
        logits_b = torch.matmul(aggregated_b, aggregated_a.T) / self.temperature
        loss_b = F.cross_entropy(logits_b, targets)
        loss = (loss_a + loss_b) / 2.0
        
        if self.reduction == "sum":
            loss = loss * logits.size(0)
        # For "mean", cross_entropy already averages over the classes.
        
        # Optionally print or log the loss
        # print("contrastive loss:", self.loss_weight * loss.item())
        
        return self.loss_weight * loss


