# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedLoss(nn.Module):
    """Configurable detection loss with scale-adaptive assignment for small objects."""

    def __init__(
        self,
        nc=80,
        device='cpu',
        num_scales=3,
        scale_area_thresholds=(0.0025, 0.0225),  # (32/640)^2, (96/640)^2
        box_weight=5.0,
        obj_weight=1.0,
        cls_weight=1.0,
        small_object_boost=2.0,
        use_scale_adaptive_assignment=True,
        use_small_object_weighting=True,
        use_tiny_neighbor_supervision=True,
    ):
        super().__init__()
        self.nc = nc
        self.device = device
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        self.num_scales = num_scales
        self.scale_area_thresholds = scale_area_thresholds
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.small_object_boost = small_object_boost

        self.use_scale_adaptive_assignment = use_scale_adaptive_assignment
        self.use_small_object_weighting = use_small_object_weighting
        self.use_tiny_neighbor_supervision = use_tiny_neighbor_supervision

    def forward(self, outputs, targets):
        """Calculate weighted detection loss over all detection scales."""
        loss_dict = {'cls_loss': 0.0, 'box_loss': 0.0, 'obj_loss': 0.0}

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        total_box_loss = torch.tensor(0.0, device=self.device)
        total_obj_loss = torch.tensor(0.0, device=self.device)
        total_cls_loss = torch.tensor(0.0, device=self.device)
        valid_scales = 0

        for scale_index, output in enumerate(outputs):
            if output is None or output.numel() == 0:
                continue

            # output shape: [B, A, H, W, 5 + nc]
            batch_size, num_anchors, grid_h, grid_w, _ = output.shape
            reg_preds = output[..., :4]   # [B, A, H, W, 4]
            obj_preds = output[..., 4]    # [B, A, H, W]
            cls_preds = output[..., 5:]   # [B, A, H, W, nc]

            obj_target, box_target, cls_target, sample_weights = self._build_targets_for_scale(
                targets=targets,
                scale_index=scale_index,
                shape=(batch_size, num_anchors, grid_h, grid_w),
                device=output.device,
            )

            obj_loss = self.bce(obj_preds, obj_target).mean()

            pos_mask = obj_target > 0.5
            if pos_mask.any():
                pred_box = torch.sigmoid(reg_preds[pos_mask])
                tgt_box = box_target[pos_mask]
                pos_weights = sample_weights[pos_mask]

                l1 = F.smooth_l1_loss(pred_box, tgt_box, reduction='none').mean(dim=-1)
                iou = self._bbox_iou_xyxy(pred_box, tgt_box)
                box_per_pos = l1 + (1.0 - iou)
                box_loss = (box_per_pos * pos_weights).mean()

                if self.nc > 0:
                    cls_logits_pos = cls_preds[pos_mask]
                    cls_target_pos = cls_target[pos_mask]
                    cls_raw = self.bce(cls_logits_pos, cls_target_pos).mean(dim=-1)
                    cls_loss = (cls_raw * pos_weights).mean()
                else:
                    cls_loss = torch.tensor(0.0, device=output.device)
            else:
                box_loss = torch.tensor(0.0, device=output.device)
                cls_loss = torch.tensor(0.0, device=output.device)

            total_box_loss = total_box_loss + box_loss
            total_obj_loss = total_obj_loss + obj_loss
            total_cls_loss = total_cls_loss + cls_loss
            valid_scales += 1

            loss_dict['box_loss'] += float(box_loss.detach().item())
            loss_dict['obj_loss'] += float(obj_loss.detach().item())
            loss_dict['cls_loss'] += float(cls_loss.detach().item())

        if valid_scales == 0:
            final_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return final_loss, loss_dict

        total_box_loss = total_box_loss / valid_scales
        total_obj_loss = total_obj_loss / valid_scales
        total_cls_loss = total_cls_loss / valid_scales

        final_loss = (
            self.box_weight * total_box_loss
            + self.obj_weight * total_obj_loss
            + self.cls_weight * total_cls_loss
        )

        for k in loss_dict:
            loss_dict[k] /= valid_scales

        return final_loss, loss_dict

    def _select_scale_index(self, area):
        """Assign GT to a scale by normalized area."""
        t0, t1 = self.scale_area_thresholds
        if area <= t0:
            return 0
        if area <= t1:
            return 1
        return min(2, self.num_scales - 1)

    def _is_scale_assigned(self, area, scale_index):
        if not self.use_scale_adaptive_assignment:
            return True
        return self._select_scale_index(area) == scale_index

    def _build_targets_for_scale(self, targets, scale_index, shape, device):
        """Build obj/box/class targets for one feature scale."""
        batch_size, num_anchors, grid_h, grid_w = shape

        obj_target = torch.zeros((batch_size, num_anchors, grid_h, grid_w), device=device)
        box_target = torch.zeros((batch_size, num_anchors, grid_h, grid_w, 4), device=device)
        cls_target = torch.zeros((batch_size, num_anchors, grid_h, grid_w, self.nc), device=device)
        sample_weights = torch.ones((batch_size, num_anchors, grid_h, grid_w), device=device)

        for b in range(batch_size):
            image_targets = targets[b] if b < len(targets) else []

            for target in image_targets:
                if target is None or len(target) < 5:
                    continue

                x1, y1, x2, y2, class_id = target[:5]
                x1 = float(max(0.0, min(1.0, x1)))
                y1 = float(max(0.0, min(1.0, y1)))
                x2 = float(max(0.0, min(1.0, x2)))
                y2 = float(max(0.0, min(1.0, y2)))

                if x2 <= x1 or y2 <= y1:
                    continue

                area = (x2 - x1) * (y2 - y1)
                if not self._is_scale_assigned(area, scale_index):
                    continue

                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                gi = max(0, min(grid_w - 1, int(cx * grid_w)))
                gj = max(0, min(grid_h - 1, int(cy * grid_h)))

                anchor_id = 0
                obj_target[b, anchor_id, gj, gi] = 1.0
                box_target[b, anchor_id, gj, gi] = torch.tensor([x1, y1, x2, y2], device=device)

                cls_id = int(class_id)
                if 0 <= cls_id < self.nc:
                    cls_target[b, anchor_id, gj, gi, cls_id] = 1.0

                weight = 1.0
                if self.use_small_object_weighting:
                    smallness = 1.0 - min(1.0, area / max(self.scale_area_thresholds[0], 1e-6))
                    weight = 1.0 + (self.small_object_boost - 1.0) * max(0.0, smallness)
                sample_weights[b, anchor_id, gj, gi] = weight

                # Tiny-object support: softly supervise 4-neighbors
                if self.use_tiny_neighbor_supervision and area <= self.scale_area_thresholds[0]:
                    for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nj = gj + dj
                        ni = gi + di
                        if 0 <= nj < grid_h and 0 <= ni < grid_w:
                            obj_target[b, anchor_id, nj, ni] = torch.maximum(
                                obj_target[b, anchor_id, nj, ni],
                                torch.tensor(0.5, device=device)
                            )
                            box_target[b, anchor_id, nj, ni] = torch.tensor([x1, y1, x2, y2], device=device)
                            if 0 <= cls_id < self.nc:
                                cls_target[b, anchor_id, nj, ni, cls_id] = 1.0
                            sample_weights[b, anchor_id, nj, ni] = max(
                                float(sample_weights[b, anchor_id, nj, ni].item()),
                                float(weight * 0.5)
                            )

        return obj_target, box_target, cls_target, sample_weights

    @staticmethod
    def _bbox_iou_xyxy(boxes1, boxes2, eps=1e-6):
        """Compute IoU for aligned xyxy box tensors of shape [N, 4]."""
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        inter_area = inter_w * inter_h

        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

        union = area1 + area2 - inter_area
        return inter_area / (union + eps)
