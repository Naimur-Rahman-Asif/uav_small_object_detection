# utils/metrics.py
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


def compute_iou(box1, box2):
    """Compute IoU between two xyxy boxes."""
    b1 = np.array(box1, dtype=np.float32)
    b2 = np.array(box2, dtype=np.float32)

    if b1.shape[0] < 4 or b2.shape[0] < 4:
        return 0.0

    if b1[2] < b1[0]:
        b1[0], b1[2] = b1[2], b1[0]
    if b1[3] < b1[1]:
        b1[1], b1[3] = b1[3], b1[1]
    if b2[2] < b2[0]:
        b2[0], b2[2] = b2[2], b2[0]
    if b2[3] < b2[1]:
        b2[1], b2[3] = b2[3], b2[1]

    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter

    if union <= 0:
        return 0.0
    return float(inter / union)


def _to_python_nested_list(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy().tolist()
    elif isinstance(values, np.ndarray):
        values = values.tolist()
    return values


def _normalize_predictions(predictions):
    """
    Normalize predictions to per-image structure:
    List[List[[x1,y1,x2,y2,conf,cls]]]
    """
    predictions = _to_python_nested_list(predictions)
    if predictions is None:
        return []

    if len(predictions) == 0:
        return []

    # If already per-image list
    if isinstance(predictions[0], list):
        if len(predictions[0]) == 0:
            return predictions
        if isinstance(predictions[0][0], (list, tuple)):
            return predictions

    # Flat list case -> single image
    return [predictions]


def _normalize_ground_truths(ground_truths):
    """
    Normalize GT to per-image structure:
    List[List[[x1,y1,x2,y2,cls]]]
    """
    ground_truths = _to_python_nested_list(ground_truths)
    if ground_truths is None:
        return []

    if len(ground_truths) == 0:
        return []

    if isinstance(ground_truths[0], list):
        if len(ground_truths[0]) == 0:
            return ground_truths
        if isinstance(ground_truths[0][0], (list, tuple)):
            return ground_truths

    return [ground_truths]


def _clip_box01(box):
    x1, y1, x2, y2 = box
    x1 = float(min(1.0, max(0.0, x1)))
    y1 = float(min(1.0, max(0.0, y1)))
    x2 = float(min(1.0, max(0.0, x2)))
    y2 = float(min(1.0, max(0.0, y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _interpolated_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Continuous interpolation AP as in COCO/VOC style integration."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def _collect_class_data(predictions, ground_truths, class_id, area_range: Optional[Tuple[float, float]] = None):
    """
    Gather class-specific predictions and GTs across dataset.
    area_range: (min_area, max_area) in normalized-area units.
    """
    class_preds = []  # (img_id, conf, box)
    class_gts = {}    # img_id -> List[box]

    min_a, max_a = (None, None)
    if area_range is not None:
        min_a, max_a = area_range

    num_images = min(len(predictions), len(ground_truths))

    for img_id in range(num_images):
        preds_img = predictions[img_id] or []
        gts_img = ground_truths[img_id] or []

        class_gts[img_id] = []

        for gt in gts_img:
            if len(gt) < 5:
                continue
            gt_cls = int(gt[4])
            if gt_cls != class_id:
                continue
            box = _clip_box01(gt[:4])
            area = _box_area(box)
            if min_a is not None and area < min_a:
                continue
            if max_a is not None and area >= max_a:
                continue
            class_gts[img_id].append(box)

        for pred in preds_img:
            if len(pred) < 6:
                continue
            pred_cls = int(pred[5])
            if pred_cls != class_id:
                continue
            box = _clip_box01(pred[:4])
            conf = float(pred[4])
            area = _box_area(box)
            if min_a is not None and area < min_a:
                continue
            if max_a is not None and area >= max_a:
                continue
            class_preds.append((img_id, conf, box))

    return class_preds, class_gts


def _evaluate_class_at_iou(class_preds, class_gts, iou_thr: float):
    """Evaluate one class at one IoU threshold."""
    num_gt = sum(len(v) for v in class_gts.values())
    if num_gt == 0:
        return 0.0, 0.0, 0.0, 0, 0, 0

    if len(class_preds) == 0:
        return 0.0, 0.0, 0.0, 0, 0, num_gt

    class_preds = sorted(class_preds, key=lambda x: x[1], reverse=True)

    matched = {img_id: np.zeros(len(gts), dtype=bool) for img_id, gts in class_gts.items()}
    tp = np.zeros(len(class_preds), dtype=np.float32)
    fp = np.zeros(len(class_preds), dtype=np.float32)

    for idx, (img_id, _, pbox) in enumerate(class_preds):
        gts = class_gts.get(img_id, [])
        if len(gts) == 0:
            fp[idx] = 1.0
            continue

        ious = np.array([compute_iou(pbox, gt) for gt in gts], dtype=np.float32)
        best_gt_idx = int(np.argmax(ious))
        best_iou = float(ious[best_gt_idx])

        if best_iou >= iou_thr and not matched[img_id][best_gt_idx]:
            tp[idx] = 1.0
            matched[img_id][best_gt_idx] = True
        else:
            fp[idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / max(float(num_gt), 1e-12)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    ap = _interpolated_ap(recalls, precisions)
    precision = float(precisions[-1]) if len(precisions) > 0 else 0.0
    recall = float(recalls[-1]) if len(recalls) > 0 else 0.0

    tp_total = int(cum_tp[-1]) if len(cum_tp) > 0 else 0
    fp_total = int(cum_fp[-1]) if len(cum_fp) > 0 else 0
    fn_total = max(0, num_gt - tp_total)

    return ap, precision, recall, tp_total, fp_total, fn_total


def evaluate_map(predictions, ground_truths, iou_threshold=0.5, num_classes=None):
    """
    COCO-style evaluation summary with AP50, AP50:95 and AP_S.
    Inputs are expected per-image:
      predictions: [[x1,y1,x2,y2,conf,cls], ...] per image
      ground_truths: [[x1,y1,x2,y2,cls], ...] per image
    """
    predictions = _normalize_predictions(predictions)
    ground_truths = _normalize_ground_truths(ground_truths)

    if len(predictions) == 0 or len(ground_truths) == 0:
        return {
            'map_50': 0.0,
            'map_50_95': 0.0,
            'ap_small': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'per_class_ap50': {}
        }

    # Ensure equal number of images
    n_img = min(len(predictions), len(ground_truths))
    predictions = predictions[:n_img]
    ground_truths = ground_truths[:n_img]

    if num_classes is None:
        class_ids = set()
        for gts in ground_truths:
            for gt in (gts or []):
                if len(gt) >= 5:
                    class_ids.add(int(gt[4]))
        if len(class_ids) == 0:
            class_ids = {0}
        num_classes = max(class_ids) + 1

    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    class_range = list(range(num_classes))

    ap_by_iou = []
    ap50_per_class = {}
    total_tp = total_fp = total_fn = 0

    # Small object range in normalized area (COCO small proxy at 640 resolution)
    # (32 / 640)^2
    small_max_area = (32.0 / 640.0) ** 2
    aps_small = []

    for cls in class_range:
        class_preds, class_gts = _collect_class_data(predictions, ground_truths, cls)

        if sum(len(v) for v in class_gts.values()) == 0:
            continue

        class_aps = []
        for iou_thr in iou_thresholds:
            ap, p, r, tp, fp, fn = _evaluate_class_at_iou(class_preds, class_gts, float(iou_thr))
            class_aps.append(ap)

            if abs(iou_thr - 0.5) < 1e-9:
                ap50_per_class[cls] = ap
                total_tp += tp
                total_fp += fp
                total_fn += fn

        ap_by_iou.append(class_aps)

        # AP_small per class at IoU=0.5:0.95
        s_preds, s_gts = _collect_class_data(
            predictions,
            ground_truths,
            cls,
            area_range=(0.0, small_max_area),
        )
        if sum(len(v) for v in s_gts.values()) > 0:
            small_class_aps = []
            for iou_thr in iou_thresholds:
                ap_s, _, _, _, _, _ = _evaluate_class_at_iou(s_preds, s_gts, float(iou_thr))
                small_class_aps.append(ap_s)
            aps_small.append(float(np.mean(small_class_aps)))

    if len(ap_by_iou) == 0:
        return {
            'map_50': 0.0,
            'map_50_95': 0.0,
            'ap_small': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'per_class_ap50': {}
        }

    ap_by_iou = np.array(ap_by_iou, dtype=np.float32)  # [num_cls_valid, num_iou]
    map_50 = float(np.mean(ap_by_iou[:, 0]))
    map_50_95 = float(np.mean(ap_by_iou))

    precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0

    ap_small = float(np.mean(aps_small)) if len(aps_small) > 0 else 0.0

    return {
        'map_50': map_50,
        'map_50_95': map_50_95,
        'ap_small': ap_small,
        'precision': precision,
        'recall': recall,
        'tp': int(total_tp),
        'fp': int(total_fp),
        'fn': int(total_fn),
        'per_class_ap50': ap50_per_class,
    }


def compute_F1(precision, recall):
    """Compute F1 score."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


class MetricsCalculator:
    """Lightweight wrapper around evaluate_map for compatibility."""

    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.predictions = []
        self.ground_truths = []

    def update(self, predictions, ground_truths, iou_threshold=0.5):
        self.predictions.extend(_normalize_predictions(predictions))
        self.ground_truths.extend(_normalize_ground_truths(ground_truths))

    def compute_metrics(self):
        metrics = evaluate_map(
            self.predictions,
            self.ground_truths,
            iou_threshold=0.5,
            num_classes=self.num_classes,
        )
        f1 = compute_F1(metrics['precision'], metrics['recall'])

        return {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': f1,
            'mAP': metrics['map_50_95'],
            'map_50': metrics['map_50'],
            'ap_small': metrics.get('ap_small', 0.0),
            'per_class_ap50': metrics.get('per_class_ap50', {}),
        }
