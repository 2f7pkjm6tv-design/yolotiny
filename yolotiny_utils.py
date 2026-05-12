import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms
from PIL import Image
import torch
import numpy as np
from collections import defaultdict

'''
Core-Elemente für YOLOv4-tiny Inferenz
'''
def decode_yolo_output(output, anchors, num_classes, stride, device="cpu"):
    batch, grid_h, grid_w, channels = output.shape
    num_anchors = len(anchors)
    bbox_attrs = 5 + num_classes

    prediction = output.view(batch, grid_h, grid_w, num_anchors, bbox_attrs)

    tx = prediction[..., 0]
    ty = prediction[..., 1]
    tw = prediction[..., 2]
    th = prediction[..., 3]
    obj_conf = torch.sigmoid(prediction[..., 4])
    class_scores = torch.sigmoid(prediction[..., 5:])

    grid_x = torch.arange(grid_w, device=device).repeat(grid_h, 1).view([1, grid_h, grid_w, 1])
    grid_y = torch.arange(grid_h, device=device).repeat(grid_w, 1).t().view([1, grid_h, grid_w, 1])

    bx = (torch.sigmoid(tx) + grid_x) * stride
    by = (torch.sigmoid(ty) + grid_y) * stride

    anchors = torch.tensor(anchors, device=device, dtype=torch.float32)
    pw = anchors[:, 0].view(1, 1, 1, num_anchors)
    ph = anchors[:, 1].view(1, 1, 1, num_anchors)

    bw = torch.exp(tw) * pw
    bh = torch.exp(th) * ph

    x1 = bx - bw / 2
    y1 = by - bh / 2
    x2 = bx + bw / 2
    y2 = by + bh / 2

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    x2 = x2.reshape(-1)
    y2 = y2.reshape(-1)
    obj_conf = obj_conf.reshape(-1)
    class_scores = class_scores.reshape(-1, num_classes)

    scores, labels = torch.max(class_scores, dim=-1)
    scores = scores * obj_conf

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return boxes, scores, labels


def raw_core_model(img_path, model):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        ])
    input_tensor = transform(img)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad(): # deaktiviert Gradientenberechnung für Inference(Model wird nur benutzt kein Training), effizienter und weniger Speicherverbrauch
        raw_outputs = model(input_batch)
        
    anchors_per_scale = [
        [(116, 90), (156, 198), (373, 326)],  # 20x20
        [(30, 61), (62, 45), (59, 119)]       # 40x40
        ]

    strides = [32, 16]

    all_boxes, all_scores, all_labels = [], [], []

    for out, anchors, stride in zip(raw_outputs, anchors_per_scale, strides):
        boxes, scores, labels = decode_yolo_output(out, anchors, 80, stride)
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    return boxes, scores, labels




'''
Visualisierungs- und Evaluierungsfunktionen für YOLOv4-tiny Inferenz
'''
def apply_nms(boxes, scores, labels, iou_threshold=0.5, conf_threshold=0.25):
    keep = scores > conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    keep_idx = nms(boxes, scores, iou_threshold)
    return boxes[keep_idx], scores[keep_idx], labels[keep_idx]


def apply_nms_per_class(boxes, scores, labels, iou_threshold=0.5, conf_threshold=0.25):
    keep_conf = scores > conf_threshold
    boxes, scores, labels = boxes[keep_conf], scores[keep_conf], labels[keep_conf]

    keep_indices = []
    for cls in labels.unique():
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        cls_keep = nms(cls_boxes, cls_scores, iou_threshold)

        # lokale Indizes -> globale Indizes
        global_idx = torch.where(cls_mask)[0][cls_keep]
        keep_indices.append(global_idx)

    if len(keep_indices) == 0:
        return boxes[:0], scores[:0], labels[:0]

    keep_indices = torch.cat(keep_indices)
    keep_indices = keep_indices[scores[keep_indices].argsort(descending=True)]

    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]


def visualize_predictions(img, boxes, scores, labels, class_names, save_path):
    if not isinstance(img, torch.Tensor):
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # [1,C,H,W]
    else:
        img_tensor = img

    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)  # [C,H,W]

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_np)

    detections = []

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        width, height = x2 - x1, y2 - y1

        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)

        label_idx = int(label.item()) if torch.is_tensor(label) else int(label)
        class_text = class_names[label_idx] if class_names else str(label_idx)
        text = f"{class_text}: {float(score):.2f}"

        ax.text(
            x1, y1 - 5, text,
            color="white", fontsize=10, backgroundcolor="black"
        )

        detections.append({
            "class": class_text,
            "confidence": float(score.item()),
            "bbox": [x1, y1, width, height]
        })

    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


    return detections


'''
Evaluierungsfunktionen für YOLOv4-tiny Inferenz
'''

def iou_xywh_torch(gt_boxes, det_boxes):
    """
    IoU-Matrix zwischen GT-Boxen und Det-Boxen.
    
    Args:
        gt_boxes:  Tensor [N, 4] im Format [x, y, w, h]
        det_boxes: Tensor [M, 4] im Format [x, y, w, h]
    
    Returns:
        iou: Tensor [N, M] mit IoU-Werten
    """
    if not isinstance(gt_boxes, torch.Tensor):
        gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float32)
    else:
        gt_boxes = gt_boxes.float()

    if not isinstance(det_boxes, torch.Tensor):
        det_boxes = torch.as_tensor(det_boxes, dtype=torch.float32)
    else:
        det_boxes = det_boxes.float()

    # Leere Eingaben abfangen
    if gt_boxes.numel() == 0 or det_boxes.numel() == 0:
        return torch.zeros((gt_boxes.shape[0], det_boxes.shape[0]), dtype=torch.float32)

    # xywh -> xyxy
    gt_x1y1 = gt_boxes[:, :2]
    gt_x2y2 = gt_boxes[:, :2] + gt_boxes[:, 2:]
    det_x1y1 = det_boxes[:, :2]
    det_x2y2 = det_boxes[:, :2] + det_boxes[:, 2:]

    # Broadcasting:
    # gt: [N,1,2], det: [1,M,2]
    inter_x1y1 = torch.maximum(gt_x1y1[:, None, :], det_x1y1[None, :, :])
    inter_x2y2 = torch.minimum(gt_x2y2[:, None, :], det_x2y2[None, :, :])

    inter_wh = (inter_x2y2 - inter_x1y1).clamp(min=0)   # [N,M,2]
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]     # [N,M]

    gt_area = (gt_boxes[:, 2] * gt_boxes[:, 3])[:, None]     # [N,1]
    det_area = (det_boxes[:, 2] * det_boxes[:, 3])[None, :]  # [1,M]

    union = gt_area + det_area - inter_area
    iou = torch.where(union > 0, inter_area / union, torch.zeros_like(union))

    return iou

import torch

def confusion_matrix_torch_xywh(det_boxes, det_labels, anns, cat_mapping, iou_threshold=0.5):
    """
    Torch-Version, orientiert an deiner confusion_matrix_np:
    - det_boxes: Tensor [N_det,4] im Format [x,y,w,h]
    - det_labels: Tensor/Array [N_det] (Labels müssen zu gt_labels passen!)
    - anns: COCO annotations (list of dicts), ann['bbox'] ist [x,y,w,h]
    - cat_mapping: dict {category_id: name} (wie bei dir)
    
    Returns:
        matrix: Tensor [[TP, FP],
                        [FN,  0]]
    """
    # --- GT aus anns bauen ---
    if len(anns) == 0:
        # Keine GT: alles sind FP
        n_fp = int(len(det_boxes))
        return torch.tensor([[0, n_fp], [0, 0]], dtype=torch.int64)

    gt_boxes = torch.as_tensor([ann["bbox"] for ann in anns], dtype=torch.float32)
    gt_labels = [cat_mapping[ann["category_id"]] for ann in anns]

    # det_labels ggf. in Python-Liste konvertieren (für String-Vergleich)
    if isinstance(det_labels, torch.Tensor):
        det_labels_list = det_labels.detach().cpu().tolist()
    else:
        det_labels_list = list(det_labels)

    # det_boxes zu Tensor
    if not isinstance(det_boxes, torch.Tensor):
        det_boxes = torch.as_tensor(det_boxes, dtype=torch.float32)
    else:
        det_boxes = det_boxes.float()

    n_gt = gt_boxes.shape[0]
    n_det = det_boxes.shape[0]

    if n_det == 0:
        return torch.tensor([[0, 0], [n_gt, 0]], dtype=torch.int64)

    iou_matrix = iou_xywh_torch(gt_boxes, det_boxes)  # [n_gt, n_det]

    matched_anns = set()

    TP = 0
    FP = 0

    for det_idx in range(n_det):
        det_label = det_labels_list[det_idx]

        best_iou = -1.0
        best_ann_idx = -1

        for ann_idx in range(n_gt):
            if ann_idx in matched_anns:
                continue

            if det_label == gt_labels[ann_idx]:
                iou = float(iou_matrix[ann_idx, det_idx].item())
                if iou > best_iou:
                    best_iou = iou
                    best_ann_idx = ann_idx

        if best_iou >= iou_threshold:
            TP += 1
            matched_anns.add(best_ann_idx)
        else:
            FP += 1

    FN = n_gt - len(matched_anns)

    matrix = torch.tensor([[TP, FP], [FN, 0]], dtype=torch.int64)
    return matrix



def precision_recall_f1(matrix):
    TP, FP = matrix[0]
    FN = matrix[1][0]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score


def match_detections_xywh_category_id(
    det_boxes: torch.Tensor,
    det_category_ids: torch.Tensor | list,
    det_scores: torch.Tensor,
    anns: list[dict],
    iou_threshold: float = 0.5,
):
    """
    Greedy Matching pro Bild für Object Detection (COCO-style), wenn det_labels bereits COCO category_id sind.

    Inputs
    ------
    det_boxes: Tensor [N_det, 4] in COCO xywh: [x, y, w, h] (auf ORIGINALBILD skaliert)
    det_category_ids: Tensor [N_det] oder list[int] (COCO category_id)
    det_scores: Tensor [N_det] (wird hier nicht genutzt außer zur Dokumentation; Sortierung sollte extern passieren)
    anns: list[dict] COCO annotations für GENAU dieses Bild
          - ann["bbox"] = [x,y,w,h]
          - ann["category_id"] = int
    iou_threshold: float

    Returns
    -------
    tp_flags:  Tensor [N_det] (0/1) in DER REIHENFOLGE der Detections (idealerweise score-sorted)
    fp_flags:  Tensor [N_det] (0/1)
    num_gt_total: int (Anzahl aller GT im Bild)
    num_gt_per_cat: dict[int,int] (GT-Anzahl pro category_id im Bild)
    matched_gt_idx: Tensor [N_det] mit gematchtem GT-Index oder -1 (Debug)
    """

    device = det_boxes.device if isinstance(det_boxes, torch.Tensor) else det_scores.device

    # --- normalize inputs ---
    if not isinstance(det_boxes, torch.Tensor):
        det_boxes = torch.as_tensor(det_boxes, dtype=torch.float32, device=device)
    else:
        det_boxes = det_boxes.to(device=device, dtype=torch.float32)

    if not isinstance(det_scores, torch.Tensor):
        det_scores = torch.as_tensor(det_scores, dtype=torch.float32, device=device)
    else:
        det_scores = det_scores.to(device=device, dtype=torch.float32)

    if isinstance(det_category_ids, torch.Tensor):
        det_cats = det_category_ids.detach().cpu().tolist()
    else:
        det_cats = list(det_category_ids)

    n_det = int(det_boxes.shape[0])

    # --- GT ---
    num_gt_total = len(anns)
    if num_gt_total == 0:
        tp_flags = torch.zeros((n_det,), dtype=torch.int64, device=device)
        fp_flags = torch.ones((n_det,), dtype=torch.int64, device=device) if n_det > 0 else torch.zeros((0,), dtype=torch.int64, device=device)
        matched_gt_idx = torch.full((n_det,), -1, dtype=torch.int64, device=device)
        return tp_flags, fp_flags, 0, {}, matched_gt_idx

    gt_boxes = torch.as_tensor([ann["bbox"] for ann in anns], dtype=torch.float32, device=device)
    gt_cats = [int(ann["category_id"]) for ann in anns]

    num_gt_per_cat: dict[int, int] = {}
    for c in gt_cats:
        num_gt_per_cat[c] = num_gt_per_cat.get(c, 0) + 1

    if n_det == 0:
        tp_flags = torch.zeros((0,), dtype=torch.int64, device=device)
        fp_flags = torch.zeros((0,), dtype=torch.int64, device=device)
        matched_gt_idx = torch.zeros((0,), dtype=torch.int64, device=device)
        return tp_flags, fp_flags, num_gt_total, num_gt_per_cat, matched_gt_idx

    # --- IoU ---
    # assumes you already have: iou_xywh_torch(gt_boxes, det_boxes) -> [G,D]
    iou_matrix = iou_xywh_torch(gt_boxes, det_boxes)

    # --- greedy match ---
    matched_gt = torch.zeros((num_gt_total,), dtype=torch.bool, device=device)

    tp_flags = torch.zeros((n_det,), dtype=torch.int64, device=device)
    fp_flags = torch.zeros((n_det,), dtype=torch.int64, device=device)
    matched_gt_idx = torch.full((n_det,), -1, dtype=torch.int64, device=device)

    for d in range(n_det):
        d_cat = int(det_cats[d])

        best_iou = -1.0
        best_g = -1

        for g in range(num_gt_total):
            if bool(matched_gt[g]):
                continue
            if gt_cats[g] != d_cat:
                continue

            iou = float(iou_matrix[g, d].item())
            if iou > best_iou:
                best_iou = iou
                best_g = g

        if best_g != -1 and best_iou >= iou_threshold:
            tp_flags[d] = 1
            matched_gt[best_g] = True
            matched_gt_idx[d] = best_g
        else:
            fp_flags[d] = 1

    return tp_flags, fp_flags, num_gt_total, num_gt_per_cat, matched_gt_idx

def _compute_ap_101(precision: np.ndarray, recall: np.ndarray) -> float:
    """101-Punkt Interpolation (COCO-Standard)."""
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recall >= t
        ap += np.max(precision[mask]) if mask.any() else 0.0
    return ap / 101

def compute_map(all_results: list[dict], iou_threshold: float = 0.5) -> dict:
    per_class = defaultdict(lambda: {'tp': [], 'fp': [], 'scores': []})
    num_gt_per_class = defaultdict(int)

    for res in all_results:
        tp     = res['tp'].cpu().numpy()
        fp     = res['fp'].cpu().numpy()
        scores = res['scores'].cpu().numpy()
        labels = res['labels'].cpu().numpy()

        for d in range(len(tp)):
            cat = int(labels[d])
            per_class[cat]['tp'].append(tp[d])
            per_class[cat]['fp'].append(fp[d])
            per_class[cat]['scores'].append(scores[d])

        for cat, n in res['num_gt_per_class'].items():
            num_gt_per_class[cat] += n

    ap_per_class = {}
    for cat, data in per_class.items():
        n_gt = num_gt_per_class.get(cat, 0)
        if n_gt == 0:
            continue

        scores_arr = np.array(data['scores'])
        tp_arr     = np.array(data['tp'])
        fp_arr     = np.array(data['fp'])

        order  = np.argsort(-scores_arr)
        tp_arr = tp_arr[order]
        fp_arr = fp_arr[order]

        cum_tp    = np.cumsum(tp_arr)
        cum_fp    = np.cumsum(fp_arr)
        precision = cum_tp / (cum_tp + cum_fp + 1e-9)
        recall    = cum_tp / n_gt

        ap_per_class[cat] = _compute_ap_101(precision, recall)

    mAP = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    return {'mAP': mAP, 'AP_per_class': ap_per_class}






#_______Ueberfluessige Funktionen________

'''
def intersection_over_union(gt_box, pred_box):
    #Berechnet die Intersection over Union (IoU) zwischen zwei Bounding Boxes.
    #bbox = [x1, y1, width, height]
    
    # Schnittpunkt-Ecken berechnen
    inter_x_min = max(gt_box[0], pred_box[0])
    inter_y_min = max(gt_box[1], pred_box[1])
    inter_x_max = min(gt_box[0] + gt_box[2], pred_box[0] + pred_box[2])
    inter_y_max = min(gt_box[1] + gt_box[3], pred_box[1] + pred_box[3])
    
    # Schnittfläche 
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)


    intersection = inter_width * inter_height
    
    union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection
    
    iou = intersection / union if union > 0 else 0
    
    return iou, intersection, union




def confusion_matrix(detections, anns, cat_mapping, iou_threshold=0.5):
    categories = list(set(cat_mapping.values()))
    matrix = {cat: {'TP': 0, 'FP': 0, 'FN': 0} for cat in categories}
    
    # ← TRACKING: Welche Annotations wurden bereits gematcht?
    matched_anns = set()
    matched_dets = set()
    
    # Erste Phase: Matche Detections zu Ground Truths
    for det_idx, det in enumerate(detections):
        det_bbox = det['bbox']
        det_label = det['class']
        
        best_iou = 0
        best_ann_idx = -1
        
        # Finde BESTE passende Ground Truth
        for ann_idx, ann in enumerate(anns):
            if ann_idx in matched_anns:  # ← Skip bereits gematcht!
                continue
            
            gt_bbox = ann['bbox']
            gt_label = cat_mapping[ann['category_id']]
            
            if det_label == gt_label:
                iou = intersection_over_union_np(gt_bbox, det_bbox)[0]
                if iou > best_iou:  # ← Nur BESTE!
                    best_iou = iou
                    best_ann_idx = ann_idx
        
        # Wenn Match gefunden
        if best_iou >= iou_threshold:
            matrix[det_label]['TP'] += 1
            matched_anns.add(best_ann_idx)
            matched_dets.add(det_idx)
        else:
            # ← Nur FP, wenn KEINE gute Detection
            if det_label in matrix:
                matrix[det_label]['FP'] += 1
    
    # Zweite Phase: Unmgematche Ground Truths sind FN
    for ann_idx, ann in enumerate(anns):
        if ann_idx not in matched_anns:
            gt_label = cat_mapping[ann['category_id']]
            matrix[gt_label]['FN'] += 1
    
    return matrix



def average_precision(boxes, labels, scores, anns, cat_mapping, conf_start=0.25, conf_end=0.75, conf_steps=5, iou_threshold=0.5, plot=False):

    is_sorted_desc = torch.all(scores[:-1] >= scores[1:])  # du willst ja absteigend

    if not bool(is_sorted_desc):
        order = torch.argsort(scores, descending=True)
        boxes = boxes[order]
        labels = labels[order]
        scores = scores[order]

    conf = torch.linspace(
    conf_start,
    conf_end,
    conf_steps
    )
    print('len(conf)', len(conf), 'conf', conf)

    precisions, recalls = [], []

    for ct in conf:
        keep_idx = scores > ct  # Bool-Tensor [N]

        boxes_kept = boxes[keep_idx]
        scores_kept = scores[keep_idx]

        keep = keep_idx.nonzero(as_tuple=True)[0].tolist()  # Indizes als Python-Liste
        labels_kept = [labels[i] for i in keep]

        # Confusion Matrix (Torch)
        # WICHTIG: Diese Funktion muss zum Boxformat passen (xywh vs xyxy)!
        cf_matrix = confusion_matrix_torch_xywh(
            boxes_kept, labels_kept, anns, cat_mapping, iou_threshold=iou_threshold
        )
        print("cf_matrix", cf_matrix)

        precision, recall, _ = precision_recall_f1(cf_matrix)
        precisions.append(float(precision))
        recalls.append(float(recall))
        
    
    precisions = torch.tensor(precisions)
    recalls = torch.tensor(recalls)

    ap = torch.trapezoid(recalls, precisions)

    print('len(precisions)', len(precisions), 'len(recalls)', len(recalls)) 
    print('precisions', precisions)
    print('recalls', recalls)
    print('AP', ap)

    if plot: 
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='o')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid()
        plt.show()


    return 0
'''