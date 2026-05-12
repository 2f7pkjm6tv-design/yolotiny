from PIL import Image
import torch
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from yolotiny_utils import precision_recall_f1, iou_xywh_torch, confusion_matrix_torch_xywh, raw_core_model, match_detections_xywh_category_id, apply_nms, compute_map
from matplotlib import patches
import argparse
import os
from onnx2torch import convert
from pathlib import Path

#evaluation_loop zur mAP Berechnung
#main laedt zeigt Ground Truth und Predictions in einem Bild


def evaluation_loop(args):
    coco = COCO(args.GroundTruthPath)
    filename_to_id = {img['file_name']: img['id'] for img in coco.dataset['images']}
    model = convert(args.model)
    model.eval()
    img_dir = Path("../cocoapi/images")

    COCO80_TO_CATID = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90
    ]

    all_results = []

    count = 0
    for img_path in sorted(img_dir.glob("*.jpg")):
        img    = Image.open(img_path).convert("RGB")
        img_id = filename_to_id.get(img_path.name, None)
        anns   = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        # 1. Inferenz
        boxes, scores, labels = raw_core_model(img_path, model)

        # 2. xyxy → xywh
        boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]

        # 3. Label-Mapping
        labels = torch.tensor(
            [COCO80_TO_CATID[int(l)] for l in labels],
            dtype=labels.dtype, device=labels.device
        )

        # 4. Score-sortieren
        mask   = torch.argsort(scores, descending=True)
        boxes  = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # 5. Confidence-Filter
        conf_mask = scores > 0.01
        boxes     = boxes[conf_mask]
        scores    = scores[conf_mask]
        labels    = labels[conf_mask]

        # 6. Auf Originalbild skalieren
        original_w, original_h = img.size
        scale_x = original_w / args.input_size
        scale_y = original_h / args.input_size
        final_boxes_scaled = boxes.clone()
        final_boxes_scaled[:, [0, 2]] *= scale_x
        final_boxes_scaled[:, [1, 3]] *= scale_y

        # 7. Matching
        tp, fp, num_gt, num_gt_per_class, _ = match_detections_xywh_category_id(
            final_boxes_scaled, labels, scores, anns, iou_threshold=args.iou
        )

        all_results.append({
            'tp':               tp,
            'fp':               fp,
            'scores':           scores,
            'labels':           labels,
            'num_gt_per_class': num_gt_per_class,
        })
        count += 1

        if count >= 10:  
            break

    # 8. mAP berechnen
    results = compute_map(all_results)
    print(f"\nmAP@{args.iou:.2f}: {results['mAP']:.4f}")
    for cat_id, ap in sorted(results['AP_per_class'].items()):
        print(f"  class {cat_id:3d}: AP = {ap:.4f}")

    print('number of pictures evaluated:', count)

    return results
# mAp funktion verstehen und evaluation_loop


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # Ground Truth laden
    coco = COCO(args.GroundTruthPath)

    filename_to_id = {img['file_name']: img['id'] for img in coco.dataset['images']}
    img_id = filename_to_id.get(os.path.basename(args.image), None)
    
    if img_id is None:
        print(f"Fehler: Bild {args.image} nicht in COCO Ground Truth gefunden.")
        return
    
    img_info = coco.loadImgs(img_id)[0]
    img_path = f'../cocoapi/images/{img_info["file_name"]}'
    img = Image.open(img_path).convert("RGB")

    anns_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(anns_ids)
    gt_boxes = torch.tensor([ann['bbox'] for ann in anns], dtype=torch.float32)  # [x, y, w, h]


    original_w, original_h = img.size
    print(f"Originalbildgröße: {original_w}x{original_h}")


    class_names = [
        "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
        "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
        "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
        "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
        "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",
        "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
        "teddy bear","hair drier","toothbrush"
    ]
    cat_mapping = {cat['id']: cat['name'] for cat in coco.dataset['categories']}

    # Predictions laden
    pred = torch.load(args.predictions, weights_only=True)

    if pred.get("image_path", None) != args.image:
        print(f"Warnung: image_path in predictions.pt ({pred.get('image_path', None)}) stimmt nicht mit args.image ({args.image}) überein.")

    boxes = pred["boxes_xyxy"].clone()
    boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2] # [x1, y1, x2, y2] -> [x1, y1, w, h]
    scores = pred["scores"].clone()
    labels = pred["labels"].clone()
   
    mask = torch.argsort(scores, descending=True)

    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    input_size = pred.get("input_size", args.input_size)
    scale_x = original_w / input_size
    scale_y = original_h / input_size

    final_boxes_scaled = boxes.clone()
    final_boxes_scaled[:, [0, 2]] *= scale_x
    final_boxes_scaled[:, [1, 3]] *= scale_y

    #nur fuer Visualisierung 
    mask_show = scores > 0.25
    final_boxes_scaled = final_boxes_scaled[mask_show]
    final_labels = labels[mask_show]
    final_scores = scores[mask_show]

    final_labels = [
    class_names[int(l.item())] if 0 <= int(l.item()) < len(class_names) else f"unknown_{int(l.item())}"
    for l in final_labels
    ]
    
    iou_torch = iou_xywh_torch(gt_boxes, final_boxes_scaled)
    print('len IoU matrix:', iou_torch.shape, '\nlen gt_boxes:', len(gt_boxes), '\nlen final_boxes_scaled:', len(final_boxes_scaled))
    print('iou-torch:', iou_torch)

    cf_matrix = confusion_matrix_torch_xywh(final_boxes_scaled, final_labels, anns, cat_mapping)
    print('cf_matrix', cf_matrix)

    precision, recall, f1 = precision_recall_f1(cf_matrix)
    print('precision', precision, 'recall', recall, 'f1', f1)


    # nur für den Plot
    final_boxes, final_scores, final_labels = apply_nms(
        pred["boxes_xyxy"], pred["scores"], pred["labels"], iou_threshold=args.iou)
    final_boxes_scaled = final_boxes.clone()
    final_boxes_scaled[:, 2:] = final_boxes_scaled[:, 2:] - final_boxes_scaled[:, :2] 
    final_boxes_scaled[:, [0, 2]] *= scale_x
    final_boxes_scaled[:, [1, 3]] *= scale_y
    final_labels = [class_names[int(l.item())] if 0 <= int(l.item()) < len(class_names) else f"unknown_{int(l.item())}"
    for l in final_labels]

    if args.Visualize:
        # Visualisierung
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Links: Ground Truth
        ax1.imshow(img)
        ax1.set_title("Ground Truth (COCO)")
        for ann in anns:
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax1.add_patch(rect)
            label = cat_mapping[ann['category_id']]
            ax1.text(x, y-5, label, color='white', fontsize=10, backgroundcolor='black')
        ax1.axis('off')

        # Rechts: YOLO Detections (skaliert)
        ax2.imshow(img)
        ax2.set_title("YOLO Detections (skaliert)")
        for i, box in enumerate(final_boxes_scaled):
            x, y, w, h = box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
            ax2.add_patch(rect)
            ax2.text(x, y-5, f"{final_labels[i]}: {final_scores[i]:.2f}", 
                    color='white', fontsize=10, backgroundcolor='black')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig("./outputs/comparison.png", bbox_inches='tight', pad_inches=0)
        plt.show() 

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    for ann in anns:
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
            label = cat_mapping[ann['category_id']]
            plt.text(x, y-5, label, color='white', fontsize=10, backgroundcolor='black')
    for i, box in enumerate(final_boxes_scaled):
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(x, y-5, f"{final_labels[i]}: {final_scores[i]:.2f}", 
                color='white', fontsize=10, backgroundcolor='black')
    plt.axis('off')
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO predictions from predictions.pt")

    parser.add_argument(
        "--predictions",
        type=str,
        default="./outputs/predictions.pt",
        help="Pfad zur predictions.pt"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="./test_images/000000324158.jpg",
        help="Fallback-Bild, falls image_path nicht in predictions.pt enthalten ist"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outputs",
        help="Output-Verzeichnis"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=416,
        help="Fallback Inputgröße des Modells (wenn nicht in predictions.pt vorhanden)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.25,
        help="IoU threshold für NMS"
    )
    parser.add_argument(
        "--nms-mode",
        type=str,
        choices=["global", "per_class"],
        default="global",
        help="NMS-Modus: global oder per_class"
    )
    parser.add_argument(
        "--image-out",
        type=str,
        default="img_predictions.png",
        help="Dateiname für visualisiertes Bild"
    )
    parser.add_argument(
        "--detections-out",
        type=str,
        default="detections.json",
        help="Dateiname für detections JSON"
    )
    parser.add_argument(
        "--GroundTruthPath",
        type=str,
        default="../cocoapi/annotations/instances_val2017.json",
        help="Pfad zur Ground Truth COCO JSON"
    )

    parser.add_argument(
        "--Visualize",
        type=bool,
        default=False,
        help="Visualisierung mit Speichern"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./yolov4-tiny.onnx",
        help="Pfad zum ONNX-Modell"
    )

    args = parser.parse_args()
    main(args)
    evaluation_loop(args)
