import os
import json
import argparse
import torch
from torchvision import transforms
from PIL import Image
from torchvision.ops import nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import subprocess
import sys
from yolotiny_utils import apply_nms, apply_nms_per_class, visualize_predictions


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    if args.run_infer_first:
        if not args.infer_script:
            raise ValueError("--run-infer-first gesetzt, aber --infer-script fehlt.")
        cmd = [sys.executable, args.infer_script]
        if args.infer_image:
            cmd += ["--image", args.infer_image]
        if args.infer_model:
            cmd += ["--model", args.infer_model]
        if args.infer_outdir:
            cmd += ["--outdir", args.infer_outdir]
        print("Starte Inference:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    pred = torch.load(args.predictions, weights_only=True)

    boxes = pred["boxes_xyxy"]
    scores = pred["scores"]
    labels = pred["labels"]

    image_path = pred.get("image_path", args.image)
    img = Image.open(image_path).convert("RGB")

    # Originalgröße
    original_w, original_h = img.size
    print(f"Originalbildgröße: {original_w}x{original_h}")

    # Inputgröße (aus pred oder fallback aus args)
    input_size = pred.get("input_size", args.input_size)
    scale_x = original_w / input_size
    scale_y = original_h / input_size

    # COCO class names (80)
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

    # NMS auswählen
    if args.nms_mode == "per_class":
        final_boxes, final_scores, final_labels = apply_nms_per_class(
            boxes, scores, labels, iou_threshold=args.iou, conf_threshold=args.conf
        )
    else:
        final_boxes, final_scores, final_labels = apply_nms(
            boxes, scores, labels, iou_threshold=args.iou, conf_threshold=args.conf
        )

    # Auf Originalbildgröße skalieren
    final_boxes_scaled = final_boxes.clone()
    final_boxes_scaled[:, [0, 2]] *= scale_x
    final_boxes_scaled[:, [1, 3]] *= scale_y

    # Visualisierung
    img_out_path = os.path.join(args.outdir, args.image_out)
    dets = visualize_predictions(
        img, final_boxes_scaled, final_scores, final_labels, class_names, img_out_path
    )

    # JSON speichern
    det_out_path = os.path.join(args.outdir, args.detections_out)
    with open(det_out_path, "w") as f:
        json.dump(dets, f, indent=2)


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
        default=0.5,
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

    # Optional anderes Skript vorher ausführen
    parser.add_argument("--run-infer-first", action="store_true",
                        help="Vorher ein Inference-Skript starten")
    parser.add_argument("--infer-script", type=str, default=None,
                        help="Pfad zum Inference-Skript (z.B. ./yolotiny_core.py)")
    parser.add_argument("--infer-image", type=str, default=None,
                        help="--image für Inference-Skript")
    parser.add_argument("--infer-model", type=str, default=None,
                        help="--model für Inference-Skript")
    parser.add_argument("--infer-outdir", type=str, default=None,
                        help="--outdir für Inference-Skript")

    args = parser.parse_args()
    main(args)