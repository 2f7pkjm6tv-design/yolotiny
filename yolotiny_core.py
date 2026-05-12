import os
import json
import argparse
import torch
from torchvision import transforms
from PIL import Image
from onnx2torch import convert

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

def main():
    '''
    Output-format: boxes = x1y1x2y2, scores, labels
    '''

    parser = argparse.ArgumentParser(description="YOLOv4-tiny inference")
    parser.add_argument(
        "--image",
        type=str,
        default="./test_images/000000324158.jpg",
        help="Pfad zum Testbild"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./yolov4-tiny.onnx",
        help="Pfad zum ONNX-Modell"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outputs",
        help="Output-Verzeichnis"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    model = convert(args.model)
    model.eval() # Layer in evaluation mode, damit stabile Outputs (z.B. BatchNorm) und kein Dropout

    img = Image.open(args.image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(img)
    input_batch = input_tensor.unsqueeze(0)

    print('Input_batch.shape:', input_batch.shape)

    with torch.no_grad(): # deaktiviert Gradientenberechnung für Inference(Model wird nur benutzt kein Training), effizienter und weniger Speicherverbrauch
        raw_outputs = model(input_batch)

    print('raw_outputs:', [out.shape for out in raw_outputs])

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

    pt_path = os.path.join(args.outdir, "predictions.pt")
    torch.save(
        {
            "image_path": args.image,
            "boxes_xyxy": boxes.cpu(),
            "scores": scores.cpu(),
            "labels": labels.cpu(),
        },
        pt_path
    )


    json_path = os.path.join(args.outdir, "predictions.json")
    json_data = {
        "image_path": args.image,
        "boxes_xyxy": boxes.cpu().tolist(),
        "scores": scores.cpu().tolist(),
        "labels": labels.cpu().tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Gespeichert: {pt_path}")
    print(f"Gespeichert: {json_path}")

if __name__ == "__main__":
    main()