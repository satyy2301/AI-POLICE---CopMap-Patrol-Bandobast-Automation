import io
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T

# COCO category names (80 classes) — truncated here to the common ones, index 1 is 'person'
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class Detector:
    def __init__(self, device: str = 'cpu', score_thresh: float = 0.6):
        self.device = torch.device(device)
        self.score_thresh = score_thresh
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, pil_image: Image.Image):
        # torchvision expects tensors in [C,H,W] 0..1
        transform = T.Compose([T.ToTensor()])
        return transform(pil_image).to(self.device)

    def detect_pil(self, pil_image: Image.Image):
        tensor = self._preprocess(pil_image)
        with torch.no_grad():
            outputs = self.model([tensor])[0]

        detections = []
        person_count = 0
        for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
            score_f = float(score.cpu().item())
            if score_f < self.score_thresh:
                continue
            label_i = int(label.cpu().item())
            name = COCO_INSTANCE_CATEGORY_NAMES[label_i] if label_i < len(COCO_INSTANCE_CATEGORY_NAMES) else str(label_i)
            bbox = [float(x) for x in box.cpu().tolist()]
            detections.append({'label': name, 'score': score_f, 'bbox': bbox})
            if name == 'person':
                person_count += 1

        return {'detections': detections, 'person_count': person_count}
