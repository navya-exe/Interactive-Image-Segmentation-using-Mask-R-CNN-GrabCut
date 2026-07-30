import cv2
import torch
import torchvision
from torchvision.transforms import functional as F


def load_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    return model


def segment_image(image_path, model, threshold=0.5):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(rgb)

    with torch.no_grad():
        prediction = model([tensor])[0]

    masks = prediction["masks"]
    scores = prediction["scores"]
    boxes = prediction["boxes"]
    labels = prediction["labels"]

    valid_masks = masks[scores > threshold]
    valid_boxes = boxes[scores > threshold]
    valid_labels = labels[scores > threshold]
    valid_scores = scores[scores > threshold]

    return image, valid_masks, valid_boxes, valid_labels, valid_scores