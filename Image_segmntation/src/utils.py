import cv2
import numpy as np

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
]


def apply_mask(image, mask):
    binary = (mask > 0.5).astype(np.uint8)

    result = image.copy()
    result[binary == 0] = 0

    return result


def draw_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    binary = (mask > 0.5).astype(np.uint8)

    overlay = image.copy()
    overlay[binary == 1] = color

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_boxes(image, boxes, labels, scores):
    output = image.copy()

    for box, label, score in zip(boxes, labels, scores):

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if int(label) < len(COCO_CLASSES):
            name = COCO_CLASSES[int(label)]
        else:
            name = str(int(label))

        text = f"{name} {score:.2f}"

        cv2.putText(
            output,
            text,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return output


def save_image(path, image):
    cv2.imwrite(path, image)


def refine_mask(image, mask):
    binary = (mask > 0.5).astype(np.uint8)

    grabcut_mask = np.where(binary == 1, cv2.GC_PR_FGD, cv2.GC_BGD).astype("uint8")

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (1, 1, image.shape[1] - 2, image.shape[0] - 2)

    cv2.grabCut(
        image,
        grabcut_mask,
        rect,
        bgdModel,
        fgdModel,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    refined = np.where(
        (grabcut_mask == cv2.GC_FGD) |
        (grabcut_mask == cv2.GC_PR_FGD),
        1,
        0
    ).astype("uint8")

    result = image.copy()
    result[refined == 0] = 0

    return result