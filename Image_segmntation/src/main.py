from abc import ABC, abstractmethod
import builtins

from segmentation import load_model, segment_image
from utils import apply_mask, draw_mask, draw_boxes, save_image, refine_mask

class PrintBase(ABC):
    @abstractmethod
    def info(self, message: str) -> None:
        pass

    @abstractmethod
    def warning(self, message: str) -> None:
        pass

    @abstractmethod
    def error(self, message: str) -> None:
        pass

    @abstractmethod
    def debug(self, message: str) -> None:
        pass

class print(PrintBase):
    def info(self, message: str) -> None:
        builtins.print(f"[INFO] {message}")

    def warning(self, message: str) -> None:
        builtins.print(f"[WARNING] {message}")

    def error(self, message: str) -> None:
        builtins.print(f"[ERROR] {message}")

    def debug(self, message: str) -> None:
        builtins.print(f"[DEBUG] {message}")

logger = print()

logger.info("Loading model...")

model = load_model()

logger.info("Model loaded.")

image, masks, boxes, labels, scores = segment_image(
    "images/input/test.jpg",
    model
)

logger.info(f"Detected {len(masks)} object(s).")

if len(masks) > 0:
    mask = masks[0, 0].cpu().numpy()

    segmented = apply_mask(image, mask)
    overlay = draw_mask(image, mask)
    logger.info("Running GrabCut...")
    refined = refine_mask(image, mask)
    logger.info("GrabCut finished.")
    boxed = draw_boxes(image, boxes, labels, scores)

    save_image("images/output/result.jpg", segmented)
    save_image("images/output/overlay.jpg", overlay)
    save_image("images/output/refined.jpg", refined)
    save_image("images/output/boxed.jpg", boxed)

    logger.info("Output saved!")

else:
    logger.warning("No object detected.")