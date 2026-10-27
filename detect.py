
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def enhance_visibility(image):
    """Apply contrast enhancement as a simple experiment."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(lightness)

    return cv2.cvtColor(
        cv2.merge((enhanced, a, b)),
        cv2.COLOR_LAB2BGR
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--enhance", action="store_true")
    args = parser.parse_args()

    image = cv2.imread(args.source)

    if image is None:
        raise ValueError("Could not load the input image.")

    if args.enhance:
        image = enhance_visibility(image)

    model = YOLO(args.model)
    results = model.predict(image, verbose=False)

    output = results[0].plot()
    Path("outputs").mkdir(exist_ok=True)
    cv2.imwrite("outputs/detection.jpg", output)

    print("Detection complete!")
    print("Output saved to outputs/detection.jpg")


if __name__ == "__main__":
    main()
