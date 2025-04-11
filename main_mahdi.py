import os
import cv2
import argparse
import math
import progressbar
import numpy as np
from pointillism import (
    limit_size,
    ColorPalette,
    VectorField,
    randomized_grid,
    compute_color_probabilities,
    color_select,
)

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

def pointillize_image(
    img_path,
    palette_size=20,
    stroke_scale=1,
    gradient_smoothing_radius=0,
    limit_image_size=0,
    extend_palette=True,
):
    """
    Apply a pointillism-style transformation to the image at `img_path`.
    Returns the processed image (as a NumPy array).
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    # Optionally downsize
    if limit_image_size > 0:
        img = limit_size(img, limit_image_size)

    # Auto stroke size if set to 0
    if stroke_scale == 0:
        stroke_scale = int(math.ceil(max(img.shape) / 1000))

    # Auto gradient smoothing if not set
    if gradient_smoothing_radius == 0:
        gradient_smoothing_radius = int(round(max(img.shape) / 50))

    # Compute color palette
    palette = ColorPalette.from_image(img, palette_size)

    # Optionally extend palette with tinted versions
    if extend_palette:
        palette = palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])

    # Compute gradient
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gradient = VectorField.from_gradient(gray)
    gradient.smooth(gradient_smoothing_radius)

    # Optionally "cartoonize" the background
    res = cv2.medianBlur(img, 11)

    # Denser grid for more "point" coverage
    grid = randomized_grid(img.shape[0], img.shape[1], scale=1)

    # Batch the drawing so we don't hog memory
    batch_size = 10000
    bar = progressbar.ProgressBar()
    for h in bar(range(0, len(grid), batch_size)):
        batch_points = grid[h : h + batch_size]
        pixels = np.array([img[y, x] for (y, x) in batch_points])
        color_prob = compute_color_probabilities(pixels, palette, k=9)

        for i, (y, x) in enumerate(batch_points):
            color = color_select(color_prob[i], palette)
            # Uniform dot size (if you want variation, change the radius formula)
            radius = stroke_scale
            cv2.circle(res, (x, y), radius, color, -1, cv2.LINE_AA)

    return res


def main():
    parser = argparse.ArgumentParser(description="Convert images (or entire folders) to pointillism style.")
    parser.add_argument("--palette-size", default=20, type=int, help="Number of colors of the base palette")
    parser.add_argument("--stroke-scale", default=0, type=int, help="Size of each point (0 = auto)")
    parser.add_argument(
        "--gradient-smoothing-radius", default=0, type=int, help="Radius of gradient smoothing filter (0 = auto)"
    )
    parser.add_argument("--limit-image-size", default=0, type=int, help="Limit the image size (0 = no limit)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (if processing a folder). If not provided, we use '<input>_pointillism/'.",
    )
    parser.add_argument(
        "input_path",
        help="Path to an image or a folder of images."
    )
    args = parser.parse_args()

    input_path = args.input_path

    # Check if input is a directory or single file
    if os.path.isdir(input_path):
        # FOLDER MODE
        out_dir = args.out_dir
        # If no output dir is specified, create one automatically
        if not out_dir:
            out_dir = input_path.rstrip("/\\") + "_pointillism"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # Gather all supported images in the directory
        files = sorted(f for f in os.listdir(input_path) if f.lower().endswith(SUPPORTED_EXTS))

        if not files:
            print(f"No supported images found in {input_path}")
            return

        print(f"Processing folder: {input_path}")
        print(f"Saving pointillism images to: {out_dir}")

        for filename in files:
            in_file = os.path.join(input_path, filename)
            out_file = os.path.join(out_dir, os.path.splitext(filename)[0] + "_pointillism.jpg")

            print(f"\nConverting {filename} ...")
            try:
                result = pointillize_image(
                    in_file,
                    palette_size=args.palette_size,
                    stroke_scale=args.stroke_scale,
                    gradient_smoothing_radius=args.gradient_smoothing_radius,
                    limit_image_size=args.limit_image_size,
                )
                cv2.imwrite(out_file, result)
                print(f"Saved -> {out_file}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    else:
        # SINGLE FILE MODE
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"No such file: {input_path}")

        print(f"Converting single image: {input_path}")
        try:
            result = pointillize_image(
                input_path,
                palette_size=args.palette_size,
                stroke_scale=args.stroke_scale,
                gradient_smoothing_radius=args.gradient_smoothing_radius,
                limit_image_size=args.limit_image_size,
            )
            # Save next to original file
            out_file = os.path.splitext(input_path)[0] + "_pointillism.jpg"
            cv2.imwrite(out_file, result)
            print(f"Saved -> {out_file}")
        except Exception as e:
            print(f"Failed to process {input_path}: {e}")


if __name__ == "__main__":
    main()
