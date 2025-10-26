#!/usr/bin/env python3
"""
preprocess_math.py

Preprocessing pipeline for noisy handwritten math images:
- optional CNN binarization hook (fast fallback if unavailable)
- median blur denoise
- adaptive threshold (white background with black text)
- morphological open/close
- connected-components speckle removal
- dilation to restore strokes
- optional deskew
- optional segmentation (lines/words)

Usage:
    python preprocess_math.py --input_folder ./raw --output_folder ./cleaned \
        --cnn_model path/to/model.pth --min_area 200 --deskew

If you don't have a CNN model, omit --cnn_model and the script uses adaptive thresholding.
"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np

# ------------------------
# Helper utilities
# ------------------------
def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img

def save_png(path, img):
    cv2.imwrite(str(path), img)

def invert_if_necessary(img):
    """
    Ensure the image has white background and black text.
    If most pixels are dark (text is white), invert the image.
    """
    # Calculate the mean pixel value
    mean_val = np.mean(img)
    
    # If mean is less than 128, it means background is dark (text is white)
    # So we invert to make background white and text black
    if mean_val < 128:
        return cv2.bitwise_not(img)
    return img

# ------------------------
# Optional CNN binarizer hook (stub)
# ------------------------
def cnn_binarize(img_gray, model_path=None, device='cpu'):
    """
    Optional: if you have a trained binarization CNN (dp-isi or one from RichSu95),
    load it and run inference here. This function is a placeholder illustrating where
    to insert that model. If model_path is None, returns None.
    """
    if model_path is None:
        return None
    # Placeholder behavior: user will implement model loading & inference here.
    # Example (pseudocode):
    #   model = load_model(model_path, device=device)
    #   pred = model.infer(img_gray)  # produce binary mask 0/255
    #   return pred
    # For now: return None to trigger fallback.
    return None

# ------------------------
# Main preprocessing stages
# ------------------------
def deskew_image(img_gray):
    # Estimate skew via moments on binary edges
    edges = cv2.Canny(img_gray, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if coords.shape[0] < 10:
        return img_gray  # nothing to deskew
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = angle + 90
    (h, w) = img_gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def adaptive_denoise_and_binarize(img_gray, median_k=3, block_size=35, C=12):
    """
    - median_k: kernel size for median blur (3 or 5)
    - block_size: odd window size for adaptive threshold (15..75)
    - C: constant subtracted from mean in adaptive threshold
    Returns binary image with WHITE BACKGROUND and BLACK TEXT (0/255).
    """
    # median blur reduces salt-and-pepper
    if median_k > 1:
        med = cv2.medianBlur(img_gray, median_k)
    else:
        med = img_gray
    # adaptive threshold GAUSSIAN - regular (not inverted) for white background
    binary = cv2.adaptiveThreshold(med, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, C)
    return binary

def morphological_cleanup(binary, open_kernel=(3,3), close_kernel=(3,3), open_iters=1, close_iters=1):
    kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel)
    kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern_open, iterations=open_iters)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kern_close, iterations=close_iters)
    return closed

def remove_small_components(binary, min_area=200):
    """
    Remove connected components smaller than min_area (in pixels).
    Assumes background is white (255) and text is black (0).
    """
    # Invert temporarily for connected components (which expects foreground=white)
    binary_inv = cv2.bitwise_not(binary)
    
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA]  # skip background
    final_inv = np.zeros_like(output, dtype=np.uint8)
    for i, size in enumerate(sizes):
        if size >= min_area:
            final_inv[output == i + 1] = 255
    
    # Convert back to white background, black text
    final = cv2.bitwise_not(final_inv)
    return final

def thicken_strokes(binary, dilate_kernel=(2,2), dilate_iters=1):
    """
    Thicken black text strokes using dilation.
    Since text is black (0), we need to invert, dilate, then invert back.
    """
    # Invert so text becomes white for dilation
    inverted = cv2.bitwise_not(binary)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    dilated_inv = cv2.dilate(inverted, kern, iterations=dilate_iters)
    # Convert back to white background, black text
    return cv2.bitwise_not(dilated_inv)

# Optional segmentation: simple horizontal projection to split lines (useful if OCR wants line-by-line)
def segment_lines(binary):
    # binary: 0 text, 255 bg - invert temporarily for projection
    binary_inv = cv2.bitwise_not(binary)
    hproj = np.sum(binary_inv == 255, axis=1)
    thresh = max(1, int(0.01 * binary.shape[1]))  # line exists if >1% width has ink
    lines = []
    in_line = False
    y0 = 0
    for y, v in enumerate(hproj):
        if v > thresh and not in_line:
            in_line = True
            y0 = y
        elif v <= thresh and in_line:
            in_line = False
            y1 = y
            lines.append((y0, y1))
    # handle case ends
    if in_line:
        lines.append((y0, binary.shape[0]))
    # return list of crops
    crops = [binary[y0:y1, :] for (y0, y1) in lines if (y1 - y0) > 5]
    return crops, lines

# ------------------------
# Pipeline wrapper
# ------------------------
def preprocess_image(path_in, path_out, params):
    img_gray = load_gray(path_in)

    # optional deskew
    if params['deskew']:
        img_gray = deskew_image(img_gray)

    # Optional heavy CNN binarization
    cnn_mask = None
    if params['cnn_model'] is not None:
        cnn_mask = cnn_binarize(img_gray, params['cnn_model'], device=params['device'])
    if cnn_mask is not None:
        binary = cnn_mask  # expected 0/255
        # Ensure white background, black text
        binary = invert_if_necessary(binary)
    else:
        binary = adaptive_denoise_and_binarize(img_gray, median_k=params['median_k'],
                                                block_size=params['block_size'], C=params['C'])
    
    cleaned = morphological_cleanup(binary, open_kernel=params['open_kernel'],
                                    close_kernel=params['close_kernel'],
                                    open_iters=params['open_iters'], close_iters=params['close_iters'])
    filtered = remove_small_components(cleaned, min_area=params['min_area'])
    thick = thicken_strokes(filtered, dilate_kernel=params['dilate_kernel'], dilate_iters=params['dilate_iters'])

    # Final check to ensure white background, black text
    final = invert_if_necessary(thick)
    
    # Save final result
    save_png(path_out, final)
    return final

# ------------------------
# CLI & batch driver
# ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_folder", required=True)
    p.add_argument("--output_folder", required=True)
    p.add_argument("--cnn_model", default=None, help="Optional path to a CNN binarizer model")
    p.add_argument("--device", default="cpu")
    p.add_argument("--median_k", type=int, default=3)
    p.add_argument("--block_size", type=int, default=35)
    p.add_argument("--C", type=int, default=12)
    p.add_argument("--open_kernel", type=int, nargs=2, default=(3,3))
    p.add_argument("--close_kernel", type=int, nargs=2, default=(3,3))
    p.add_argument("--open_iters", type=int, default=1)
    p.add_argument("--close_iters", type=int, default=1)
    p.add_argument("--min_area", type=int, default=200)
    p.add_argument("--dilate_kernel", type=int, nargs=2, default=(2,2))
    p.add_argument("--dilate_iters", type=int, default=1)
    p.add_argument("--deskew", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    in_folder = Path(args.input_folder)
    out_folder = Path(args.output_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    params = {
        'cnn_model': args.cnn_model,
        'device': args.device,
        'median_k': args.median_k,
        'block_size': args.block_size,
        'C': args.C,
        'open_kernel': tuple(args.open_kernel),
        'close_kernel': tuple(args.close_kernel),
        'open_iters': args.open_iters,
        'close_iters': args.close_iters,
        'min_area': args.min_area,
        'dilate_kernel': tuple(args.dilate_kernel),
        'dilate_iters': args.dilate_iters,
        'deskew': args.deskew
    }
    for img_path in sorted(in_folder.glob("*.*")):
        try:
            out_path = out_folder / (img_path.stem + ".jpg")
            preprocess_image(img_path, out_path, params)
            print("Processed:", img_path.name, "->", out_path.name)
        except Exception as e:
            print("Error processing", img_path, e)

if __name__ == "__main__":
    main()