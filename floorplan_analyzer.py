# floorplan_analyzer.py

import cv2
import numpy as np
import fitz  # PyMuPDF
import re  # ← Add this here
import json
import os
from pdf2image import convert_from_path


def pdf_to_image(pdf_path, dpi=200):
    """Convert PDF to OpenCV image (grayscale)"""
    images = convert_from_path(pdf_path, dpi)
    img = images[0]  # Take first page
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def extract_text_with_fitz(pdf_path):
    """Extract text and its position using PyMuPDF"""
    doc = fitz.open(pdf_path)
    text_data = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        bbox = block["bbox"]
                        text_data.append({
                            "text": span["text"].strip(),
                            "x0": bbox[0],
                            "y0": bbox[1],
                            "x1": bbox[2],
                            "y1": bbox[3],
                            "width": bbox[2] - bbox[0],
                            "height": bbox[3] - bbox[1],
                            "font_size": span["size"]
                        })
    doc.close()
    return text_data


def detect_walls_and_doors(gray):
    """Detect walls (thick lines) and door gaps"""
    # Enhance image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    # Use morphological closing to connect wall lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours (potential walls)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    walls = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > 1000 and max(w, h) > 30:
            walls.append({"x": x, "y": y, "width": w, "height": h})

    # Detect doors: look for medium-length lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=20)
    doors = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if 40 < length < 150:
                doors.append({
                    "x": int((x1 + x2) / 2),
                    "y": int((y1 + y2) / 2),
                    "length": int(length)
                })

    return walls, doors


def estimate_room_size_from_text(text_data):
    """Look for dimension labels like '3.5m x 2.8m'"""
    for item in text_data:
        text = item["text"]
        match = re.search(r'(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*m', text, re.IGNORECASE)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None, None

def detect_sanitary_ware(gray):
    """Detect toilet, basin, tub using image contours"""
    sanitary_ware = {
        "toilet": [],
        "basin": [],
        "tub": []
    }

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # Too small
            continue

        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h

        # Try to classify based on shape and size
        if 0.8 < aspect_ratio < 1.2:  # Square-like
            # Likely a basin/sink
            if area < 2000:
                sanitary_ware["basin"].append({"x": x, "y": y, "width": w, "height": h})
        elif 1.5 < aspect_ratio < 3.0:  # Long rectangle
            # Likely a bathtub
            if area > 3000:
                sanitary_ware["tub"].append({"x": x, "y": y, "width": w, "height": h})
        elif len(approx) == 4:  # Rectangle
            # Could be a toilet or tub
            if area > 2000 and aspect_ratio > 1.0:
                sanitary_ware["toilet"].append({"x": x, "y": y, "width": w, "height": h})

    return sanitary_ware

def analyze_floorplan(pdf_path):
    """Main function: Analyze PDF and return layout JSON"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Step 1: Convert PDF to image
    original_img, gray = pdf_to_image(pdf_path)

    # Step 2: Extract text (labels, dimensions)
    text_data = extract_text_with_fitz(pdf_path)

    # Step 3: Detect walls and doors
    walls, doors = detect_walls_and_doors(gray)

    # Step 4: Try to get room size from text
    width_m, height_m = estimate_room_size_from_text(text_data)

    # Fallback: estimate from wall positions
    if width_m is None:
        xs = [w["x"] + w["width"] for w in walls]
        ys = [w["y"] + w["height"] for w in walls]
        width_m = max(xs) - min(xs)
        height_m = max(ys) - min(ys)

    # Step 5: Detect sanitary ware using image analysis
    sanitary_ware = detect_sanitary_ware(gray)

    # Build result
    result = {
        "room_width_m": round(width_m, 2),
        "room_length_m": round(height_m, 2) if height_m else None,
        "door_count": len(doors),
        "doors": doors,
        "wall_count": len(walls),
        "detected_labels": list(set(rooms)),
        "dimensions_text": [t["text"] for t in text_data if "m" in t["text"].lower() and re.search(r'\d', t["text"])],
        "sanitary_ware": sanitary_ware,
        "status": "success"
    }

    return result