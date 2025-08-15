# floorplan_analyzer.py
import cv2
import numpy as np
import fitz  # PyMuPDF
import json
import os
import re
from pdf2image import convert_from_path


def to_native(obj):
    """
    Convert NumPy data types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    else:
        return obj


def pdf_to_image(pdf_path, dpi=200):
    """Convert PDF to OpenCV image (grayscale)"""
    try:
        images = convert_from_path(pdf_path, dpi)
        img = images[0]  # Take first page
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to image: {e}")


def extract_text_with_fitz(pdf_path):
    """Extract text and its position using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text_data = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            bbox = block["bbox"]
                            text = span["text"].strip()
                            if text:  # Only add non-empty text
                                text_data.append({
                                    "text": text,
                                    "x0": float(bbox[0]),
                                    "y0": float(bbox[1]),
                                    "x1": float(bbox[2]),
                                    "y1": float(bbox[3]),
                                    "width": float(bbox[2] - bbox[0]),
                                    "height": float(bbox[3] - bbox[1]),
                                    "font_size": float(span["size"])
                                })
        doc.close()
        return text_data
    except Exception as e:
        raise RuntimeError(f"Failed to extract text: {e}")


def detect_walls_and_doors(gray):
    """Detect walls (thick lines) and door gaps"""
    try:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

        # Connect wall lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find walls
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        walls = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area > 1000 and max(w, h) > 30:
                walls.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                })

        # Detect doors (medium-length gaps)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=40, maxLineGap=250)
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
    except Exception as e:
        print(f"[ERROR] Wall/Door detection failed: {e}")
        return [], []


def detect_windows(gray):
    """Detect windows as pairs of short, close, parallel lines"""
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=15, maxLineGap=10)
        windows = []

        if lines is None:
            return windows

        line_list = [l[0] for l in lines]

        for i, line1 in enumerate(line_list):
            x1, y1, x2, y2 = line1
            len1 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if not (15 < len1 < 80):
                continue

            for j, line2 in enumerate(line_list):
                if i == j:
                    continue
                x3, y3, x4, y4 = line2
                len2 = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
                if not (15 < len2 < 80):
                    continue

                # Midpoint distance
                mx1, my1 = (x1 + x2) / 2, (y1 + y2) / 2
                mx2, my2 = (x3 + x4) / 2, (y3 + y4) / 2
                dist = np.sqrt((mx2 - mx1)**2 + (my2 - my1)**2)

                # Angle difference
                angle1 = np.arctan2(y2 - y1, x2 - x1)
                angle2 = np.arctan2(y4 - y3, x4 - x3)
                angle_diff = abs(angle1 - angle2) % np.pi

                if dist < 40 and angle_diff < 0.3:  # Close and parallel
                    windows.append({
                        "x": int(mx1),
                        "y": int(my1),
                        "gap_px": int(dist),
                        "type": "window"
                    })
                    break
        return windows
    except Exception as e:
        print(f"[ERROR] Window detection failed: {e}")
        return []


def detect_circular_fixtures(gray):
    """Detect circular fixtures (toilet, sink)"""
    try:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=50, param2=30, minRadius=8, maxRadius=30
        )
        fixtures = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if r > 8:
                    fixtures.append({
                        "x": int(x),
                        "y": int(y),
                        "radius": int(r),
                        "type": "sink_or_toilet"
                    })
        return fixtures
    except Exception as e:
        print(f"[ERROR] Circular fixture detection failed: {e}")
        return []


def detect_rectangular_fixtures(gray):
    """Detect rectangular fixtures (bathtub, shower pan)"""
    try:
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fixtures = []
        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if len(approx) == 4 and area > 600:
                aspect_ratio = max(w, h) / min(w, h)
                if 1.8 <= aspect_ratio <= 3.5:  # Typical tub ratio
                    fixtures.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "aspect_ratio": round(float(aspect_ratio), 2),
                        "type": "bathtub_or_shower_pan"
                    })
        return fixtures
    except Exception as e:
        print(f"[ERROR] Rectangular fixture detection failed: {e}")
        return []


def parse_dimensions(text_data):
    """Parse dimensions in feet ('') or meters (m)"""
    dimensions = []
    for item in text_data:
        text = item["text"].replace("’", "'").replace("”", '"').strip()

        # Match feet: 13'0", 17'0", etc.
        feet_match = re.search(r"(\d+)\s*['′]\s*(\d*)\s*['″]", text)
        if feet_match:
            ft = int(feet_match.group(1))
            inch = int(feet_match.group(2)) if feet_match.group(2).isdigit() else 0
            total_ft = ft + inch / 12
            dimensions.append({
                "text": text,
                "value_m": round(total_ft * 0.3048, 2),
                "unit": "feet",
                "x0": float(item["x0"]),
                "y0": float(item["y0"])
            })

        # Match meters: 3.5m x 2.8m
        meter_match = re.search(r"(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*m", text, re.IGNORECASE)
        if meter_match:
            w = float(meter_match.group(1))
            h = float(meter_match.group(2))
            dimensions.append({
                "text": text,
                "value_m": max(w, h),
                "unit": "meters",
                "x0": float(item["x0"]),
                "y0": float(item["y0"])
            })
    return dimensions


def detect_sanitary_fixtures_from_text(text_data):
    """Detect fixtures from text labels"""
    fixtures = []
    keywords = {
        "bathtub": ["tub", "bathtub", "shower pan", "custom shower pan"],
        "toilet": ["toilet", "wc", "water closet"],
        "sink": ["sink", "vanity", "basin", "lavatory"]
    }
    for item in text_data:
        text_lower = item["text"].lower()
        for ftype, words in keywords.items():
            if any(w in text_lower for w in words):
                fixtures.append({
                    "type": ftype,
                    "text": item["text"],
                    "x": float(item["x0"]),
                    "y": float(item["y0"])
                })
                break
    # Remove duplicates using tuple key
    return list({(f["type"], f["text"]): f for f in fixtures}.values())


def detect_labels(text_data):
    """Extract room and feature labels"""
    keywords = ["bathroom", "kitchen", "bedroom", "living", "hall", "wc", "shower", "window"]
    labels = []
    for item in text_data:
        text = item["text"].lower()
        if any(kw in text for kw in keywords) and len(text) < 30:
            labels.append(text.strip())
    return list(set(labels))


def analyze_floorplan(pdf_path):
    """
    Main function: Analyze floor plan and return structured JSON.
    Returns a JSON-serializable dict.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        # Step 1: Convert PDF to image
        original_img, gray = pdf_to_image(pdf_path)

        # Step 2: Extract text
        text_data = extract_text_with_fitz(pdf_path)

        # Step 3: Detect structural elements
        walls, doors = detect_walls_and_doors(gray)
        windows = detect_windows(gray)
        circular_fixtures = detect_circular_fixtures(gray)
        rectangular_fixtures = detect_rectangular_fixtures(gray)

        # Step 4: Parse dimensions
        dimensions = parse_dimensions(text_data)
        room_size_m = None
        for dim in dimensions:
            if dim["unit"] in ["feet", "meters"]:
                room_size_m = dim["value_m"]
                break

        # Fallback room size from walls
        if room_size_m is None and walls:
            xs = [w["x"] + w["width"] for w in walls]
            ys = [w["y"] + w["height"] for w in walls]
            if xs and ys:
                width_px = max(xs) - min([w["x"] for w in walls])
                room_size_m = round(width_px * 0.05, 2)  # 20 px ≈ 1m

        # Step 5: Detect fixtures and labels
        textual_fixtures = detect_sanitary_fixtures_from_text(text_data)
        labels = detect_labels(text_data)

        # Build result
        result = {
            "room_size_m": round(float(room_size_m), 2) if room_size_m else None,
            "dimensions": dimensions,
            "wall_count": len(walls),
            "walls": walls,
            "door_count": len(doors),
            "doors": doors,
            "window_count": len(windows),
            "windows": windows,
            "fixtures": {
                "from_text": textual_fixtures,
                "circular": circular_fixtures,
                "rectangular": rectangular_fixtures
            },
            "detected_labels": labels,
            "status": "success"
        }

        # Ensure all values are JSON-serializable
        result = to_native(result)
        return result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# Example usage (for local testing)
if __name__ == "__main__":
    pdf_path = "floorplan.pdf"
    try:
        result = analyze_floorplan(pdf_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")