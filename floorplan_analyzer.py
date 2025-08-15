# floorplan_analyzer.py
import cv2
import numpy as np
import fitz  # PyMuPDF
import json
import os
import re
from pdf2image import convert_from_path


def to_native(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
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


def create_templates():
    """Generate standard fixture templates and save to 'templates/'"""
    os.makedirs("templates", exist_ok=True)

    # 1. Toilet: Circle with line
    toilet = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(toilet, (25, 25), 15, 255, 2)
    cv2.line(toilet, (25, 25), (25, 35), 255, 2)
    cv2.imwrite("templates/toilet.png", toilet)

    # 2. Sink: Small rectangle
    sink = np.zeros((40, 40), dtype=np.uint8)
    cv2.rectangle(sink, (10, 10), (30, 25), 255, -1)
    cv2.imwrite("templates/sink.png", sink)

    # 3. Bathtub: Long rectangle
    bathtub = np.zeros((60, 120), dtype=np.uint8)
    cv2.rectangle(bathtub, (15, 15), (105, 45), 255, -1)
    cv2.imwrite("templates/bathtub.png", bathtub)

    # 4. Window: Two parallel lines
    window = np.zeros((50, 50), dtype=np.uint8)
    cv2.line(window, (10, 20), (40, 20), 255, 2)
    cv2.line(window, (10, 30), (40, 30), 255, 2)
    cv2.imwrite("templates/window.png", window)

    print("✅ Templates created in 'templates/' folder!")


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
                            if text:
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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        walls = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area > 1000 and max(w, h) > 30:
                walls.append({
                    "x": int(x), "y": int(y), "width": int(w), "height": int(h)
                })
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=80, maxLineGap=50)
        doors = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if 80 < length < 150:
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    near_wall = any(
                        abs(cx - (w["x"] + w["width"]/2)) < 100 and
                        abs(cy - (w["y"] + w["height"]/2)) < 100
                        for w in walls
                    )
                    if near_wall:
                        doors.append({
                            "x": cx, "y": cy, "length": int(length)
                        })
        return walls, doors
    except Exception as e:
        print(f"[ERROR] Wall/Door detection failed: {e}")
        return [], []


def detect_fixture_by_template(gray, template_path, threshold=0.7):
    """Detect fixture using template matching"""
    if not os.path.exists(template_path):
        print(f"❌ Template not found: {template_path}")
        return []

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        return []

    w, h = template.shape[::-1]
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)

    matches = []
    for pt in zip(*loc[::-1]):  # Switch x and y
        x, y = pt[0], pt[1]
        matches.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "confidence": float(result[y, x]),
            "type": os.path.basename(template_path).replace(".png", "")
        })

    # Remove duplicates (nearby detections)
    filtered = []
    for match in matches:
        if not any(
            abs(match["x"] - m["x"]) < 20 and abs(match["y"] - m["y"]) < 20
            for m in filtered
        ):
            filtered.append(match)
    return filtered


def parse_dimensions(text_data):
    """Parse dimensions in feet ('') or meters (m)"""
    dimensions = []
    for item in text_data:
        text = item["text"].replace("’", "'").replace("”", '"').strip()
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
    """Main function: Analyze floor plan and return structured JSON"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        # Create templates on first run
        if not os.path.exists("templates"):
            create_templates()

        # Step 1: Convert PDF to image
        original_img, gray = pdf_to_image(pdf_path)

        # Step 2: Extract text
        text_data = extract_text_with_fitz(pdf_path)

        # Step 3: Detect walls and doors
        walls, doors = detect_walls_and_doors(gray)

        # Step 4: Parse dimensions
        dimensions = parse_dimensions(text_data)
        room_size_m = None
        for dim in dimensions:
            if dim["unit"] in ["feet", "meters"]:
                room_size_m = dim["value_m"]
                break
        if room_size_m is None and walls:
            xs = [w["x"] + w["width"] for w in walls]
            ys = [w["y"] + w["height"] for w in walls]
            width_px = max(xs) - min([w["x"] for w in walls])
            room_size_m = round(width_px * 0.05, 2)

        # Step 5: Template Matching for Fixtures
        fixtures_from_template = {}
        for name in ["toilet", "sink", "bathtub", "window"]:
            path = f"templates/{name}.png"
            matches = detect_fixture_by_template(gray, path, threshold=0.7)
            fixtures_from_template[name] = matches

        # Step 6: Text-based fixtures
        textual_fixtures = detect_sanitary_fixtures_from_text(text_data)
        labels = detect_labels(text_data)

        # Final result
        result = {
            "room_size_m": round(float(room_size_m), 2) if room_size_m else None,
            "dimensions": dimensions,
            "wall_count": len(walls),
            "walls": walls,
            "door_count": len(doors),
            "doors": doors,
            "fixtures_from_template": fixtures_from_template,
            "fixtures_from_text": textual_fixtures,
            "detected_labels": labels,
            "status": "success"
        }
        result = to_native(result)
        return result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# Example usage
if __name__ == "__main__":
    pdf_path = "floorplan.pdf"
    try:
        result = analyze_floorplan(pdf_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")