# floorplan_analyzer_with_scale_and_templates.py
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


def create_all_bathroom_templates():
    """Generate comprehensive templates for all bathroom fixtures"""
    os.makedirs("templates", exist_ok=True)

    # 1. TOILETS
    toilet1 = np.zeros((50, 60), dtype=np.uint8)
    cv2.ellipse(toilet1, (30, 25), (25, 18), 0, 0, 360, 255, -1)
    cv2.line(toilet1, (30, 25), (30, 40), 255, 2)
    cv2.imwrite("templates/toilet_one_piece.png", toilet1)

    toilet2 = np.zeros((70, 60), dtype=np.uint8)
    cv2.rectangle(toilet2, (15, 10), (45, 30), 255, -1)
    cv2.rectangle(toilet2, (20, 30), (40, 50), 255, -1)
    cv2.line(toilet2, (30, 30), (30, 40), 255, 2)
    cv2.imwrite("templates/toilet_two_piece.png", toilet2)

    toilet3 = np.zeros((40, 60), dtype=np.uint8)
    cv2.ellipse(toilet3, (30, 20), (25, 15), 0, 0, 360, 255, 2)
    cv2.line(toilet3, (30, 20), (30, 35), 255, 2)
    cv2.imwrite("templates/toilet_wall_hung.png", toilet3)

    toilet4 = np.zeros((50, 70), dtype=np.uint8)
    cv2.ellipse(toilet4, (35, 25), (30, 20), 0, 0, 360, 255, -1)
    cv2.line(toilet4, (35, 25), (35, 40), 255, 2)
    cv2.rectangle(toilet4, (55, 15), (65, 35), 255, 1)
    cv2.imwrite("templates/toilet_smart.png", toilet4)

    # 2. URINALS
    urinal1 = np.zeros((50, 40), dtype=np.uint8)
    cv2.ellipse(urinal1, (20, 25), (15, 20), 0, 0, 180, 255, 2)
    cv2.line(urinal1, (20, 5), (20, 25), 255, 2)
    cv2.imwrite("templates/urinal_bowl.png", urinal1)

    urinal2 = np.zeros((40, 100), dtype=np.uint8)
    cv2.rectangle(urinal2, (10, 10), (90, 30), 255, -1)
    cv2.imwrite("templates/urinal_trough.png", urinal2)

    # 3. SINKS
    sink1 = np.zeros((50, 60), dtype=np.uint8)
    cv2.ellipse(sink1, (30, 25), (25, 15), 0, 0, 180, 255, 2)
    cv2.line(sink1, (10, 25), (50, 25), 255, 1)
    cv2.imwrite("templates/sink_wall_mounted.png", sink1)

    sink2 = np.zeros((70, 60), dtype=np.uint8)
    cv2.ellipse(sink2, (30, 25), (25, 15), 0, 0, 180, 255, 2)
    cv2.line(sink2, (30, 45), (30, 65), 255, 2)
    cv2.rectangle(sink2, (25, 65), (35, 70), 255, -1)
    cv2.imwrite("templates/sink_pedestal.png", sink2)

    sink3 = np.zeros((50, 50), dtype=np.uint8)
    cv2.circle(sink3, (25, 25), 20, 255, 2)
    cv2.circle(sink3, (25, 25), 10, 255, 1)
    cv2.imwrite("templates/sink_vessel_round.png", sink3)

    sink4 = np.zeros((50, 60), dtype=np.uint8)
    cv2.ellipse(sink4, (30, 25), (25, 18), 0, 0, 360, 255, 2)
    cv2.ellipse(sink4, (30, 25), (15, 10), 0, 0, 360, 255, 1)
    cv2.imwrite("templates/sink_vessel_oval.png", sink4)

    sink5 = np.zeros((40, 80), dtype=np.uint8)
    cv2.rectangle(sink5, (10, 10), (70, 30), 255, -1)
    cv2.line(sink5, (10, 20), (70, 20), 255, 1)
    cv2.imwrite("templates/sink_console.png", sink5)

    # 4. BATHTUBS
    tub1 = np.zeros((60, 120), dtype=np.uint8)
    cv2.rectangle(tub1, (15, 15), (105, 45), 255, -1)
    cv2.line(tub1, (15, 15), (105, 15), 255, 1)
    cv2.line(tub1, (15, 45), (105, 45), 255, 1)
    cv2.imwrite("templates/bathtub_standard.png", tub1)

    tub2 = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(tub2, (10, 10), (90, 30), 255, -1)
    cv2.rectangle(tub2, (10, 10), (30, 90), 255, -1)
    cv2.imwrite("templates/bathtub_corner.png", tub2)

    tub3 = np.zeros((70, 130), dtype=np.uint8)
    cv2.rectangle(tub3, (15, 15), (115, 55), 255, 3)
    cv2.rectangle(tub3, (25, 25), (105, 45), 255, -1)
    cv2.imwrite("templates/bathtub_soaking.png", tub3)

    tub4 = np.zeros((80, 140), dtype=np.uint8)
    cv2.rectangle(tub4, (20, 20), (120, 60), 255, -1)
    cv2.rectangle(tub4, (0, 0), (140, 10), 255, -1)
    cv2.rectangle(tub4, (0, 0), (10, 80), 255, -1)
    cv2.rectangle(tub4, (130, 0), (140, 80), 255, -1)
    cv2.imwrite("templates/bathtub_alcove.png", tub4)

    # 5. SHOWERS
    shower1 = np.zeros((70, 70), dtype=np.uint8)
    cv2.rectangle(shower1, (15, 15), (55, 55), 255, 2)
    cv2.line(shower1, (15, 15), (55, 55), 255, 1)
    cv2.line(shower1, (15, 55), (55, 15), 255, 1)
    cv2.imwrite("templates/shower_stall_square.png", shower1)

    shower2 = np.zeros((80, 100), dtype=np.uint8)
    cv2.rectangle(shower2, (10, 10), (90, 70), 255, 2)
    cv2.line(shower2, (10, 10), (90, 70), 255, 1)
    cv2.line(shower2, (10, 70), (90, 10), 255, 1)
    cv2.imwrite("templates/shower_stall_rect.png", shower2)

    shower3 = np.zeros((100, 120), dtype=np.uint8)
    cv2.rectangle(shower3, (10, 10), (110, 90), 255, 2)
    cv2.line(shower3, (60, 10), (60, 25), 255, 2)
    cv2.imwrite("templates/shower_walk_in.png", shower3)

    shower4 = np.zeros((90, 110), dtype=np.uint8)
    cv2.rectangle(shower4, (15, 15), (95, 75), 255, 2)
    cv2.circle(shower4, (25, 25), 5, 255, 1)
    cv2.imwrite("templates/shower_pan_outline.png", shower4)

    shower5 = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(shower5, (10, 10), (90, 90), 255, 2)
    cv2.circle(shower5, (50, 50), 10, 255, 1)
    cv2.line(shower5, (50, 40), (50, 30), 255, 1)
    cv2.imwrite("templates/shower_steam.png", shower5)

    print("✅ All bathroom fixture templates created in 'templates/' folder!")


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


def parse_scale_info(text_data):
    """
    Parse scale info like '1 inch = 4'0"'
    Returns pixels_per_foot
    """
    for item in text_data:
        text = item["text"].lower()
        match = re.search(r"(\d+)\s*inch\s*=\s*(\d+)'(\d*)\"", text)
        if match:
            inches_per_unit = int(match.group(1))
            ft = int(match.group(2))
            inch = int(match.group(3)) if match.group(3) else 0
            real_length_ft = ft + inch / 12

            # At 200 DPI, 1 inch = 200 pixels
            pixels_per_inch = 200  # Based on dpi in pdf_to_image()
            pixels_per_unit = inches_per_unit * pixels_per_inch
            pixels_per_foot = pixels_per_unit / real_length_ft

            return pixels_per_foot
    return None  # Fallback to 20 px per foot


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
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
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
                    doors.append({
                        "x": cx,
                        "y": cy,
                        "length_px": int(length)
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
    for pt in zip(*loc[::-1]):
        x, y = pt[0], pt[1]
        matches.append({
            "template": os.path.basename(template_path),
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "confidence": float(result[y, x])
        })

    return matches


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
                "value_ft": total_ft,
                "unit": "feet",
                "x0": float(item["x0"]),
                "y0": float(item["y0"])
            })
    return dimensions


def detect_sanitary_fixtures_from_text(text_data):
    """Detect fixtures from text labels"""
    fixtures = []
    keywords = {
        "toilet": ["toilet", "wc", "water closet", "one piece", "two piece", "wall hung", "smart toilet"],
        "urinal": ["urinal", "trough"],
        "sink": ["sink", "basin", "lavatory", "vanity", "vessel", "pedestal", "console"],
        "bathtub": ["tub", "bathtub", "soaking tub", "corner tub", "alcove"],
        "shower": ["shower", "shower pan", "custom shower pan", "steam shower", "walk-in"]
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
        # Step 0: Create templates if not exist
        if not os.path.exists("templates"):
            create_all_bathroom_templates()

        # Step 1: Convert PDF to image
        original_img, gray = pdf_to_image(pdf_path)

        # Step 2: Extract text
        text_data = extract_text_with_fitz(pdf_path)

        # Step 3: Parse scale
        pixels_per_foot = parse_scale_info(text_data)
        if pixels_per_foot is None:
            pixels_per_foot = 20.0  # Fallback: ~20px per foot

        # Step 4: Detect walls and doors
        walls, doors = detect_walls_and_doors(gray)

        # Step 5: Parse dimensions
        dimensions = parse_dimensions(text_data)
        room_size_ft = None
        for dim in dimensions:
            if dim["unit"] == "feet":
                room_size_ft = dim["value_ft"]
                break

        # Fallback: estimate from wall size
        if room_size_ft is None and walls:
            wall_width_px = max(w["x"] + w["width"] for w in walls) - min(w["x"] for w in walls)
            room_size_ft = wall_width_px / pixels_per_foot

        # Convert to meters
        room_size_m = round(room_size_ft * 0.3048, 2)

        # Step 6: Template Matching for All Fixtures
        template_groups = {
            "toilet": [
                "toilet_one_piece.png", "toilet_two_piece.png", "toilet_wall_hung.png", "toilet_smart.png"
            ],
            "urinal": [
                "urinal_bowl.png", "urinal_trough.png"
            ],
            "sink": [
                "sink_wall_mounted.png", "sink_pedestal.png",
                "sink_vessel_round.png", "sink_vessel_oval.png", "sink_console.png"
            ],
            "bathtub": [
                "bathtub_standard.png", "bathtub_corner.png",
                "bathtub_soaking.png", "bathtub_alcove.png"
            ],
            "shower": [
                "shower_stall_square.png", "shower_stall_rect.png",
                "shower_walk_in.png", "shower_pan_outline.png", "shower_steam.png"
            ]
        }

        all_matches = []
        for ftype, templates in template_groups.items():
            for t in templates:
                path = f"templates/{t}"
                matches = detect_fixture_by_template(gray, path, threshold=0.7)
                for m in matches:
                    m["type"] = ftype
                all_matches.extend(matches)

        # Remove duplicates
        unique_matches = []
        for match in all_matches:
            is_duplicate = False
            for existing in unique_matches:
                dist = ((match["x"] - existing["x"])**2 + (match["y"] - existing["y"])**2)**0.5
                if dist < 30:
                    if match["confidence"] > existing["confidence"]:
                        unique_matches.remove(existing)
                        unique_matches.append(match)
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_matches.append(match)

        # Step 7: Text-based Fixtures
        textual_fixtures = detect_sanitary_fixtures_from_text(text_data)
        labels = detect_labels(text_data)

        # Final result
        result = {
            "room_size_m": round(float(room_size_m), 2),
            "room_size_ft": round(float(room_size_ft), 2),
            "scale": {
                "text": "1 inch = 4'0\"",
                "pixels_per_foot": round(float(pixels_per_foot), 2)
            },
            "dimensions": dimensions,
            "wall_count": len(walls),
            "walls": walls,
            "door_count": len(doors),
            "doors": doors,
            "fixtures": {
                "from_text": textual_fixtures,
                "from_template": unique_matches
            },
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