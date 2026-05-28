import os
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np

BRAND_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Tropicana": (0, 140, 255), "Real": (0, 50, 255), "Minute Maid": (50, 205, 50), "Coca-Cola": (0, 0, 200), "Pepsi": (255, 50, 50),
    "Sprite": (0, 255, 128), "Fanta": (0, 165, 255), "7UP": (0, 128, 0), "Thums Up": (128, 0, 0), "Limca": (180, 220, 50),
    "Mountain Dew": (0, 255, 0), "Mirinda": (0, 100, 255), "Nestea": (100, 150, 200), "Lipton": (0, 255, 255), "B Natural": (200, 100, 200),
    "Paper Boat": (180, 128, 0), "Gatorade": (255, 128, 0), "Red Bull": (0, 215, 255), "Nescafe": (19, 69, 139), "Amul Kool": (128, 0, 128),
    "Lay's": (0, 200, 255), "Doritos": (0, 60, 200), "Cheetos": (0, 120, 255), "Kurkure": (0, 80, 240), "Pringles": (30, 30, 160),
    "Bingo": (0, 40, 200), "Uncle Chipps": (200, 100, 0), "Parle-G": (0, 180, 255), "Good Day": (0, 220, 255), "Marie Gold": (20, 160, 240),
    "Oreo": (50, 50, 50), "Britannia": (0, 200, 220), "Tiger": (0, 140, 230), "Hide & Seek": (130, 0, 80), "Dark Fantasy": (30, 40, 100), "Malkist": (0, 160, 255),
    "Amul": (0, 200, 255), "Nestle": (150, 150, 150), "Mother Dairy": (200, 120, 0), "Activia": (80, 180, 80), "Yakult": (100, 100, 220),
    "Hershey's": (30, 60, 100), "Epigamia": (60, 100, 200), "Actimel": (80, 80, 200), "Milky Mist": (180, 220, 180), "Go Cheese": (0, 180, 255),
    "Diet Coke": (180, 180, 180), "Monaco": (0, 210, 255), "Other": (128, 128, 128)
}

HEADER_H, FOOTER_H, SIDEBAR_W, BG_COLOR, FONT = 60, 80, 260, (24, 24, 24), cv2.FONT_HERSHEY_SIMPLEX

class ShelfVisualizer:
    def draw(self, image_path: str, detections: List[Dict[str, Any]], ocr_labels: List[str], shelf_space: Dict[str, float], output_path: str) -> str:
        orig = cv2.imread(image_path)
        if orig is None: raise ValueError(f"Could not load image: {image_path}")
        h, w = orig.shape[:2]
        canvas = np.full((h + HEADER_H + FOOTER_H, w + SIDEBAR_W, 3), BG_COLOR, dtype=np.uint8)
        canvas[HEADER_H : HEADER_H + h, 0 : w] = orig

        for i, det in enumerate(detections):
            bbox = det.get("bbox", [0, 0, 0, 0])
            brand = det.get("brand") or "Other"
            if det.get("brand_confidence", 0.0) < 0.35:
                brand = "Other"
            color = BRAND_COLORS.get(brand, BRAND_COLORS["Other"])
            x1, y1 = int(round(bbox[0])), int(round(bbox[1])) + HEADER_H
            x2, y2 = int(round(bbox[2])), int(round(bbox[3])) + HEADER_H
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            tag = f"{brand} {det.get('brand_confidence', 0.0):.2f}"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.4, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (min(w + SIDEBAR_W, x1 + tw + 6), y1), color, -1)
            cv2.putText(canvas, tag, (x1 + 3, y1 - 4), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"#{i + 1}", (x1 + 5, y2 - 5), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, "RETAIL SHELF ANALYTICS PIPELINE  |  YOLOv8 + Hybrid OCR-SigLIP + docTR", (20, 38), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "EXTRACTED PRICE / PROMOTION TAGS:", (20, HEADER_H + h + 26), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        # Use ASCII-safe labels for OpenCV rendering (Rs. instead of ₹)
        ocr_display = [lbl.replace('₹', 'Rs.') for lbl in ocr_labels]
        ocr_str = "  |  ".join(ocr_display) if ocr_display else "No distinct price points isolated."
        max_chars = max(10, int((w - 40) / 7))
        if len(ocr_str) > max_chars: ocr_str = ocr_str[:max_chars - 3] + "..."
        cv2.putText(canvas, ocr_str, (20, HEADER_H + h + 55), FONT, 0.45, (0, 220, 0), 1, cv2.LINE_AA)

        sx = w
        cv2.putText(canvas, "SHARE OF SHELF", (sx + 18, HEADER_H + 32), FONT, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Total Facings: {len(detections)}", (sx + 18, HEADER_H + 56), FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.line(canvas, (sx + 10, HEADER_H + 65), (sx + SIDEBAR_W - 10, HEADER_H + 65), (60, 60, 60), 1)
        y_leg = HEADER_H + 85
        for brand, share in list(shelf_space.items())[:14]:
            color = BRAND_COLORS.get(brand, BRAND_COLORS["Other"])
            cv2.rectangle(canvas, (sx + 14, y_leg - 12), (sx + 28, y_leg + 2), color, -1)
            cv2.putText(canvas, f"{brand[:16]}: {share:.1f}%", (sx + 34, y_leg), FONT, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (sx + 14, y_leg + 6), (sx + 14 + max(2, min(200, int(200 * share / 100))), y_leg + 11), color, -1)
            y_leg += 30
            if y_leg > HEADER_H + h - 20: break

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, canvas)
        return output_path
