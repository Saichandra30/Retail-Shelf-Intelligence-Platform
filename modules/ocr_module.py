import os
import re
import cv2
from typing import List, Dict, Any

PRICE_REGEX = re.compile(r'[₹Rs\.]*\s*\d{1,4}')

class ShelfOCR:
    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    def extract_clean_rail_prices(self, image_path: str, detections: List[Dict[str, Any]]) -> List[str]:
        if not os.path.exists(image_path) or not detections:
            return []

        img = cv2.imread(image_path)
        if img is None:
            return []

        h, w, _ = img.shape

        # Group product bottom-edges by shelf row (40px tolerance)
        y_bottoms = sorted([d["bbox"][3] for d in detections])
        unique_rails = []
        for y in y_bottoms:
            if not any(abs(y - ry) < 40 for ry in unique_rails):
                unique_rails.append(y)

        rail_labels = []
        seen_nums = set()

        for y_rail in unique_rails:
            # Crop strictly the shelf rail strip (just below the product bounding boxes)
            # Expanded to 120px to account for high-resolution images where rails are thick
            y1 = max(0, int(y_rail) - 10)
            y2 = min(h, int(y_rail) + 120)

            if y2 <= y1 or (y2 - y1) < 5:
                continue

            roi = img[y1:y2, 0:w]
            if roi.size == 0:
                continue

            # Upscale for better OCR accuracy on small text
            scale = 2
            roi_up = cv2.resize(roi, (roi.shape[1] * scale, roi.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

            try:
                result = self.reader.readtext(roi_up)
            except Exception:
                continue

            for box, text, prob in result:
                # Drop low-confidence tokens
                if prob < 0.45:
                    continue

                # Filter: Must either contain ₹/Rs/* or just be a clean 1-3 digit price
                text_clean = text.strip()
                if not re.search(r'(?:₹|rs\.|rs\s|\*|price)', text_clean, re.IGNORECASE):
                    # If it's a long barcode/packaging number, skip it
                    if len(re.sub(r'\D', '', text_clean)) >= 4:
                        continue

                # Extract only digit sequences (price numbers)
                clean_num = re.sub(r'\D', '', text_clean)
                if not clean_num:
                    continue

                num_val = int(clean_num)
                # Only keep realistic FMCG prices (Rs 5 – Rs 499)
                if num_val < 5 or num_val >= 500:
                    continue

                if num_val not in seen_nums:
                    seen_nums.add(num_val)
                    rail_labels.append(str(num_val))  # Raw number without symbol

        # Sort numerically
        rail_labels.sort(key=lambda s: int(re.sub(r'\D', '', s)) if re.sub(r'\D', '', s) else 0)
        return rail_labels
