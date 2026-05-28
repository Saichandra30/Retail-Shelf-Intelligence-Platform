import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

class ShelfSegmentor:
    def __init__(self):
        pass

    def analyze_shelf_arrangements(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Maps out whether items are standing vertically (bottles) or lying flat (biscuit boxes)."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            aspect_ratio = w / float(h) if h > 0 else 1.0
            
            if aspect_ratio < 0.75:
                # Tall and narrow structure
                det["orientation"] = "Vertical Facing (e.g., Beverage Bottle / Milk Carton)"
            elif aspect_ratio > 1.35:
                # Wide and flat structure
                det["orientation"] = "Horizontal Facing (e.g., Biscuit Box / Snack Multipack)"
            else:
                # Square-ish profile
                det["orientation"] = "Square Facing (e.g., Chips Pouch / Dahi Tub)"
                
        return detections

    def estimate_space(self, image_path: str, detections: List[Dict[str, Any]]) -> Dict[str, float]:
        if not detections:
            return {}
        
        brand_areas = {}
        total_area = 0.0
        
        for det in detections:
            bbox = det["bbox"]
            brand = det.get("brand") or "Other"
            # Geometric bounding box area aggregation
            area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            
            brand_areas[brand] = brand_areas.get(brand, 0.0) + area
            total_area += area
            
        if total_area == 0:
            return {}
            
        return {b: round((a / total_area) * 100, 2) for b, a in brand_areas.items()}

    def detect_shelf_rows(self, detections: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        if not detections:
            return []
        y_centers = sorted([((d["bbox"][1] + d["bbox"][3]) / 2.0) for d in detections])
        rows = []
        if y_centers:
            current_row = [y_centers[0]]
            for y in y_centers[1:]:
                if y - current_row[-1] < 75:  # Row separation tolerance
                    current_row.append(y)
                else:
                    rows.append((int(min(current_row) - 25), int(max(current_row) + 25)))
                    current_row = [y]
            rows.append((int(min(current_row) - 25), int(max(current_row) + 25)))
        return rows

    def compute_planogram_score(self, detections: List[Dict[str, Any]], img_w: int) -> Dict[str, Any]:
        if not detections:
            return {"score": 0.0, "observation": "Shelf empty."}
        
        left_zone = img_w / 3.0
        right_zone = 2.0 * left_zone
        l, m, r = 0, 0, 0
        
        for det in detections:
            cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
            if cx < left_zone: l += 1
            elif cx < right_zone: m += 1
            else: r += 1
            
        variance = (abs(l - len(detections)/3) + abs(m - len(detections)/3) + abs(r - len(detections)/3)) / len(detections)
        score = max(0.0, min(1.0, 1.0 - (variance / 2.0)))
        return {"score": round(score, 2), "observation": "Distribution evaluated across horizontal shelf zones."}
