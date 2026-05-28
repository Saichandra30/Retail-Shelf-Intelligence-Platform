import os
import cv2
from ultralytics import YOLO
from typing import List, Dict, Any

class ProductDetector:
    def __init__(self):
        # Initialize YOLOv8 loaded with SKU-110K pre-trained weights
        self.model = YOLO('yolov8n_sku110k.pt')
        self.conf_threshold = 0.01
        self.iou_threshold = 0.45

        self.model.to("cpu")

    def detect(self, image_path: str) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=640,
            device="cpu",
            verbose=False
        )
        
        detections = []
        if not results or not results[0].boxes:
            return []

        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            
            detections.append({
                "bbox": [int(coord) for coord in xyxy],
                "confidence": round(conf, 2),
                "brand": None,
                "brand_confidence": 0.0,
                "orientation": "Unknown"
            })
            
        # Spatial sorting: Top-to-Bottom, then Left-to-Right
        detections.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        return detections

    def get_model_info(self):
        return {"model_name": "YOLOv8 SKU-110K", "device": "CPU"}