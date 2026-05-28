from ultralytics import YOLO

# Load YOLOv10n model from scratch
model = YOLO("yolov10n.yaml")

# Train the model
results = model.train(data="coco8.yaml", epochs=10, imgsz=640, batch=8)
