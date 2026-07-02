from ultralytics import YOLO

model = YOLO(r"C:\Users\zuizui\mc\runs\cirsium_detect\weights\best.pt")
model.export(format="imx", imgsz=640)