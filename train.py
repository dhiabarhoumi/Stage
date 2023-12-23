from ultralytics import YOLO


model = YOLO("best.pt")

model.train(data="C:/Users/dhiab/OneDrive/Bureau/Stage/generated from V3.v1i.yolov8/data.yaml", epochs=150,imgsz=640) 

