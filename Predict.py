from ultralytics import YOLO


model=YOLO("C:/Users/dhiab/OneDrive/Bureau/Stage/VS.pt")


model.predict(source=0,show=True,conf=0.5)

#model.export(format='onnx')
