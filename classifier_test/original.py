from ultralytics import YOLO

# Load your custom model using the new classification YAML
model = YOLO("/Users/abhijithks/Downloads/yolov5s-cls.pt")

metrics = model.val(data="/Users/abhijithks/tmpZsh/final-dataset/pig-imageFolder/", split="test")
