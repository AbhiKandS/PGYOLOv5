from ultralytics import YOLO

# Load your custom model using the new classification YAML
model = YOLO('/Users/abhijithks/tmpZsh/sca-yolov4/runs/classify/train/pig_experiment_2/weights/best.pt', task="classify")

metrics = model.val(
   data='/Users/abhijithks/tmpZsh/final-dataset/pig-imageFolder/', 
   split="test"
)