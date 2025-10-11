from ultralytics import YOLO
import multiprocessing

# Load your custom model using the new classification YAML
model = YOLO('/Users/abhijithks/tmpZsh/sca-yolov4/ultralytics/cfg/models/v5/pg-cls.yaml', task="classify")

# Train the model on your classification dataset
results = model.train(
   data='/Users/abhijithks/tmpZsh/final-dataset/pig-imageFolder/', # Path to your ImageFolder
   epochs=50,
   imgsz=224,
   val=True,
   save=True,
    batch=32,
    workers=7,
    device='mps'
    project='/Users/abhijithks/tmpZsh/sca-yolov4/runs/classify/train',  # Set the main project directory
    name='pig_experiment_2',                 # Set the specific run name
)