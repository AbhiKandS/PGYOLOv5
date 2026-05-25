import time

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchvision import datasets, models
from tqdm import tqdm


def run_evaluation():
    # --- 1. Configuration ---
    MODEL_PATH = "/Users/abhijithks/Downloads/Swin_b.pth"
    DATA_DIR = "/Users/abhijithks/tmpZsh/final-dataset/pig-imageFolder/"
    BATCH_SIZE = 32
    NUM_CLASSES = 9

    # --- 2. Device Setup ---
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")

    # --- 3. Model Loading ---
    weights = models.Swin_B_Weights.DEFAULT
    model = models.swin_b()

    # Get the number of input features from the original head
    num_ftrs = model.head.in_features

    # Replace the head with a new Linear layer for your number of classes
    model.head = nn.Linear(num_ftrs, NUM_CLASSES)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # --- 4. Data Loading ---
    # ✅ FIX: Use the correct weights and transforms for ResNet-18
    auto_transforms = weights.transforms()
    test_dataset = datasets.ImageFolder(root=f"{DATA_DIR}/test", transform=auto_transforms)

    # Set num_workers to 0 if on Windows/macOS and not in a main block,
    # or keep it > 0 inside the main block.
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # --- 5. Initialize Metrics and Timers ---
    top1_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(DEVICE)
    top5_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=5).to(DEVICE)
    inference_times = []

    # --- 6. The Validation Loop ---
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            start_time = time.perf_counter()
            outputs = model(inputs)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            batch_time = end_time - start_time
            per_image_time = batch_time / inputs.size(0)
            inference_times.append(per_image_time)

            top1_acc_metric.update(outputs, labels)
            top5_acc_metric.update(outputs, labels)

    # --- 7. Calculate and Print Final Results ---
    final_top1_acc = top1_acc_metric.compute()
    final_top5_acc = top5_acc_metric.compute()
    avg_inference_time_ms = (sum(inference_times) / len(inference_times)) * 1000

    print("\n--- Test Results ---")
    print(f"Top-1 Accuracy (Acc@1): {final_top1_acc.item():.4f}")
    print(f"Top-5 Accuracy (Acc@5): {final_top5_acc.item():.4f}")
    print(f"Average Inference Time per Image: {avg_inference_time_ms:.2f} ms")


# ✅ FIX: Wrap the code that runs the script in this block
if __name__ == "__main__":
    run_evaluation()
