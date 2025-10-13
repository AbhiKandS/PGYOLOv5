# PG-YOLOv5: Pruning-Guided YOLOv5

This repository provides information and usage instructions for **PG-YOLOv5**, a novel, lightweight detection framework optimized for high-fidelity livestock facial analysis, as presented in the paper *"Explainable Pruning-Focused Attention Network for High-Fidelity Livestock Facial Detection and Identification"*.

PG-YOLOv5 is built upon YOLOv5 and integrates specialized modules to enhance discriminative feature representation and eliminate redundant model components, making it both highly accurate and computationally efficient for real-time deployment.

## Key Features

The core innovations of PG-YOLOv5 are designed to balance accuracy with efficiency, making it ideal for challenging on-device agricultural applications.

  * **StrucFocus-Channel Attention (SFCA):** A sequential attention module that significantly enhances feature discriminability. It first models inter-channel dependencies to weigh the importance of each feature channel, then applies spatial attention to identify *where* the most informative regions are located, effectively suppressing noise and amplifying critical features.

  * **Attention-Induced Channel Pruning (AICP):** A novel pruning mechanism that uses batch-level self-attention to learn inter-sample channel correlations. It dynamically identifies and suppresses redundant or uninformative channels during training, leading to a compact, hardware-friendly model without sacrificing performance.

  * **Learnable Robust Attention Loss (LRAL):** A custom loss function that incorporates an attention-based regularization term. It encourages sparsity in the learned attention maps, guiding the model to focus on the most salient features. This stabilizes training, improves robustness against occlusion and noise, and facilitates more effective pruning.

## Getting Started: Using the Pre-trained Model

The following instructions explain how to use a pre-trained PG-YOLOv5 model for inference within the Ultralytics framework.

### 1\. Setup Environment

First, clone the PGYOLOv5 repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/AbhiKandS/PGYOLOv5.git

# Navigate into the project directory
cd PGYOLOv5
```

### 2\. Run Inference with PG-YOLOv5

You can easily load the pre-trained PG-YOLOv5 model and run predictions on your images. You will need the pre-trained model weights file (i.e., `pgyolov5.pt`).

```python
from ultralytics import YOLO

# Load the pre-trained PG-YOLOv5 model weights
model = YOLO('./pgyolov5.pt')

# Run inference on image data
metrics = model.val(data='/path/to/data.yaml')
```

## Performance on Public Dataset (PCVD)

PG-YOLOv5 was benchmarked on the public **Pig Face Computer Vision Dataset (PCVD)**.

  * **Dataset Link:** [Pig Face Computer Vision Dataset on Roboflow](https://universe.roboflow.com/project-zsqs6/pig-zsqs6)

The framework demonstrates a superior trade-off between accuracy and computational efficiency compared to other state-of-the-art models.

### Comparative Analysis

The table below summarizes the performance of PG-YOLOv5 against other popular object detection models on the PCVD test set. Arrows indicate whether a higher (${\color{green}\uparrow}$) or lower (${\color{red}\downarrow}$) value is preferable.

| Model               | Param (M) ${\color{red}\downarrow}$ | FLOPs (G) ${\color{red}\downarrow}$ | mIoU ${\color{green}\uparrow}$ | mAP$_{50}$ ${\color{green}\uparrow}$ | mAP$_{50-95}$ ${\color{green}\uparrow}$ |
| ------------------- | :---------------------------------: | :--------------------------------: | :---------------------------: | :--------------------------------: | :-----------------------------------: |
| RetinaNet (2017)    |                31.17                |               151.55               |             0.767             |               0.988                |                 0.702                 |
| YOLOv5s (2017)      |                9.10                 |               24.04                |             0.822             |               0.989                |                 0.752                 |
| FCOS (2019)         |                30.57                |               128.35               |             0.689             |               0.976                |                 0.576                 |
| YOLOv3-SPP (2021)   |                62.54                |               32.82                |             0.785             |               0.995                |                 0.733                 |
| YOLOv8n (2023)      |                3.27                 |              **4.26** |             0.811             |               0.989                |                 0.737                 |
| Mamba Yolo-T (2024) |                5.98                 |               13.61                |             0.815             |               0.991                |                 0.709                 |
| Hyper-Yolo (2025)   |                3.95                 |               11.0                 |             0.802             |               0.991                |                 0.726                 |
| **PG-YOLOv5** |              **9.01** |             **23.40** |           **0.828** |             **0.995** |               **0.759** |


