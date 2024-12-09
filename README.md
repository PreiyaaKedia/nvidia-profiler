# Image Classifier Training with NVIDIA Profiling

This repository contains a script for training an image classifier on the FashionMNIST dataset using PyTorch. The script includes NVIDIA profiling to analyze the performance of the training process.

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- NVIDIA Nsight Systems

## Setup

1. **Clone the repository**:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create a virtual environment** (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    pip install -r requirements.txt
    
## Running the Training Script

To run the training script, use the following command:

python [train.py]

## Profiling with NVIDIA Nsight Systems
To profile the training script using NVIDIA Nsight Systems, use the following command:
```
nsys profile --trace=nvtx,cuda,osrt,cudnn --capture-range=cudaProfilerApi --export=sqlite --output=profile_report --force-overwrite=true python train.py
```
This command will generate a profiling report in SQLite format, which can be analyzed using NVIDIA Nsight Systems.

### Script Overview
The train.py script performs the following steps:

Imports necessary libraries.
Defines the data transformation and loads the FashionMNIST dataset.
Creates a DataLoader for the training set.
Defines the ImageClassifier model.
Initializes the model, loss function, and optimizer.
Sets up CUDA streams and AMP (Automatic Mixed Precision).
Starts the NVIDIA profiler.
Runs the training loop with NVTX markers for profiling.
Stops the NVIDIA profiler and prints the training time.
License
This project is licensed under the MIt License
