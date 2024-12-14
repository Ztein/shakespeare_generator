# Shakespeare Text Generator

A neural network-based text generator trained on Shakespeare's complete works. The model uses LSTM architecture to generate Shakespeare-style text.

## Features

- Downloads Shakespeare's complete works automatically
- Uses PyTorch with CUDA/MPS/CPU support
- Optimized for different platforms (Windows, Mac, Linux)
- Supports Apple Silicon (M-series) processors
- Automatic worker optimization for data loading
- Learning rate scheduling for better training

## Requirements

- Python 3.8-3.11 (tested with 3.11.9)
- PyTorch 2.0.1
- CUDA 11.8
- MPS 1.4
- Apple Silicon (M-series) processors
- TensorFlow 2.12.0 (for TF implementation)

```bash
pip install -r requirements.txt
```
## Usage

### Option 1: Use Pre-trained Model
A pre-trained model (`best_model.pt`) is included in the repository. To generate text immediately:
```bash
python generate.py
```

### Option 2: Train Your Own Model
1. Train the model:
```bash
python train.py
```
This will download Shakespeare's text, preprocess it, and train the model. The model will be saved as `best_model.pt`. That means you will over write the pre-trained model. If you want to keep the pre-trained model, you can copy that file to a different name.

2. Generate text:
```bash
python generate.py
```



## Model Architecture

- LSTM-based neural network
- Character-level text generation
- Embedding layer for character encoding
- Dropout for regularization
- AdamW optimizer with OneCycleLR scheduler

## Files

- `data_loader.py`: Downloads and preprocesses Shakespeare's text
- `model.py`: Neural network model definition
- `train.py`: Training script with optimization
- `generate.py`: Text generation script
- `requirements.txt`: Required Python packages
- `best_model.pt`: Pre-trained model

## Performance

The model automatically selects the best available hardware:
- CUDA for NVIDIA GPUs
- MPS for Apple Silicon
- CPU for other systems

Data loading is optimized based on the platform:
- Windows: Single worker for stability
- Other platforms: Automatically uses 80% of available CPU cores
