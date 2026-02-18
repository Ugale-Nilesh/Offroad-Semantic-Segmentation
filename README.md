# TerraSeg — Off‑Road Semantic Scene Segmentation

## Overview

TerraSeg is a deep learning–based semantic segmentation project designed for off‑road autonomous navigation. The system trains a computer vision model to understand terrain structure using synthetic desert digital‑twin data and evaluates its ability to generalize to unseen environments.

The goal is to enable unmanned ground vehicles (UGVs) to interpret surroundings by classifying each pixel into terrain categories such as vegetation, rocks, and sky, allowing safer path planning in unstructured environments.

---

## Problem Statement

Autonomous vehicles operating off‑road cannot rely on lane markings or predefined roads. Instead, they must understand terrain composition. This project builds a perception pipeline that performs pixel‑level classification of terrain elements so that navigation algorithms can make decisions based on environmental understanding.

---

## Features

* Semantic segmentation of desert terrain
* Synthetic‑to‑novel environment generalization
* IoU‑based performance evaluation
* Training & inference pipeline
* Failure case analysis capability

---

## Classes Detected

* Trees
* Lush Bushes
* Dry Grass
* Dry Bushes
* Ground Clutter
* Flowers
* Logs
* Rocks
* Landscape (general ground)
* Sky

---

## Project Structure

```
project/
│── dataset/
│   ├── train/
│   ├── val/
│   └── testImages/
│
│── runs/                  # training logs & checkpoints
│── ENV_SETUP/             # environment setup files
│── train.py               # training script
│── test.py                # inference & evaluation
│── config.yaml            # training configuration
│── README.txt
```

---

## Installation

### 1. Create Environment

Install Anaconda/Miniconda and run:

```
cd ENV_SETUP
setup_env.bat
```

(For Linux/Mac use equivalent shell script)

Activate environment:

```
conda activate EDU
```

---

## Training

Train the model using:

```
python train.py
```

This will:

* Train on train & validation dataset
* Save model checkpoints
* Generate logs in `runs/`

---

## Testing / Evaluation

Evaluate on unseen images:

```
python test.py
```

Outputs:

* Predicted segmentation masks
* Loss metrics
* IoU score

---

## Metrics Used

* Intersection over Union (IoU)
* Training Loss
* Validation Performance

---

## Expected Output

The model predicts a colored segmentation map where each color represents a terrain class. Higher IoU indicates better overlap with ground truth labels.

---

## Reproducibility

To reproduce results:

1. Download dataset
2. Setup environment
3. Run training
4. Run testing
5. Compare IoU score

---

## Future Improvements

* Domain adaptation for real‑world images
* Faster inference for real‑time robotics
* Multi‑sensor fusion (LiDAR + RGB)
* Self‑supervised learning

---

## Authors

Team Project — AI Off‑Road Perception

---

## License

For academic and educational use only.
