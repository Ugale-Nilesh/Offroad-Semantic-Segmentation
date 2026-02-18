import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10
IMAGE_SIZE = 512
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 40

# TRAIN
TRAIN_IMG = "Offroad_Segmentation_Training_Dataset/train/Color_Images"
TRAIN_MASK = "Offroad_Segmentation_Training_Dataset/train/Segmentation"

# VALIDATION
VAL_IMG = "Offroad_Segmentation_Training_Dataset/val/Color_Images"
VAL_MASK = "Offroad_Segmentation_Training_Dataset/val/Segmentation"

# TEST (NO MASKS)
TEST_IMG = "Offroad_Segmentation_testImages/Color_Images"

CLASS_MAP = {
    100:0,200:1,300:2,500:3,550:4,
    600:5,700:6,800:7,7100:8,10000:9
}
