# Clean working training script (DeepLabV3)
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import torch.optim as optim

# ---------------- DATASET ----------------
value_map = {0:0,100:1,200:2,300:3,500:4,550:5,700:6,800:7,7100:8,10000:9}
n_classes = len(value_map)

def convert_mask(mask):
    arr=np.array(mask)
    new=np.zeros_like(arr,dtype=np.uint8)
    for k,v in value_map.items():
        new[arr==k]=v
    return Image.fromarray(new)

class MaskDataset(Dataset):
    def __init__(self,data_dir,transform=None,mask_transform=None):
        self.image_dir=os.path.join(data_dir,'Color_Images')
        self.mask_dir=os.path.join(data_dir,'Segmentation')
        self.ids=os.listdir(self.image_dir)
        self.transform=transform
        self.mask_transform=mask_transform

    def __len__(self): return len(self.ids)

    def __getitem__(self,idx):
        name=self.ids[idx]
        img=Image.open(os.path.join(self.image_dir,name)).convert('RGB')
        mask=convert_mask(Image.open(os.path.join(self.mask_dir,name)))
        if self.transform: img=self.transform(img)
        if self.mask_transform: mask=self.mask_transform(mask)*255
        return img,mask

# ---------------- TRAIN ----------------
def main():
    BASE_DIR=os.path.dirname(os.path.abspath(__file__))
    train_dir=os.path.join(BASE_DIR,'Offroad_Segmentation_Training_Dataset','train')
    val_dir=os.path.join(BASE_DIR,'Offroad_Segmentation_Training_Dataset','val')

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device)

    transform=transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    mask_t=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])

    train_loader=DataLoader(MaskDataset(train_dir,transform,mask_t),batch_size=2,shuffle=True)
    val_loader=DataLoader(MaskDataset(val_dir,transform,mask_t),batch_size=2)

    print('Train samples:',len(train_loader.dataset))

    model=deeplabv3_resnet50(weights=None,num_classes=n_classes).to(device)
    opt=optim.Adam(model.parameters(),lr=1e-4)
    loss_fn=torch.nn.CrossEntropyLoss()

    epochs = 1

    for e in range(epochs):
        # ---- TRAIN ----
        model.train()
        total = 0
        for imgs, labels in tqdm(train_loader, desc=f'Epoch {e+1}/{epochs}'):
            imgs = imgs.to(device)
            labels = labels.squeeze(1).long().to(device)

            out = model(imgs)['out']
            loss = loss_fn(out, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print('Train loss:', total / len(train_loader))

        # ---- VALIDATION ----
        model.eval()
        vloss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.squeeze(1).long().to(device)
                out = model(imgs)['out']
                vloss += loss_fn(out, labels).item()

        print('Val loss:', vloss / len(val_loader))

    torch.save(model.state_dict(),'model.pth')
    print('Model saved')

if __name__=='__main__': main()