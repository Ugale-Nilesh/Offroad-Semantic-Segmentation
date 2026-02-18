import os, cv2, torch, numpy as np
from torch.utils.data import Dataset
from core.configs.config import CLASS_MAP, IMAGE_SIZE


def encode_mask(mask):
    out = np.zeros_like(mask)
    for k,v in CLASS_MAP.items():
        out[mask==k]=v
    return out

class OffroadDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))/255.0
        img = torch.tensor(img).permute(2,0,1).float()

        if self.mask_dir is None:
            return img,self.images[idx]

        mask_path = os.path.join(self.mask_dir,self.images[idx])
        mask = cv2.imread(mask_path,0)
        mask = cv2.resize(mask,(IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_NEAREST)
        mask = torch.tensor(encode_mask(mask)).long()
        return img,mask