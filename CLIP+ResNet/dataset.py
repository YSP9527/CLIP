# dataset.py
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    import PIL
    BICUBIC = PIL.Image.BICUBIC

class MarsCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        :param csv_file: CSV 文件路径。假设包含 'image_path' 和 'label' 两列
        :param transform: 图像预处理操作
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_frame.iloc[idx]['image_path']
        
        # 以 "/" 为分隔符将路径拆分成列表
        path_parts = img_path.split(os.sep)
        # 找到 "MRT-43K" 的索引并截取后续部分
        target_index = path_parts.index("MRT-43K")
        img_path = os.path.join(*path_parts[target_index:])

        label = int(self.data_frame.iloc[idx]['label'])
        
        # 强制转换为 RGB，防止读取到灰度图报错
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(cfg):
    # 统一使用 CLIP 的标准化参数
    shared_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), interpolation=BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    # 使用自定义的 CSV Dataset
    train_dataset = MarsCSVDataset(csv_file=cfg.TRAIN_CSV, transform=shared_transform)
    val_dataset = MarsCSVDataset(csv_file=cfg.VAL_CSV, transform=shared_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, val_loader