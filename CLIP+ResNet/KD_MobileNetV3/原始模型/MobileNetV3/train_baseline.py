import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import logging
from tqdm import tqdm
#CUDA_VISIBLE_DEVICES=0 python 研究拓展/CLIP+ResNet/KD_MobileNetV3/原始模型/MobileNetV3/train_baseline.py
# 尝试导入较新版本的插值模式，兼容老版本
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    import PIL
    BICUBIC = PIL.Image.BICUBIC

# ==========================================
# 1. 统一配置区 (Configuration)
# ==========================================
class Config:
    # 数据集 CSV 路径 (请根据实际路径修改)
    TRAIN_CSV = r'研究拓展/dataset/train_subset.csv'  
    VAL_CSV = r'研究拓展/dataset/val_subset.csv'      
    
    # 输出与日志路径
    SAVE_DIR = r'研究拓展/CLIP+ResNet/KD_MobileNetV3/原始模型/MobileNetV3/checkpoints'
    LOG_FILE = r'研究拓展/CLIP+ResNet/KD_MobileNetV3/原始模型/MobileNetV3/runs/baseline_mobilenetv3.log'
    
    # 图像与类别
    IMAGE_SIZE = 224  
    NUM_CLASSES = 5
    
    # 训练超参数
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-4      
    WEIGHT_DECAY = 1e-4
    
    # 设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 数据加载区 (Dataset & DataLoader)
# ==========================================
class MarsCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
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
        try:
            # 找到 "MRT-43K" 的索引并截取后续部分
            target_index = path_parts.index("MRT-43K")
            img_path = os.path.join(*path_parts[target_index:])
        except ValueError:
            pass # 如果路径里没有 MRT-43K 就按原路径读取

        label = int(self.data_frame.iloc[idx]['label'])
        
        # 强制转换为 RGB，防止灰度图报错
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(cfg):
    # 标准的 ImageNet 归一化参数，同样适用于预训练的 MobileNet
    transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), interpolation=BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    train_dataset = MarsCSVDataset(csv_file=cfg.TRAIN_CSV, transform=transform)
    val_dataset = MarsCSVDataset(csv_file=cfg.VAL_CSV, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

# ==========================================
# 3. 日志与工具函数
# ==========================================
def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('MobileNetV3_Baseline')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # 防止重复添加 handler
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger

# ==========================================
# 4. 核心训练引擎 (Training Loop)
# ==========================================
def main():
    cfg = Config()
    
    # 确保保存目录存在
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    
    logger = setup_logger(cfg.LOG_FILE)
    logger.info("===========================================")
    logger.info(f"Starting MobileNetV3-Small Baseline Training on {cfg.DEVICE}")
    logger.info("===========================================")
    
    # 加载数据
    train_loader, val_loader = get_dataloaders(cfg)
    logger.info(f"Loaded {len(train_loader.dataset)} Train images and {len(val_loader.dataset)} Val images.")
    
    # === 修改处：初始化 MobileNetV3-Small ===
    logger.info("Loading pre-trained MobileNetV3-Small model...")
    # 推荐使用新的权重量写法，代替 pretrained=True (已被抛弃)
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    
    # 修改最后的全连接层适配火星地形分类 (5类)
    # MobileNetV3 的分类器包含一个 Dropout 和一个 Linear 层
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, cfg.NUM_CLASSES)
    
    model = model.to(cfg.DEVICE)
    # =======================================
    
    # 优化器与损失函数
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0 
    
    # 开始训练
    for epoch in range(cfg.EPOCHS):
        # -- 训练阶段 --
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
        for images, labels in pbar_train:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            pbar_train.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = 100. * correct_train / total_train
        
        # -- 验证阶段 --
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Val]  ")
            for images, labels in pbar_val:
                images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = 100. * correct_val / total_val
        
        # -- 日志输出 --
        logger.info(f"Epoch [{epoch+1:02d}/{cfg.EPOCHS:02d}] "
                    f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
                    f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
        
        # -- 保存最佳模型 --
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # 修复了原来保存路径字符串格式化的问题
            save_filename = f"(Acc_{best_val_acc:.2f}%)_mobilenetv3_baseline_best.pth"
            save_path = os.path.join(cfg.SAVE_DIR, save_filename)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_path)
            logger.info(f" >>> New best model saved to {save_path} (Acc: {best_val_acc:.2f}%)")

if __name__ == "__main__":
    main()