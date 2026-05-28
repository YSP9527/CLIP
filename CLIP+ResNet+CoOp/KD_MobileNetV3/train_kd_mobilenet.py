import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import logging
from tqdm import tqdm
import clip
#CUDA_VISIBLE_DEVICES=0 python 研究拓展/CLIP+ResNet/KD_MobileNetV3/train_kd_mobilenet.py
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
    TRAIN_CSV = r'研究拓展/dataset/train_subset.csv'
    VAL_CSV = r'研究拓展/dataset/val_subset.csv'       
    SAVE_DIR = r'研究拓展/CLIP+ResNet+CoOp/KD_MobileNetV3/checkpoints'
    LOG_FILE = r'研究拓展/CLIP+ResNet+CoOp/KD_MobileNetV3/runs/kd_mobilenetv3.log'
    
    IMAGE_SIZE = 224  
    NUM_CLASSES = 5
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-4      
    WEIGHT_DECAY = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # === 核心修改 1：蒸馏超参数 ===
    TEACHER_WEIGHT_PATH = r'研究拓展/CLIP+ResNet+CoOp/checkpoints/(Acc: 90.80%)_best_model.pth' # 替换为真实路径
    ALPHA = 0.5        # 真实标签(CE) Loss 的权重
    BETA = 0.5         # 蒸馏(KD) Loss 的权重 (通常 ALPHA + BETA = 1)
    TEMPERATURE = 4.0  # 温度系数 T，越大软标签越平滑，推荐 3~5 之间

# ==========================================
# 2. 蒸馏损失函数定义 (KL Divergence)
# ==========================================
def distillation_loss(student_logits, teacher_logits, temperature):
    """
    计算知识蒸馏的 KL 散度损失。
    注意：PyTorch 的 kl_div 期望输入是 log_softmax，目标是 softmax。
    """
    # 学生侧需要 log_softmax
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    # 老师侧作为目标概率，需要 softmax (不需要梯度)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    # 计算 KL 散度，需乘以 T^2 保持梯度量级与 CE Loss 匹配
    loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss

# ==========================================
# 数据加载与日志 (与之前完全相同)
# ==========================================

class DualBranchMarsClassifier(nn.Module):
    def __init__(self, num_classes, clip_model_name="ViT-B/32"):
        super(DualBranchMarsClassifier, self).__init__()
        
        # 1. ResNet50 分支
        resnet = models.resnet50(pretrained=True)
        # 剥离最后的 fc 层
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        resnet_dim = 2048 
        
        # 2. CLIP 分支
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")
        self.clip_image_encoder = self.clip_model.visual
        clip_dim = self.clip_model.visual.output_dim
        
        # 3. 融合后的分类头 (Classifier Head)
        fused_dim = resnet_dim + clip_dim # 2048 + 512 = 2560
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # 提取 ResNet 特征
        res_feat = self.resnet_features(x)
        res_feat = torch.flatten(res_feat, 1) # [B, 2048]
        
        # 提取 CLIP 特征
        clip_feat = self.clip_image_encoder(x.type(self.clip_model.dtype)) # [B, 512]
        # 对 CLIP 特征进行 L2 归一化 (对齐对比学习的要求)
        clip_feat = clip_feat / clip_feat.norm(dim=1, keepdim=True)
        
        # 特征拼接 (Concatenation)
        fused_feat = torch.cat((res_feat, clip_feat), dim=1) # [B, 2560]
        
        # 计算分类 logits
        logits = self.classifier(fused_feat)
        
        # 返回 logits 用于算分类损失(Loss2)；返回归一化后的 clip_feat 用于算对比损失(Loss1)
        return logits, clip_feat


class MarsCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.data_frame)
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        img_path = self.data_frame.iloc[idx]['image_path']
        path_parts = img_path.split(os.sep)
        try:
            target_index = path_parts.index("MRT-43K")
            img_path = os.path.join(*path_parts[target_index:])
        except ValueError:
            pass 
        label = int(self.data_frame.iloc[idx]['label'])
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, label

def get_dataloaders(cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), interpolation=BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    train_loader = DataLoader(MarsCSVDataset(cfg.TRAIN_CSV, transform), batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(MarsCSVDataset(cfg.VAL_CSV, transform), batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('KD_MobileNetV3')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

# ==========================================
# 3. 核心训练引擎 (Training Loop)
# ==========================================
def main():
    cfg = Config()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    logger = setup_logger(cfg.LOG_FILE)
    logger.info("===========================================")
    logger.info(f"Starting Dual-Teacher Distillation on {cfg.DEVICE}")
    logger.info(f"Alpha(CE): {cfg.ALPHA}, Beta(KD): {cfg.BETA}, Temp: {cfg.TEMPERATURE}")
    logger.info("===========================================")
    
    train_loader, val_loader = get_dataloaders(cfg)
    
    # === 核心修改 2：加载教师模型 (你的 90.80% 融合模型) ===
    logger.info("Loading Teacher Model...")
    # TODO: 实例化你的 CLIP+ResNet50 融合模型
    teacher_model = DualBranchMarsClassifier(num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)
    checkpoint = torch.load(cfg.TEACHER_WEIGHT_PATH, map_location=cfg.DEVICE)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model.eval()

    # 冻结教师模型所有参数，不计算梯度
    for param in teacher_model.parameters():
        param.requires_grad = False

    # === 核心修改 3：加载学生模型 ===
    logger.info("Loading Student Model (MobileNetV3-Small)...")
    student_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = student_model.classifier[3].in_features
    student_model.classifier[3] = nn.Linear(in_features, cfg.NUM_CLASSES)
    student_model = student_model.to(cfg.DEVICE)
    
    # 优化器只传给学生的参数！
    optimizer = optim.AdamW(student_model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    criterion_ce = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0 
    
    for epoch in range(cfg.EPOCHS):
        # -- 训练阶段 --
        student_model.train()
        train_loss, train_loss_ce, train_loss_kd = 0.0, 0.0, 0.0
        correct_train, total_train = 0, 0
        
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
        for images, labels in pbar_train:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            # 1. 教师生成软标签 (不需要梯度)
            with torch.no_grad():
                teacher_logits, _ = teacher_model(images)
                
            # 2. 学生前向传播
            optimizer.zero_grad()
            student_logits = student_model(images)
            
            # 3. 计算两部分 Loss
            loss_ce = criterion_ce(student_logits, labels)
            loss_kd = distillation_loss(student_logits, teacher_logits, cfg.TEMPERATURE)
            
            # 4. 联合优化
            loss = cfg.ALPHA * loss_ce + cfg.BETA * loss_kd
            
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * images.size(0)
            train_loss_ce += loss_ce.item() * images.size(0)
            train_loss_kd += loss_kd.item() * images.size(0)
            
            _, predicted = torch.max(student_logits.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            pbar_train.set_postfix({
                'Loss': f"{loss.item():.4f}", 
                'CE': f"{loss_ce.item():.4f}", 
                'KD': f"{loss_kd.item():.4f}"
            })
            
        epoch_train_acc = 100. * correct_train / total_train
        
        # -- 验证阶段 (只验证学生模型) --
        student_model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Val]  ")
            for images, labels in pbar_val:
                images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                
                outputs = student_model(images)
                loss = criterion_ce(outputs, labels) # 验证集只看真实准确率，不用 KD Loss
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = 100. * correct_val / total_val
        
        # -- 日志输出 --
        logger.info(f"Epoch [{epoch+1:02d}/{cfg.EPOCHS:02d}] "
                    f"Train Acc: {epoch_train_acc:.2f}% (CE: {train_loss_ce/total_train:.4f}, KD: {train_loss_kd/total_train:.4f}) | "
                    f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
        
        # -- 保存最佳模型 --
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_filename = f"(Acc_{best_val_acc:.2f}%)_kd_mobilenetv3_best.pth"
            save_path = os.path.join(cfg.SAVE_DIR, save_filename)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_path)
            logger.info(f" >>> New best KD model saved to {save_path} (Acc: {best_val_acc:.2f}%)")

if __name__ == "__main__":
    main()