import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import clip  # 需要 pip install git+https://github.com/openai/CLIP.git
from tqdm import tqdm
import logging
#python 研究拓展/CLIP_Zero_Shot/eval_zeroshot.py
# ==========================================
# 1. 数据加载区 (Dataset)
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

        # 路径截取逻辑，保留原样
        path_parts = img_path.split(os.sep)
        try:
            target_index = path_parts.index("MRT-43K")
            img_path = os.path.join(*path_parts[target_index:])
        except ValueError:
            pass # 如果路径里没有 MRT-43K 就按原路径读取

        label = int(self.data_frame.iloc[idx]['label'])
        image = Image.open(img_path).convert('RGB')
        
        # Zero-shot 必须使用 CLIP 官方的 transform 进行预处理
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==========================================
# 2. 日志工具
# ==========================================
def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('CLIP_ZeroShot')
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
# 3. 核心 Zero-shot 验证流程
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_csv_path = r'研究拓展/dataset/val_subset.csv' # 替换为你的验证集 CSV 路径
    log_path = r"研究拓展/CLIP_Zero_Shot/runs/zeroshot_eval.log"
    batch_size = 32
    
    logger = setup_logger(log_path)
    logger.info(f"Starting CLIP Zero-Shot Evaluation on {device}")
    
    # 步骤 A: 加载官方 CLIP 模型和其配套的图像预处理函数
    model_name = "ViT-B/32"  # 可以换成 "ViT-B/16" 或 "RN50"
    model, preprocess = clip.load(model_name, device=device)
    logger.info(f"Loaded CLIP Model: {model_name}")
    
    # 加载数据 (注意：此处传入的是 CLIP 的 preprocess)
    val_dataset = MarsCSVDataset(csv_file=val_csv_path, transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info(f"Loaded {len(val_dataset)} Validation images.")
    
    # 步骤 B: 准备文本特征
    # 这里是 Zero-shot 的灵魂：构造文本 Prompt
    CLASSES = ['bedrock', 'gravel', 'looserock', 'norulerock', 'sand']
    # 模板工程 (Prompt Engineering): 越贴近训练数据的语境效果越好
    text_descriptions =[f"A photo of Mars terrain showing {c.lower()}." for c in CLASSES]
    logger.info(f"Using text prompts: {text_descriptions}")
    
    text_tokens = clip.tokenize(text_descriptions).to(device)
    
    # 提取并归一化文本特征 (因为不需要训练，这里用 torch.no_grad)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # 归一化，为了后续算余弦相似度
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 步骤 C: 开始遍历图片进行预测
    model.eval()
    correct_val, total_val = 0, 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Zero-Shot Eval]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # 提取图像特征并归一化
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度 (Cosine Similarity)
            # image_features 维度 [Batch, Feature_Dim]
            # text_features 维度 [Class_Num, Feature_Dim]
            # 矩阵乘法得到相似度矩阵 [Batch, Class_Num]
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # 取概率最大的类别索引作为预测结果
            _, predicted = similarity.max(dim=-1)
            
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
            pbar.set_postfix({'Acc': f"{100.*correct_val/total_val:.2f}%"})
            
    final_acc = 100. * correct_val / total_val
    logger.info(f"Zero-Shot Evaluation Finished! Final Accuracy: {final_acc:.2f}%")

if __name__ == "__main__":
    main()