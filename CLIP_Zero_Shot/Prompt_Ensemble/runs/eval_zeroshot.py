import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import clip  # 需要 pip install git+https://github.com/openai/CLIP.git
from tqdm import tqdm
import logging
# python 研究拓展/CLIP_Zero_Shot/Prompt_Ensemble/runs/eval_zeroshot.py
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
# 3. 核心 Zero-shot 验证流程 (Prompt Ensemble)
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_csv_path = r'研究拓展/dataset/val_subset.csv' # 替换为你的验证集 CSV 路径
    log_path = r"研究拓展/CLIP_Zero_Shot/Prompt_Ensemble/runs/zeroshot_ensemble_eval.log"
    batch_size = 32
    
    logger = setup_logger(log_path)
    logger.info(f"Starting CLIP Zero-Shot Evaluation (Prompt Ensemble) on {device}")
    
    # 步骤 A: 加载官方 CLIP 模型和其配套的图像预处理函数
    model_name = "ViT-B/32"  
    model, preprocess = clip.load(model_name, device=device)
    logger.info(f"Loaded CLIP Model: {model_name}")
    
    # 加载数据 
    val_dataset = MarsCSVDataset(csv_file=val_csv_path, transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info(f"Loaded {len(val_dataset)} Validation images.")
    
    # ==========================================================
    # 步骤 B: 准备文本特征 (Prompt Ensemble 核心逻辑)
    # ==========================================================
    # 根据 MRT-43K 论文 Table B.1 定义扩展的高质量地质学提示词
    prompt_templates = {
        'bedrock': [
            "A close-up view of a massive, flat bedrock formation on the Martian surface.",
            "A photo of solid and stable Martian bedrock with a platy structure.",
            "Martian terrain showing continuous, exposed bedrock captured by a rover.",
            "A high-resolution image of smooth, stable rock layers on Mars.",
            "Rover-perspective view of unbroken Martian bedrock without loose fragments.",
            "A satellite or rover photo highlighting the solid bedrock terrain of Mars.",
            "Martian ground characterized by large, flat, and firmly embedded rocks.",
            "An image of Martian bedrock, indicating a low traversability risk area.",
            "A view of Martian terrain featuring a stable, platy rock surface.",
            "Clear Martian geological structures showing solid bedrock formations."
        ],
        'gravel': [
            "Martian terrain featuring rocky fragments mixed with sand and soil.",
            "A relatively firm Martian surface covered in fine gravel and pebbles.",
            "A photo of Martian ground characterized by gravel and small rocky fragments.",
            "Rover view of a level Martian surface composed of mixed gravel.",
            "Martian soil with scattered tiny rocks and gravel particles.",
            "A close-up of Martian gravel terrain with a firm, navigable structure.",
            "Martian landscape showing a mix of coarse sand and small gravel.",
            "An image of the Martian surface covered in pebble-sized rock fragments.",
            "Terrain on Mars displaying a uniform distribution of fine gravel.",
            "A photo of compact Martian dirt mixed with small gravel pieces."
        ],
        'looserock': [
            "Martian surface scattered with discrete, medium-to-large loose rocks.",
            "A photo showing Martian terrain with scattered rocks posing puncture risks.",
            "Rover perspective of discrete rocks resting loosely on the Martian soil.",
            "Martian terrain featuring individual, distinct loose rocks that require detours.",
            "A view of the Martian surface littered with independent, movable rocks.",
            "Scattered loose rocks on Mars, separated by patches of sand or soil.",
            "An image of Martian terrain with hazardous loose stones on the surface.",
            "Medium-sized individual rocks scattered across the red Martian landscape.",
            "A photo of discrete Martian rocks that are not embedded in the ground.",
            "Martian ground showing a moderate density of loose, detached rocks."
        ],
        'norulerock': [
            "Rugged Martian terrain densely populated with irregular large boulders.",
            "A highly obstructed Martian surface filled with chaotic, no-rule rocks.",
            "A photo of dangerous Martian terrain packed with massive, jagged rocks.",
            "Rover view of an untraversable area densely covered in irregular boulders.",
            "Chaotic Martian geological formations with overlapping large rocks.",
            "A rugged, heavily rock-strewn Martian landscape presenting significant navigation obstacles.",
            "An image of intensely rocky Martian terrain with no clear path.",
            "Densely clustered, irregular boulders dominating the Martian surface.",
            "A harsh Martian environment covered completely by jagged, unpatterned rocks.",
            "A close-up of a high-risk Martian area packed with formidable no-rule boulders."
        ],
        'sand': [
            "A photo of fine, loose Martian sand covering the terrain.",
            "Martian terrain showing loose sandy areas prone to causing wheel sinkage.",
            "A clear view of smooth, red Martian sand dunes or ripples.",
            "Rover perspective of soft, loose sand on the surface of Mars.",
            "A photo of featureless, fine-grained Martian sandy terrain.",
            "Martian landscape dominated by loose soil and wind-blown sand.",
            "An image of soft sandy terrain on Mars, presenting slipping hazards for rovers.",
            "A close-up of smooth, undisturbed Martian sand.",
            "Expansive Martian terrain covered entirely by fine red sand.",
            "A photo of a Martian sand patch without rocks or solid structures."
        ]
    }
    
    logger.info("Extracting ensembled text features...")
    ensembled_features_list = []
    
    # 按照 CLASSES 的顺序提取特征，确保标签索引 0~4 一一对应
    CLASSES = ['bedrock', 'gravel', 'looserock', 'norulerock', 'sand']
    
    with torch.no_grad():
        for class_name in CLASSES:
            prompts = prompt_templates[class_name]
            
            # 1. 词元化 (Tokenize) 这 10 句话
            text_tokens = clip.tokenize(prompts).to(device)
            
            # 2. 提取特征 -> 维度: [10, 512]
            class_features = model.encode_text(text_tokens)
            
            # 3. 沿着第 0 维度求平均，融合 10 句话的语义 -> 维度: [1, 512]
            class_features = class_features.mean(dim=0, keepdim=True)
            
            # 4. 求平均后必须重新进行 L2 归一化
            class_features /= class_features.norm(dim=-1, keepdim=True)
            
            ensembled_features_list.append(class_features)
            
    # 5. 将 5 个类别的综合特征拼接起来 -> 维度: [5, 512]
    text_features = torch.cat(ensembled_features_list, dim=0)
    logger.info("Ensembled text features extracted successfully.")

    # ==========================================================
    # 步骤 C: 开始遍历图片进行预测
    # ==========================================================
    model.eval()
    correct_val, total_val = 0, 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Zero-Shot Ensemble Eval]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # 提取图像特征并归一化
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度 (Cosine Similarity)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # 取概率最大的类别索引作为预测结果
            _, predicted = similarity.max(dim=-1)
            
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
            pbar.set_postfix({'Acc': f"{100.*correct_val/total_val:.2f}%"})
            
    final_acc = 100. * correct_val / total_val
    logger.info(f"Zero-Shot Ensemble Evaluation Finished! Final Accuracy: {final_acc:.2f}%")

if __name__ == "__main__":
    main()