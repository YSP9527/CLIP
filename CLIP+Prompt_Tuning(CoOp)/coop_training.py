import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import clip  # pip install git+https://github.com/openai/CLIP.git
import logging
from tqdm import tqdm
# CUDA_VISIBLE_DEVICES=0 python 研究拓展/CLIP+Prompt_Tuning_CoOp/coop_training.py
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    import PIL
    BICUBIC = PIL.Image.BICUBIC

# ==========================================
# 1. 统一配置区
# ==========================================
class Config:
    TRAIN_CSV = r'研究拓展/dataset/train_subset.csv'
    VAL_CSV = r'研究拓展/dataset/val_subset.csv'
    SAVE_DIR = r'研究拓展/CLIP+Prompt_Tuning_CoOp/checkpoints'
    LOG_FILE = r'研究拓展/CLIP+Prompt_Tuning_CoOp/runs/coop_training.log'

    IMAGE_SIZE = 224
    NUM_CLASSES = 5
    BATCH_SIZE = 32
    EPOCHS = 50

    # === 核心改动 1：CoOp 的超参数 ===
    # 提示词向量的长度，原论文推荐 16 或 4
    CTX_INIT = ""      # 如果不用随机初始化，可以给一句起始话，比如 "a photo of a"
    N_CTX = 16         # 也就是我们说的 V1, V2 ... V16
    CTX_DIM = 512      # CLIP ViT-B/32 的特征维度

    # 学习率要稍微设大一点，因为只有几千个参数在训练
    LR = 2e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 数据加载与日志 (与之前相同)
# ==========================================
class MarsCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self): return len(self.data_frame)
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

def get_dataloaders(cfg, preprocess):
    train_loader = DataLoader(MarsCSVDataset(cfg.TRAIN_CSV, preprocess), batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(MarsCSVDataset(cfg.VAL_CSV, preprocess), batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('CoOp')
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
# 2. CoOp 核心模块 (The Magic)
# ==========================================

# === 核心改动 2：Prompt Learner (提示词学习器) ===
# 作用：它的唯一任务就是维护并生成那 16 个可学习的向量，并将它们与类别名拼接。
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, cfg):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.N_CTX
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype

        # 1. 随机初始化可学习的 Context 向量 (n_ctx=16)
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]

        # === 终极修复：使用占位符 (Dummy) 生成正确的 Token 序列 ===
        # 我们用 "X " 重复 16 次来占位，这样生成的 tokenized_prompts 长度永远是 77
        # 且 argmax() 能绝对精准地找到 16 个上下文之后的 [EOT] 标签！
        dummy_prompts = [" ".join(["X"] * n_ctx) + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in dummy_prompts]).to(cfg.DEVICE)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # 提取各个部分 (SOT 在 0; 占位符在 1 到 n_ctx; 剩下的是后缀)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # [SOT]
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # [CLASS] + [EOT] + PAD

        self.tokenized_prompts = tokenized_prompts # 带有正确 EOT 位置的 Token 索引
        self.n_cls = n_cls
        self.n_ctx = n_ctx

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # 此时：1 (SOT) + 16 (ctx) + 60 (suffix) = 严格等于 77！
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

# === 核心改动 3：自定义 CLIP 前向传播 ===
# 作用：因为我们拼接出来的 prompt 是连续的浮点向量，不能直接调用 clip.encode_text (它只接受整数ID)。
# 我们需要写一个底层的 forward 来处理这批浮点向量。
class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model, cfg):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, cfg)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # 引用 CLIP 的组件
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model, cfg.N_CTX)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        # 1. 提取图像特征并归一化
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 2. 让 Prompt Learner 生成最新的 (SOT + V + CLASS + EOT) 向量
        prompts = self.prompt_learner()

        # 3. 提取文本特征并归一化
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 4. 计算 Cosine 相似度，并乘以温度系数 logit_scale
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

# CLIP 内部文本编码器的底层逻辑
class TextEncoder(nn.Module):
    def __init__(self, clip_model, n_ctx):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.n_ctx = n_ctx

    def forward(self, prompts, tokenized_prompts):
        # 获取所需的序列长度 (现在由于占位符魔法，这里会严格等于 77)
        seq_len = prompts.shape[1]

        # 调整 positional embedding 的大小以匹配当前序列长度 (完美的防御性编程)
        adjusted_pos_embedding = self.positional_embedding[:seq_len].type(self.dtype)
        x = prompts + adjusted_pos_embedding

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # 因为 tokenized_prompts 被占位符撑开了，这里的 argmax 能精准定位到真实的 EOT！
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

# ==========================================
# 3. 核心训练引擎
# ==========================================
def main():
    cfg = Config()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    logger = setup_logger(cfg.LOG_FILE)

    logger.info("===========================================")
    logger.info("Starting CoOp (Prompt Tuning) Training")
    logger.info("===========================================")

    # 加载 CLIP (冻结所有参数)
    clip_model, preprocess = clip.load("ViT-B/32", device=cfg.DEVICE)
    for param in clip_model.parameters():
        param.requires_grad = False

    train_loader, val_loader = get_dataloaders(cfg, preprocess)

    # 类别名称 (供 Prompt Learner 使用)
    CLASSES = ['bedrock', 'gravel', 'looserock', 'norulerock', 'sand']

    # 初始化 CustomCLIP
    model = CustomCLIP(CLASSES, clip_model, cfg).to(cfg.DEVICE)

    # === 核心改动 4：只优化 prompt_learner.ctx ===
    # 注意：绝对不要把整个 model.parameters() 传给优化器！
    optimizer = optim.SGD(model.prompt_learner.parameters(), lr=cfg.LR, momentum=0.9)
    # Cosine 学习率衰减 (CoOp 标配)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(cfg.EPOCHS):
        # -- 训练阶段 --
        model.train()
        model.image_encoder.eval()
        model.text_encoder.eval()
        train_loss, correct_train, total_train = 0.0, 0, 0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
        for images, labels in pbar_train:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits.float(), labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            pbar_train.set_postfix({'Loss': f"{loss.item():.4f}"})

        scheduler.step()
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = 100. * correct_train / total_train

        # -- 验证阶段 --
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Val]  ")
            for images, labels in pbar_val:
                images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)

                logits = model(images)
                loss = criterion(logits.float(), labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = 100. * correct_val / total_val

        logger.info(f"Epoch [{epoch+1:02d}/{cfg.EPOCHS:02d}] "
                    f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
                    f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_path = os.path.join(cfg.SAVE_DIR, f"(Acc_{best_val_acc:.2f}%)_coop_best.pth")
            # 在 Prompt Tuning 中，我们通常只保存那 16 个向量，因为大模型根本没变
            torch.save({
                'epoch': epoch + 1,
                'prompt_learner': model.prompt_learner.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_path)
            logger.info(f" >>> New best CoOp prompts saved to {save_path} (Acc: {best_val_acc:.2f}%)")

if __name__ == "__main__":
    main()
