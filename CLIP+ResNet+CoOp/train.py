# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import clip
import logging
import os
from config import Config
from dataset import get_dataloaders
from model import DualBranchMarsClassifier
from tqdm import tqdm
from CoOp import PromptLearner, TextEncoder

#CUDA_VISIBLE_DEVICES=0 python 研究拓展/CLIP+ResNet+CoOp/train.py
def setup_logger(log_file):
    """配置日志，使其同时输出到控制台和文件"""
    logger = logging.getLogger('MarsTraining')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件 Handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台 Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def train():
    cfg = Config()
    cfg.make_dirs() # 创建保存目录
    logger = setup_logger(cfg.LOG_FILE)
    logger.info("Starting training process...")

    # 1. 加载数据
    train_loader, val_loader = get_dataloaders(cfg)
    logger.info(f"Loaded {len(train_loader.dataset)} training images and {len(val_loader.dataset)} validation images.")

    # 2. 初始化模型
    model = DualBranchMarsClassifier(num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)

    # 3. 准备 CLIP 文本特征
    prompt_learner = PromptLearner(cfg.CLASSES, model.clip_model, cfg).to(cfg.DEVICE)
    text_encoder = TextEncoder(model.clip_model, cfg.N_CTX).to(cfg.DEVICE)

    # 2. 加载你跑到 80.6% 的 CoOp 权重
    coop_checkpoint = torch.load(cfg.coop_checkpoint)
    prompt_learner.load_state_dict(coop_checkpoint['prompt_learner'])

    # 3. 极其关键：将它们设置为 eval 模式，并计算出 5 个类别的终极文本特征
    prompt_learner.eval()
    text_encoder.eval()

    with torch.no_grad(): # 这个文本特征是完美的“答案”，不需要更新！
        prompts = prompt_learner()
        # 得到 [5, 512] 的终极多模态文本锚点
        coop_text_features = text_encoder(prompts, prompt_learner.tokenized_prompts)
        # 归一化！
        coop_text_features = coop_text_features / coop_text_features.norm(dim=-1, keepdim=True)

    # 4. 优化器配置
    clip_params = list(model.clip_image_encoder.parameters())
    cnn_and_head_params = list(model.resnet_features.parameters()) + list(model.classifier.parameters())
    optimizer = optim.AdamW([
        {'params': cnn_and_head_params, 'lr': cfg.LR_CNN},
        {'params': clip_params, 'lr': cfg.LR_CLIP}
    ], weight_decay=cfg.WEIGHT_DECAY)

    # 5. 损失函数
    criterion_ce = nn.CrossEntropyLoss()
    logit_scale = model.clip_model.logit_scale

    # 记录最佳验证集精度
    best_val_acc = 0.0

    # 6. 开始训练循环
    for epoch in range(cfg.EPOCHS):
        # ----------------- 训练阶段 -----------------
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
        for images, labels in pbar_train:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)

            optimizer.zero_grad()
            logits, clip_image_features = model(images)

            loss_2 = criterion_ce(logits, labels)
            sim_logits = logit_scale.exp() * clip_image_features @ coop_text_features.t()
            loss_1 = criterion_ce(sim_logits, labels)

            loss = cfg.LAMBDA_1 * loss_1 + cfg.LAMBDA_2 * loss_2

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            pbar_train.set_postfix({'Loss': loss.item()})

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = 100. * correct_train / total_train

        # ----------------- 验证阶段 -----------------
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # 验证时不计算梯度，节省显存并加速
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Val]")
            for images, labels in pbar_val:
                images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)

                logits, clip_image_features = model(images)

                loss_2 = criterion_ce(logits, labels)
                sim_logits = logit_scale.exp() * clip_image_features @ coop_text_features.t()
                loss_1 = criterion_ce(sim_logits, labels)

                loss = cfg.LAMBDA_1 * loss_1 + cfg.LAMBDA_2 * loss_2

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = 100. * correct_val / total_val

        # ----------------- 日志打印与模型保存 -----------------
        logger.info(f"Epoch [{epoch+1}/{cfg.EPOCHS}] "
                    f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
                    f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

        # 只有当前验证集精度更高时，才保存权重
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_path = os.path.join(cfg.SAVE_DIR, f'(Acc: {best_val_acc:.2f}%)_best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_path)
            logger.info(f"--> Saved new best model to {save_path} (Acc: {best_val_acc:.2f}%)")

if __name__ == "__main__":
    train()