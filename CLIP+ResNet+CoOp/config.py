# config.py
import torch
import os

class Config:
    # --- 数据集路径 (改为 CSV 模式) ---
    TRAIN_CSV = r'研究拓展/dataset/train_subset.csv'  # 你的训练集 CSV 路径
    VAL_CSV = r'研究拓展/dataset/val_subset.csv'      # 你的验证集 CSV 路径

    # --- 实验输出配置 ---
    SAVE_DIR = r'研究拓展/CLIP+ResNet+CoOp/checkpoints'             # 模型权重保存目录
    LOG_FILE = r'研究拓展/CLIP+ResNet+CoOp/runs/training_log.txt'        # 训练日志保存路径

    # --- 图像预处理 ---
    IMAGE_SIZE = 224

    # --- MRT-43K 类别与文本提示 ---
    CLASSES = ['bedrock', 'gravel', 'looserock', 'norulerock', 'sand']
    NUM_CLASSES = len(CLASSES)

    # --- 训练超参数 ---
    BATCH_SIZE = 32
    EPOCHS = 50
    LR_CNN = 1e-4
    LR_CLIP = 1e-6
    WEIGHT_DECAY = 1e-4

    # --- 损失函数权重 ---
    LAMBDA_1 = 1.0
    LAMBDA_2 = 1.0

    # ---CoOp所需配置--- #
    N_CTX = 16
    coop_checkpoint = r"研究拓展/CLIP+Prompt_Tuning_CoOp/checkpoints/(Acc_80.90%)_coop_best.pth"

    # --- 设备 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def make_dirs(cls):
        """确保保存权重的文件夹存在"""
        if not os.path.exists(cls.SAVE_DIR):
            os.makedirs(cls.SAVE_DIR)