import torch
import torch.nn as nn
import clip

class CLIPMLPClassifier(nn.Module):
    def __init__(self, num_classes=5, clip_model_name="ViT-B/32", freeze_backbone=True):
        super(CLIPMLPClassifier, self).__init__()
        
        # 加载 CLIP 模型
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")
        self.image_encoder = self.clip_model.visual
        
        # 获取 CLIP 视觉特征维度
        if "RN" in clip_model_name:
            clip_dim = self.image_encoder.output_dim
        else: # ViT models
            clip_dim = self.image_encoder.output_dim

        # 冻结 CLIP 视觉编码器 (MLP Probe 范式)
        if freeze_backbone:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
                
        # 定义 MLP 分类头
        self.mlp_head = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 1. 获取 CLIP 视觉特征
        with torch.set_grad_enabled(not self.image_encoder.parameters().__next__().requires_grad == False):
            # 如果冻结了骨干网络，使用 torch.no_grad() 可以节省显存并加速
            img_features = self.image_encoder(x.type(self.clip_model.dtype))
            
        # 可选：对特征进行 L2 归一化，通常对 CLIP 提取的特征有帮助
        img_features = img_features / img_features.norm(dim=1, keepdim=True)
        
        # 2. 通过 MLP 分类器
        logits = self.mlp_head(img_features)
        
        return logits