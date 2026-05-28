# model.py
import torch
import torch.nn as nn
import torchvision.models as models
import clip

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
        fused_feat = torch.cat((res_feat, clip_feat.detach()), dim=1) # [B, 2560]
        
        # 计算分类 logits
        logits = self.classifier(fused_feat)
        
        # 返回 logits 用于算分类损失(Loss2)；返回归一化后的 clip_feat 用于算对比损失(Loss1)
        return logits, clip_feat