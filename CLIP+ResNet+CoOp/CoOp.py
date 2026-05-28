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