import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class GeMPool(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
    
    def gem(self, x, p: int = 3, eps: float = 1e-6):
        kernel_size = (x.size(-2), x.size(-1))
        return F.avg_pool2d(x.clamp(min=eps).pow(p), kernel_size).pow(1. / p)
    
    def __repr__(self):
        return self.__class__.__name__ + \
            f'(p={self.p.data.tolist()[0] :.4f}, eps={str(self.eps)})'

class ArcMarginProduct(nn.Module):
    # ref: https://github.com/smly/kaggle-book-gokui/blob/main/chapter4/model.py
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        input = input.float()

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

class SakeNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if hasattr(timm.models, cfg.model_name):
            print(f'load model_name: {cfg.model_name}')
            print(f'load imagenet pretrained: {cfg.pretrained}')
            
            # timmのモデルをバックボーンとして、GeM Poolingで埋め込みベクトルを得る
            self.backborn = timm.create_model(
                cfg.model_name,
                pretrained=cfg.pretrained,
                features_only=True,
                in_chans=cfg.in_channels
            )
            self.pooling = GeMPool()
            self.fc = nn.Linear(cfg.embedding_dim, cfg.num_target_class)
        else:
            raise NotImplementedError
    
    def get_embedding(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.size(0)
        
        features = self.backborn(images)[-1]
        emb = self.pooling(features).view(batch_size, -1)
        return emb
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        x = self.backborn(x)[-1]
        x = self.pooling(x).view(batch_size, -1)
        output = self.fc(x)
        return output

class AngularModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if hasattr(timm.models, cfg.model_name):
            print(f'load model_name: {cfg.model_name}')
            print(f'load imagenet pretrained: {cfg.pretrained}')
            
            # timmのモデルをバックボーンとして、GeM Poolingで埋め込みベクトルを得る
            self.backborn = timm.create_model(
                cfg.model_name,
                pretrained=cfg.pretrained,
                features_only=True,
                in_chans=cfg.in_channels
            )
            self.pooling = GeMPool()
            
            self.fc = nn.Linear(cfg.embedding_dim, cfg.fc_dim)
            self.bn = nn.BatchNorm1d(cfg.fc_dim)
            self._init_params()
            
            self.final = ArcMarginProduct(cfg.fc_dim, cfg.num_target_class)
        else:
            raise NotImplementedError
        
    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
    
    def get_embedding(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.size(0)
        
        x = self.backborn(images)[-1]
        x = self.pooling(x).view(batch_size, -1)
        
        # 全結合層
        x = self.fc(x)
        x = self.bn(x)
        
        return x
    
    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        features = self.get_embedding(images)
        logits = self.final(features, labels)
        return logits

if __name__ == '__main__':
    from pathlib import Path

    from config import Config
    
    EXP_NAME='efficientnet_b0_base'
    cfg = Config
    batch_size = cfg.batch_size
    image_size = cfg.image_size
    
    model = SakeNet(cfg=Config)
    model = model.to(cfg.device)
    model_path = Path(__file__).parents[1].joinpath('models', f'{EXP_NAME}.pth')
    torch.save(model.state_dict(), model_path)
    
    dummy_size = (cfg.batch_size, cfg.in_channels, cfg.image_size, cfg.image_size)
    summary(model, input_size=dummy_size)
    
    emb = model(torch.rand(dummy_size).to(cfg.device))
    print(emb.shape)
