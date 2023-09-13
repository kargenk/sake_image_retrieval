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

class SakeNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if hasattr(timm.models, cfg.model_name):
            print(f'load model_name: {cfg.model_name}')
            print(f'load imagenet pretrained: {cfg.pretrained}')
            
            # timmのモデルをバックボーンとして、GeM Poolingで埋め込みベクトルを得る
            self.backborn = timm.create_model(
                cfg.model_name, num_classes=0,
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

if __name__ == '__main__':
    from pathlib import Path

    from config import Config
    
    EXP_NAME='convnext_base'
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
