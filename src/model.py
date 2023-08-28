import timm
import torch
import torch.nn as nn
from torchinfo import summary


class SakeNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if hasattr(timm.models, cfg.model_name):
            base_model = timm.create_model(
                cfg.model_name, num_classes=0,
                pretrained=cfg.pretrained, in_chans=cfg.in_channels)
            self.backborn = base_model
            in_features = base_model.num_features
            print(f'load imagenet model_name: {cfg.model_name}')
            print(f'load imagenet pretrained: {cfg.pretrained}')
        else:
            raise NotImplementedError
        self.in_features = in_features
        self.fc = nn.Linear(self.in_features, cfg.embedding_dim)
    
    def get_embedding(self, image: torch.Tensor) -> torch.Tensor:
        emb = self.backborn(image)
        emb = self.fc(emb)
        return emb
    
    def forward(self, x) -> torch.Tensor:
        out = self.backborn(x)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    from pathlib import Path

    from main import Config
    
    EXP_NAME="convnext_base"
    device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
    cfg = Config
    batch_size = cfg.batch_size
    image_size = cfg.image_size
    
    model = SakeNet(cfg = Config)
    model = model.to(device)
    model_path = Path(__file__).parents[1].joinpath('models', f'{EXP_NAME}.pth')
    torch.save(model.state_dict(), model_path)
    
    dummy_tensor = (cfg.batch_size, cfg.in_channels, cfg.image_size, cfg.image_size)
    summary(model, input_size=dummy_tensor)
