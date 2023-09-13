import torch


class Config:
    model_name = 'convnext_base'
    debug = False
    pretrained = True
    seed = 3407
    device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
    
    image_size = 224
    in_channels = 3
    embedding_dim = 1024
    num_target_class = 10
    batch_size = 256
    num_workers  = 1
    num_epochs = 300
    n_fold = 5
    
    optimizer_name = 'AdamW'
    optimizer_params = dict(
        lr=1e-4,
        weight_decay=1e-2,
        eps=1e-6,
        betas=(0.9, 0.999),
    )

    scheduler_name = 'CosineAnnealingLR'
    scheduler_params = dict(
        T_max=500,
        eta_min=1e-6,
    )