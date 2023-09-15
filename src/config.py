import torch


class Config:
    model_name = 'convnext_base'  # tf_efficientnet_b0_ns or convnext_base
    device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
    debug = False
    seed = 3407
    
    # Model params
    pretrained = True
    image_size = 224
    in_channels = 3
    embedding_dim = 1024  # efficientnet: 320, convnext-base: 1024
    num_target_class = 2499
    
    # Data params
    batch_size = 128
    num_workers  = 4
    num_epochs = 20
    n_fold = 5
    
    # for ArcFace params
    use_arc = True
    margin = 0.3
    scale = 30
    fc_dim = 512
    
    # Optimizer params
    optimizer_name = 'AdamW'
    optimizer_params = dict(
        lr=1e-4,
        weight_decay=1e-2,
        eps=1e-6,
        betas=(0.9, 0.999),
    )

    # Scheduler params
    scheduler_name = 'CosineAnnealingLR'
    scheduler_params = dict(
        T_max=500,
        eta_min=1e-6,
    )