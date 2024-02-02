from torch.optim import AdamW


def get_optimizer_grouped_parameters_no_llrd(model, cfg):

    no_decay = ['bias', "LayerNorm.bias", "LayerNorm.weight"]
    backbone_params = model.backbone.named_parameters()

    optimizer_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": cfg.optimizer.lr,
            "weight_decay": cfg.optimizer.weight_decay,
        },
        {
            "params": [p for n, p in backbone_params if not any(nd in n for nd in no_decay)],
            "lr": cfg.optimizer.lr,
            "weight_decay": cfg.optimizer.weight_decay,
        },
        {
            "params": [p for n, p in backbone_params if any(nd in n for nd in no_decay)],
            "lr": cfg.optimizer.lr,
            "weight_decay": 0.0,
        },
    ]

    return optimizer_parameters


def get_optimizer_grouped_parameters_with_llrd(model, cfg):
    """layerwise learning rate decay implementation
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": cfg.optimizer.head_lr,
            "weight_decay": cfg.optimizer.weight_decay,
        },
    ]

    # initialize lrs for backbone layers
    layers = [model.backbone.embeddings] + list(model.backbone.encoder.layer)
    layers.reverse()
    lr = cfg.optimizer.lr

    for layer in layers:
        lr *= cfg.optimizer.llrd

        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.optimizer.weight_decay,
                "lr": lr,
            },

            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

    return optimizer_grouped_parameters


def get_optimizer(model, cfg):
    """optimizer for model training
    """

    if cfg.optimizer.use_llrd:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters_with_llrd(model, cfg)
    else:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters_no_llrd(model, cfg)

    optimizer_kwargs = {
        "betas": (cfg.optimizer.beta1, cfg.optimizer.beta2),
        "eps": cfg.optimizer.eps,
        "lr": cfg.optimizer.lr
    }

    if cfg.optimizer.use_bnb:
        import bitsandbytes as bnb

        optimizer = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            **optimizer_kwargs
        )
        return optimizer
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters,
            **optimizer_kwargs
        )

    return optimizer
