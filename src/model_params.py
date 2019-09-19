import torch

def get_model_data_gan(dataset):
    if dataset.lower() == "mnist":
        params = {
            "epochs": 50,
            "batch_size": 64,
            "z_dim": 100,
            "optimizer_G": torch.optim.Adam,
            "optimizer_D": torch.optim.Adam,
            "optim_config_G": {
                "lr": 2e-4,
                "weight_decay": None,
                "betas": (0.5, 0.999)
            },
            "optim_config_D": {
                "lr": 2e-4,
                "weight_decay": None,
                "betas": (0.5, 0.999)
            },
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "step_config": {
                "step_size" : 300,
                "gamma" : 0.1
            }
        }
    else:
        raise ValueError("Dataset not known!")
    return params