import torch

def get_model_data_gan(dataset):
    if dataset.lower() == "mnist":
        params = {
            "optimizer_G": torch.optim.Adam,
            "optimizer_D": torch.optim.Adam,
            "batch_size": 64,
            "epochs": 5,
            "z_dim": 100,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR,
            "step_config": {
                "step_size" : 300,
                "gamma" : 0.1
            },
            "optim_config": {
                "lr": 1e-3,
                "weight_decay": 1
            }
        }
    else:
        raise ValueError("Dataset not known!")
    return params