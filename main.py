import numpy as np
import torch
from tqdm import tqdm

from torchvision.transforms import v2 as T

from src.dicts import TASKS, MODELS, OPTIMS, LOSSES
from src.util import (
    get_root_dir,
    mkdir,
    create_hyperparam_dirname,
    seed_everything,
    accuracy,
    parse_and_process_cfg,
    HyperParamSearcher,
)


def train_single_run(cfg, task, model, optimizer, loss_fn, torch_device):
    losses = np.zeros(cfg["n_timesteps"])
    accuracies = np.zeros(cfg["n_timesteps"])
    
    model = model.to(torch_device)
    model.train()
    for t, (x, y) in enumerate(task):
        x, y = x.to(torch_device), y.to(torch_device)
        pred = model(x)
        
        optimizer.zero_grad()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        losses[t] = loss.detach().cpu().item()
        
        with torch.no_grad():
            p = model(x)
            p = p.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            accuracies[t] = accuracy(p, y)
            
        if t + 1 >= cfg["n_timesteps"]:
            break
    
    res_dict = {"losses": losses, "accuracies": accuracies}
    return res_dict


def main():
    # Parse arguments
    orig_cfg = parse_and_process_cfg()

    # Create hyperparameter searcher
    search_keys = orig_cfg["search_keys"]
    hyperparam_searcher = HyperParamSearcher(orig_cfg, search_keys)

    # Random seeds
    seed_everything(orig_cfg["seed"])
    seeds = np.random.randint(0, 1000000000, size=orig_cfg["n_runs"])

    # Device
    torch_device = torch.device(orig_cfg["device"])
    orig_cfg["model_params"]["torch_device"] = orig_cfg["device"]

    # Run training
    for cfg in hyperparam_searcher:
        # Make directory for each hyperparameters
        if len(search_keys) > 0:
            dir_name = create_hyperparam_dirname(search_keys, cfg)
        else:
            dir_name = "run"
            
        cfg["res_dir"] = cfg["exp_dir"] / dir_name
        mkdir(cfg["res_dir"])
        for run in tqdm(range(1, cfg["n_runs"] + 1)):
            seed_everything(seeds[run-1])

            task = TASKS[cfg["task_name"]](**cfg["task_params"])
            model = MODELS[cfg["model_name"]](**cfg["model_params"])
            optimizer = OPTIMS[cfg["optimizer"]](model.parameters(), cfg["step_size"])
            loss_fn = LOSSES[cfg["loss"]]

            res_dict = train_single_run(cfg, task, model, optimizer, loss_fn, torch_device)
            
            for k, v in res_dict.items():
                np.save(cfg["res_dir"] / f"{k}_run_{run}.npy", v)


if __name__ == "__main__":
    main()

