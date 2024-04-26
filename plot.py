import matplotlib.pyplot as plt
import numpy as np

from src.util import parse_and_process_cfg, mkdir


if __name__ == "__main__":
    cfg = parse_and_process_cfg()
    cfg["plot_dir"] = cfg["root_dir"] / f"plots/{cfg['exp_name']}"
    mkdir(cfg["plot_dir"])

    sub_dir = [d for d in cfg["exp_dir"].iterdir() if d.is_dir()]
    
    final_acc = []
    for d in sub_dir:
        # Plot accuracy
        acc = None
        paths = list(d.glob("accuracies*.npy"))
        for p in paths:
            data = np.load(p)
            final_acc.append(data)

    final_acc = np.array(final_acc)
    acc_mean = final_acc.mean(axis=0)
    acc_err = 1.96 * final_acc.std(axis=0) / np.sqrt(final_acc.shape[0])

    fig, ax = plt.subplots()
    ax.plot(acc_mean)
    ax.fill_between(range(final_acc.shape[1]), acc_mean - acc_err, acc_mean + acc_err, alpha=0.3)
    plt.savefig(cfg["plot_dir"] / "accuracy.png")
