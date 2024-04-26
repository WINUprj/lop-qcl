import matplotlib.pyplot as plt
import numpy as np

from src.util import parse_and_process_cfg, mkdir
from functools import cmp_to_key


def comp_classical(x,y):
    d1, d2 = x.as_posix().split('/')[-1], y.as_posix().split('/')[-1]
    k1, k2 = float(d1.split('_')[2]), float(d2.split('_')[2])
    return k1 - k2


if __name__ == "__main__":
    cfg = parse_and_process_cfg()
    cfg["plot_dir"] = cfg["root_dir"] / f"plots/{cfg['exp_name']}"
    mkdir(cfg["plot_dir"])

    # NOTE: Manually specify the result directories to use in plots
    exp_dirs = [cfg["exp_dir"], cfg["root_dir"] / "results/classical3", cfg["root_dir"] / "results/classical5"]

    cmap = [plt.cm.tab20(i) for i in range(0, 10*len(exp_dirs), 2)]
    fig, ax = plt.subplots(3, 5, figsize=(20, 15))
    cnt = 0
    col = 0
    for i, dir in enumerate(exp_dirs):
        final_acc = []
        sub_dir = sorted([d for d in dir.iterdir() if d.is_dir()], key=cmp_to_key(comp_classical))
        for d in sub_dir:
            # Plot accuracy
            temp_acc = []
            acc = None
            paths = list(d.glob("accuracies*.npy"))
            for p in paths:
                data = np.load(p)
                temp_acc.append(data)

            temp_acc = np.array(temp_acc)
            temp_avg = temp_acc.mean(axis=0)
            temp_std = 1.96 * temp_acc.std(axis=0) / np.sqrt(temp_acc.shape[0])
            ax[cnt//5, cnt%5].plot(temp_avg, alpha=0.7, color=cmap[col])
            ax[cnt//5, cnt%5].fill_between(range(temp_acc.shape[1]), temp_avg - temp_std, temp_avg + temp_std, alpha=0.3, color=cmap[col])

            cnt += 1
            if cnt % 5 == 0:
                col += 1

    for a, l in zip(ax[:, 0], [1, 3, 5]):
        a.set_ylabel(r"$N_l$="+f"{l}", size="large")
    for a, l in zip(ax[0], ["1e-5", 0.0001, 0.001, 0.01, 0.1]):
        a.set_title(f"Step size={l}")
    plt.savefig(cfg["plot_dir"] / "classical_accuracies.png")



