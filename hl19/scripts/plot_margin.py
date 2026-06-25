import argparse
import json
import pickle
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

import xobjects as xo
import xdeps as xd
import xtrack as xt
import xpart as xp
import xfields as xf
import xcoll as xc
import xsuite as xs


parser = argparse.ArgumentParser()
parser.add_argument(
    "study_name", 
    type=str, 
    help="name of study"
)
parser.add_argument(
    "result_path", 
    type=str, 
    help="path to the results folder"
)
parser.add_argument(
    "seed", 
    type=int, 
    help="error seed"
)


cases = [
    "Q1L1", "Q1L5", "Q1R1", "Q1R5", 
    "Q2L1", "Q2L5", "Q2R1", "Q2R5", 
    "Q3L1", "Q3L5", "Q3R1", "Q3R5"
]
colors = {
    "positive": {
        "mean": {
            "2D": "royalblue", 
            "4D": "mediumpurple", 
        }, 
        "conservative": {
            "2D": "blue", 
            "4D": "blueviolet", 
        }, 
    }, 
    "negative": {
        "mean": {
            "2D": "slateblue", 
            "4D": "mediumorchid", 
        }, 
        "conservative": {
            "2D": "darkslateblue", 
            "4D": "darkmagenta", 
        }, 
    }, 
}
markers = {
    "positive": {
        "mean": {
            "2D": "^", 
            "4D": "^", 
        }, 
        "conservative": {
            "2D": "o", 
            "4D": "o", 
        }, 
    }, 
    "negative": {
        "mean": {
            "2D": "v", 
            "4D": "v", 
        }, 
        "conservative": {
            "2D": "s", 
            "4D": "s", 
        }, 
    }, 
}


f_rev = 11245
n_bunch = 2808
bunch_intensity = 2.2e11
turn_step = 0.11245
max_turn = 10 + 9 * turn_step
charge = 1.60218e-19


def get_max_collimator_losses(lm_file):
    with open(lm_file, "r") as file:
        lm = json.load(file)

    num_init = lm["num_initial"]

    coll_losses = lm["collimator"]

    max_idx = np.argmax(coll_losses["e"])
    name = coll_losses["name"][max_idx]
    energy = coll_losses["e"][max_idx] / num_init * bunch_intensity * n_bunch * charge

    return name, energy


def get_1MJ_turn_margin(part_fin_file):
    with open(part_fin_file, "rb") as file:
        part_fin = pickle.load(file)
    p_fin = xp.Particles.from_dict(part_fin)
    energy = p_fin.energy[np.argsort(p_fin.particle_id)]
    at_turn = part_fin["at_turn"][np.argsort(part_fin["particle_id"])]
    num_part = len(at_turn)
    mask = part_fin["state"][np.argsort(part_fin["particle_id"])] != 1
    energy = energy[mask]
    at_turn = at_turn[mask]
    lost_energy, turns = np.histogram(at_turn, bins=np.arange(0, max_turn+turn_step, turn_step), weights=energy)
    turns = turns[:-1] + (turns[1] - turns[0]) / 2.0
    cum_lost_energy = np.cumsum(lost_energy) / num_part * bunch_intensity * n_bunch * charge

    exceed_idx = np.where(cum_lost_energy >= 1e6)[0]
    if len(exceed_idx):
        return turns[exceed_idx[0]]
    else:
        return np.nan
    

def main():
    '''
    Parse arguments
    '''
    args = parser.parse_args()
    result_path = args.result_path
    study_name = args.study_name
    seed = args.seed

    result_path = Path(result_path).resolve() / study_name

    fs = 20
    markersize = 8

    margins = {
        "positive": {
            "mean": {
                "2D": [], 
                "4D": [], 
            }, 
            "conservative": {
                "2D": [], 
                "4D": [], 
            }, 
        }, 
        "negative": {
            "mean": {
                "2D": [], 
                "4D": [], 
            }, 
            "conservative": {
                "2D": [], 
                "4D": [], 
            }, 
        }, 
    }

    max_losses = {
        "positive": {
            "mean": {
                "2D": {
                    "name": [], 
                    "energy": []
                }, 
                "4D": {
                    "name": [], 
                    "energy": []
                }, 
            }, 
            "conservative": {
                "2D": {
                    "name": [], 
                    "energy": []
                }, 
                "4D": {
                    "name": [], 
                    "energy": []
                }, 
            }, 
        }, 
        "negative": {
            "mean": {
                "2D": {
                    "name": [], 
                    "energy": []
                }, 
                "4D": {
                    "name": [], 
                    "energy": []
                }, 
            }, 
            "conservative": {
                "2D": {
                    "name": [], 
                    "energy": []
                }, 
                "4D": {
                    "name": [], 
                    "energy": []
                }, 
            }, 
        }, 
    }

    unique_colls = []
    for mo_pol in margins.keys():
        for halo in margins[mo_pol].keys():
            for mode in margins[mo_pol][halo].keys():
                for case in cases:
                    part_fin_file = result_path / f"{mo_pol}_mo_{halo}_halo/part_fin_seed{seed}_{case}_{mode}.pkl"
                    margins[mo_pol][halo][mode].append(get_1MJ_turn_margin(part_fin_file))
                    lm_file = result_path / f"{mo_pol}_mo_{halo}_halo/lossmap_seed{seed}_{case}_{mode}.json"
                    name, e = get_max_collimator_losses(lm_file)
                    if name not in unique_colls:
                        unique_colls.append(name)
                    max_losses[mo_pol][halo][mode]["name"].append(name)
                    max_losses[mo_pol][halo][mode]["energy"].append(e)
    unique_colls = np.sort(np.asarray(unique_colls))

    # Margins
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    ax_b = ax.twinx()

    ax.axhline(420, color="firebrick", linestyle='-.', linewidth=3)
    ax_b.axhline(420e-6 * f_rev, color="None", linestyle='None', linewidth=3)
    xx = np.arange(len(cases))
    for mo_pol in margins.keys():
        for halo in margins[mo_pol].keys():
            for mode in margins[mo_pol][halo].keys():
                ax.plot(
                    xx, 
                    np.asarray(margins[mo_pol][halo][mode]) / f_rev * 1e6, 
                    color=colors[mo_pol][halo][mode], 
                    marker=markers[mo_pol][halo][mode], 
                    markersize=markersize, 
                    linestyle="None", 
                    label=r"$q=1.3$, $\beta=0.9$" + f" ({mode}), {"+" if mo_pol == "positive" else "-"} MO" if halo == "mean" else r"$q=1.5$, $\beta=1.68$" + f" ({mode}), {"+" if mo_pol == "positive" else "-"} MO"
                )
                ax_b.plot(
                    xx, 
                    margins[mo_pol][halo][mode], 
                    color="None", 
                    marker="None",  
                    linestyle="None"
                )
    ax.set_xlim(-0.5, len(cases)-0.5)
    ax.set_ylim(4 / f_rev * 1e6, 10 / f_rev * 1e6)
    ax.set_xticks(xx, labels=cases)
    ax.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax.set_ylabel(r"1 MJ exceeded [$\mu$s]", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs-2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax_b.set_ylim(4, 10)
    ax_b.set_ylabel(r"1 MJ exceeded [turn]", fontsize=fs)
    ax_b.tick_params(axis="both", labelsize=fs-2)
    ax_b.grid(True, which='major', axis='x', color='gray', linestyle='-')

    fig.subplots_adjust(top=0.9)

    fig.savefig(result_path / f"margins.png", dpi=300, format='png', bbox_inches='tight')

    # Max collimator losses
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    ax.axhline(1, color="firebrick", linestyle='-.', linewidth=3)
    xx = np.arange(len(cases))
    for mo_pol in max_losses.keys():
        for halo in max_losses[mo_pol].keys():
            for mode in max_losses[mo_pol][halo].keys():
                ax.plot(
                    xx, 
                    np.asarray(max_losses[mo_pol][halo][mode]["energy"]) * 1e-6, 
                    color=colors[mo_pol][halo][mode], 
                    marker=markers[mo_pol][halo][mode], 
                    markersize=markersize, 
                    linestyle="None", 
                    label=r"$q=1.3$, $\beta=0.9$" + f" ({mode}), {"+" if mo_pol == "positive" else "-"} MO" if halo == "mean" else r"$q=1.5$, $\beta=1.68$" + f" ({mode}), {"+" if mo_pol == "positive" else "-"} MO"
                )
    ax.set_xlim(-0.5, len(cases)-0.5)
    ax.set_xticks(xx, labels=cases)
    ax.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax.set_ylabel("max. collimator loss [MJ]\nin 445 s (5 turns)", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs-2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)
    ax.set_yscale("log")
    ax.grid(True, which='major', axis='y', color='gray', linestyle='-')
    ax.grid(True, which='minor', axis='y', color='gray', linestyle='--')

    fig.subplots_adjust(top=0.9)

    fig.savefig(result_path / f"max_collimator_losses.png", dpi=300, format='png', bbox_inches='tight')

    # Max collimator loss location
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    xx = np.arange(len(cases))
    for mo_pol in max_losses.keys():
        for halo in max_losses[mo_pol].keys():
            for mode in max_losses[mo_pol][halo].keys():
                lookup = {v: i for i, v in enumerate(unique_colls)}
                indices = np.asarray([lookup[v] for v in np.asarray(max_losses[mo_pol][halo][mode]["name"])])
                ax.plot(
                    xx, 
                    indices, 
                    markeredgecolor=colors[mo_pol][halo][mode], 
                    markerfacecolor='None', 
                    marker=markers[mo_pol][halo][mode], 
                    markersize=markersize * 2, 
                    linestyle="None", 
                    label=r"$q=1.3$, $\beta=0.9$" + f" ({mode}), {"+" if mo_pol == "positive" else "-"} MO" if halo == "mean" else r"$q=1.5$, $\beta=1.68$" + f" ({mode}), {"+" if mo_pol == "positive" else "-"} MO"
                )
    ax.set_xlim(-0.5, len(cases)-0.5)
    ax.set_xticks(xx, labels=cases)
    ax.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax.set_ylim(-0.5, len(unique_colls)-0.5)
    ax.set_yticks(np.arange(len(unique_colls)), labels=unique_colls)
    ax.set_ylabel("max. collimator loss locations", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs-2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)
    ax.grid(True, which='major', axis='y', color='gray', linestyle='-')

    fig.subplots_adjust(top=0.9)

    fig.savefig(result_path / f"max_collimator_losses_loc.png", dpi=300, format='png', bbox_inches='tight')


if __name__ == "__main__":
    main()