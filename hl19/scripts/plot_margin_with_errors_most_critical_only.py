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


cases = [
    "Q2L1", "Q2L5", "Q2R5", "Q3L5", "Q3R1"
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


def get_max_collimator_losses(coll_loss_file):
    coll_losses = np.loadtxt(coll_loss_file, delimiter=',', dtype=str)

    names, counts = np.unique(coll_losses[0, :], return_counts=True)
    name = names[np.argmax(counts)]
    energy_avg = np.nanmean(coll_losses[1, np.where(coll_losses[0, :] == name)].astype(float))
    energy_std = np.nanstd(coll_losses[1, np.where(coll_losses[0, :] == name)].astype(float))
    energy_min = np.nanmin(coll_losses[1, np.where(coll_losses[0, :] == name)].astype(float))
    energy_max = np.nanmax(coll_losses[1, np.where(coll_losses[0, :] == name)].astype(float))

    return name, energy_avg, energy_std, energy_min, energy_max


def get_1MJ_turn_margin(losses_vs_time_file):
    losses_vs_time = np.loadtxt(losses_vs_time_file, delimiter=',')
    
    turns = losses_vs_time[0, :]
    cum_lost_energy_avg = losses_vs_time[1, :]
    cum_lost_energy_std = losses_vs_time[2, :]
    cum_lost_energy_min = losses_vs_time[3, :]
    cum_lost_energy_max = losses_vs_time[4, :]

    exceed_idx_avg = np.where(cum_lost_energy_avg >= 1e6)[0]
    exceed_idx_stdm = np.where(cum_lost_energy_avg-cum_lost_energy_std >= 1e6)[0]
    exceed_idx_stdp = np.where(cum_lost_energy_avg+cum_lost_energy_std >= 1e6)[0]
    exceed_idx_min = np.where(cum_lost_energy_min >= 1e6)[0]
    exceed_idx_max = np.where(cum_lost_energy_max >= 1e6)[0]
    
    if len(exceed_idx_avg):
        turn_avg = turns[exceed_idx_avg[0]]
    else:
        turn_avg = np.nan
    if len(exceed_idx_stdm):
        turn_stdm = turns[exceed_idx_stdm[0]]
    else:
        turn_stdm = np.nan
    if len(exceed_idx_stdp):
        turn_stdp = turns[exceed_idx_stdp[0]]
    else:
        turn_stdp = np.nan
    if len(exceed_idx_min):
        turn_min = turns[exceed_idx_min[0]]
    else:
        turn_min = np.nan
    if len(exceed_idx_max):
        turn_max = turns[exceed_idx_max[0]]
    else:
        turn_max = np.nan

    return turn_avg, turn_stdm, turn_stdp, turn_min, turn_max

    if turn_stdm < turn_stdp:
        return turn_avg, turn_stdm, turn_stdp, turn_min, turn_max
    else:
        return turn_avg, turn_stdp, turn_stdm, turn_max, turn_min
    

def main():
    '''
    Parse arguments
    '''
    args = parser.parse_args()
    result_path = args.result_path
    study_name = args.study_name

    result_path = Path(result_path).resolve() / study_name

    fs = 20
    markersize = 8

    margins = {
        "positive": {
            "conservative": {
                "4D": [], 
            }, 
        }, 
        "negative": {
            "conservative": {
                "4D": [], 
            }, 
        }, 
    }

    max_losses = {
        "positive": {
            "conservative": {
                "4D": {
                    "name": [], 
                    "energy": []
                }, 
            }, 
        }, 
        "negative": {
            "conservative": {
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
                    losses_vs_time_file = result_path / f"{mo_pol}_mo_{halo}_halo/losses_vs_time_{case}_{mode}.csv"
                    turn_avg, turn_stdm, turn_stdp, turn_min, turn_max = get_1MJ_turn_margin(losses_vs_time_file)
                    margins[mo_pol][halo][mode].append({
                        "label": case, 
                        "mean": turn_avg / f_rev * 1e6, 
                        "med": turn_avg / f_rev * 1e6, 
                        "q1": turn_stdm / f_rev * 1e6, 
                        "q3": turn_stdp / f_rev * 1e6, 
                        "whislo": turn_min / f_rev * 1e6, 
                        "whishi": turn_max / f_rev * 1e6, 
                        "fliers": []
                    })
                    
                    coll_loss_file = result_path / f"{mo_pol}_mo_{halo}_halo/max_coll_losses_{case}_{mode}.csv"
                    name, e_avg, e_std, e_min, e_max = get_max_collimator_losses(coll_loss_file)
                    if name not in unique_colls:
                        unique_colls.append(name)
                    max_losses[mo_pol][halo][mode]["name"].append(name)
                    max_losses[mo_pol][halo][mode]["energy"].append({
                        "label": case, 
                        "mean": e_avg * 1e-6, 
                        "med": e_avg * 1e-6, 
                        "q1": (e_avg-e_std)  * 1e-6, 
                        "q3": (e_avg+e_std)  * 1e-6, 
                        "whislo": e_min * 1e-6, 
                        "whishi": e_max * 1e-6, 
                        "fliers": []
                    })
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
                ax.bxp(
                    margins[mo_pol][halo][mode],
                    showmeans=True,
                    meanline=True,
                    boxprops=dict(color=colors[mo_pol][halo][mode], linewidth=2),
                    whiskerprops=dict(color=colors[mo_pol][halo][mode], linewidth=2),
                    capprops=dict(color=colors[mo_pol][halo][mode], linewidth=2),
                    medianprops=dict(color=colors[mo_pol][halo][mode], linewidth=2, linestyle='--'),
                    meanprops=dict(color=colors[mo_pol][halo][mode], linewidth=2, linestyle='--'),
                    label=r"$q=1.3$, $\beta=0.9$" + f" ({mode}), {"+" if mo_pol == "positive" else "-"} MO" if halo == "mean" else r"$q=1.5$, $\beta=1.68$" + f" ({mode}), {"+" if mo_pol == "positive" else "-"} MO"
                )
                ax_b.plot(
                    xx, 
                    np.asarray([v["mean"] for v in margins[mo_pol][halo][mode]]) * f_rev / 1e6, 
                    color="None", 
                    marker="None",  
                    linestyle="None"
                )
    ax.set_xlim(0.5, len(cases)+0.5)
    ax.set_ylim(4 / f_rev * 1e6, 10 / f_rev * 1e6)
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
                ax.bxp(
                    max_losses[mo_pol][halo][mode]["energy"],
                    showmeans=True,
                    meanline=True,
                    boxprops=dict(color=colors[mo_pol][halo][mode], linewidth=2),
                    whiskerprops=dict(color=colors[mo_pol][halo][mode], linewidth=2),
                    capprops=dict(color=colors[mo_pol][halo][mode], linewidth=2),
                    medianprops=dict(color=colors[mo_pol][halo][mode], linewidth=2, linestyle='--'),
                    meanprops=dict(color=colors[mo_pol][halo][mode], linewidth=2, linestyle='--'),
                    label=r"$q=1.3$, $\beta=0.9$" + f" ({mode}), {"+" if mo_pol == "positive" else "-"} MO" if halo == "mean" else r"$q=1.5$, $\beta=1.68$" + f" ({mode}), {"+" if mo_pol == "positive" else "-"} MO"
                )
    ax.set_xlim(0.5, len(cases)+0.5)
    ax.set_ylim(bottom=1e-1)
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