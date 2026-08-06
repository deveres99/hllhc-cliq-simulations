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
parser.add_argument(
    "cliq_case", 
    type=str, 
    help="which magnet was assigned field errors ('Q1R5a', 'Q1L5a', 'Q2R5a', " \
    "'Q2L5a', 'Q3R5a', 'Q3L5a', 'Q1R1a', 'Q1L1a', 'Q2R1a', 'Q2L1a', 'Q3R1a', " \
    "'Q3L1a', 'Q1R5b', 'Q1L5b', 'Q2R5b', 'Q2L5b', 'Q3R5b', 'Q3L5b', 'Q1R1b', " \
    "'Q1L1b', 'Q2R1b', 'Q2L1b', 'Q3R1b', 'Q3L1b')"
)
parser.add_argument(
    "i_mo", 
    type=float, 
    help="octupole current"
)
parser.add_argument(
    "save", 
    type=int, 
    help="should plots be saved? (0 or 1)"
)
parser.add_argument(
    "show", 
    type=int, 
    help="should plots be shown? (0 or 1)"
)


f_rev = 11245
n_bunch = 2760
bunch_intensity = 2.2e11
turn_step = 0.11245
max_turn = 60 + 9 * turn_step
charge = 1.60218e-19

s_PDSU_interlock = 120e-6
t_PDSU_interlock = s_PDSU_interlock * f_rev
delta_s_to_dump = 300e-6
delta_t_to_dump = delta_s_to_dump * f_rev

halo_models = ['mean', 'mean', 'cons', 'cons']
dist_modes = ['2D', '4D', '2D', '4D']

colors = {
    "mean": {
        "2D": "royalblue", 
        "4D": "mediumpurple", 
    }, 
    "cons": {
        "2D": "blue", 
        "4D": "blueviolet", 
    }
}
markers = {
    "mean": {
        "2D": "^", 
        "4D": "o", 
    }, 
    "cons": {
        "2D": "v", 
        "4D": "s", 
    } 
}


def main():
    '''
    Parse arguments
    '''
    args = parser.parse_args()
    result_path = args.result_path
    study_name = args.study_name
    seed = args.seed
    cliq_case = args.cliq_case
    i_mo = args.i_mo
    save = args.save
    show = args.show

    result_path = Path(result_path).resolve() / study_name

    fs = 20
    markersize = 4

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    ax_b = ax.twiny()

    ax.text(
        5, 99, 
        "Failure onset", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax.axvspan(
        0, 120, 
        facecolor="dimgrey", 
        edgecolor="None", 
        alpha=0.5
    )
    ax.axvline(120, color="black", linestyle=':', linewidth=3)
    ax.text(
        115, 99, 
        "PDSU interlock", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax.text(
        125, 99, 
        r"120 $\mu$s", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax.axvspan(
        120, 242, 
        facecolor="grey", 
        edgecolor="None", 
        alpha=0.5
    )
    ax.axvline(242, color="black", linestyle=':', linewidth=3)
    ax.text(
        237, 99, 
        "BIS propagation", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax.text(
        247, 99, 
        r"242 $\mu$s", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax.axvspan(
        242, 331, 
        facecolor="darkgrey", 
        edgecolor="None", 
        alpha=0.5
    )
    ax.axvline(331, color="black", linestyle=':', linewidth=3)
    ax.text(
        326, 99, 
        "Abort gap synch.", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax.text(
        336, 99, 
        r"331 $\mu$s", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax.axvspan(
        331, 420, 
        facecolor="lightgrey", 
        edgecolor="None", 
        alpha=0.5
    )
    ax.axvline(420, color="black", linestyle=':', linewidth=3)
    ax.text(
        415, 99, 
        "Beam dumped", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax.text(
        425, 99, 
        r"420 $\mu$s", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax.axhline(0.125, color="red", linestyle='--', linewidth=3)
    ax.text(
        19.99 / f_rev * 1e6, 0.115, 
        "BLM thresholds (125 kJ)", 
        fontsize=fs-8, 
        rotation="horizontal", 
        ha="right", 
        va="top", 
        color="red"
    )
    ax.axhline(1, color="firebrick", linestyle='-.', linewidth=3)
    ax.text(
        19.99 / f_rev * 1e6, 1.1, 
        "Critical losses (1 MJ)", 
        fontsize=fs-8, 
        rotation="horizontal", 
        ha="right", 
        va="bottom", 
        color="firebrick"
    )

    '''
    Cumulative losses over time
    '''
    for halo_model, dist_mode in zip(halo_models, dist_modes):
        part_fin_file = result_path / f"{halo_model}_halo_{dist_mode}" / f"part_fin_seed{seed}_{cliq_case}_{halo_model}_{dist_mode}_imo_{int(i_mo)}.pkl"
        with open(part_fin_file, "rb") as file:
            part_fin = pickle.load(file)
        p_fin = xt.Particles.from_dict(part_fin)
        energy = p_fin.energy[np.argsort(p_fin.particle_id)]
        at_turn = part_fin["at_turn"][np.argsort(part_fin["particle_id"])]
        num_part = len(at_turn)
        mask = part_fin["state"][np.argsort(part_fin["particle_id"])] != 1
        energy = energy[mask]
        at_turn = at_turn[mask]
        lost_energy, turns = np.histogram(at_turn, bins=np.arange(0, max_turn+turn_step, turn_step), weights=energy)
        turns = turns[:-1] + (turns[1] - turns[0]) / 2.0
        cum_lost_energy = np.cumsum(lost_energy) / num_part * bunch_intensity * n_bunch * charge # J
        lost_energy = lost_energy / num_part * bunch_intensity * n_bunch * charge # J

        if halo_model == "mean":
            label = rf"$q=1.3$, $\beta=0.9$ ({dist_mode})"
        elif halo_model == "cons":
            label = rf"$q=1.5$, $\beta=1.68$ ({dist_mode})"
        else:
            raise ValueError(f"Not supported `halo_model` ({halo_model})!")

        ax.step(
            turns / f_rev * 1e6, 
            cum_lost_energy * 1e-6, 
            color=colors[halo_model][dist_mode], 
            marker=markers[halo_model][dist_mode], 
            markersize=markersize,
            where='mid', 
            label=label
        )

        ax.plot(
            turns, 
            cum_lost_energy * 1e-6, 
            color='black', 
            marker='None', 
            linestyle='None'
        )

    ax.set_xlim((0, 20 / f_rev * 1e6))
    ax.set_ylim((1e-4, 1e2))
    ax.set_xlabel(r"time from failure onset [$\mu$s]", fontsize=fs)
    ax.set_ylabel(r"beam losses [MJ]", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs-2)
    ax.set_yscale("log")
    ax.grid(True, which='major', axis='y', color='gray', linestyle='-')
    ax.grid(True, which='minor', axis='y', color='gray', linestyle='--')
    # ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.11), ncol=6, fontsize=fs-4)
    ax.legend(loc="lower right", ncol=1, fontsize=fs-2)

    ax_b.set_xticks(np.arange(int(max_turn)), np.arange(int(max_turn)))
    ax_b.set_xlim((0, 20))
    ax_b.set_ylim((1e-4, 1e2))
    ax_b.set_xlabel(r"turn from failure onset", fontsize=fs)
    ax_b.tick_params(axis="both", labelsize=fs-2)
    ax_b.set_yscale("log")
    ax_b.grid(True, which='major', axis='x', color='gray', linestyle='-')

    if save:
        fig.savefig(result_path / f"losses_vs_time_log_{cliq_case}_imo_{int(i_mo)}.png", dpi=150, format='png', bbox_inches='tight')

    if show:
        plt.show()


if __name__ == "__main__":
    main()