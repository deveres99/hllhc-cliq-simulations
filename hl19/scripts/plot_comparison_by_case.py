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
parser.add_argument(
    "dist_cons", 
    type=str, 
    help="distribution conservativism ('mean', 'conservative')"
)
parser.add_argument(
    "dist_mode", 
    type=str, 
    help="distribution generation mode ('2D', '4D')"
)
parser.add_argument(
    "mo_pol", 
    type=str, 
    help="octupole polarity, can be 'positive' or 'negative'"
)


colors = {
    "Q1L1": "yellowgreen", 
    "Q1L5": "deepskyblue", 
    "Q1R1": "gold", 
    "Q1R5": "hotpink", 
    "Q2L1": "forestgreen", 
    "Q2L5": "dodgerblue", 
    "Q2R1": "orange", 
    "Q2R5": "deeppink", 
    "Q3L1": "darkgreen", 
    "Q3L5": "blue", 
    "Q3R1": "darkorange", 
    "Q3R5": "mediumvioletred", 
}


f_rev = 11245
n_bunch = 2808
bunch_intensity = 2.2e11
turn_step = 0.11245
max_turn = 10 + 9 * turn_step
charge = 1.60218e-19


def get_losses_vs_turn(part_fin_file):
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

    return cum_lost_energy, turns


def main():
    '''
    Parse arguments
    '''
    args = parser.parse_args()
    result_path = args.result_path
    study_name = args.study_name
    seed = args.seed
    dist_cons = args.dist_cons
    dist_mode = args.dist_mode
    mo_pol = args.mo_pol

    result_path = Path(result_path).resolve() / study_name

    fs = 20
    markersize = 4


    '''
    Losses over time
    '''
    part_fin_Q1L1_file = result_path / f"{mo_pol}_mo_{dist_cons}_halo/part_fin_seed{seed}_Q1L1_{dist_mode}.pkl"
    cum_lost_energy_Q1L1, turns_Q1L1 = get_losses_vs_turn(part_fin_Q1L1_file)
    
    part_fin_Q1L5_file = result_path / f"{mo_pol}_mo_{dist_cons}_halo/part_fin_seed{seed}_Q1L5_{dist_mode}.pkl"
    cum_lost_energy_Q1L5, turns_Q1L5 = get_losses_vs_turn(part_fin_Q1L5_file)

    part_fin_Q1R1_file = result_path / f"{mo_pol}_mo_{dist_cons}_halo/part_fin_seed{seed}_Q1R1_{dist_mode}.pkl"
    cum_lost_energy_Q1R1, turns_Q1R1 = get_losses_vs_turn(part_fin_Q1R1_file)

    part_fin_Q1R5_file = result_path / f"{mo_pol}_mo_{dist_cons}_halo/part_fin_seed{seed}_Q1R5_{dist_mode}.pkl"
    cum_lost_energy_Q1R5, turns_Q1R5 = get_losses_vs_turn(part_fin_Q1R5_file)

    part_fin_Q2L1_file = result_path / f"{mo_pol}_mo_{dist_cons}_halo/part_fin_seed{seed}_Q2L1_{dist_mode}.pkl"
    cum_lost_energy_Q2L1, turns_Q2L1 = get_losses_vs_turn(part_fin_Q2L1_file)
    
    part_fin_Q2L5_file = result_path / f"{mo_pol}_mo_{dist_cons}_halo/part_fin_seed{seed}_Q2L5_{dist_mode}.pkl"
    cum_lost_energy_Q2L5, turns_Q2L5 = get_losses_vs_turn(part_fin_Q2L5_file)

    part_fin_Q2R1_file = result_path / f"{mo_pol}_mo_{dist_cons}_halo/part_fin_seed{seed}_Q2R1_{dist_mode}.pkl"
    cum_lost_energy_Q2R1, turns_Q2R1 = get_losses_vs_turn(part_fin_Q2R1_file)

    part_fin_Q2R5_file = result_path / f"{mo_pol}_mo_{dist_cons}_halo/part_fin_seed{seed}_Q2R5_{dist_mode}.pkl"
    cum_lost_energy_Q2R5, turns_Q2R5 = get_losses_vs_turn(part_fin_Q2R5_file)

    part_fin_Q3L1_file = result_path / f"{mo_pol}_mo_{dist_cons}_halo/part_fin_seed{seed}_Q3L1_{dist_mode}.pkl"
    cum_lost_energy_Q3L1, turns_Q3L1 = get_losses_vs_turn(part_fin_Q3L1_file)
    
    part_fin_Q3L5_file = result_path / f"{mo_pol}_mo_{dist_cons}_halo/part_fin_seed{seed}_Q3L5_{dist_mode}.pkl"
    cum_lost_energy_Q3L5, turns_Q3L5 = get_losses_vs_turn(part_fin_Q3L5_file)

    part_fin_Q3R1_file = result_path / f"{mo_pol}_mo_{dist_cons}_halo/part_fin_seed{seed}_Q3R1_{dist_mode}.pkl"
    cum_lost_energy_Q3R1, turns_Q3R1 = get_losses_vs_turn(part_fin_Q3R1_file)

    part_fin_Q3R5_file = result_path / f"{mo_pol}_mo_{dist_cons}_halo/part_fin_seed{seed}_Q3R5_{dist_mode}.pkl"
    cum_lost_energy_Q3R5, turns_Q3R5 = get_losses_vs_turn(part_fin_Q3R5_file)
    
    
    # Linear scale
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    ax_b = ax.twiny()

    ax.text(
        5, 1.99, 
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
        115, 1.99, 
        "PSDU interlock", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax.text(
        125, 1.99, 
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
        237, 1.99, 
        "BIS propagation", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax.text(
        247, 1.99, 
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
        326, 1.99, 
        "Abort gap synch.", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax.text(
        336, 1.99, 
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
        415, 1.99, 
        "Extraction", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax.text(
        425, 1.99, 
        r"420 $\mu$s", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax.axhline(0.125, color="red", linestyle='--', linewidth=3)
    ax.text(
        9.99 / f_rev * 1e6, 0.115, 
        "BLM thresholds (125 kJ)", 
        fontsize=fs-8, 
        rotation="horizontal", 
        ha="right", 
        va="top", 
        color="red"
    )
    ax.axhline(1, color="firebrick", linestyle='-.', linewidth=3)
    ax.text(
        9.99 / f_rev * 1e6, 1.01, 
        "Critical losses (1 MJ)", 
        fontsize=fs-8, 
        rotation="horizontal", 
        ha="right", 
        va="bottom", 
        color="firebrick"
    )
    ax.step(
        turns_Q1L1 / f_rev * 1e6, 
        cum_lost_energy_Q1L1 * 1e-6, 
        color=colors["Q1L1"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q1L1"
    )
    ax.step(
        turns_Q1L5 / f_rev * 1e6, 
        cum_lost_energy_Q1L5 * 1e-6, 
        color=colors["Q1L5"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q1L5"
    )
    ax.step(
        turns_Q1R1 / f_rev * 1e6, 
        cum_lost_energy_Q1R1 * 1e-6, 
        color=colors["Q1R1"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q1R1"
    )
    ax.step(
        turns_Q1R5 / f_rev * 1e6, 
        cum_lost_energy_Q1R5 * 1e-6, 
        color=colors["Q1R5"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q1R5"
    )
    ax.step(
        turns_Q2L1 / f_rev * 1e6, 
        cum_lost_energy_Q2L1 * 1e-6, 
        color=colors["Q2L1"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q2L1"
    )
    ax.step(
        turns_Q2L5 / f_rev * 1e6, 
        cum_lost_energy_Q2L5 * 1e-6, 
        color=colors["Q2L5"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q2L5"
    )
    ax.step(
        turns_Q2R1 / f_rev * 1e6, 
        cum_lost_energy_Q2R1 * 1e-6, 
        color=colors["Q2R1"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q2R1"
    )
    ax.step(
        turns_Q2R5 / f_rev * 1e6, 
        cum_lost_energy_Q2R5 * 1e-6, 
        color=colors["Q2R5"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q2R5"
    )
    ax.step(
        turns_Q3L1 / f_rev * 1e6, 
        cum_lost_energy_Q3L1 * 1e-6, 
        color=colors["Q3L1"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q3L1"
    )
    ax.step(
        turns_Q3L5 / f_rev * 1e6, 
        cum_lost_energy_Q3L5 * 1e-6, 
        color=colors["Q3L5"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q3L5"
    )
    ax.step(
        turns_Q3R1 / f_rev * 1e6, 
        cum_lost_energy_Q3R1 * 1e-6, 
        color=colors["Q3R1"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q3R1"
    )
    ax.step(
        turns_Q3R5 / f_rev * 1e6, 
        cum_lost_energy_Q3R5 * 1e-6, 
        color=colors["Q3R5"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q3R5"
    )
    ax.set_xlim((0, 10 / f_rev * 1e6))
    ax.set_ylim((0, 2))
    ax.set_xlabel(r"time from failure onset [$\mu$s]", fontsize=fs)
    ax.set_ylabel("beam losses [MJ]", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs-2)
    ax.grid(True, which='major', axis='y', color='gray', linestyle='-')
    ax.grid(True, which='minor', axis='y', color='gray', linestyle='--')
    ax.legend(loc="upper right", fontsize=fs-2)

    ax_b.plot(
        turns_Q1L1, 
        cum_lost_energy_Q1L1 * 1e-6, 
        color=colors["Q1L1"], 
        marker='None', 
        linestyle='None'
    )
    ax_b.set_xticks(np.arange(11), np.arange(11))
    ax_b.set_xlim((0, 10))
    ax_b.set_ylim((0, 2))
    ax_b.set_xlabel(r"turn from failure onset", fontsize=fs)
    ax_b.tick_params(axis="both", labelsize=fs-2)
    ax_b.grid(True, which='major', axis='x', color='gray', linestyle='-')

    fig.savefig(result_path / f"losses_vs_time_lin_comparison_by_case_{mo_pol}_mo_seed{seed}_{dist_cons}_halo_{dist_mode}.png", dpi=300, format='png', bbox_inches='tight')

    # Log scale
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
        "PSDU interlock", 
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
        "Extraction", 
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
        9.99 / f_rev * 1e6, 0.115, 
        "BLM thresholds (125 kJ)", 
        fontsize=fs-8, 
        rotation="horizontal", 
        ha="right", 
        va="top", 
        color="red"
    )
    ax.axhline(1, color="firebrick", linestyle='-.', linewidth=3)
    ax.text(
        9.99 / f_rev * 1e6, 1.1, 
        "Critical losses (1 MJ)", 
        fontsize=fs-8, 
        rotation="horizontal", 
        ha="right", 
        va="bottom", 
        color="firebrick"
    )
    ax.step(
        turns_Q1L1 / f_rev * 1e6, 
        cum_lost_energy_Q1L1 * 1e-6, 
        color=colors["Q1L1"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q1L1"
    )
    ax.step(
        turns_Q1L5 / f_rev * 1e6, 
        cum_lost_energy_Q1L5 * 1e-6, 
        color=colors["Q1L5"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q1L5"
    )
    ax.step(
        turns_Q1R1 / f_rev * 1e6, 
        cum_lost_energy_Q1R1 * 1e-6, 
        color=colors["Q1R1"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q1R1"
    )
    ax.step(
        turns_Q1R5 / f_rev * 1e6, 
        cum_lost_energy_Q1R5 * 1e-6, 
        color=colors["Q1R5"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q1R5"
    )
    ax.step(
        turns_Q2L1 / f_rev * 1e6, 
        cum_lost_energy_Q2L1 * 1e-6, 
        color=colors["Q2L1"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q2L1"
    )
    ax.step(
        turns_Q2L5 / f_rev * 1e6, 
        cum_lost_energy_Q2L5 * 1e-6, 
        color=colors["Q2L5"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q2L5"
    )
    ax.step(
        turns_Q2R1 / f_rev * 1e6, 
        cum_lost_energy_Q2R1 * 1e-6, 
        color=colors["Q2R1"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q2R1"
    )
    ax.step(
        turns_Q2R5 / f_rev * 1e6, 
        cum_lost_energy_Q2R5 * 1e-6, 
        color=colors["Q2R5"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q2R5"
    )
    ax.step(
        turns_Q3L1 / f_rev * 1e6, 
        cum_lost_energy_Q3L1 * 1e-6, 
        color=colors["Q3L1"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q3L1"
    )
    ax.step(
        turns_Q3L5 / f_rev * 1e6, 
        cum_lost_energy_Q3L5 * 1e-6, 
        color=colors["Q3L5"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q3L5"
    )
    ax.step(
        turns_Q3R1 / f_rev * 1e6, 
        cum_lost_energy_Q3R1 * 1e-6, 
        color=colors["Q3R1"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q3R1"
    )
    ax.step(
        turns_Q3R5 / f_rev * 1e6, 
        cum_lost_energy_Q3R5 * 1e-6, 
        color=colors["Q3R5"], 
        marker='s', 
        markersize=markersize, 
        where='mid', 
        label="Q3R5"
    )
    ax.set_xlim((0, 10 / f_rev * 1e6))
    ax.set_ylim((1e-4, 1e2))
    ax.set_xlabel(r"time from failure onset [$\mu$s]", fontsize=fs)
    ax.set_ylabel(r"beam losses [MJ]", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs-2)
    ax.set_yscale("log")
    ax.grid(True, which='major', axis='y', color='gray', linestyle='-')
    ax.grid(True, which='minor', axis='y', color='gray', linestyle='--')
    ax.legend(loc="lower right", fontsize=fs-2, ncol=3)

    ax_b.plot(
        turns_Q1L1, 
        cum_lost_energy_Q1L1 * 1e-6, 
        color=colors["Q1L1"], 
        marker='None', 
        linestyle='None'
    )
    ax_b.set_xticks(np.arange(11), np.arange(11))
    ax_b.set_xlim((0, 10))
    ax_b.set_ylim((1e-4, 1e2))
    ax_b.set_xlabel(r"turn from failure onset", fontsize=fs)
    ax_b.tick_params(axis="both", labelsize=fs-2)
    ax_b.set_yscale("log")
    ax_b.grid(True, which='major', axis='x', color='gray', linestyle='-')

    fig.savefig(result_path / f"losses_vs_time_log_comparison_by_case_{mo_pol}_mo_seed{seed}_{dist_cons}_halo_{dist_mode}.png", dpi=300, format='png', bbox_inches='tight')


if __name__ == "__main__":
    main()