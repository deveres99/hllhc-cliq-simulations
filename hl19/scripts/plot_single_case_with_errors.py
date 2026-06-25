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
    "cliq_case", 
    type=str, 
    help="which magnet to assign field errors to ('Q1R5', 'Q1L5', 'Q2R5', 'Q2L5', 'Q3R5', 'Q3L5', 'Q1R1', 'Q1L1', 'Q2R1', 'Q2L1', 'Q3R1', 'Q3L1')"
)
parser.add_argument(
    "dist_mode", 
    type=str, 
    help="distribution generation mode ('2D', '4D')"
)


f_rev = 11245
n_bunch = 2808
bunch_intensity = 2.2e11
turn_step = 0.11245
max_turn = 10 + 9 * turn_step
charge = 1.60218e-19


def Cq(q):
    """
    Normalisation for q-Gaussian function.

    Parameters
    ----------
    q : float
        q parameter (1 < q < 3).

    Returns
    -------
    float
        Normalisation factor.
    """

    if q <= 1 or q >= 3:
        raise ValueError("This implementation only supports 1 < q < 3!")
    else:
        return np.sqrt(np.pi) * gamma((3-q)/(2*(q-1))) / (np.sqrt(q-1) * gamma(1/(q-1)))
    

def eq(x, q):
    """
    q-exponention function.

    Parameters
    ----------
    x : float | ndarray
        Value(s) to calculate q-exponential for.
    q : float
        q parameter (1 < q < 3).

    Returns
    -------
    float | ndarray
        Value(s) of q-exponential for given x.
    """

    if q <= 1 or q >= 3:
        raise ValueError("This implementation only supports 1 < q < 3!")
    else:
        return np.where(1+(1-q)*x>0, (1+(1-q)*x)**(1/(1-q)), 0)


def q_Gaussian(x, q, beta):
    """
    q-Gaussian function.

    Parameters
    ----------
    x : float | ndarray
        Value(s) to calculate q-Gaussian for.
    q : float
        q parameter (1 < q < 3).
    beta : float
        Scale parameter.
    
    Returns
    -------
    float | ndarray
        Value(s) of q-Gaussian for given x.
    """

    return np.sqrt(beta) * eq(-beta*x**2, q) / Cq(q)


def get_max_collimator_losses(lm_file):
    with open(lm_file, "r") as file:
        lm = json.load(file)

    num_init = lm["num_initial"]

    coll_losses = lm["collimator"]

    max_idx = np.argmax(coll_losses["e"])
    name = coll_losses["name"][max_idx]
    energy = coll_losses["e"][max_idx] / num_init * bunch_intensity * n_bunch * charge

    return name, energy


def main():
    '''
    Parse arguments
    '''
    args = parser.parse_args()
    result_path = args.result_path
    study_name = args.study_name
    cliq_case = args.cliq_case
    dist_mode = args.dist_mode

    result_path = Path(result_path).resolve() / study_name

    fs = 20
    markersize = 4


    plot_name = f"{cliq_case}_{dist_mode}"


    '''
    Losses over time
    '''
    seed = 1

    print(f"###### SEED {seed} ######")
    part_fin_file = result_path / f"part_fin_seed{seed}_{cliq_case}_{dist_mode}.pkl"
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

    lm_file = result_path / f"lossmap_seed{seed}_{cliq_case}_{dist_mode}.json"
    name, e = get_max_collimator_losses(lm_file)

    max_coll_loss_loc_all_seeds = np.empty(60, dtype="S100")
    max_coll_loss_all_seeds = np.ones(60) * np.nan
    max_coll_loss_loc_all_seeds[0] = name
    max_coll_loss_all_seeds[0] = e

    turns_all_seeds = turns
    cum_lost_energy_all_seeds = np.ones((60, len(cum_lost_energy))) * np.nan
    cum_lost_energy_all_seeds[0, :] = cum_lost_energy

    for seed in range(2, 61, 1):
        print(f"###### SEED {seed} ######")
        part_fin_file = result_path / f"part_fin_seed{seed}_{cliq_case}_{dist_mode}.pkl"
        try:
            with open(part_fin_file, "rb") as file:
                part_fin = pickle.load(file)
        except FileNotFoundError:
            max_coll_loss_loc_all_seeds[seed-1] = ""
            print("No data, moving on to next seed.")
            continue
        p_fin = xp.Particles.from_dict(part_fin)
        energy = p_fin.energy[np.argsort(p_fin.particle_id)]
        at_turn = part_fin["at_turn"][np.argsort(part_fin["particle_id"])]
        num_part = len(at_turn)
        mask = part_fin["state"][np.argsort(part_fin["particle_id"])] != 1
        energy = energy[mask]
        at_turn = at_turn[mask]
        lost_energy, turns = np.histogram(at_turn, bins=np.arange(0, max_turn+turn_step, turn_step), weights=energy)
        cum_lost_energy = np.cumsum(lost_energy) / num_part * bunch_intensity * n_bunch * charge
        cum_lost_energy_all_seeds[seed-1, :] = cum_lost_energy

        lm_file = result_path / f"lossmap_seed{seed}_{cliq_case}_{dist_mode}.json"
        name, e = get_max_collimator_losses(lm_file)
        max_coll_loss_loc_all_seeds[seed-1] = name
        max_coll_loss_all_seeds[seed-1] = e

    cum_lost_energy_avg = np.nanmean(cum_lost_energy_all_seeds, axis=0)
    cum_lost_energy_std = np.nanstd(cum_lost_energy_all_seeds, axis=0)
    cum_lost_energy_min = np.nanmin(cum_lost_energy_all_seeds, axis=0)
    cum_lost_energy_max = np.nanmax(cum_lost_energy_all_seeds, axis=0)

    np.savetxt(
        result_path / f"losses_vs_time_{plot_name}.csv", 
        np.vstack((turns_all_seeds, cum_lost_energy_avg, cum_lost_energy_std, cum_lost_energy_min, cum_lost_energy_max)), 
        delimiter=','
    )
    np.savetxt(
        result_path / f"max_coll_losses_{plot_name}.csv", 
        np.vstack((max_coll_loss_loc_all_seeds.astype(str), max_coll_loss_all_seeds.astype(str))), 
        delimiter=',', 
        fmt="%s"
    )

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
        turns_all_seeds / f_rev * 1e6, 
        cum_lost_energy_min * 1e-6, 
        color='blue', 
        linestyle=':', 
        where='mid'
    )
    ax.step(
        turns_all_seeds / f_rev * 1e6, 
        cum_lost_energy_max * 1e-6, 
        color='blue', 
        linestyle=':', 
        where='mid'
    )
    ax.fill_between(
        turns_all_seeds / f_rev * 1e6, 
        (cum_lost_energy_avg-cum_lost_energy_std) * 1e-6, 
        (cum_lost_energy_avg+cum_lost_energy_std) * 1e-6,
        step='mid', 
        alpha=0.5,
        color='blue'
    )
    ax.step(
        turns_all_seeds / f_rev * 1e6, 
        cum_lost_energy_avg * 1e-6, 
        color='blue', 
        marker='s', 
        markersize=markersize, 
        where='mid'
    )
    ax.set_xlim((0, 10 / f_rev * 1e6))
    ax.set_ylim((0, 2))
    ax.set_xlabel(r"time from failure onset [$\mu$s]", fontsize=fs)
    ax.set_ylabel("beam losses [MJ]", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs-2)
    ax.grid(True, which='major', axis='y', color='gray', linestyle='-')
    ax.grid(True, which='minor', axis='y', color='gray', linestyle='--')

    ax_b.plot(
        turns_all_seeds, 
        cum_lost_energy_avg * 1e-6, 
        color='blue', 
        marker='None', 
        linestyle='None'
    )
    ax_b.set_xticks(np.arange(11), np.arange(11))
    ax_b.set_xlim((0, 10))
    ax_b.set_ylim((0, 2))
    ax_b.set_xlabel(r"turn from failure onset", fontsize=fs)
    ax_b.tick_params(axis="both", labelsize=fs-2)
    ax_b.grid(True, which='major', axis='x', color='gray', linestyle='-')

    # Log scale
    fig.savefig(result_path / f"losses_vs_time_lin_{plot_name}.png", dpi=300, format='png', bbox_inches='tight')

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
        turns_all_seeds / f_rev * 1e6, 
        cum_lost_energy_min * 1e-6, 
        color='blue', 
        linestyle=':', 
        where='mid'
    )
    ax.step(
        turns_all_seeds / f_rev * 1e6, 
        cum_lost_energy_max * 1e-6, 
        color='blue', 
        linestyle=':', 
        where='mid'
    )
    ax.fill_between(
        turns_all_seeds / f_rev * 1e6, 
        (cum_lost_energy_avg-cum_lost_energy_std) * 1e-6, 
        (cum_lost_energy_avg+cum_lost_energy_std) * 1e-6,
        step='mid', 
        alpha=0.5,
        color='blue'
    )
    ax.step(
        turns_all_seeds / f_rev * 1e6, 
        cum_lost_energy_avg * 1e-6, 
        color='blue', 
        marker='s', 
        markersize=markersize, 
        where='mid'
    )
    ax.set_xlim((0, 10 / f_rev * 1e6))
    ax.set_ylim((1e-4, 1e2))
    ax.set_xlabel(r"time from failure onset [$\mu$s]", fontsize=fs)
    ax.set_ylabel(r"beam losses [MJ]", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs-2)
    ax.set_yscale("log")
    ax.grid(True, which='major', axis='y', color='gray', linestyle='-')
    ax.grid(True, which='minor', axis='y', color='gray', linestyle='--')

    ax_b.plot(
        turns_all_seeds, 
        cum_lost_energy_avg * 1e-6, 
        color='blue', 
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

    fig.savefig(result_path / f"losses_vs_time_log_{plot_name}.png", dpi=300, format='png', bbox_inches='tight')


if __name__ == "__main__":
    main()