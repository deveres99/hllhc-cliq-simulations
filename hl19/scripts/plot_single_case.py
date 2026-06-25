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


def main():
    '''
    Parse arguments
    '''
    args = parser.parse_args()
    result_path = args.result_path
    study_name = args.study_name
    seed = args.seed
    cliq_case = args.cliq_case
    dist_mode = args.dist_mode

    result_path = Path(result_path).resolve() / study_name

    fs = 20
    markersize = 4


    '''
    Lossmap
    '''
    with open("data/lhc_collimation_metadata.json", "r") as file:
        metadata = json.load(file)

    lm_file = result_path / f"lossmap_seed{seed}_{cliq_case}_{dist_mode}.json"
    plot_name = lm_file.stem[8:]
    lm = xc.LossMap.from_json(lm_file)
    lm.update_metadata(metadata)
    lm.plot(show=False, norm="deposited_energy", beam_intensity=bunch_intensity*n_bunch, savefig=result_path / f"lossmap_de_{plot_name}.pdf", zoom="betatron")
    lm.plot(show=False, norm="deposited_energy_per_length", beam_intensity=bunch_intensity*n_bunch, savefig=result_path / f"lossmap_depl_{plot_name}.pdf", zoom="betatron")


    '''
    Initial distribution
    '''
    part_init_file = result_path / f"part_init_seed{seed}_{cliq_case}_{dist_mode}.pkl"
    with open(part_init_file, "rb") as file:
        part_init = pickle.load(file)
    p_init = xp.Particles.from_dict(part_init)

    # Load line
    if "positive_mo" in study_name:
        line = xt.load("data/hllhc_b1_v19_round_imo300_no_errors.json")
    elif "negative_mo" in study_name:
        line = xt.load("data/hllhc_b1_v19_round_imo-300_no_errors.json")
    else:
        raise ValueError("Octupole polarity cannot be determined from study name!")
    tw = line.twiss()
    p_init_norm = tw.get_normalized_coordinates(p_init, nemitt_x=2.5e-6, nemitt_y=2.5e-6)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), layout="tight")

    ax.hist(
        p_init_norm.x_norm, 
        bins=100, 
        range=[-7, 7], 
        density=True, 
        histtype="step", 
        color="black", 
        label=f"q-Gaussian {dist_mode} (projection)"
    )
    xx = np.linspace(-7, 7, 1000)
    if "mean_halo" in study_name:
        q = 1.3
        beta = 0.9
    elif "conservative_halo" in study_name:
        q = 1.5
        beta = 1.68
    else:
        raise ValueError("Halo conservativism cannot be determined from study name!")
    ax.plot(
        xx, 
        q_Gaussian(xx, q, beta), 
        color="red", 
        label=fr"q-Gaussian PDF ($q=${q}, $\beta=${beta})"
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$x$ [$\sigma$]", fontsize=fs)
    ax.set_ylabel("density", fontsize=fs)
    ax.legend(loc="lower center", fontsize=fs-2)
    ax.tick_params(axis="both", labelsize=fs-2)
    ax.grid(True, which='major', axis='both', color='gray', linestyle='-')
    ax.grid(True, which='minor', axis='y', color='gray', linestyle='--')

    fig.savefig(result_path / f"init_dist_{plot_name}.png", dpi=300, format='png', bbox_inches='tight')


    '''
    Losses over time
    '''
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
        turns / f_rev * 1e6, 
        cum_lost_energy * 1e-6, 
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
        turns, 
        cum_lost_energy * 1e-6, 
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
        turns / f_rev * 1e6, 
        cum_lost_energy * 1e-6, 
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
        turns, 
        cum_lost_energy * 1e-6, 
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