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
    "halo_model", 
    type=str, 
    help="halo model ('mean', 'cons')"
)
parser.add_argument(
    "dist_mode", 
    type=str, 
    help="distribution generation mode ('2D', '4D')"
)
parser.add_argument(
    "i_mo", 
    type=float, 
    help="octupole current"
)
parser.add_argument(
    "line_path", 
    type=str, 
    help="xsuite line path"
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
n_bunch = 2730
bunch_intensity = 2.2e11
turn_step = 0.11245
max_turn = 60 + 9 * turn_step
charge = 1.60218e-19

s_PDSU_interlock = 120e-6
t_PDSU_interlock = s_PDSU_interlock * f_rev
delta_s_to_dump = 300e-6
delta_t_to_dump = delta_s_to_dump * f_rev


def main():
    '''
    Parse arguments
    '''
    args = parser.parse_args()
    result_path = args.result_path
    study_name = args.study_name
    seed = args.seed
    cliq_case = args.cliq_case
    halo_model = args.halo_model
    dist_mode = args.dist_mode
    i_mo = args.i_mo
    line_path = args.line_path
    save = args.save
    show = args.show

    result_path = Path(result_path).resolve() / study_name


    '''
    Load line
    '''
    line = xt.load(line_path)
    if "b1" in line_path:
        beam = 1
    elif "b2" in line_path:
        beam = 2
    else:
        raise ValueError("Beam cannot be determined from line path!")
    line_is_reversed = True if f'{beam}' == '2' else False

    with open("data/lhc_collimation_metadata.json", "r") as file:
        metadata = json.load(file)

    colldb = xc.CollimatorDatabase.from_yaml("data/CollDB_tight_150mm.yaml", beam=beam)
    colldb.install_everest_collimators(line=line, verbose=True)
    tw = line.twiss()
    line.xcoll.collimators.assign_optics(twiss=tw)
    line.xcoll.scattering.identify_primary_losses()
    

    '''
    Cumulative losses over time
    '''
    part_fin_file = result_path / f"part_fin_seed{seed}_{cliq_case}_{halo_model}_{dist_mode}_imo_{int(i_mo)}.pkl"
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

    # Time of PDSU dump
    t_PDSU = t_PDSU_interlock + delta_t_to_dump
    s_PDSU = s_PDSU_interlock + delta_s_to_dump

    try:
        t_BLM_interlock = turns[cum_lost_energy * 1e-3 >= 125][0]
        s_BLM_interlock = t_BLM_interlock / f_rev
        t_BLM = t_BLM_interlock + delta_t_to_dump
        s_BLM = s_BLM_interlock + delta_s_to_dump
    except IndexError:
        t_BLM_interlock = None


    '''
    Loss map at PDSU dump
    '''
    mask_PDSU = part_fin["at_turn"] <= (t_PDSU_interlock + delta_t_to_dump)
    part_fin_PDSU = {
        key: value[mask_PDSU] if isinstance(value, np.ndarray) else value for key, value in part_fin.items()
    }
    p_fin_PDSU = xt.Particles.from_dict(part_fin_PDSU)
    lm_PDSU = xc.LossMap(line, line_is_reversed=line_is_reversed, part=p_fin_PDSU)
    lm_PDSU.update_metadata(metadata)

    int_scale_PDSU = len(p_fin_PDSU.state) / 1e6
    if save:
        lm_PDSU.plot(
            savefig=result_path / f"lm_PDSU_{cliq_case}_{halo_model}_{dist_mode}_imo_{int(i_mo)}.png", 
            show=True if show else False, 
            norm="deposited_energy", 
            zoom="betatron", 
            beam_intensity=bunch_intensity*n_bunch*int_scale_PDSU
        )
    else:
        lm_PDSU.plot(
            show=True if show else False, 
            norm="deposited_energy", 
            zoom="betatron", 
            beam_intensity=bunch_intensity*n_bunch*int_scale_PDSU
        )


    '''
    Loss map at BLM dump
    '''
    if t_BLM_interlock is not None:
        mask_BLM = part_fin["at_turn"] <= (t_BLM_interlock + delta_t_to_dump)
        part_fin_BLM = {
            key: value[mask_BLM] if isinstance(value, np.ndarray) else value for key, value in part_fin.items()
        }
        p_fin_BLM = xt.Particles.from_dict(part_fin_BLM)
        lm_BLM = xc.LossMap(line, line_is_reversed=line_is_reversed, part=p_fin_BLM)
        lm_BLM.update_metadata(metadata)

        int_scale_BLM = len(p_fin_BLM.state) / 1e6
        if save:
            lm_BLM.plot(
                savefig=result_path / f"lm_BLM_{cliq_case}_{halo_model}_{dist_mode}_imo_{int(i_mo)}.png", 
                show=True if show else False, 
                norm="deposited_energy", 
                zoom="betatron", 
                beam_intensity=bunch_intensity*n_bunch*int_scale_BLM
            )
        else:
            lm_BLM.plot(
                show=True if show else False, 
                norm="deposited_energy", 
                zoom="betatron", 
                beam_intensity=bunch_intensity*n_bunch*int_scale_BLM
            )


    '''
    Loss evolution plot
    '''
    fs = 20
    markersize = 4

    # Linear scale
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    ax1_b = ax1.twiny()

    ax1.text(
        5, 1.99, 
        "Failure onset", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax1.axvspan(
        0, 120, 
        facecolor="dimgrey", 
        edgecolor="None", 
        alpha=0.5
    )
    ax1.axvline(120, color="black", linestyle=':', linewidth=3)
    ax1.text(
        115, 1.99, 
        "PSDU interlock", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax1.text(
        125, 1.99, 
        r"120 $\mu$s", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax1.axvspan(
        120, 242, 
        facecolor="grey", 
        edgecolor="None", 
        alpha=0.5
    )
    ax1.axvline(242, color="black", linestyle=':', linewidth=3)
    ax1.text(
        237, 1.99, 
        "BIS propagation", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax1.text(
        247, 1.99, 
        r"242 $\mu$s", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax1.axvspan(
        242, 331, 
        facecolor="darkgrey", 
        edgecolor="None", 
        alpha=0.5
    )
    ax1.axvline(331, color="black", linestyle=':', linewidth=3)
    ax1.text(
        326, 1.99, 
        "Abort gap synch.", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax1.text(
        336, 1.99, 
        r"331 $\mu$s", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax1.axvspan(
        331, 420, 
        facecolor="lightgrey", 
        edgecolor="None", 
        alpha=0.5
    )
    ax1.axvline(420, color="black", linestyle=':', linewidth=3)
    ax1.text(
        415, 1.99, 
        "Beam dumped", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax1.text(
        425, 1.99, 
        r"420 $\mu$s", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax1.axhline(0.125, color="red", linestyle='--', linewidth=3)
    ax1.text(
        19.99 / f_rev * 1e6, 0.115, 
        "BLM thresholds (125 kJ)", 
        fontsize=fs-8, 
        rotation="horizontal", 
        ha="right", 
        va="top", 
        color="red"
    )
    ax1.axhline(1, color="firebrick", linestyle='-.', linewidth=3)
    ax1.text(
        19.99 / f_rev * 1e6, 1.01, 
        "Critical losses (1 MJ)", 
        fontsize=fs-8, 
        rotation="horizontal", 
        ha="right", 
        va="bottom", 
        color="firebrick"
    )
    ax1.step(
        turns / f_rev * 1e6, 
        cum_lost_energy * 1e-6, 
        color='blue', 
        marker='s', 
        markersize=markersize, 
        where='mid'
    )
    ax1.set_xlim((0, 20 / f_rev * 1e6))
    ax1.set_ylim((0, 2))
    ax1.set_xlabel(r"time from failure onset [$\mu$s]", fontsize=fs)
    ax1.set_ylabel("beam losses [MJ]", fontsize=fs)
    ax1.tick_params(axis="both", labelsize=fs-2)
    ax1.grid(True, which='major', axis='y', color='gray', linestyle='-')
    ax1.grid(True, which='minor', axis='y', color='gray', linestyle='--')

    ax1_b.plot(
        turns, 
        cum_lost_energy * 1e-6, 
        color='blue', 
        marker='None', 
        linestyle='None'
    )
    ax1_b.set_xticks(np.arange(int(max_turn)), np.arange(int(max_turn)))
    ax1_b.set_xlim((0, 20))
    ax1_b.set_ylim((0, 2))
    ax1_b.set_xlabel(r"turn from failure onset", fontsize=fs)
    ax1_b.tick_params(axis="both", labelsize=fs-2)
    ax1_b.grid(True, which='major', axis='x', color='gray', linestyle='-')

    # Log scale
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    ax2_b = ax2.twiny()

    ax2.text(
        5, 99, 
        "Failure onset", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax2.axvspan(
        0, 120, 
        facecolor="dimgrey", 
        edgecolor="None", 
        alpha=0.5
    )
    ax2.axvline(120, color="black", linestyle=':', linewidth=3)
    ax2.text(
        115, 99, 
        "PSDU interlock", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax2.text(
        125, 99, 
        r"120 $\mu$s", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax2.axvspan(
        120, 242, 
        facecolor="grey", 
        edgecolor="None", 
        alpha=0.5
    )
    ax2.axvline(242, color="black", linestyle=':', linewidth=3)
    ax2.text(
        237, 99, 
        "BIS propagation", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax2.text(
        247, 99, 
        r"242 $\mu$s", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax2.axvspan(
        242, 331, 
        facecolor="darkgrey", 
        edgecolor="None", 
        alpha=0.5
    )
    ax2.axvline(331, color="black", linestyle=':', linewidth=3)
    ax2.text(
        326, 99, 
        "Abort gap synch.", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax2.text(
        336, 99, 
        r"331 $\mu$s", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax2.axvspan(
        331, 420, 
        facecolor="lightgrey", 
        edgecolor="None", 
        alpha=0.5
    )
    ax2.axvline(420, color="black", linestyle=':', linewidth=3)
    ax2.text(
        415, 99, 
        "Beam dumped", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="right", 
        va="top", 
        color="black"
    )
    ax2.text(
        425, 99, 
        r"420 $\mu$s", 
        fontsize=fs-8, 
        rotation="vertical", 
        ha="left", 
        va="top", 
        color="black"
    )
    ax2.axhline(0.125, color="red", linestyle='--', linewidth=3)
    ax2.text(
        19.99 / f_rev * 1e6, 0.115, 
        "BLM thresholds (125 kJ)", 
        fontsize=fs-8, 
        rotation="horizontal", 
        ha="right", 
        va="top", 
        color="red"
    )
    ax2.axhline(1, color="firebrick", linestyle='-.', linewidth=3)
    ax2.text(
        19.99 / f_rev * 1e6, 1.1, 
        "Critical losses (1 MJ)", 
        fontsize=fs-8, 
        rotation="horizontal", 
        ha="right", 
        va="bottom", 
        color="firebrick"
    )
    ax2.step(
        turns / f_rev * 1e6, 
        cum_lost_energy * 1e-6, 
        color='blue', 
        marker='s', 
        markersize=markersize, 
        where='mid'
    )
    ax2.set_xlim((0, 20 / f_rev * 1e6))
    ax2.set_ylim((1e-4, 1e2))
    ax2.set_xlabel(r"time from failure onset [$\mu$s]", fontsize=fs)
    ax2.set_ylabel(r"beam losses [MJ]", fontsize=fs)
    ax2.tick_params(axis="both", labelsize=fs-2)
    ax2.set_yscale("log")
    ax2.grid(True, which='major', axis='y', color='gray', linestyle='-')
    ax2.grid(True, which='minor', axis='y', color='gray', linestyle='--')

    ax2_b.plot(
        turns, 
        cum_lost_energy * 1e-6, 
        color='blue', 
        marker='None', 
        linestyle='None'
    )
    ax2_b.set_xticks(np.arange(int(max_turn)), np.arange(int(max_turn)))
    ax2_b.set_xlim((0, 20))
    ax2_b.set_ylim((1e-4, 1e2))
    ax2_b.set_xlabel(r"turn from failure onset", fontsize=fs)
    ax2_b.tick_params(axis="both", labelsize=fs-2)
    ax2_b.set_yscale("log")
    ax2_b.grid(True, which='major', axis='x', color='gray', linestyle='-')

    if save:
        fig1.savefig(result_path / f"losses_vs_time_lin_{cliq_case}_{halo_model}_{dist_mode}_imo_{int(i_mo)}.png", dpi=300, format='png', bbox_inches='tight')
        fig2.savefig(result_path / f"losses_vs_time_log_{cliq_case}_{halo_model}_{dist_mode}_imo_{int(i_mo)}.png", dpi=150, format='png', bbox_inches='tight')

    if show:
        plt.show()


if __name__ == "__main__":
    main()