import argparse
import json
import pickle
import numpy as np
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
    "halo_model", 
    type=str, 
    help="halo model ('mean', 'cons', or 'all')"
)
parser.add_argument(
    "dist_mode", 
    type=str, 
    help="distribution generation mode ('2D', '4D', or 'all')"
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

cliq_cases = [
    'Q1R5a', 'Q1R5b', 'Q1L5a', 'Q1L5b', 
    'Q2R5a', 'Q2R5b', 'Q2L5a', 'Q2L5b', 
    'Q3R5a', 'Q3R5b', 'Q3L5a', 'Q3L5b', 
    'Q1R1a', 'Q1R1b', 'Q1L1a', 'Q1L1b', 
    'Q2R1a', 'Q2R1b', 'Q2L1a', 'Q2L1b', 
    'Q3R1a', 'Q3R1b', 'Q3L1a', 'Q3L1b'
]
halo_models = ['mean', 'cons']
dist_modes = ['2D', '4D']

s_PDSU_interlock = 120e-6
t_PDSU_interlock = s_PDSU_interlock * f_rev
delta_s_to_dump = 300e-6
delta_t_to_dump = delta_s_to_dump * f_rev


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
x_ticks = np.arange(len(cliq_cases))


fs = 20
markersize = 8


def plot_1MJ_exceeded(
    ax, axb, 
    y_s, y_s_PDSU, y_s_BLM, 
    y_t, y_t_PDSU, y_t_BLM, 
    halo_model, dist_mode
):
    if halo_model == "mean":
        label = rf"$q=1.3$, $\beta=0.9$ ({dist_mode})"
    elif halo_model == "cons":
        label = rf"$q=1.5$, $\beta=1.68$ ({dist_mode})"
    else:
        raise ValueError(f"Not supported `halo_model` ({halo_model})!")

    if y_s_PDSU:
        ax.axhline(y_s_PDSU * 1e6, color="firebrick", linestyle='-.', linewidth=3)
    for x, y in zip(x_ticks, y_s_BLM * 1e6):
        ax.plot(
            [x-0.25, x+0.25], 
            [y, y], 
            color=colors[halo_model][dist_mode], 
            marker='None', 
            linestyle='--'
        )
    for x, y_blm, y in zip(x_ticks, y_s_BLM * 1e6, y_s * 1e6):
        ax.plot(
            [x, x], 
            [y_blm, y], 
            color=colors[halo_model][dist_mode], 
            marker='None', 
            linestyle='--'
        )
        if x == 0:
            ax.plot(
                x, y, 
                color=colors[halo_model][dist_mode], 
                marker=markers[halo_model][dist_mode], 
                markersize=markersize, 
                linestyle='None', 
                label=label
            )
        else:
            ax.plot(
                x, y, 
                color=colors[halo_model][dist_mode], 
                marker=markers[halo_model][dist_mode], 
                markersize=markersize, 
                linestyle='None'
            )

    if y_t_PDSU:
        axb.axhline(y_t_PDSU, linestyle='None')
    axb.plot(
        x_ticks, y_t_BLM, 
        marker='None', 
        linestyle='None'
    )
    axb.plot(
        x_ticks, y_t, 
        marker='None', 
        linestyle='None'
    )


def plot_1MJ_exceeded_wrt_dump(
    ax, axb, 
    y_s, y_t, 
    halo_model, dist_mode
):
    if halo_model == "mean":
        label = rf"$q=1.3$, $\beta=0.9$ ({dist_mode})"
    elif halo_model == "cons":
        label = rf"$q=1.5$, $\beta=1.68$ ({dist_mode})"
    else:
        raise ValueError(f"Not supported `dist_mode` ({dist_mode})!")

    ax.axhline(0, color="firebrick", linestyle='-.', linewidth=3)
    ax.plot(
        x_ticks, y_s * 1e6, 
        color=colors[halo_model][dist_mode], 
        marker=markers[halo_model][dist_mode], 
        markersize=markersize, 
        linestyle='None', 
        label=label
    )

    axb.axhline(0, linestyle='None')
    axb.plot(
        x_ticks, y_t, 
        marker='None', 
        linestyle='None'
    )


def plot_total_losses(ax, y, halo_model, dist_mode):
    if halo_model == "mean":
        label = rf"$q=1.3$, $\beta=0.9$ ({dist_mode})"
    elif halo_model == "cons":
        label = rf"$q=1.5$, $\beta=1.68$ ({dist_mode})"
    else:
        raise ValueError(f"Not supported `dist_mode` ({dist_mode})!")
    
    ax.plot(
        x_ticks, y, 
        color=colors[halo_model][dist_mode], 
        marker=markers[halo_model][dist_mode], 
        markersize=markersize, 
        linestyle='None', 
        label=label
    )


def plot_max_collimator_losses(ax, y_primary, y_total, halo_model, dist_mode):
    if halo_model == "mean":
        label = rf"$q=1.3$, $\beta=0.9$ ({dist_mode})"
    elif halo_model == "cons":
        label = rf"$q=1.5$, $\beta=1.68$ ({dist_mode})"
    else:
        raise ValueError(f"Not supported `dist_mode` ({dist_mode})!")
    
    for x, y_p, y_t in zip(x_ticks, y_primary, y_total):
        ax.plot(
            [x, x], 
            [y_p, y_t], 
            color=colors[halo_model][dist_mode], 
            marker='None', 
            linestyle='--'
        )
        ax.plot(
            x, y_p, 
            color=colors[halo_model][dist_mode], 
            marker=markers[halo_model][dist_mode], 
            markersize=markersize, 
            markerfacecolor=colors[halo_model][dist_mode], 
            fillstyle='none', 
            linestyle='None'
        )
        if x == 0:
            ax.plot(
                x, y_t, 
                color=colors[halo_model][dist_mode], 
                marker=markers[halo_model][dist_mode], 
                markersize=markersize, 
                linestyle='None', 
                label=label
            )
        else:
            ax.plot(
                x, y_t, 
                color=colors[halo_model][dist_mode], 
                marker=markers[halo_model][dist_mode], 
                markersize=markersize, 
                linestyle='None'
            )


def plot_max_collimator_losses2(ax, y_total, halo_model, dist_mode):
    if halo_model == "mean":
        label = rf"$q=1.3$, $\beta=0.9$ ({dist_mode})"
    elif halo_model == "cons":
        label = rf"$q=1.5$, $\beta=1.68$ ({dist_mode})"
    else:
        raise ValueError(f"Not supported `dist_mode` ({dist_mode})!")

    for x, y_t in zip(x_ticks, y_total):
        if x == 0:
            ax.plot(
                x, y_t, 
                color=colors[halo_model][dist_mode], 
                marker=markers[halo_model][dist_mode], 
                markersize=markersize, 
                linestyle='None', 
                label=label
            )
        else:
            ax.plot(
                x, y_t, 
                color=colors[halo_model][dist_mode], 
                marker=markers[halo_model][dist_mode], 
                markersize=markersize, 
                linestyle='None'
            )

    # data = {
    #     "TCP": y_TCP, 
    #     "TCSG/TCSPM": y_TCSG_TCSPM, 
    #     "TCL": y_TCL, 
    #     "TCT": y_TCT
    # }

    # width = 0.13
    # multiplier = -2
    # colours = ["green", "orange", "blue", "red"]

    # idx = 0
    # for coll, fraction in data.items():
    #     offset = width * multiplier
    #     rects = ax.bar(x_ticks + offset, fraction, width, color=colours[idx], label=coll, zorder=3)
    #     ax.bar_label(rects, padding=5, rotation=90, color='black', fontsize=fs-2, zorder=4)
    #     multiplier += 1
    #     idx += 1


def plot_collimator_names(
    ax, 
    y, 
    halo_model, dist_mode
):
    if halo_model == "mean":
        label = rf"$q=1.3$, $\beta=0.9$ ({dist_mode})"
    elif halo_model == "cons":
        label = rf"$q=1.5$, $\beta=1.68$ ({dist_mode})"
    else:
        raise ValueError(f"Not supported `dist_mode` ({dist_mode})!")

    ax.plot(
        x_ticks, y, 
        color=colors[halo_model][dist_mode], 
        marker=markers[halo_model][dist_mode], 
        markersize=markersize * 2, 
        markerfacecolor=colors[halo_model][dist_mode], 
        fillstyle='none', 
        linestyle='None', 
        label=label
    )


def main():
    '''
    Parse arguments
    '''
    args = parser.parse_args()
    result_path = args.result_path
    study_name = args.study_name
    halo_model = args.halo_model
    dist_mode = args.dist_mode
    save = args.save
    show = args.show


    '''
    Make plots
    '''

    # Time/turn of 1 MJ exceeded
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    ax1b = ax1.twinx()

    fig1_a, ax1_a = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    ax1b_a = ax1_a.twinx()

    fig1_b, ax1_b = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    ax1b_b = ax1_b.twinx()

    # Total losses at time of PDSU dump
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    # Total losses at time of BLM dump
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    # Maximum losses (primary and total) on a single collimator at time of PDSU dump
    fig4, ax4 = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    # Maximum losses (primary and total) on a single collimator at time of BLM dump
    fig5, ax5 = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    # Maximum losses (primary and total) on a single collimator when 1 MJ exceeded
    fig6, ax6 = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    # Name of collimator with maximum primary losses at time of PDSU dump
    fig7, ax7 = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    # Name of collimator with maximum primary losses at time of BLM dump
    fig8, ax8 = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    # Name of collimator with maximum primary losses when 1 MJ exceeded
    fig9, ax9 = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    # Name of collimator with maximum total losses at time of PDSU dump
    fig10, ax10 = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    # Name of collimator with maximum total losses at time of BLM dump
    fig11, ax11 = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    # Name of collimator with maximum total losses when 1 MJ exceeded
    fig12, ax12 = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    # Maximum losses on a single collimator by type at time of PDSU dump
    fig13a, ax13a = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    fig13b, ax13b = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    fig13c, ax13c = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    fig13d, ax13d = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    # Maximum losses on a single collimator by type at time of BLM dump
    fig14a, ax14a = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    fig14b, ax14b = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    fig14c, ax14c = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    fig14d, ax14d = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    # Name of collimator with maximum total losses by type at time of PDSU dump
    fig15a, ax15a = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    fig15b, ax15b = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    fig15c, ax15c = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    fig15d, ax15d = plt.subplots(1, 1, figsize=(12, 6), layout="tight")

    # Name of collimator with maximum total losses by type at time of BLM dump
    fig16a, ax16a = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    fig16b, ax16b = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    fig16c, ax16c = plt.subplots(1, 1, figsize=(12, 6), layout="tight")
    fig16d, ax16d = plt.subplots(1, 1, figsize=(12, 6), layout="tight")


    '''
    Load observables and plot
    '''
    result_path = Path(result_path).resolve() / study_name

    if halo_model == "all" and dist_mode == "all":
        with open(result_path / "mean_halo_2D/observables.pkl", "rb") as file:
            obs_mean_2D = pickle.load(file)
        with open(result_path / "mean_halo_4D/observables.pkl", "rb") as file:
            obs_mean_4D = pickle.load(file)
        with open(result_path / "cons_halo_2D/observables.pkl", "rb") as file:
            obs_cons_2D = pickle.load(file)
        with open(result_path / "cons_halo_4D/observables.pkl", "rb") as file:
            obs_cons_4D = pickle.load(file)

        # Time/turn of 1 MJ exceeded
        plot_1MJ_exceeded(
            ax1, ax1b, 
            obs_mean_2D["s_1MJ"], obs_mean_2D["s_PDSU"], obs_mean_2D["s_BLM"], 
            obs_mean_2D["t_1MJ"], obs_mean_2D["t_PDSU"], obs_mean_2D["t_BLM"], 
            "mean", "2D"
        )
        plot_1MJ_exceeded(
            ax1, ax1b, 
            obs_mean_4D["s_1MJ"], None, obs_mean_4D["s_BLM"], 
            obs_mean_4D["t_1MJ"], None, obs_mean_4D["t_BLM"], 
            "mean", "4D"
        )
        plot_1MJ_exceeded(
            ax1, ax1b, 
            obs_cons_2D["s_1MJ"], None, obs_cons_2D["s_BLM"], 
            obs_cons_2D["t_1MJ"], None, obs_cons_2D["t_BLM"], 
            "cons", "2D"
        )
        plot_1MJ_exceeded(
            ax1, ax1b, 
            obs_cons_4D["s_1MJ"], None, obs_cons_4D["s_BLM"], 
            obs_cons_4D["t_1MJ"], None, obs_cons_4D["t_BLM"], 
            "cons", "4D"
        )

        plot_1MJ_exceeded_wrt_dump(
            ax1_a, ax1b_a, obs_mean_2D["s_1MJ_wrt_PDSU"], obs_mean_2D["t_1MJ_wrt_PDSU"], 
            "mean", "2D"
        )
        plot_1MJ_exceeded_wrt_dump(
            ax1_a, ax1b_a, obs_mean_4D["s_1MJ_wrt_PDSU"], obs_mean_2D["t_1MJ_wrt_PDSU"], 
            "mean", "4D"
        )
        plot_1MJ_exceeded_wrt_dump(
            ax1_a, ax1b_a, obs_cons_2D["s_1MJ_wrt_PDSU"], obs_cons_2D["t_1MJ_wrt_PDSU"], 
            "cons", "2D"
        )
        plot_1MJ_exceeded_wrt_dump(
            ax1_a, ax1b_a, obs_cons_4D["s_1MJ_wrt_PDSU"], obs_cons_2D["t_1MJ_wrt_PDSU"], 
            "cons", "4D"
        )

        plot_1MJ_exceeded_wrt_dump(
            ax1_b, ax1b_b, obs_mean_2D["s_1MJ_wrt_BLM"], obs_mean_2D["t_1MJ_wrt_BLM"], 
            "mean", "2D"
        )
        plot_1MJ_exceeded_wrt_dump(
            ax1_b, ax1b_b, obs_mean_4D["s_1MJ_wrt_BLM"], obs_mean_2D["t_1MJ_wrt_BLM"], 
            "mean", "4D"
        )
        plot_1MJ_exceeded_wrt_dump(
            ax1_b, ax1b_b, obs_cons_2D["s_1MJ_wrt_BLM"], obs_cons_2D["t_1MJ_wrt_BLM"], 
            "cons", "2D"
        )
        plot_1MJ_exceeded_wrt_dump(
            ax1_b, ax1b_b, obs_cons_4D["s_1MJ_wrt_BLM"], obs_cons_2D["t_1MJ_wrt_BLM"], 
            "cons", "4D"
        )

        # Total losses at time of PDSU dump
        plot_total_losses(
            ax2, obs_mean_2D["total_MJ_at_PDSU"], "mean", "2D"
        )
        plot_total_losses(
            ax2, obs_mean_4D["total_MJ_at_PDSU"], "mean", "4D"
        )
        plot_total_losses(
            ax2, obs_cons_2D["total_MJ_at_PDSU"], "cons", "2D"
        )
        plot_total_losses(
            ax2, obs_cons_4D["total_MJ_at_PDSU"], "cons", "4D"
        )

        # Total losses at time of BLM dump
        plot_total_losses(
            ax3, obs_mean_2D["total_MJ_at_BLM"], "mean", "2D"
        )
        plot_total_losses(
            ax3, obs_mean_4D["total_MJ_at_BLM"], "mean", "4D"
        )
        plot_total_losses(
            ax3, obs_cons_2D["total_MJ_at_BLM"], "cons", "2D"
        )
        plot_total_losses(
            ax3, obs_cons_4D["total_MJ_at_BLM"], "cons", "4D"
        )

        # Maximum losses (primary and total) on a single collimator at time of PDSU dump
        plot_max_collimator_losses(
            ax4, 
            obs_mean_2D["max_primary_coll_MJ_at_PDSU"], obs_mean_2D["max_total_coll_MJ_at_PDSU"], 
            "mean", "2D"
        )
        plot_max_collimator_losses(
            ax4, 
            obs_mean_4D["max_primary_coll_MJ_at_PDSU"], obs_mean_4D["max_total_coll_MJ_at_PDSU"], 
            "mean", "4D"
        )
        plot_max_collimator_losses(
            ax4, 
            obs_cons_2D["max_primary_coll_MJ_at_PDSU"], obs_cons_2D["max_total_coll_MJ_at_PDSU"], 
            "cons", "2D"
        )
        plot_max_collimator_losses(
            ax4, 
            obs_cons_4D["max_primary_coll_MJ_at_PDSU"], obs_cons_4D["max_total_coll_MJ_at_PDSU"], 
            "cons", "4D"
        )

        # Maximum losses (primary and total) on a single collimator at time of BLM dump
        plot_max_collimator_losses(
            ax5, 
            obs_mean_2D["max_primary_coll_MJ_at_BLM"], obs_mean_2D["max_total_coll_MJ_at_BLM"], 
            "mean", "2D"
        )
        plot_max_collimator_losses(
            ax5, 
            obs_mean_4D["max_primary_coll_MJ_at_BLM"], obs_mean_4D["max_total_coll_MJ_at_BLM"], 
            "mean", "4D"
        )
        plot_max_collimator_losses(
            ax5, 
            obs_cons_2D["max_primary_coll_MJ_at_BLM"], obs_cons_2D["max_total_coll_MJ_at_BLM"], 
            "cons", "2D"
        )
        plot_max_collimator_losses(
            ax5, 
            obs_cons_4D["max_primary_coll_MJ_at_BLM"], obs_cons_4D["max_total_coll_MJ_at_BLM"], 
            "cons", "4D"
        )

        # Maximum losses (primary and total) on a single collimator when 1 MJ exceeded
        plot_max_collimator_losses(
            ax6, 
            obs_mean_2D["max_primary_coll_MJ_at_1MJ"], obs_mean_2D["max_total_coll_MJ_at_1MJ"], 
            "mean", "2D"
        )
        plot_max_collimator_losses(
            ax6, 
            obs_mean_4D["max_primary_coll_MJ_at_1MJ"], obs_mean_4D["max_total_coll_MJ_at_1MJ"], 
            "mean", "4D"
        )
        plot_max_collimator_losses(
            ax6, 
            obs_cons_2D["max_primary_coll_MJ_at_1MJ"], obs_cons_2D["max_total_coll_MJ_at_1MJ"], 
            "cons", "2D"
        )
        plot_max_collimator_losses(
            ax6, 
            obs_cons_4D["max_primary_coll_MJ_at_1MJ"], obs_cons_4D["max_total_coll_MJ_at_1MJ"], 
            "cons", "4D"
        )

        # Name of collimator with maximum primary losses at time of PDSU dump
        names_primary_PDSU = np.unique(np.hstack((
            obs_mean_2D["max_primary_coll_name_at_PDSU"], 
            obs_mean_4D["max_primary_coll_name_at_PDSU"], 
            obs_cons_2D["max_primary_coll_name_at_PDSU"], 
            obs_cons_4D["max_primary_coll_name_at_PDSU"], 
        )))
        if "" in names_primary_PDSU:
            names_primary_PDSU = np.delete(names_primary_PDSU, np.where(names_primary_PDSU == "")[0][0])
        idxs_primary_PDSU_mean_2D = np.full(len(cliq_cases), np.nan)
        idxs_primary_PDSU_mean_4D = np.full(len(cliq_cases), np.nan)
        idxs_primary_PDSU_cons_2D = np.full(len(cliq_cases), np.nan)
        idxs_primary_PDSU_cons_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean_2D = np.where(names_primary_PDSU == obs_mean_2D["max_primary_coll_name_at_PDSU"][i])[0]
            try:
                idxs_primary_PDSU_mean_2D[i] = idx_mean_2D[0]
            except IndexError:
                pass
            idx_mean_4D = np.where(names_primary_PDSU == obs_mean_4D["max_primary_coll_name_at_PDSU"][i])[0]
            try:
                idxs_primary_PDSU_mean_4D[i] = idx_mean_4D[0]
            except IndexError:
                pass
            idx_cons_2D = np.where(names_primary_PDSU == obs_cons_2D["max_primary_coll_name_at_PDSU"][i])[0]
            try:
                idxs_primary_PDSU_cons_2D[i] = idx_cons_2D[0]
            except IndexError:
                pass
            idx_cons_4D = np.where(names_primary_PDSU == obs_cons_4D["max_primary_coll_name_at_PDSU"][i])[0]
            try:
                idxs_primary_PDSU_cons_4D[i] = idx_cons_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax7, idxs_primary_PDSU_mean_2D, "mean", "2D")
        plot_collimator_names(ax7, idxs_primary_PDSU_mean_4D, "mean", "4D")
        plot_collimator_names(ax7, idxs_primary_PDSU_cons_2D, "cons", "2D")
        plot_collimator_names(ax7, idxs_primary_PDSU_cons_4D, "cons", "4D")
        ax7.set_yticks(np.arange(len(names_primary_PDSU)), labels=names_primary_PDSU)

        # Name of collimator with maximum primary losses at time of BLM dump
        names_primary_BLM = np.unique(np.hstack((
            obs_mean_2D["max_primary_coll_name_at_BLM"], 
            obs_mean_4D["max_primary_coll_name_at_BLM"], 
            obs_cons_2D["max_primary_coll_name_at_BLM"], 
            obs_cons_4D["max_primary_coll_name_at_BLM"], 
        )))
        if "" in names_primary_BLM:
            names_primary_BLM = np.delete(names_primary_BLM, np.where(names_primary_BLM == "")[0][0])
        idxs_primary_BLM_mean_2D = np.full(len(cliq_cases), np.nan)
        idxs_primary_BLM_mean_4D = np.full(len(cliq_cases), np.nan)
        idxs_primary_BLM_cons_2D = np.full(len(cliq_cases), np.nan)
        idxs_primary_BLM_cons_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean_2D = np.where(names_primary_BLM == obs_mean_2D["max_primary_coll_name_at_BLM"][i])[0]
            try:
                idxs_primary_BLM_mean_2D[i] = idx_mean_2D[0]
            except IndexError:
                pass
            idx_mean_4D = np.where(names_primary_BLM == obs_mean_4D["max_primary_coll_name_at_BLM"][i])[0]
            try:
                idxs_primary_BLM_mean_4D[i] = idx_mean_4D[0]
            except IndexError:
                pass
            idx_cons_2D = np.where(names_primary_BLM == obs_cons_2D["max_primary_coll_name_at_BLM"][i])[0]
            try:
                idxs_primary_BLM_cons_2D[i] = idx_cons_2D[0]
            except IndexError:
                pass
            idx_cons_4D = np.where(names_primary_BLM == obs_cons_4D["max_primary_coll_name_at_BLM"][i])[0]
            try:
                idxs_primary_BLM_cons_4D[i] = idx_cons_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax8, idxs_primary_BLM_mean_2D, "mean", "2D")
        plot_collimator_names(ax8, idxs_primary_BLM_mean_4D, "mean", "4D")
        plot_collimator_names(ax8, idxs_primary_BLM_cons_2D, "cons", "2D")
        plot_collimator_names(ax8, idxs_primary_BLM_cons_4D, "cons", "4D")
        ax8.set_yticks(np.arange(len(names_primary_BLM)), labels=names_primary_BLM)

        # Name of collimator with maximum primary losses when 1MJ exceeded
        names_primary_1MJ = np.unique(np.hstack((
            obs_mean_2D["max_primary_coll_name_at_1MJ"], 
            obs_mean_4D["max_primary_coll_name_at_1MJ"], 
            obs_cons_2D["max_primary_coll_name_at_1MJ"], 
            obs_cons_4D["max_primary_coll_name_at_1MJ"], 
        )))
        if "" in names_primary_1MJ:
            names_primary_1MJ = np.delete(names_primary_1MJ, np.where(names_primary_1MJ == "")[0][0])
        idxs_primary_1MJ_mean_2D = np.full(len(cliq_cases), np.nan)
        idxs_primary_1MJ_mean_4D = np.full(len(cliq_cases), np.nan)
        idxs_primary_1MJ_cons_2D = np.full(len(cliq_cases), np.nan)
        idxs_primary_1MJ_cons_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean_2D = np.where(names_primary_1MJ == obs_mean_2D["max_primary_coll_name_at_1MJ"][i])[0]
            try:
                idxs_primary_1MJ_mean_2D[i] = idx_mean_2D[0]
            except IndexError:
                pass
            idx_mean_4D = np.where(names_primary_1MJ == obs_mean_4D["max_primary_coll_name_at_1MJ"][i])[0]
            try:
                idxs_primary_1MJ_mean_4D[i] = idx_mean_4D[0]
            except IndexError:
                pass
            idx_cons_2D = np.where(names_primary_1MJ == obs_cons_2D["max_primary_coll_name_at_1MJ"][i])[0]
            try:
                idxs_primary_1MJ_cons_2D[i] = idx_cons_2D[0]
            except IndexError:
                pass
            idx_cons_4D = np.where(names_primary_1MJ == obs_cons_4D["max_primary_coll_name_at_1MJ"][i])[0]
            try:
                idxs_primary_1MJ_cons_4D[i] = idx_cons_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax9, idxs_primary_1MJ_mean_2D, "mean", "2D")
        plot_collimator_names(ax9, idxs_primary_1MJ_mean_4D, "mean", "4D")
        plot_collimator_names(ax9, idxs_primary_1MJ_cons_2D, "cons", "2D")
        plot_collimator_names(ax9, idxs_primary_1MJ_cons_4D, "cons", "4D")
        ax9.set_yticks(np.arange(len(names_primary_1MJ)), labels=names_primary_1MJ)

        # Name of collimator with maximum total losses at time of PDSU dump
        names_total_PDSU = np.unique(np.hstack((
            obs_mean_2D["max_total_coll_name_at_PDSU"], 
            obs_mean_4D["max_total_coll_name_at_PDSU"], 
            obs_cons_2D["max_total_coll_name_at_PDSU"], 
            obs_cons_4D["max_total_coll_name_at_PDSU"], 
        )))
        if "" in names_total_PDSU:
            names_total_PDSU = np.delete(names_total_PDSU, np.where(names_total_PDSU == "")[0][0])
        idxs_total_PDSU_mean_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_mean_4D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_cons_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_cons_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean_2D = np.where(names_total_PDSU == obs_mean_2D["max_total_coll_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_mean_2D[i] = idx_mean_2D[0]
            except IndexError:
                pass
            idx_mean_4D = np.where(names_total_PDSU == obs_mean_4D["max_total_coll_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_mean_4D[i] = idx_mean_4D[0]
            except IndexError:
                pass
            idx_cons_2D = np.where(names_total_PDSU == obs_cons_2D["max_total_coll_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_cons_2D[i] = idx_cons_2D[0]
            except IndexError:
                pass
            idx_cons_4D = np.where(names_total_PDSU == obs_cons_4D["max_total_coll_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_cons_4D[i] = idx_cons_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax10, idxs_total_PDSU_mean_2D, "mean", "2D")
        plot_collimator_names(ax10, idxs_total_PDSU_mean_4D, "mean", "4D")
        plot_collimator_names(ax10, idxs_total_PDSU_cons_2D, "cons", "2D")
        plot_collimator_names(ax10, idxs_total_PDSU_cons_4D, "cons", "4D")
        ax10.set_yticks(np.arange(len(names_total_PDSU)), labels=names_total_PDSU)

        # Name of collimator with maximum total losses at time of BLM dump
        names_total_BLM = np.unique(np.hstack((
            obs_mean_2D["max_total_coll_name_at_BLM"], 
            obs_mean_4D["max_total_coll_name_at_BLM"], 
            obs_cons_2D["max_total_coll_name_at_BLM"], 
            obs_cons_4D["max_total_coll_name_at_BLM"], 
        )))
        if "" in names_total_BLM:
            names_total_BLM = np.delete(names_total_BLM, np.where(names_total_BLM == "")[0][0])
        idxs_total_BLM_mean_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_mean_4D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_cons_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_cons_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean_2D = np.where(names_total_BLM == obs_mean_2D["max_total_coll_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_mean_2D[i] = idx_mean_2D[0]
            except IndexError:
                pass
            idx_mean_4D = np.where(names_total_BLM == obs_mean_4D["max_total_coll_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_mean_4D[i] = idx_mean_4D[0]
            except IndexError:
                pass
            idx_cons_2D = np.where(names_total_BLM == obs_cons_2D["max_total_coll_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_cons_2D[i] = idx_cons_2D[0]
            except IndexError:
                pass
            idx_cons_4D = np.where(names_total_BLM == obs_cons_4D["max_total_coll_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_cons_4D[i] = idx_cons_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax11, idxs_total_BLM_mean_2D, "mean", "2D")
        plot_collimator_names(ax11, idxs_total_BLM_mean_4D, "mean", "4D")
        plot_collimator_names(ax11, idxs_total_BLM_cons_2D, "cons", "2D")
        plot_collimator_names(ax11, idxs_total_BLM_cons_4D, "cons", "4D")
        ax11.set_yticks(np.arange(len(names_total_BLM)), labels=names_total_BLM)

        # Name of collimator with maximum total losses when 1MJ exceeded
        names_total_1MJ = np.unique(np.hstack((
            obs_mean_2D["max_total_coll_name_at_1MJ"], 
            obs_mean_4D["max_total_coll_name_at_1MJ"], 
            obs_cons_2D["max_total_coll_name_at_1MJ"], 
            obs_cons_4D["max_total_coll_name_at_1MJ"], 
        )))
        if "" in names_total_1MJ:
            names_total_1MJ = np.delete(names_total_1MJ, np.where(names_total_1MJ == "")[0][0])
        idxs_total_1MJ_mean_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_1MJ_mean_4D = np.full(len(cliq_cases), np.nan)
        idxs_total_1MJ_cons_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_1MJ_cons_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean_2D = np.where(names_total_1MJ == obs_mean_2D["max_total_coll_name_at_1MJ"][i])[0]
            try:
                idxs_total_1MJ_mean_2D[i] = idx_mean_2D[0]
            except IndexError:
                pass
            idx_mean_4D = np.where(names_total_1MJ == obs_mean_4D["max_total_coll_name_at_1MJ"][i])[0]
            try:
                idxs_total_1MJ_mean_4D[i] = idx_mean_4D[0]
            except IndexError:
                pass
            idx_cons_2D = np.where(names_total_1MJ == obs_cons_2D["max_total_coll_name_at_1MJ"][i])[0]
            try:
                idxs_total_1MJ_cons_2D[i] = idx_cons_2D[0]
            except IndexError:
                pass
            idx_cons_4D = np.where(names_total_1MJ == obs_cons_4D["max_total_coll_name_at_1MJ"][i])[0]
            try:
                idxs_total_1MJ_cons_4D[i] = idx_cons_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax12, idxs_total_1MJ_mean_2D, "mean", "2D")
        plot_collimator_names(ax12, idxs_total_1MJ_mean_4D, "mean", "4D")
        plot_collimator_names(ax12, idxs_total_1MJ_cons_2D, "cons", "2D")
        plot_collimator_names(ax12, idxs_total_1MJ_cons_4D, "cons", "4D")
        ax12.set_yticks(np.arange(len(names_total_1MJ)), labels=names_total_1MJ)

        # Maximum total losses on a single collimator by type at time of PDSU dump
        plot_max_collimator_losses2(ax13a, obs_mean_2D["max_total_TCP_MJ_at_PDSU"], "mean", "2D")
        plot_max_collimator_losses2(ax13a, obs_mean_4D["max_total_TCP_MJ_at_PDSU"], "mean", "4D")
        plot_max_collimator_losses2(ax13a, obs_cons_2D["max_total_TCP_MJ_at_PDSU"], "cons", "2D")
        plot_max_collimator_losses2(ax13a, obs_cons_4D["max_total_TCP_MJ_at_PDSU"], "cons", "4D")
        plot_max_collimator_losses2(ax13b, obs_mean_2D["max_total_TCSG/TCSPM_MJ_at_PDSU"], "mean", "2D")
        plot_max_collimator_losses2(ax13b, obs_mean_4D["max_total_TCSG/TCSPM_MJ_at_PDSU"], "mean", "4D")
        plot_max_collimator_losses2(ax13b, obs_cons_2D["max_total_TCSG/TCSPM_MJ_at_PDSU"], "cons", "2D")
        plot_max_collimator_losses2(ax13b, obs_cons_4D["max_total_TCSG/TCSPM_MJ_at_PDSU"], "cons", "4D")
        plot_max_collimator_losses2(ax13c, obs_mean_2D["max_total_TCL_MJ_at_PDSU"], "mean", "2D")
        plot_max_collimator_losses2(ax13c, obs_mean_4D["max_total_TCL_MJ_at_PDSU"], "mean", "4D")
        plot_max_collimator_losses2(ax13c, obs_cons_2D["max_total_TCL_MJ_at_PDSU"], "cons", "2D")
        plot_max_collimator_losses2(ax13c, obs_cons_4D["max_total_TCL_MJ_at_PDSU"], "cons", "4D")
        plot_max_collimator_losses2(ax13d, obs_mean_2D["max_total_TCT_MJ_at_PDSU"], "mean", "2D")
        plot_max_collimator_losses2(ax13d, obs_mean_4D["max_total_TCT_MJ_at_PDSU"], "mean", "4D")
        plot_max_collimator_losses2(ax13d, obs_cons_2D["max_total_TCT_MJ_at_PDSU"], "cons", "2D")
        plot_max_collimator_losses2(ax13d, obs_cons_4D["max_total_TCT_MJ_at_PDSU"], "cons", "4D")

        # Maximum total losses on a single collimator by type at time of BLM dump
        plot_max_collimator_losses2(ax14a, obs_mean_2D["max_total_TCP_MJ_at_BLM"], "mean", "2D")
        plot_max_collimator_losses2(ax14a, obs_mean_4D["max_total_TCP_MJ_at_BLM"], "mean", "4D")
        plot_max_collimator_losses2(ax14a, obs_cons_2D["max_total_TCP_MJ_at_BLM"], "cons", "2D")
        plot_max_collimator_losses2(ax14a, obs_cons_4D["max_total_TCP_MJ_at_BLM"], "cons", "4D")
        plot_max_collimator_losses2(ax14b, obs_mean_2D["max_total_TCSG/TCSPM_MJ_at_BLM"], "mean", "2D")
        plot_max_collimator_losses2(ax14b, obs_mean_4D["max_total_TCSG/TCSPM_MJ_at_BLM"], "mean", "4D")
        plot_max_collimator_losses2(ax14b, obs_cons_2D["max_total_TCSG/TCSPM_MJ_at_BLM"], "cons", "2D")
        plot_max_collimator_losses2(ax14b, obs_cons_4D["max_total_TCSG/TCSPM_MJ_at_BLM"], "cons", "4D")
        plot_max_collimator_losses2(ax14c, obs_mean_2D["max_total_TCL_MJ_at_BLM"], "mean", "2D")
        plot_max_collimator_losses2(ax14c, obs_mean_4D["max_total_TCL_MJ_at_BLM"], "mean", "4D")
        plot_max_collimator_losses2(ax14c, obs_cons_2D["max_total_TCL_MJ_at_BLM"], "cons", "2D")
        plot_max_collimator_losses2(ax14c, obs_cons_4D["max_total_TCL_MJ_at_BLM"], "cons", "4D")
        plot_max_collimator_losses2(ax14d, obs_mean_2D["max_total_TCT_MJ_at_BLM"], "mean", "2D")
        plot_max_collimator_losses2(ax14d, obs_mean_4D["max_total_TCT_MJ_at_BLM"], "mean", "4D")
        plot_max_collimator_losses2(ax14d, obs_cons_2D["max_total_TCT_MJ_at_BLM"], "cons", "2D")
        plot_max_collimator_losses2(ax14d, obs_cons_4D["max_total_TCT_MJ_at_BLM"], "cons", "4D")

        # Name of collimator with maximum total losses by type at time of PDSU dump
        names_total_TCP_PDSU = np.unique(np.hstack((
            obs_mean_2D["max_total_TCP_name_at_PDSU"], 
            obs_mean_4D["max_total_TCP_name_at_PDSU"], 
            obs_cons_2D["max_total_TCP_name_at_PDSU"], 
            obs_cons_4D["max_total_TCP_name_at_PDSU"], 
        )))
        if "" in names_total_TCP_PDSU:
            names_total_TCP_PDSU = np.delete(names_total_TCP_PDSU, np.where(names_total_TCP_PDSU == "")[0][0])
        idxs_total_PDSU_mean_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_mean_4D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_cons_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_cons_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean_2D = np.where(names_total_TCP_PDSU == obs_mean_2D["max_total_TCP_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_mean_2D[i] = idx_mean_2D[0]
            except IndexError:
                pass
            idx_mean_4D = np.where(names_total_TCP_PDSU == obs_mean_4D["max_total_TCP_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_mean_4D[i] = idx_mean_4D[0]
            except IndexError:
                pass
            idx_cons_2D = np.where(names_total_TCP_PDSU == obs_cons_2D["max_total_TCP_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_cons_2D[i] = idx_cons_2D[0]
            except IndexError:
                pass
            idx_cons_4D = np.where(names_total_TCP_PDSU == obs_cons_4D["max_total_TCP_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_cons_4D[i] = idx_cons_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax15a, idxs_total_PDSU_mean_2D, "mean", "2D")
        plot_collimator_names(ax15a, idxs_total_PDSU_mean_4D, "mean", "4D")
        plot_collimator_names(ax15a, idxs_total_PDSU_cons_2D, "cons", "2D")
        plot_collimator_names(ax15a, idxs_total_PDSU_cons_4D, "cons", "4D")
        ax15a.set_yticks(np.arange(len(names_total_TCP_PDSU)), labels=names_total_TCP_PDSU)

        names_total_TCSG_TCSPM_PDSU = np.unique(np.hstack((
            obs_mean_2D["max_total_TCSG/TCSPM_name_at_PDSU"], 
            obs_mean_4D["max_total_TCSG/TCSPM_name_at_PDSU"], 
            obs_cons_2D["max_total_TCSG/TCSPM_name_at_PDSU"], 
            obs_cons_4D["max_total_TCSG/TCSPM_name_at_PDSU"], 
        )))
        if "" in names_total_TCSG_TCSPM_PDSU:
            names_total_TCSG_TCSPM_PDSU = np.delete(names_total_TCSG_TCSPM_PDSU, np.where(names_total_TCSG_TCSPM_PDSU == "")[0][0])
        idxs_total_PDSU_mean_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_mean_4D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_cons_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_cons_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean_2D = np.where(names_total_TCSG_TCSPM_PDSU == obs_mean_2D["max_total_TCSG/TCSPM_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_mean_2D[i] = idx_mean_2D[0]
            except IndexError:
                pass
            idx_mean_4D = np.where(names_total_TCSG_TCSPM_PDSU == obs_mean_4D["max_total_TCSG/TCSPM_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_mean_4D[i] = idx_mean_4D[0]
            except IndexError:
                pass
            idx_cons_2D = np.where(names_total_TCSG_TCSPM_PDSU == obs_cons_2D["max_total_TCSG/TCSPM_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_cons_2D[i] = idx_cons_2D[0]
            except IndexError:
                pass
            idx_cons_4D = np.where(names_total_TCSG_TCSPM_PDSU == obs_cons_4D["max_total_TCSG/TCSPM_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_cons_4D[i] = idx_cons_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax15b, idxs_total_PDSU_mean_2D, "mean", "2D")
        plot_collimator_names(ax15b, idxs_total_PDSU_mean_4D, "mean", "4D")
        plot_collimator_names(ax15b, idxs_total_PDSU_cons_2D, "cons", "2D")
        plot_collimator_names(ax15b, idxs_total_PDSU_cons_4D, "cons", "4D")
        ax15b.set_yticks(np.arange(len(names_total_TCSG_TCSPM_PDSU)), labels=names_total_TCSG_TCSPM_PDSU)

        names_total_TCL_PDSU = np.unique(np.hstack((
            obs_mean_2D["max_total_TCL_name_at_PDSU"], 
            obs_mean_4D["max_total_TCL_name_at_PDSU"], 
            obs_cons_2D["max_total_TCL_name_at_PDSU"], 
            obs_cons_4D["max_total_TCL_name_at_PDSU"], 
        )))
        if "" in names_total_TCL_PDSU:
            names_total_TCL_PDSU = np.delete(names_total_TCL_PDSU, np.where(names_total_TCL_PDSU == "")[0][0])
        idxs_total_PDSU_mean_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_mean_4D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_cons_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_cons_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean_2D = np.where(names_total_TCL_PDSU == obs_mean_2D["max_total_TCL_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_mean_2D[i] = idx_mean_2D[0]
            except IndexError:
                pass
            idx_mean_4D = np.where(names_total_TCL_PDSU == obs_mean_4D["max_total_TCL_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_mean_4D[i] = idx_mean_4D[0]
            except IndexError:
                pass
            idx_cons_2D = np.where(names_total_TCL_PDSU == obs_cons_2D["max_total_TCL_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_cons_2D[i] = idx_cons_2D[0]
            except IndexError:
                pass
            idx_cons_4D = np.where(names_total_TCL_PDSU == obs_cons_4D["max_total_TCL_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_cons_4D[i] = idx_cons_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax15c, idxs_total_PDSU_mean_2D, "mean", "2D")
        plot_collimator_names(ax15c, idxs_total_PDSU_mean_4D, "mean", "4D")
        plot_collimator_names(ax15c, idxs_total_PDSU_cons_2D, "cons", "2D")
        plot_collimator_names(ax15c, idxs_total_PDSU_cons_4D, "cons", "4D")
        ax15c.set_yticks(np.arange(len(names_total_TCL_PDSU)), labels=names_total_TCL_PDSU)

        names_total_TCT_PDSU = np.unique(np.hstack((
            obs_mean_2D["max_total_TCT_name_at_PDSU"], 
            obs_mean_4D["max_total_TCT_name_at_PDSU"], 
            obs_cons_2D["max_total_TCT_name_at_PDSU"], 
            obs_cons_4D["max_total_TCT_name_at_PDSU"], 
        )))
        if "" in names_total_TCT_PDSU:
            names_total_TCT_PDSU = np.delete(names_total_TCT_PDSU, np.where(names_total_TCT_PDSU == "")[0][0])
        idxs_total_PDSU_mean_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_mean_4D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_cons_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_cons_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean_2D = np.where(names_total_TCT_PDSU == obs_mean_2D["max_total_TCT_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_mean_2D[i] = idx_mean_2D[0]
            except IndexError:
                pass
            idx_mean_4D = np.where(names_total_TCT_PDSU == obs_mean_4D["max_total_TCT_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_mean_4D[i] = idx_mean_4D[0]
            except IndexError:
                pass
            idx_cons_2D = np.where(names_total_TCT_PDSU == obs_cons_2D["max_total_TCT_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_cons_2D[i] = idx_cons_2D[0]
            except IndexError:
                pass
            idx_cons_4D = np.where(names_total_TCT_PDSU == obs_cons_4D["max_total_TCT_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_cons_4D[i] = idx_cons_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax15d, idxs_total_PDSU_mean_2D, "mean", "2D")
        plot_collimator_names(ax15d, idxs_total_PDSU_mean_4D, "mean", "4D")
        plot_collimator_names(ax15d, idxs_total_PDSU_cons_2D, "cons", "2D")
        plot_collimator_names(ax15d, idxs_total_PDSU_cons_4D, "cons", "4D")
        ax15d.set_yticks(np.arange(len(names_total_TCT_PDSU)), labels=names_total_TCT_PDSU)

        # Name of collimator with maximum total losses by type at time of BLM dump
        names_total_TCP_BLM = np.unique(np.hstack((
            obs_mean_2D["max_total_TCP_name_at_BLM"], 
            obs_mean_4D["max_total_TCP_name_at_BLM"], 
            obs_cons_2D["max_total_TCP_name_at_BLM"], 
            obs_cons_4D["max_total_TCP_name_at_BLM"], 
        )))
        if "" in names_total_TCP_BLM:
            names_total_TCP_BLM = np.delete(names_total_TCP_BLM, np.where(names_total_TCP_BLM == "")[0][0])
        idxs_total_BLM_mean_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_mean_4D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_cons_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_cons_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean_2D = np.where(names_total_TCP_BLM == obs_mean_2D["max_total_TCP_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_mean_2D[i] = idx_mean_2D[0]
            except IndexError:
                pass
            idx_mean_4D = np.where(names_total_TCP_BLM == obs_mean_4D["max_total_TCP_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_mean_4D[i] = idx_mean_4D[0]
            except IndexError:
                pass
            idx_cons_2D = np.where(names_total_TCP_BLM == obs_cons_2D["max_total_TCP_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_cons_2D[i] = idx_cons_2D[0]
            except IndexError:
                pass
            idx_cons_4D = np.where(names_total_TCP_BLM == obs_cons_4D["max_total_TCP_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_cons_4D[i] = idx_cons_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax16a, idxs_total_BLM_mean_2D, "mean", "2D")
        plot_collimator_names(ax16a, idxs_total_BLM_mean_4D, "mean", "4D")
        plot_collimator_names(ax16a, idxs_total_BLM_cons_2D, "cons", "2D")
        plot_collimator_names(ax16a, idxs_total_BLM_cons_4D, "cons", "4D")
        ax16a.set_yticks(np.arange(len(names_total_TCP_BLM)), labels=names_total_TCP_BLM)

        names_total_TCSG_TCSPM_BLM = np.unique(np.hstack((
            obs_mean_2D["max_total_TCSG/TCSPM_name_at_BLM"], 
            obs_mean_4D["max_total_TCSG/TCSPM_name_at_BLM"], 
            obs_cons_2D["max_total_TCSG/TCSPM_name_at_BLM"], 
            obs_cons_4D["max_total_TCSG/TCSPM_name_at_BLM"], 
        )))
        if "" in names_total_TCSG_TCSPM_BLM:
            names_total_TCSG_TCSPM_BLM = np.delete(names_total_TCSG_TCSPM_BLM, np.where(names_total_TCSG_TCSPM_BLM == "")[0][0])
        idxs_total_BLM_mean_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_mean_4D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_cons_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_cons_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean_2D = np.where(names_total_TCSG_TCSPM_BLM == obs_mean_2D["max_total_TCSG/TCSPM_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_mean_2D[i] = idx_mean_2D[0]
            except IndexError:
                pass
            idx_mean_4D = np.where(names_total_TCSG_TCSPM_BLM == obs_mean_4D["max_total_TCSG/TCSPM_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_mean_4D[i] = idx_mean_4D[0]
            except IndexError:
                pass
            idx_cons_2D = np.where(names_total_TCSG_TCSPM_BLM == obs_cons_2D["max_total_TCSG/TCSPM_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_cons_2D[i] = idx_cons_2D[0]
            except IndexError:
                pass
            idx_cons_4D = np.where(names_total_TCSG_TCSPM_BLM == obs_cons_4D["max_total_TCSG/TCSPM_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_cons_4D[i] = idx_cons_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax16b, idxs_total_BLM_mean_2D, "mean", "2D")
        plot_collimator_names(ax16b, idxs_total_BLM_mean_4D, "mean", "4D")
        plot_collimator_names(ax16b, idxs_total_BLM_cons_2D, "cons", "2D")
        plot_collimator_names(ax16b, idxs_total_BLM_cons_4D, "cons", "4D")
        ax16b.set_yticks(np.arange(len(names_total_TCSG_TCSPM_BLM)), labels=names_total_TCSG_TCSPM_BLM)

        names_total_TCL_BLM = np.unique(np.hstack((
            obs_mean_2D["max_total_TCL_name_at_BLM"], 
            obs_mean_4D["max_total_TCL_name_at_BLM"], 
            obs_cons_2D["max_total_TCL_name_at_BLM"], 
            obs_cons_4D["max_total_TCL_name_at_BLM"], 
        )))
        if "" in names_total_TCL_BLM:
            names_total_TCL_BLM = np.delete(names_total_TCL_BLM, np.where(names_total_TCL_BLM == "")[0][0])
        idxs_total_BLM_mean_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_mean_4D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_cons_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_cons_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean_2D = np.where(names_total_TCL_BLM == obs_mean_2D["max_total_TCL_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_mean_2D[i] = idx_mean_2D[0]
            except IndexError:
                pass
            idx_mean_4D = np.where(names_total_TCL_BLM == obs_mean_4D["max_total_TCL_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_mean_4D[i] = idx_mean_4D[0]
            except IndexError:
                pass
            idx_cons_2D = np.where(names_total_TCL_BLM == obs_cons_2D["max_total_TCL_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_cons_2D[i] = idx_cons_2D[0]
            except IndexError:
                pass
            idx_cons_4D = np.where(names_total_TCL_BLM == obs_cons_4D["max_total_TCL_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_cons_4D[i] = idx_cons_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax16c, idxs_total_BLM_mean_2D, "mean", "2D")
        plot_collimator_names(ax16c, idxs_total_BLM_mean_4D, "mean", "4D")
        plot_collimator_names(ax16c, idxs_total_BLM_cons_2D, "cons", "2D")
        plot_collimator_names(ax16c, idxs_total_BLM_cons_4D, "cons", "4D")
        ax16c.set_yticks(np.arange(len(names_total_TCL_BLM)), labels=names_total_TCL_BLM)

        names_total_TCT_BLM = np.unique(np.hstack((
            obs_mean_2D["max_total_TCT_name_at_BLM"], 
            obs_mean_4D["max_total_TCT_name_at_BLM"], 
            obs_cons_2D["max_total_TCT_name_at_BLM"], 
            obs_cons_4D["max_total_TCT_name_at_BLM"], 
        )))
        if "" in names_total_TCT_BLM:
            names_total_TCT_BLM = np.delete(names_total_TCT_BLM, np.where(names_total_TCT_BLM == "")[0][0])
        idxs_total_BLM_mean_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_mean_4D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_cons_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_cons_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean_2D = np.where(names_total_TCT_BLM == obs_mean_2D["max_total_TCT_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_mean_2D[i] = idx_mean_2D[0]
            except IndexError:
                pass
            idx_mean_4D = np.where(names_total_TCT_BLM == obs_mean_4D["max_total_TCT_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_mean_4D[i] = idx_mean_4D[0]
            except IndexError:
                pass
            idx_cons_2D = np.where(names_total_TCT_BLM == obs_cons_2D["max_total_TCT_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_cons_2D[i] = idx_cons_2D[0]
            except IndexError:
                pass
            idx_cons_4D = np.where(names_total_TCT_BLM == obs_cons_4D["max_total_TCT_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_cons_4D[i] = idx_cons_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax16d, idxs_total_BLM_mean_2D, "mean", "2D")
        plot_collimator_names(ax16d, idxs_total_BLM_mean_4D, "mean", "4D")
        plot_collimator_names(ax16d, idxs_total_BLM_cons_2D, "cons", "2D")
        plot_collimator_names(ax16d, idxs_total_BLM_cons_4D, "cons", "4D")
        ax16d.set_yticks(np.arange(len(names_total_TCT_BLM)), labels=names_total_TCT_BLM)
    elif halo_model != "all" and dist_mode == "all":
        with open(result_path / f"{halo_model}_halo_2D/observables.pkl", "rb") as file:
            obs_2D = pickle.load(file)
        with open(result_path / f"{halo_model}_halo_4D/observables.pkl", "rb") as file:
            obs_4D = pickle.load(file)

        # Time/turn of 1 MJ exceeded
        plot_1MJ_exceeded(
            ax1, ax1b, 
            obs_2D["s_1MJ"], obs_2D["s_PDSU"], obs_2D["s_BLM"], 
            obs_2D["t_1MJ"], obs_2D["t_PDSU"], obs_2D["t_BLM"], 
            halo_model, "2D"
        )
        plot_1MJ_exceeded(
            ax1, ax1b, 
            obs_4D["s_1MJ"], None, obs_4D["s_BLM"], 
            obs_4D["t_1MJ"], None, obs_4D["t_BLM"], 
            halo_model, "4D"
        )

        plot_1MJ_exceeded_wrt_dump(
            ax1_a, ax1b_a, obs_2D["s_1MJ_wrt_PDSU"], obs_2D["t_1MJ_wrt_PDSU"], 
            halo_model, "2D"
        )
        plot_1MJ_exceeded_wrt_dump(
            ax1_a, ax1b_a, obs_4D["s_1MJ_wrt_PDSU"], obs_2D["t_1MJ_wrt_PDSU"], 
            halo_model, "4D"
        )

        plot_1MJ_exceeded_wrt_dump(
            ax1_b, ax1b_b, obs_2D["s_1MJ_wrt_BLM"], obs_2D["t_1MJ_wrt_BLM"], 
            halo_model, "2D"
        )
        plot_1MJ_exceeded_wrt_dump(
            ax1_b, ax1b_b, obs_4D["s_1MJ_wrt_BLM"], obs_2D["t_1MJ_wrt_BLM"], 
            halo_model, "4D"
        )

        # Total losses at time of PDSU dump
        plot_total_losses(
            ax2, obs_2D["total_MJ_at_PDSU"], halo_model, "2D"
        )
        plot_total_losses(
            ax2, obs_4D["total_MJ_at_PDSU"], halo_model, "4D"
        )

        # Total losses at time of BLM dump
        plot_total_losses(
            ax3, obs_2D["total_MJ_at_BLM"], halo_model, "2D"
        )
        plot_total_losses(
            ax3, obs_4D["total_MJ_at_BLM"], halo_model, "4D"
        )

        # Maximum losses (primary and total) on a single collimator at time of PDSU dump
        plot_max_collimator_losses(
            ax4, 
            obs_2D["max_primary_coll_MJ_at_PDSU"], obs_2D["max_total_coll_MJ_at_PDSU"], 
            halo_model, "2D"
        )
        plot_max_collimator_losses(
            ax4, 
            obs_4D["max_primary_coll_MJ_at_PDSU"], obs_4D["max_total_coll_MJ_at_PDSU"], 
            halo_model, "4D"
        )

        # Maximum losses (primary and total) on a single collimator at time of BLM dump
        plot_max_collimator_losses(
            ax5, 
            obs_2D["max_primary_coll_MJ_at_BLM"], obs_2D["max_total_coll_MJ_at_BLM"], 
            halo_model, "2D"
        )
        plot_max_collimator_losses(
            ax5, 
            obs_4D["max_primary_coll_MJ_at_BLM"], obs_4D["max_total_coll_MJ_at_BLM"], 
            halo_model, "4D"
        )

        # Maximum losses (primary and total) on a single collimator when 1 MJ exceeded
        plot_max_collimator_losses(
            ax6, 
            obs_2D["max_primary_coll_MJ_at_1MJ"], obs_2D["max_total_coll_MJ_at_1MJ"], 
            halo_model, "2D"
        )
        plot_max_collimator_losses(
            ax6, 
            obs_4D["max_primary_coll_MJ_at_1MJ"], obs_4D["max_total_coll_MJ_at_1MJ"], 
            halo_model, "4D"
        )

        # Name of collimator with maximum primary losses at time of PDSU dump
        names_primary_PDSU = np.unique(np.hstack((
            obs_2D["max_primary_coll_name_at_PDSU"], 
            obs_4D["max_primary_coll_name_at_PDSU"], 
        )))
        if "" in names_primary_PDSU:
            names_primary_PDSU = np.delete(names_primary_PDSU, np.where(names_primary_PDSU == "")[0][0])
        idxs_primary_PDSU_2D = np.full(len(cliq_cases), np.nan)
        idxs_primary_PDSU_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_2D = np.where(names_primary_PDSU == obs_2D["max_primary_coll_name_at_PDSU"][i])[0]
            try:
                idxs_primary_PDSU_2D[i] = idx_2D[0]
            except IndexError:
                pass
            idx_4D = np.where(names_primary_PDSU == obs_4D["max_primary_coll_name_at_PDSU"][i])[0]
            try:
                idxs_primary_PDSU_4D[i] = idx_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax7, idxs_primary_PDSU_2D, halo_model, "2D")
        plot_collimator_names(ax7, idxs_primary_PDSU_4D, halo_model, "4D")
        ax7.set_yticks(np.arange(len(names_primary_PDSU)), labels=names_primary_PDSU)

        # Name of collimator with maximum primary losses at time of BLM dump
        names_primary_BLM = np.unique(np.hstack((
            obs_2D["max_primary_coll_name_at_BLM"], 
            obs_4D["max_primary_coll_name_at_BLM"], 
        )))
        if "" in names_primary_BLM:
            names_primary_BLM = np.delete(names_primary_BLM, np.where(names_primary_BLM == "")[0][0])
        idxs_primary_BLM_2D = np.full(len(cliq_cases), np.nan)
        idxs_primary_BLM_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_2D = np.where(names_primary_BLM == obs_2D["max_primary_coll_name_at_BLM"][i])[0]
            try:
                idxs_primary_BLM_2D[i] = idx_2D[0]
            except IndexError:
                pass
            idx_4D = np.where(names_primary_BLM == obs_4D["max_primary_coll_name_at_BLM"][i])[0]
            try:
                idxs_primary_BLM_4D[i] = idx_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax8, idxs_primary_BLM_2D, halo_model, "2D")
        plot_collimator_names(ax8, idxs_primary_BLM_4D, halo_model, "4D")
        ax8.set_yticks(np.arange(len(names_primary_BLM)), labels=names_primary_BLM)

        # Name of collimator with maximum primary losses when 1MJ exceeded
        names_primary_1MJ = np.unique(np.hstack((
            obs_2D["max_primary_coll_name_at_1MJ"], 
            obs_4D["max_primary_coll_name_at_1MJ"], 
        )))
        if "" in names_primary_1MJ:
            names_primary_1MJ = np.delete(names_primary_1MJ, np.where(names_primary_1MJ == "")[0][0])
        idxs_primary_1MJ_2D = np.full(len(cliq_cases), np.nan)
        idxs_primary_1MJ_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_2D = np.where(names_primary_1MJ == obs_2D["max_primary_coll_name_at_1MJ"][i])[0]
            try:
                idxs_primary_1MJ_2D[i] = idx_2D[0]
            except IndexError:
                pass
            idx_4D = np.where(names_primary_1MJ == obs_4D["max_primary_coll_name_at_1MJ"][i])[0]
            try:
                idxs_primary_1MJ_4D[i] = idx_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax9, idxs_primary_1MJ_2D, halo_model, "2D")
        plot_collimator_names(ax9, idxs_primary_1MJ_4D, halo_model, "4D")
        ax9.set_yticks(np.arange(len(names_primary_1MJ)), labels=names_primary_1MJ)

        # Name of collimator with maximum total losses at time of PDSU dump
        names_total_PDSU = np.unique(np.hstack((
            obs_2D["max_total_coll_name_at_PDSU"], 
            obs_4D["max_total_coll_name_at_PDSU"], 
        )))
        if "" in names_total_PDSU:
            names_total_PDSU = np.delete(names_total_PDSU, np.where(names_total_PDSU == "")[0][0])
        idxs_total_PDSU_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_2D = np.where(names_total_PDSU == obs_2D["max_total_coll_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_2D[i] = idx_2D[0]
            except IndexError:
                pass
            idx_4D = np.where(names_total_PDSU == obs_4D["max_total_coll_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_4D[i] = idx_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax10, idxs_total_PDSU_2D, halo_model, "2D")
        plot_collimator_names(ax10, idxs_total_PDSU_4D, halo_model, "4D")
        ax10.set_yticks(np.arange(len(names_total_PDSU)), labels=names_total_PDSU)

        # Name of collimator with maximum total losses at time of BLM dump
        names_total_BLM = np.unique(np.hstack((
            obs_2D["max_total_coll_name_at_BLM"], 
            obs_4D["max_total_coll_name_at_BLM"], 
        )))
        if "" in names_total_BLM:
            names_total_BLM = np.delete(names_total_BLM, np.where(names_total_BLM == "")[0][0])
        idxs_total_BLM_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_2D = np.where(names_total_BLM == obs_2D["max_total_coll_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_2D[i] = idx_2D[0]
            except IndexError:
                pass
            idx_4D = np.where(names_total_BLM == obs_4D["max_total_coll_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_4D[i] = idx_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax11, idxs_total_BLM_2D, halo_model, "2D")
        plot_collimator_names(ax11, idxs_total_BLM_4D, halo_model, "4D")
        ax11.set_yticks(np.arange(len(names_total_BLM)), labels=names_total_BLM)

        # Name of collimator with maximum total losses when 1MJ exceeded
        names_total_1MJ = np.unique(np.hstack((
            obs_2D["max_total_coll_name_at_1MJ"], 
            obs_4D["max_total_coll_name_at_1MJ"], 
        )))
        if "" in names_total_1MJ:
            names_total_1MJ = np.delete(names_total_1MJ, np.where(names_total_1MJ == "")[0][0])
        idxs_total_1MJ_2D = np.full(len(cliq_cases), np.nan)
        idxs_total_1MJ_4D = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_2D = np.where(names_total_1MJ == obs_2D["max_total_coll_name_at_1MJ"][i])[0]
            try:
                idxs_total_1MJ_2D[i] = idx_2D[0]
            except IndexError:
                pass
            idx_4D = np.where(names_total_1MJ == obs_4D["max_total_coll_name_at_1MJ"][i])[0]
            try:
                idxs_total_1MJ_4D[i] = idx_4D[0]
            except IndexError:
                pass
        plot_collimator_names(ax12, idxs_total_1MJ_2D, halo_model, "2D")
        plot_collimator_names(ax12, idxs_total_1MJ_4D, halo_model, "4D")
        ax12.set_yticks(np.arange(len(names_total_1MJ)), labels=names_total_1MJ)
    elif halo_model == "all" and dist_mode != "all":
        with open(result_path / f"mean_halo_{dist_mode}/observables.pkl", "rb") as file:
            obs_mean = pickle.load(file)
        with open(result_path / f"cons_halo_{dist_mode}/observables.pkl", "rb") as file:
            obs_cons = pickle.load(file)

        # Time/turn of 1 MJ exceeded
        plot_1MJ_exceeded(
            ax1, ax1b, 
            obs_mean["s_1MJ"], obs_mean["s_PDSU"], obs_mean["s_BLM"], 
            obs_mean["t_1MJ"], obs_mean["t_PDSU"], obs_mean["t_BLM"], 
            "mean", dist_mode
        )
        plot_1MJ_exceeded(
            ax1, ax1b, 
            obs_cons["s_1MJ"], None, obs_cons["s_BLM"], 
            obs_cons["t_1MJ"], None, obs_cons["t_BLM"], 
            "cons", dist_mode
        )

        plot_1MJ_exceeded_wrt_dump(
            ax1_a, ax1b_a, obs_mean["s_1MJ_wrt_PDSU"], obs_mean["t_1MJ_wrt_PDSU"], 
            "mean", dist_mode
        )
        plot_1MJ_exceeded_wrt_dump(
            ax1_a, ax1b_a, obs_cons["s_1MJ_wrt_PDSU"], obs_cons["t_1MJ_wrt_PDSU"], 
            "cons", dist_mode
        )

        plot_1MJ_exceeded_wrt_dump(
            ax1_b, ax1b_b, obs_mean["s_1MJ_wrt_BLM"], obs_mean["t_1MJ_wrt_BLM"], 
            "mean", dist_mode
        )
        plot_1MJ_exceeded_wrt_dump(
            ax1_b, ax1b_b, obs_cons["s_1MJ_wrt_BLM"], obs_cons["t_1MJ_wrt_BLM"], 
            "cons", dist_mode
        )

        # Total losses at time of PDSU dump
        plot_total_losses(
            ax2, obs_mean["total_MJ_at_PDSU"], "mean", dist_mode
        )
        plot_total_losses(
            ax2, obs_cons["total_MJ_at_PDSU"], "cons", dist_mode
        )

        # Total losses at time of BLM dump
        plot_total_losses(
            ax3, obs_mean["total_MJ_at_BLM"], "mean", dist_mode
        )
        plot_total_losses(
            ax3, obs_cons["total_MJ_at_BLM"], "cons", dist_mode
        )

        # Maximum losses (primary and total) on a single collimator at time of PDSU dump
        plot_max_collimator_losses(
            ax4, 
            obs_mean["max_primary_coll_MJ_at_PDSU"], obs_mean["max_total_coll_MJ_at_PDSU"], 
            "mean", dist_mode
        )
        plot_max_collimator_losses(
            ax4, 
            obs_cons["max_primary_coll_MJ_at_PDSU"], obs_cons["max_total_coll_MJ_at_PDSU"], 
            "cons", dist_mode
        )

        # Maximum losses (primary and total) on a single collimator at time of BLM dump
        plot_max_collimator_losses(
            ax5, 
            obs_mean["max_primary_coll_MJ_at_BLM"], obs_mean["max_total_coll_MJ_at_BLM"], 
            "mean", dist_mode
        )
        plot_max_collimator_losses(
            ax5, 
            obs_cons["max_primary_coll_MJ_at_BLM"], obs_cons["max_total_coll_MJ_at_BLM"], 
            "cons", dist_mode
        )

        # Maximum losses (primary and total) on a single collimator when 1 MJ exceeded
        plot_max_collimator_losses(
            ax6, 
            obs_mean["max_primary_coll_MJ_at_1MJ"], obs_mean["max_total_coll_MJ_at_1MJ"], 
            "mean", dist_mode
        )
        plot_max_collimator_losses(
            ax6, 
            obs_cons["max_primary_coll_MJ_at_1MJ"], obs_cons["max_total_coll_MJ_at_1MJ"], 
            "cons", dist_mode
        )

        # Name of collimator with maximum primary losses at time of PDSU dump
        names_primary_PDSU = np.unique(np.hstack((
            obs_mean["max_primary_coll_name_at_PDSU"], 
            obs_cons["max_primary_coll_name_at_PDSU"], 
        )))
        if "" in names_primary_PDSU:
            names_primary_PDSU = np.delete(names_primary_PDSU, np.where(names_primary_PDSU == "")[0][0])
        idxs_primary_PDSU_mean = np.full(len(cliq_cases), np.nan)
        idxs_primary_PDSU_cons = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean = np.where(names_primary_PDSU == obs_mean["max_primary_coll_name_at_PDSU"][i])[0]
            try:
                idxs_primary_PDSU_mean[i] = idx_mean[0]
            except IndexError:
                pass
            idx_cons = np.where(names_primary_PDSU == obs_cons["max_primary_coll_name_at_PDSU"][i])[0]
            try:
                idxs_primary_PDSU_cons[i] = idx_cons[0]
            except IndexError:
                pass
        plot_collimator_names(ax7, idxs_primary_PDSU_mean, "mean", dist_mode)
        plot_collimator_names(ax7, idxs_primary_PDSU_cons, "cons", dist_mode)
        ax7.set_yticks(np.arange(len(names_primary_PDSU)), labels=names_primary_PDSU)

        # Name of collimator with maximum primary losses at time of BLM dump
        names_primary_BLM = np.unique(np.hstack((
            obs_mean["max_primary_coll_name_at_BLM"], 
            obs_cons["max_primary_coll_name_at_BLM"], 
        )))
        if "" in names_primary_BLM:
            names_primary_BLM = np.delete(names_primary_BLM, np.where(names_primary_BLM == "")[0][0])
        idxs_primary_BLM_mean = np.full(len(cliq_cases), np.nan)
        idxs_primary_BLM_cons = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean = np.where(names_primary_BLM == obs_mean["max_primary_coll_name_at_BLM"][i])[0]
            try:
                idxs_primary_BLM_mean[i] = idx_mean[0]
            except IndexError:
                pass
            idx_cons = np.where(names_primary_BLM == obs_cons["max_primary_coll_name_at_BLM"][i])[0]
            try:
                idxs_primary_BLM_cons[i] = idx_cons[0]
            except IndexError:
                pass
        plot_collimator_names(ax8, idxs_primary_BLM_mean, "mean", dist_mode)
        plot_collimator_names(ax8, idxs_primary_BLM_cons, "cons", dist_mode)
        ax8.set_yticks(np.arange(len(names_primary_BLM)), labels=names_primary_BLM)

        # Name of collimator with maximum primary losses when 1MJ exceeded
        names_primary_1MJ = np.unique(np.hstack((
            obs_mean["max_primary_coll_name_at_1MJ"], 
            obs_cons["max_primary_coll_name_at_1MJ"], 
        )))
        if "" in names_primary_1MJ:
            names_primary_1MJ = np.delete(names_primary_1MJ, np.where(names_primary_1MJ == "")[0][0])
        idxs_primary_1MJ_mean = np.full(len(cliq_cases), np.nan)
        idxs_primary_1MJ_cons = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean = np.where(names_primary_1MJ == obs_mean["max_primary_coll_name_at_1MJ"][i])[0]
            try:
                idxs_primary_1MJ_mean[i] = idx_mean[0]
            except IndexError:
                pass
            idx_cons = np.where(names_primary_1MJ == obs_cons["max_primary_coll_name_at_1MJ"][i])[0]
            try:
                idxs_primary_1MJ_cons[i] = idx_cons[0]
            except IndexError:
                pass
        plot_collimator_names(ax9, idxs_primary_1MJ_mean, "mean", dist_mode)
        plot_collimator_names(ax9, idxs_primary_1MJ_cons, "cons", dist_mode)
        ax9.set_yticks(np.arange(len(names_primary_1MJ)), labels=names_primary_1MJ)

        # Name of collimator with maximum total losses at time of PDSU dump
        names_total_PDSU = np.unique(np.hstack((
            obs_mean["max_total_coll_name_at_PDSU"], 
            obs_cons["max_total_coll_name_at_PDSU"], 
        )))
        if "" in names_total_PDSU:
            names_total_PDSU = np.delete(names_total_PDSU, np.where(names_total_PDSU == "")[0][0])
        idxs_total_PDSU_mean = np.full(len(cliq_cases), np.nan)
        idxs_total_PDSU_cons = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean = np.where(names_total_PDSU == obs_mean["max_total_coll_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_mean[i] = idx_mean[0]
            except IndexError:
                pass
            idx_cons = np.where(names_total_PDSU == obs_cons["max_total_coll_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU_cons[i] = idx_cons[0]
            except IndexError:
                pass
        plot_collimator_names(ax10, idxs_total_PDSU_mean, "mean", dist_mode)
        plot_collimator_names(ax10, idxs_total_PDSU_cons, "cons", dist_mode)
        ax10.set_yticks(np.arange(len(names_total_PDSU)), labels=names_total_PDSU)

        # Name of collimator with maximum total losses at time of BLM dump
        names_total_BLM = np.unique(np.hstack((
            obs_mean["max_total_coll_name_at_BLM"], 
            obs_cons["max_total_coll_name_at_BLM"], 
        )))
        if "" in names_total_BLM:
            names_total_BLM = np.delete(names_total_BLM, np.where(names_total_BLM == "")[0][0])
        idxs_total_BLM_mean = np.full(len(cliq_cases), np.nan)
        idxs_total_BLM_cons = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean = np.where(names_total_BLM == obs_mean["max_total_coll_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_mean[i] = idx_mean[0]
            except IndexError:
                pass
            idx_cons = np.where(names_total_BLM == obs_cons["max_total_coll_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM_cons[i] = idx_cons[0]
            except IndexError:
                pass
        plot_collimator_names(ax11, idxs_total_BLM_mean, "mean", dist_mode)
        plot_collimator_names(ax11, idxs_total_BLM_cons, "cons", dist_mode)
        ax11.set_yticks(np.arange(len(names_total_BLM)), labels=names_total_BLM)

        # Name of collimator with maximum total losses when 1MJ exceeded
        names_total_1MJ = np.unique(np.hstack((
            obs_mean["max_total_coll_name_at_1MJ"], 
            obs_cons["max_total_coll_name_at_1MJ"], 
        )))
        if "" in names_total_1MJ:
            names_total_1MJ = np.delete(names_total_1MJ, np.where(names_total_1MJ == "")[0][0])
        idxs_total_1MJ_mean = np.full(len(cliq_cases), np.nan)
        idxs_total_1MJ_cons = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx_mean = np.where(names_total_1MJ == obs_mean["max_total_coll_name_at_1MJ"][i])[0]
            try:
                idxs_total_1MJ_mean[i] = idx_mean[0]
            except IndexError:
                pass
            idx_cons = np.where(names_total_1MJ == obs_cons["max_total_coll_name_at_1MJ"][i])[0]
            try:
                idxs_total_1MJ_cons[i] = idx_cons[0]
            except IndexError:
                pass
        plot_collimator_names(ax12, idxs_total_1MJ_mean, "mean",dist_mode)
        plot_collimator_names(ax12, idxs_total_1MJ_cons, "cons",dist_mode)
        ax12.set_yticks(np.arange(len(names_total_1MJ)), labels=names_total_1MJ)
    else:
        with open(result_path / f"{halo_model}_halo_{dist_mode}/observables.pkl", "rb") as file:
            obs = pickle.load(file)

        # Time/turn of 1 MJ exceeded
        plot_1MJ_exceeded(
            ax1, ax1b, 
            obs["s_1MJ"], obs["s_PDSU"], obs["s_BLM"], 
            obs["t_1MJ"], obs["t_PDSU"], obs["t_BLM"], 
            halo_model, dist_mode
        )

        plot_1MJ_exceeded_wrt_dump(
            ax1_a, ax1b_a, obs["s_1MJ_wrt_PDSU"], obs["t_1MJ_wrt_PDSU"], 
            halo_model, dist_mode
        )

        plot_1MJ_exceeded_wrt_dump(
            ax1_b, ax1b_b, obs["s_1MJ_wrt_BLM"], obs["t_1MJ_wrt_BLM"], 
            halo_model, dist_mode
        )

        # Total losses at time of PDSU dump
        plot_total_losses(
            ax3, obs["total_MJ_at_PDSU"], halo_model, dist_mode
        )

        # Total losses at time of BLM dump
        plot_total_losses(
            ax3, obs["total_MJ_at_BLM"], halo_model, dist_mode
        )

        # Maximum losses (primary and total) on a single collimator at time of PDSU dump
        plot_max_collimator_losses(
            ax4, 
            obs["max_primary_coll_MJ_at_PDSU"], obs["max_total_coll_MJ_at_PDSU"], 
            halo_model, dist_mode
        )

        # Maximum losses (primary and total) on a single collimator at time of BLM dump
        plot_max_collimator_losses(
            ax5, 
            obs["max_primary_coll_MJ_at_BLM"], obs["max_total_coll_MJ_at_BLM"], 
            halo_model, dist_mode
        )

        # Maximum losses (primary and total) on a single collimator when 1 MJ exceeded
        plot_max_collimator_losses(
            ax6, 
            obs["max_primary_coll_MJ_at_1MJ"], obs["max_total_coll_MJ_at_1MJ"], 
            halo_model, dist_mode
        )

        # Name of collimator with maximum primary losses at time of PDSU dump
        names_primary_PDSU = np.unique(obs["max_primary_coll_name_at_PDSU"])
        if "" in names_primary_PDSU:
            names_primary_PDSU = np.delete(names_primary_PDSU, np.where(names_primary_PDSU == "")[0][0])
        idxs_primary_PDSU = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx = np.where(names_primary_PDSU == obs["max_primary_coll_name_at_PDSU"][i])[0]
            try:
                idxs_primary_PDSU[i] = idx[0]
            except IndexError:
                pass
        plot_collimator_names(ax7, idxs_primary_PDSU, halo_model, dist_mode)
        ax7.set_yticks(np.arange(len(names_primary_PDSU)), labels=names_primary_PDSU)

        # Name of collimator with maximum primary losses at time of BLM dump
        names_primary_BLM = np.unique(obs["max_primary_coll_name_at_BLM"])
        if "" in names_primary_BLM:
            names_primary_BLM = np.delete(names_primary_BLM, np.where(names_primary_BLM == "")[0][0])
        idxs_primary_BLM = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx = np.where(names_primary_BLM == obs["max_primary_coll_name_at_BLM"][i])[0]
            try:
                idxs_primary_BLM[i] = idx[0]
            except IndexError:
                pass
        plot_collimator_names(ax8, idxs_primary_BLM, halo_model, dist_mode)
        ax8.set_yticks(np.arange(len(names_primary_BLM)), labels=names_primary_BLM)

        # Name of collimator with maximum primary losses when 1MJ exceeded
        names_primary_1MJ = np.unique(obs["max_primary_coll_name_at_1MJ"])
        if "" in names_primary_1MJ:
            names_primary_1MJ = np.delete(names_primary_1MJ, np.where(names_primary_1MJ == "")[0][0])
        idxs_primary_1MJ = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx = np.where(names_primary_1MJ == obs["max_primary_coll_name_at_1MJ"][i])[0]
            try:
                idxs_primary_1MJ[i] = idx[0]
            except IndexError:
                pass
        plot_collimator_names(ax9, idxs_primary_1MJ, halo_model, dist_mode)
        ax9.set_yticks(np.arange(len(names_primary_1MJ)), labels=names_primary_1MJ)

        # Name of collimator with maximum total losses at time of PDSU dump
        names_total_PDSU = np.unique(obs["max_total_coll_name_at_PDSU"])
        if "" in names_total_PDSU:
            names_total_PDSU = np.delete(names_total_PDSU, np.where(names_total_PDSU == "")[0][0])
        idxs_total_PDSU = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx = np.where(names_total_PDSU == obs["max_total_coll_name_at_PDSU"][i])[0]
            try:
                idxs_total_PDSU[i] = idx[0]
            except IndexError:
                pass
        plot_collimator_names(ax10, idxs_total_PDSU, halo_model, dist_mode)
        ax10.set_yticks(np.arange(len(names_total_PDSU)), labels=names_total_PDSU)

        # Name of collimator with maximum total losses at time of BLM dump
        names_total_BLM = np.unique(obs["max_total_coll_name_at_BLM"])
        if "" in names_total_BLM:
            names_total_BLM = np.delete(names_total_BLM, np.where(names_total_BLM == "")[0][0])
        idxs_total_BLM = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx = np.where(names_total_BLM == obs["max_total_coll_name_at_BLM"][i])[0]
            try:
                idxs_total_BLM[i] = idx[0]
            except IndexError:
                pass
        plot_collimator_names(ax11, idxs_total_BLM, halo_model, dist_mode)
        ax11.set_yticks(np.arange(len(names_total_BLM)), labels=names_total_BLM)

        # Name of collimator with maximum total losses when 1MJ exceeded
        names_total_1MJ = np.unique(obs["max_total_coll_name_at_1MJ"])
        if "" in names_total_1MJ:
            names_total_1MJ = np.delete(names_total_1MJ, np.where(names_total_1MJ == "")[0][0])
        idxs_total_1MJ = np.full(len(cliq_cases), np.nan)
        for i in range(len(cliq_cases)):
            idx = np.where(names_total_1MJ == obs["max_total_coll_name_at_1MJ"][i])[0]
            try:
                idxs_total_1MJ[i] = idx[0]
            except IndexError:
                pass
        plot_collimator_names(ax12, idxs_total_1MJ, halo_model,dist_mode)
        ax12.set_yticks(np.arange(len(names_total_1MJ)), labels=names_total_1MJ)

    
    '''
    Set legends etc.
    '''
    # Time/turn of 1 MJ exceeded
    ax1.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax1.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax1.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax1.set_ylabel(r"1 MJ exceeded [$\mu$s]", fontsize=fs)
    ax1.tick_params(axis="both", labelsize=fs-2)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax1b.set_ylabel(r"1 MJ exceeded [turn]", fontsize=fs)
    ax1b.tick_params(axis="both", labelsize=fs-2)
    ax1_limits = tuple(np.asarray(ax1.get_ylim()) * 1e-6 * f_rev)
    ax1b.set_ylim(ax1_limits)
    ax1b.grid(True, which='major', axis='x', color='gray', linestyle='-')

    ax1_a.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax1_a.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax1_a.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax1_a.set_ylabel(r"1 MJ exceeded [$\mu$s]" + "\nw.r.t. PDSU dump", fontsize=fs)
    ax1_a.tick_params(axis="both", labelsize=fs-2)
    ax1_a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax1b_a.set_ylabel("1 MJ exceeded [turn]\nw.r.t. PDSU dump", fontsize=fs)
    ax1b_a.tick_params(axis="both", labelsize=fs-2)
    ax1_a_limits = tuple(np.asarray(ax1_a.get_ylim()) * 1e-6 * f_rev)
    ax1b_a.set_ylim(ax1_a_limits)
    ax1b_a.grid(True, which='major', axis='x', color='gray', linestyle='-')

    ax1_b.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax1_b.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax1_b.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax1_b.set_ylabel(r"1 MJ exceeded [$\mu$s]" + "\nw.r.t. BLM dump", fontsize=fs)
    ax1_b.tick_params(axis="both", labelsize=fs-2)
    ax1_b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax1b_b.set_ylabel("1 MJ exceeded [turn]\nw.r.t. BLM dump", fontsize=fs)
    ax1b_b.tick_params(axis="both", labelsize=fs-2)
    ax1_b_limits = tuple(np.asarray(ax1_b.get_ylim()) * 1e-6 * f_rev)
    ax1b_b.set_ylim(ax1_b_limits)
    ax1b_b.grid(True, which='major', axis='x', color='gray', linestyle='-')

    # Total losses at time of PDSU dump
    ax2.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax2.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax2.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax2.set_ylabel("total beam loss [MJ]\nat time of PDSU dump", fontsize=fs)
    ax2.tick_params(axis="both", labelsize=fs-2)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    # Total losses at time of BLM dump
    ax3.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax3.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax3.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax3.set_ylabel("total beam loss [MJ]\nat time of BLM dump", fontsize=fs)
    ax3.tick_params(axis="both", labelsize=fs-2)
    ax3.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    # Maximum losses (primary and total) on a single collimator at time of PDSU dump
    ax4.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax4.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax4.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax4.set_ylabel("max beam loss [MJ]\non single collimator\nat time of PDSU dump", fontsize=fs)
    ax4.tick_params(axis="both", labelsize=fs-2)
    ax4.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    # Maximum losses (primary and total) on a single collimator at time of BLM dump
    ax5.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax5.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax5.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax5.set_ylabel("max beam loss [MJ]\non single collimator\nat time of BLM dump", fontsize=fs)
    ax5.tick_params(axis="both", labelsize=fs-2)
    ax5.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    # Maximum losses (primary and total) on a single collimator when 1 MJ exceeded
    ax6.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax6.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax6.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax6.set_ylabel("max beam loss [MJ]\non single collimator\nwhen 1 MJ total loss is exceeded", fontsize=fs)
    ax6.tick_params(axis="both", labelsize=fs-2)
    ax6.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    # Name of collimator with maximum primary losses at time of PDSU dump
    ax7.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax7.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax7.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax7.set_ylabel("collimator with max primary loss\nat time of PDSU dump", fontsize=fs)
    ax7.tick_params(axis="both", labelsize=fs-2)
    ax7.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    # Name of collimator with maximum primary losses at time of BLM dump
    ax8.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax8.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax8.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax8.set_ylabel("collimator with max primary loss\nat time of BLM dump", fontsize=fs)
    ax8.tick_params(axis="both", labelsize=fs-2)
    ax8.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    # Name of collimator with maximum primary losses when 1 MJ exceeded
    ax9.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax9.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax9.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax9.set_ylabel("collimator with max primary loss\nwhen 1 MJ total loss is exceeded", fontsize=fs)
    ax9.tick_params(axis="both", labelsize=fs-2)
    ax9.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    # Name of collimator with maximum total losses at time of PDSU dump
    ax10.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax10.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax10.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax10.set_ylabel("collimator with max total loss\nat time of PDSU dump", fontsize=fs)
    ax10.tick_params(axis="both", labelsize=fs-2)
    ax10.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    # Name of collimator with maximum total losses at time of BLM dump
    ax11.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax11.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax11.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax11.set_ylabel("collimator with max total loss\nat time of BLM dump", fontsize=fs)
    ax11.tick_params(axis="both", labelsize=fs-2)
    ax11.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    # Name of collimator with maximum total losses when 1 MJ exceeded
    ax12.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax12.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax12.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax12.set_ylabel("collimator with max total loss\nwhen 1 MJ total loss is exceeded", fontsize=fs)
    ax12.tick_params(axis="both", labelsize=fs-2)
    ax12.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    # Maximum total losses on a single collimator by type at time of PDSU dump
    ax13a.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax13a.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax13a.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax13a.set_ylabel("max beam loss [MJ]\non single TCP\nat time of PDSU dump", fontsize=fs)
    ax13a.tick_params(axis="both", labelsize=fs-2)
    ax13a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax13b.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax13b.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax13b.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax13b.set_ylabel("max beam loss [MJ]\non single TCSG/TCSPM\nat time of PDSU dump", fontsize=fs)
    ax13b.tick_params(axis="both", labelsize=fs-2)
    ax13b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax13c.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax13c.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax13c.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax13c.set_ylabel("max beam loss [MJ]\non single TCL\nat time of PDSU dump", fontsize=fs)
    ax13c.tick_params(axis="both", labelsize=fs-2)
    ax13c.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax13d.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax13d.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax13d.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax13d.set_ylabel("max beam loss [MJ]\non single TCT\nat time of PDSU dump", fontsize=fs)
    ax13d.tick_params(axis="both", labelsize=fs-2)
    ax13d.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    # Maximum total losses on a single collimator by type at time of BLM dump
    ax14a.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax14a.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax14a.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax14a.set_ylabel("max beam loss [MJ]\non single TCP\nat time of BLM dump", fontsize=fs)
    ax14a.tick_params(axis="both", labelsize=fs-2)
    ax14a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax14b.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax14b.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax14b.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax14b.set_ylabel("max beam loss [MJ]\non single TCSG/TCSPM\nat time of BLM dump", fontsize=fs)
    ax14b.tick_params(axis="both", labelsize=fs-2)
    ax14b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax14c.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax14c.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax14c.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax14c.set_ylabel("max beam loss [MJ]\non single TCL\nat time of BLM dump", fontsize=fs)
    ax14c.tick_params(axis="both", labelsize=fs-2)
    ax14c.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax14d.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax14d.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax14d.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax14d.set_ylabel("max beam loss [MJ]\non single TCT\nat time of BLM dump", fontsize=fs)
    ax14d.tick_params(axis="both", labelsize=fs-2)
    ax14d.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    # Name of collimator with maximum total losses by type at time of PDSU dump
    ax15a.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax15a.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax15a.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax15a.set_ylabel("TCP with max total loss\nat time of PDSU dump", fontsize=fs)
    ax15a.tick_params(axis="both", labelsize=fs-2)
    ax15a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax15b.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax15b.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax15b.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax15b.set_ylabel("TCSG/TCSPM with max total loss\nat time of PDSU dump", fontsize=fs)
    ax15b.tick_params(axis="both", labelsize=fs-2)
    ax15b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax15c.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax15c.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax15c.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax15c.set_ylabel("TCL with max total loss\nat time of PDSU dump", fontsize=fs)
    ax15c.tick_params(axis="both", labelsize=fs-2)
    ax15c.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax15d.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax15d.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax15d.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax15d.set_ylabel("TCT with max total loss\nat time of PDSU dump", fontsize=fs)
    ax15d.tick_params(axis="both", labelsize=fs-2)
    ax15d.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax16a.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax16a.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax16a.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax16a.set_ylabel("TCP with max total loss\nat time of BLM dump", fontsize=fs)
    ax16a.tick_params(axis="both", labelsize=fs-2)
    ax16a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax16b.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax16b.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax16b.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax16b.set_ylabel("TCSG/TCSPM with max total loss\nat time of BLM dump", fontsize=fs)
    ax16b.tick_params(axis="both", labelsize=fs-2)
    ax16b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax16c.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax16c.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax16c.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax16c.set_ylabel("TCL with max total loss\nat time of BLM dump", fontsize=fs)
    ax16c.tick_params(axis="both", labelsize=fs-2)
    ax16c.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    ax16d.set_xlim(-0.5, len(cliq_cases)-0.5)
    ax16d.set_xticks(x_ticks, labels=cliq_cases, rotation=45, ha="center")
    ax16d.set_xlabel("magnet CLIQ fired in", fontsize=fs)
    ax16d.set_ylabel("TCT with max total loss\nat time of BLM dump", fontsize=fs)
    ax16d.tick_params(axis="both", labelsize=fs-2)
    ax16d.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fs-2)

    if save:
        save_name_tag = f"{halo_model}_halo_{dist_mode}_dist"
        
        fig1.savefig(result_path / f"time_1MJ_exceeded_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig1_a.savefig(result_path / f"time_1MJ_exceeded_wrt_PDSU_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig1_b.savefig(result_path / f"time_1MJ_exceeded_wrt_BLM_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')

        fig2.savefig(result_path / f"total_loss_at_PDSU_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig3.savefig(result_path / f"total_loss_at_BLM_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')

        fig4.savefig(result_path / f"max_collimator_loss_at_PDSU_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig5.savefig(result_path / f"max_collimator_loss_at_BLM_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig6.savefig(result_path / f"max_collimator_loss_at_1MJ_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')

        fig7.savefig(result_path / f"max_collimator_primary_loss_name_at_PDSU_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig8.savefig(result_path / f"max_collimator_primary_loss_name_at_BLM_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig9.savefig(result_path / f"max_collimator_primary_loss_name_at_1MJ_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')

        fig10.savefig(result_path / f"max_collimator_total_loss_name_at_PDSU_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig11.savefig(result_path / f"max_collimator_total_loss_name_at_BLM_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig12.savefig(result_path / f"max_collimator_total_loss_name_at_1MJ_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')

        fig13a.savefig(result_path / f"max_TCP_total_loss_at_PDSU_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig13b.savefig(result_path / f"max_TCSG_TCSPM_total_loss_at_PDSU_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig13c.savefig(result_path / f"max_TCL_total_loss_at_PDSU_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig13d.savefig(result_path / f"max_TCT_total_loss_at_PDSU_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig14a.savefig(result_path / f"max_TCP_total_loss_at_BLM_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig14b.savefig(result_path / f"max_TCSG_TCSPM_total_loss_at_BLM_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig14c.savefig(result_path / f"max_TCL_total_loss_at_BLM_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig14d.savefig(result_path / f"max_TCT_total_loss_at_BLM_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')

        fig15a.savefig(result_path / f"max_TCP_total_loss_name_at_PDSU_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig15b.savefig(result_path / f"max_TCSG_TCSPM_total_loss_name_at_PDSU_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig15c.savefig(result_path / f"max_TCL_total_loss_name_at_PDSU_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig15d.savefig(result_path / f"max_TCT_total_loss_name_at_PDSU_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig16a.savefig(result_path / f"max_TCP_total_loss_name_at_BLM_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig16b.savefig(result_path / f"max_TCSG_TCSPM_total_loss_name_at_BLM_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig16c.savefig(result_path / f"max_TCL_total_loss_name_at_BLM_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')
        fig16d.savefig(result_path / f"max_TCT_total_loss_name_at_BLM_{save_name_tag}.png", dpi=150, format='png', bbox_inches='tight')

    if show:
        plt.show()


if __name__ == "__main__":
    main()