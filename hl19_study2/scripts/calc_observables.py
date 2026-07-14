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
    "cliq_cases", 
    type=str, 
    help="either 'all' or list of cases separated by a comma as one string"
)
parser.add_argument(
    "line_path", 
    type=str, 
    help="xsuite line path"
)


f_rev = 11245
n_bunch = 2730
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
    requested_cliq_cases = args.cliq_cases
    line_path = args.line_path

    if requested_cliq_cases != "all":
        cliq_cases = requested_cliq_cases.split(',')

    result_path = Path(result_path).resolve() / study_name

    all_files_ = sorted(list(result_path.glob(f"part_fin*.pkl")))
    idxs = []
    all_files = []
    for filename in all_files_:
        for i, case in enumerate(cliq_cases):
            if case in str(filename):
                idxs.append(i)
                all_files.append(filename)
                break


    '''
    Final result dict
    '''
    final_results = {
        "magnets": np.asarray(cliq_cases, dtype=str), 
        "s_PDSU": np.full(len(cliq_cases), np.nan, dtype=float), 
        "t_PDSU": np.full(len(cliq_cases), np.nan, dtype=float), 
        "s_BLM": np.full(len(cliq_cases), np.nan, dtype=float), 
        "t_BLM": np.full(len(cliq_cases), np.nan, dtype=float), 
        "s_1MJ": np.full(len(cliq_cases), np.nan, dtype=float), 
        "t_1MJ": np.full(len(cliq_cases), np.nan, dtype=float), 
        "s_1MJ_wrt_PDSU": np.full(len(cliq_cases), np.nan, dtype=float), 
        "t_1MJ_wrt_PDSU": np.full(len(cliq_cases), np.nan, dtype=float), 
        "s_1MJ_wrt_BLM": np.full(len(cliq_cases), np.nan, dtype=float), 
        "t_1MJ_wrt_BLM": np.full(len(cliq_cases), np.nan, dtype=float), 
        "total_MJ_at_PDSU": np.full(len(cliq_cases), np.nan, dtype=float), 
        "total_MJ_at_BLM": np.full(len(cliq_cases), np.nan, dtype=float), 
        "max_total_W_before_PDSU": np.full(len(cliq_cases), np.nan, dtype=float), 
        "max_total_W_before_BLM": np.full(len(cliq_cases), np.nan, dtype=float), 
        "max_total_W": np.full(len(cliq_cases), np.nan, dtype=float), 
        "t_max_total_W": np.full(len(cliq_cases), np.nan, dtype=float), 
        "s_max_total_W": np.full(len(cliq_cases), np.nan, dtype=float), 
        "max_primary_coll_MJ_at_PDSU": np.full(len(cliq_cases), np.nan, dtype=float), 
        "max_total_coll_MJ_at_PDSU": np.full(len(cliq_cases), np.nan, dtype=float), 
        "max_primary_coll_name_at_PDSU": np.full(len(cliq_cases), "", dtype="U40"), 
        "max_total_coll_name_at_PDSU": np.full(len(cliq_cases), "", dtype="U40"), 
        "max_primary_coll_MJ_at_BLM": np.full(len(cliq_cases), np.nan, dtype=float), 
        "max_total_coll_MJ_at_BLM": np.full(len(cliq_cases), np.nan, dtype=float), 
        "max_primary_coll_name_at_BLM": np.full(len(cliq_cases), "", dtype="U40"), 
        "max_total_coll_name_at_BLM": np.full(len(cliq_cases), "", dtype="U40"), 
        "max_primary_coll_MJ_at_1MJ": np.full(len(cliq_cases), np.nan, dtype=float), 
        "max_total_coll_MJ_at_1MJ": np.full(len(cliq_cases), np.nan, dtype=float), 
        "max_primary_coll_name_at_1MJ": np.full(len(cliq_cases), "", dtype="U40"), 
        "max_total_coll_name_at_1MJ": np.full(len(cliq_cases), "", dtype="U40"), 
    }

    
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


    for idx, part_fin_file in zip(idxs, all_files):
        print(f"~~~~~~~~~~~~ Processing {cliq_cases[idx]} ~~~~~~~~~~~~")

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
        inst_tot_power = lost_energy / ((turns[1] - turns[0]) / f_rev)


        # Time of PDSU dump
        final_results["t_PDSU"] = t_PDSU_interlock + delta_t_to_dump
        final_results["s_PDSU"] = s_PDSU_interlock + delta_s_to_dump

        # Time/turn 1MJ is reached
        try:
            t_1MJ = turns[cum_lost_energy * 1e-6 >= 1][0]
        except IndexError:
            t_1MJ = None
            print("1 MJ not exceeded!")
        final_results["t_1MJ"][idx] = t_1MJ
        if t_1MJ is not None:
            s_1MJ = t_1MJ / f_rev
            print(f"1 MJ exceeded at turn {t_1MJ} / {s_1MJ * 1e6} us!")
            final_results["s_1MJ"][idx] = s_1MJ

        # Time/turn 1MJ is reached w.r.t. PDSU and BLM interlock
        if t_1MJ is not None:
            t_1MJ_wrt_PDSU = t_1MJ - (t_PDSU_interlock + delta_t_to_dump)
            final_results["t_1MJ_wrt_PDSU"][idx] = t_1MJ_wrt_PDSU
            s_1MJ_wrt_PDSU = s_1MJ - (s_PDSU_interlock + delta_s_to_dump)
            final_results["s_1MJ_wrt_PDSU"][idx] = s_1MJ_wrt_PDSU

        try:
            t_BLM_interlock = turns[cum_lost_energy * 1e-3 >= 125][0]
            s_BLM_interlock = t_BLM_interlock / f_rev
            final_results["t_BLM"][idx] = t_BLM_interlock + delta_t_to_dump
            final_results["s_BLM"][idx] = s_BLM_interlock + delta_s_to_dump
        except IndexError:
            t_BLM_interlock = None
        if t_1MJ is not None:
            t_1MJ_wrt_BLM = t_1MJ - (t_BLM_interlock + delta_t_to_dump)
            final_results["t_1MJ_wrt_BLM"][idx] = t_1MJ_wrt_BLM
            s_1MJ_wrt_BLM = s_1MJ - (s_BLM_interlock + delta_s_to_dump)
            final_results["s_1MJ_wrt_BLM"][idx] = s_1MJ_wrt_BLM

        # Total losses in MJ at PDSU and BLM dumps
        total_MJ_at_PDSU = cum_lost_energy[turns <= (t_PDSU_interlock + delta_t_to_dump)][-1] * 1e-6
        final_results["total_MJ_at_PDSU"][idx] = total_MJ_at_PDSU
        if t_BLM_interlock is not None:
            total_MJ_at_BLM = cum_lost_energy[turns <= (t_BLM_interlock + delta_t_to_dump)][-1] * 1e-6
            final_results["total_MJ_at_BLM"][idx] = total_MJ_at_BLM

        # Power load
        max_total_W_before_PDSU = np.max(inst_tot_power[turns <= (t_PDSU_interlock + delta_t_to_dump)])
        final_results["max_total_W_before_PDSU"][idx] = max_total_W_before_PDSU
        if t_BLM_interlock is not None:
            max_total_W_before_BLM = np.max(inst_tot_power[turns <= (t_BLM_interlock + delta_t_to_dump)])
            final_results["max_total_W_before_BLM"][idx] = max_total_W_before_BLM
        max_total_W = np.max(inst_tot_power)
        final_results["max_total_W"][idx] = max_total_W
        t_max_total_W = turns[np.argmax(inst_tot_power)]
        final_results["t_max_total_W"][idx] = t_max_total_W
        s_max_total_W = t_max_total_W / f_rev
        final_results["s_max_total_W"][idx] = s_max_total_W

        # Collimator losses
        mask_PDSU = part_fin["at_turn"] <= (t_PDSU_interlock + delta_t_to_dump)
        part_fin_PDSU = {
            key: value[mask_PDSU] if isinstance(value, np.ndarray) else value for key, value in part_fin.items()
        }
        p_fin_PDSU = xt.Particles.from_dict(part_fin_PDSU)
        lm_PDSU = xc.LossMap(line, line_is_reversed=line_is_reversed, part=p_fin_PDSU)
        lm_PDSU.update_metadata(metadata)
        lm_PDSU = lm_PDSU.lossmap

        try:
            max_primary_coll_MJ_at_PDSU = np.max(lm_PDSU["collimator"]["e_prim"]) / num_part * bunch_intensity * n_bunch * charge
            idx_max_primary_coll_MJ_at_PDSU = np.argmax(lm_PDSU["collimator"]["e_prim"])
            final_results["max_primary_coll_MJ_at_PDSU"][idx] = max_primary_coll_MJ_at_PDSU * 1e-6
            max_primary_coll_name_at_PDSU = lm_PDSU["collimator"]["name"][idx_max_primary_coll_MJ_at_PDSU]
            final_results["max_primary_coll_name_at_PDSU"][idx] = max_primary_coll_name_at_PDSU
        except KeyError:
            pass
        try:
            max_total_coll_MJ_at_PDSU = np.max(lm_PDSU["collimator"]["e"]) / num_part * bunch_intensity * n_bunch * charge
            idx_max_total_coll_MJ_at_PDSU = np.argmax(lm_PDSU["collimator"]["e"])
            final_results["max_total_coll_MJ_at_PDSU"][idx] = max_total_coll_MJ_at_PDSU * 1e-6
            max_total_coll_name_at_PDSU = lm_PDSU["collimator"]["name"][idx_max_total_coll_MJ_at_PDSU]
            final_results["max_total_coll_name_at_PDSU"][idx] = max_total_coll_name_at_PDSU
        except KeyError:
            pass

        
        if t_BLM_interlock is not None:
            mask_BLM = part_fin["at_turn"] <= (t_BLM_interlock + delta_t_to_dump)
            part_fin_BLM = {
                key: value[mask_BLM] if isinstance(value, np.ndarray) else value for key, value in part_fin.items()
            }
            p_fin_BLM = xt.Particles.from_dict(part_fin_BLM)
            lm_BLM = xc.LossMap(line, line_is_reversed=line_is_reversed, part=p_fin_BLM)
            lm_BLM.update_metadata(metadata)
            lm_BLM = lm_BLM.lossmap

            try:
                max_primary_coll_MJ_at_BLM = np.max(lm_BLM["collimator"]["e_prim"]) / num_part * bunch_intensity * n_bunch * charge
                idx_max_primary_coll_MJ_at_BLM = np.argmax(lm_BLM["collimator"]["e_prim"])
                final_results["max_primary_coll_MJ_at_BLM"][idx] = max_primary_coll_MJ_at_BLM * 1e-6
                max_primary_coll_name_at_BLM = lm_BLM["collimator"]["name"][idx_max_primary_coll_MJ_at_BLM]
                final_results["max_primary_coll_name_at_BLM"][idx] = max_primary_coll_name_at_BLM
            except KeyError:
                pass
            try:
                max_total_coll_MJ_at_BLM = np.max(lm_BLM["collimator"]["e"]) / num_part * bunch_intensity * n_bunch * charge
                idx_max_total_coll_MJ_at_BLM = np.argmax(lm_BLM["collimator"]["e"])
                final_results["max_total_coll_MJ_at_BLM"][idx] = max_total_coll_MJ_at_BLM * 1e-6
                max_total_coll_name_at_BLM = lm_BLM["collimator"]["name"][idx_max_total_coll_MJ_at_BLM]
                final_results["max_total_coll_name_at_BLM"][idx] = max_total_coll_name_at_BLM
            except KeyError:
                pass


        if t_1MJ is not None:
            mask_1MJ = part_fin["at_turn"] <= t_1MJ
            part_fin_1MJ = {
                key: value[mask_1MJ] if isinstance(value, np.ndarray) else value for key, value in part_fin.items()
            }
            p_fin_1MJ = xt.Particles.from_dict(part_fin_1MJ)
            lm_1MJ = xc.LossMap(line, line_is_reversed=line_is_reversed, part=p_fin_1MJ)
            lm_1MJ.update_metadata(metadata)
            lm_1MJ = lm_1MJ.lossmap

            try:
                max_primary_coll_MJ_at_1MJ = np.max(lm_1MJ["collimator"]["e_prim"]) / num_part * bunch_intensity * n_bunch * charge
                idx_max_primary_coll_MJ_at_1MJ = np.argmax(lm_1MJ["collimator"]["e_prim"])
                final_results["max_primary_coll_MJ_at_1MJ"][idx] = max_primary_coll_MJ_at_1MJ * 1e-6
                max_primary_coll_name_at_1MJ = lm_1MJ["collimator"]["name"][idx_max_primary_coll_MJ_at_1MJ]
                final_results["max_primary_coll_name_at_1MJ"][idx] = max_primary_coll_name_at_1MJ
            except KeyError:
                pass
            try:
                max_total_coll_MJ_at_1MJ = np.max(lm_1MJ["collimator"]["e"]) / num_part * bunch_intensity * n_bunch * charge
                idx_max_total_coll_MJ_at_1MJ = np.argmax(lm_1MJ["collimator"]["e"])
                final_results["max_total_coll_MJ_at_1MJ"][idx] = max_total_coll_MJ_at_1MJ * 1e-6
                max_total_coll_name_at_1MJ = lm_1MJ["collimator"]["name"][idx_max_total_coll_MJ_at_1MJ]
                final_results["max_total_coll_name_at_1MJ"][idx] = max_total_coll_name_at_1MJ
            except KeyError:
                pass

    save_name = "observables.pkl" if requested_cliq_cases == "all" else f"observables_{requested_cliq_cases}.pkl"
    with open(result_path / save_name, "wb") as file:
        pickle.dump(final_results, file)


if __name__ == "__main__":
    main()