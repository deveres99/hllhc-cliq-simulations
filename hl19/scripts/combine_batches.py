import argparse
import pickle
import numpy as np
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
    "study_path", 
    type=str, 
    help="path to the study"
)
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
    "start_seed", 
    type=int, 
    help="error seed to start with"
)
parser.add_argument(
    "end_seed", 
    type=int, 
    help="error seed to end with (not included)"
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
parser.add_argument(
    "id_shift", 
    type=int, 
    help="integer shift of particle ids to make unique"
)


turn_shift = 0.11245

keys_of_interest = [
    'rvv', 
    'mass0', 
    'particle_id', 
    'beta0', 
    'ax', 
    'px', 
    'charge_ratio', 
    'rpp', 
    'x', 
    'spin_y', 
    'y', 
    'at_turn', 
    'delta', 
    't_sim', 
    'chi', 
    'at_element', 
    'q0', 
    'py', 
    'weight', 
    'pdg_id', 
    'gamma0', 
    'zeta', 
    'parent_particle_id', 
    'ay', 
    'spin_z', 
    's', 
    'p0c', 
    'spin_x', 
    'ptau', 
    'anomalous_magnetic_moment', 
    'state', 
]


def main():
    '''
    Parse arguments
    '''
    args = parser.parse_args()
    study_path = args.study_path
    result_path = args.result_path
    study_name = args.study_name
    id_shift = args.id_shift
    start_seed = args.start_seed
    end_seed = args.end_seed
    cliq_case = args.cliq_case
    dist_mode = args.dist_mode


    '''
    Create results folder is not already existing
    '''
    study_path = Path(study_path).resolve() / study_name
    result_path = Path(result_path).resolve() / study_name
    result_path.mkdir(parents=True, exist_ok=True)


    '''
    Combine batches
    '''
    for seed in range(start_seed, end_seed):
        print(f"###### Combining batches for seed {seed} ######")
        if len(list(study_path.glob(f"job_0/lossmap_batch*_seed{seed}_{cliq_case}_{dist_mode}.json"))) == 0:
            print("No files found, moving on to next seed!")
            continue
        
        lm = xc.LossMap.from_json(study_path.glob(f"job_0/lossmap_batch*_seed{seed}_{cliq_case}_{dist_mode}.json"))
        lm.to_json(result_path / f"lossmap_seed{seed}_{cliq_case}_{dist_mode}.json")

        init_files = np.array(list(study_path.glob(f"job_0/part_init_batch*_seed{seed}_{cliq_case}_{dist_mode}.pkl")))
        fin_files = np.array(list(study_path.glob(f"job_0/part_fin_batch*_seed{seed}_{cliq_case}_{dist_mode}.pkl")))
        id_shift_nos = [int(file.name.split("batch")[1][0]) for file in fin_files]
        if len(init_files) != len(fin_files):
            raise ValueError(f"Found {len(init_files)} initial particle files, but {len(fin_files)} final particles files!")
        elif len(fin_files) == 0:
            print(f"No files found!")
        else:
            print(f"Found {len(fin_files)} files!")
            combined_init_data = {key: [] for key in keys_of_interest}
            combined_fin_data = {key: [] for key in keys_of_interest}
            for i in range(len(fin_files)):
                init_file = init_files[i]
                fin_file = fin_files[i]
                id_shift_no = id_shift_nos[i]

                with open(init_file, "rb") as f:
                    init_data = pickle.load(f)
                with open(fin_file, "rb") as f:
                    fin_data = pickle.load(f)

                for key in keys_of_interest:
                    init_arr = np.asarray(init_data[key])
                    fin_arr = np.asarray(fin_data[key])
                    if key == "particle_id":
                        init_arr = init_arr + id_shift_no * id_shift
                        fin_arr = fin_arr + id_shift_no * id_shift
                    if key == "at_turn":
                        init_arr = init_arr + id_shift_no * turn_shift
                        fin_arr = fin_arr + id_shift_no * turn_shift
                    combined_init_data[key].append(init_arr)
                    combined_fin_data[key].append(fin_arr)

            for key in keys_of_interest:
                try:
                    combined_init_data[key] = np.concatenate(combined_init_data[key])
                    combined_fin_data[key] = np.concatenate(combined_fin_data[key])
                except ValueError:
                    combined_init_data[key] = combined_init_data[key][0]
                    combined_fin_data[key] = combined_fin_data[key][0]

            with open(result_path / f"part_init_seed{seed}_{cliq_case}_{dist_mode}.pkl", "wb") as f:
                pickle.dump(combined_init_data, f)
            with open(result_path / f"part_fin_seed{seed}_{cliq_case}_{dist_mode}.pkl", "wb") as f:
                pickle.dump(combined_fin_data, f)


if __name__ == "__main__":
    main()