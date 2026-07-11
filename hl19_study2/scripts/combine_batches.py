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
    help="which magnet to assign field errors to ('all', 'Q1R5a', 'Q1L5a', 'Q2R5a', " \
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


turn_shift = 0.11245
id_shift = int(1e5)

cliq_cases = [
    'Q1R5a', 'Q1L5a', 'Q2R5a', 'Q2L5a', 'Q3R5a', 'Q3L5a', 
    'Q1R1a', 'Q1L1a', 'Q2R1a', 'Q2L1a', 'Q3R1a', 'Q3L1a', 
    'Q1R5b', 'Q1L5b', 'Q2R5b', 'Q2L5b', 'Q3R5b', 'Q3L5b', 
    'Q1R1b', 'Q1L1b', 'Q2R1b', 'Q2L1b', 'Q3R1b', 'Q3L1b'
]


def combine_single_case(
    study_path, 
    result_path, 
    start_seed, 
    end_seed, 
    cliq_case, 
    halo_model, 
    dist_mode, 
    i_mo
):
    for seed in range(start_seed, end_seed+1):
        print(f"###### Combining batches for seed {seed} ######")
        if len(list(study_path.glob(f"job_0/part_fin*_seed{seed}_{cliq_case}_{halo_model}_{dist_mode}_imo_{int(i_mo)}.pkl"))) == 0:
            print("No files found, moving on to next seed!")
            continue
        if len(list(study_path.glob(f"job_0/part_fin*_seed{seed}_{cliq_case}_{halo_model}_{dist_mode}_imo_{int(i_mo)}.pkl"))) != 10:
            print("Did not find expected number of files, moving on to next seed!")
            continue

        fin_files = np.array(sorted(list(study_path.glob(f"job_0/part_fin*_seed{seed}_{cliq_case}_{halo_model}_{dist_mode}_imo_{int(i_mo)}.pkl"))))
        id_shift_nos = [int(file.name.split("batch")[1][0]) for file in fin_files]
        
        print(f"Found {len(fin_files)} files!")
        
        fin_file = fin_files[0]

        all_fin_part = []
        for i in range(0, len(fin_files)):
            fin_file = fin_files[i]
            with open(fin_file, "rb") as f:
                fin_data = pickle.load(f)
            fin_part = xt.Particles.from_dict(fin_data)
            all_fin_part.append(fin_part)

        combined_fin_part = xt.Particles.merge(all_fin_part)
        combined_fin_data = combined_fin_part.to_dict()
        combined_fin_data["at_turn"] = combined_fin_data["at_turn"].astype(float)

        for shift_no in id_shift_nos:
            mask = (combined_fin_data["particle_id"] >= (shift_no * id_shift)) & (combined_fin_data["particle_id"] < ((shift_no+1) * id_shift))
            combined_fin_data["at_turn"][mask] = combined_fin_data["at_turn"][mask] + shift_no * turn_shift

        with open(result_path / f"part_fin_seed{seed}_{cliq_case}_{halo_model}_{dist_mode}_imo_{int(i_mo)}.pkl", "wb") as f:
            pickle.dump(combined_fin_data, f)


def main():
    '''
    Parse arguments
    '''
    args = parser.parse_args()
    study_path = args.study_path
    result_path = args.result_path
    study_name = args.study_name
    start_seed = args.start_seed
    end_seed = args.end_seed
    cliq_case = args.cliq_case
    halo_model = args.halo_model
    dist_mode = args.dist_mode
    i_mo = args.i_mo


    '''
    Create results folder is not already existing
    '''
    study_path = Path(study_path).resolve() / study_name
    result_path = Path(result_path).resolve() / study_name
    result_path.mkdir(parents=True, exist_ok=True)


    '''
    Combine batches
    '''
    if cliq_case == "all":
        for curr_cliq_case in cliq_cases:
            print(f"~~~~~~~~~~~~ Combining batches for CLIQ case {curr_cliq_case} ~~~~~~~~~~~~")
            combine_single_case(
                study_path, 
                result_path, 
                start_seed, 
                end_seed, 
                curr_cliq_case, 
                halo_model, 
                dist_mode, 
                i_mo
            )
    else:
        print(f"~~~~~~~~~~~~ Combining batches for CLIQ case {cliq_case} ~~~~~~~~~~~~")
        combine_single_case(
            study_path, 
            result_path, 
            start_seed, 
            end_seed, 
            cliq_case, 
            halo_model, 
            dist_mode, 
            i_mo
        )


if __name__ == "__main__":
    main()