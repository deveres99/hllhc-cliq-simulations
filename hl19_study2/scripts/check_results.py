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


def check_single_case(
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
        num_files = len(list(study_path.glob(f"job_0/part_fin*_seed{seed}_{cliq_case}_{halo_model}_{dist_mode}_imo_{int(i_mo)}.pkl")))
        if num_files == 0:
            print("No files found!")
        elif num_files != 10:
            print(f"Only {num_files}/10 files found!")
        else:
            continue


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
            check_single_case(
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
        check_single_case(
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