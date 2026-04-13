import os
import yaml
import argparse

import xobjects as xo
import xdeps as xd
import xtrack as xt
import xpart as xp
import xfields as xf
import xcoll as xc
import xmask as xm
import xmask.lhc as xmlhc
import xsuite as xs


parser = argparse.ArgumentParser()
parser.add_argument(
    "beam", 
    type=int, 
    help="1 or 2"
)
parser.add_argument(
    "start_seed", 
    type=int, 
    help="Error seed to start with."
)
parser.add_argument(
    "end_seed", 
    type=int, 
    help="Error seed to end with."
)
parser.add_argument(
    "mode", 
    type=str, 
    help="Can be 'collision' or 'injection'."
)
parser.add_argument(
    "line_path", 
    type=str, 
    help="Path to existing xsuite line without errors/"
)


def install_and_correct(beam, seed, mode, line_path):
    '''
    Read error tables
    '''
    min_order = 0
    max_order = 15

    fname_rotations = './madx/errors/LHC/rotations.tab'
    fname_err_table = f'./madx/errors/LHC/wise/{mode}_errors-emfqcs-{seed}.tfs'

    multipole_errors_arc, tt_err_arc = xmlhc.load_wise_table_arc_magnets(
        fname_err_table=fname_err_table,
        fname_rotations=fname_rotations,
        min_order=min_order, max_order=max_order
    )

    # Association knob_name -> multipole errors
    multipole_errors_to_apply = {
        'on_error_arc': multipole_errors_arc
    }


    '''
    Load line without error, make an env
    '''
    line = xt.load(line_path)
    env = line.env
    env[f'lhcb{beam}'] = line


    '''
    Apply errors
    '''
    for knob_name, multipole_errors in multipole_errors_to_apply.items():
        for line_name in ['lhcb1', 'lhcb2']:
            try:
                line = env[line_name]
            except KeyError:
                continue
            xm.set_multipole_errors_in_line(
                line, multipole_errors,
                min_order=min_order, max_order=max_order,
                error_knob_name=knob_name,
                append_order_to_knob_name=True
            )

    # Switch off errors of order 0 and 1
    env['on_error_arc_k0'] = 0
    env['on_error_arc_k0s'] = 0
    env['on_error_arc_k1'] = 0
    env['on_error_arc_k1s'] = 0


    '''
    Correct errors
    '''
    # Status of error knobs
    tt_err_knobs = env.vars.get_table().rows[r'on_error_.*']
    print("Error knobs in the environment:")
    tt_err_knobs.show()

    # Errors off to get reference twiss
    env.set(tt_err_knobs.name, 0)
    tw_b12 = {}
    for line_name in ['lhcb1', 'lhcb2']:
        try:
            tw = env[line_name].twiss4d(reverse=False) # Reference twiss
            tw_b12[line_name[-2:]] = tw
        except KeyError:
            tw_b12[line_name[-2:]] = None

    # errors back on
    for nn in tt_err_knobs.name:
        env[nn] = tt_err_knobs['value', nn]

    # Spool piece correctors (MCS, MC0, MCD)
    xmlhc.set_arc_spool_piece_correctors(env, twiss_b1=tw_b12['b1'], twiss_b2=tw_b12['b2'])

    # k1s local + global correction (uses MQS)
    xmlhc.correct_k1s(env, twiss_b1=tw_b12['b1'], twiss_b2=tw_b12['b2'])

    # k2s local + global correction (uses MSS)
    xmlhc.correct_k2s(env, twiss_b1=tw_b12['b1'], twiss_b2=tw_b12['b2'])


    '''
    Save line
    '''
    env[f'lhcb{beam}'].to_json(line_path[:-15] + f"_seed{seed}.json")


def main():
    '''
    Print xsuite package versions
    '''
    print(f"xobjects: {xo.__version__}")
    print(f"xdeps: {xd.__version__}")
    print(f"xtrack: {xt.__version__}")
    print(f"xpart: {xp.__version__}")
    print(f"xfields: {xf.__version__}")
    print(f"xcoll: {xc.__version__}")
    print(f"xsuite: {xs.__version__}")


    '''
    Parse arguments
    '''
    args = parser.parse_args()
    beam = args.beam
    start_seed = args.start_seed
    end_seed = args.end_seed
    mode = args.mode
    line_path = args.line_path


    '''
    Install all
    '''
    for seed in range(start_seed, end_seed+1, 1):
        print("##############################################################")
        print(f"#######################     SEED {seed}    #######################")
        print("##############################################################")
        install_and_correct(beam, seed, mode, line_path)


if __name__ == "__main__":
    main()