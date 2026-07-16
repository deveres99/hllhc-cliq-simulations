import argparse
import re
import pickle
import numpy as np

import xobjects as xo
import xdeps as xd
import xtrack as xt
import xpart as xp
import xfields as xf
import xcoll as xc


parser = argparse.ArgumentParser()
parser.add_argument(
    "line", 
    type=str, 
    help="path to xsuite line to track"
)
parser.add_argument(
    "colldb", 
    type=str, 
    help="path to colldb yaml file"
)
parser.add_argument(
    "beam", 
    type=int, 
    help="beam, can be 1 or 2"
)
parser.add_argument(
    "seed", 
    type=int, 
    help="error seed"
)
parser.add_argument(
    "init_dist_path", 
    type=str, 
    help="path to initial distribution file"
)
parser.add_argument(
    "mqxf_fieldmap_file", 
    type=str, 
    help="path to csv file containing fit of MQXF (Q2 R5) magnetic field over time"
)
parser.add_argument(
    "cliq_case", 
    type=str, 
    help="which magnet to assign field errors to ('Q1R5a', 'Q1L5a', 'Q2R5a', 'Q2L5a', 'Q3R5a', 'Q3L5a', 'Q1R1a', 'Q1L1a', 'Q2R1a', 'Q2L1a', 'Q3R1a', 'Q3L1a', " \
    "'Q1R5b', 'Q1L5b', 'Q2R5b', 'Q2L5b', 'Q3R5b', 'Q3L5b', 'Q1R1b', 'Q1L1b', 'Q2R1b', 'Q2L1b', 'Q3R1b', 'Q3L1b')"
)
parser.add_argument(
    "halo_model", 
    type=str, 
    help="halo model ('mean', 'conservative')"
)
parser.add_argument(
    "dist_mode", 
    type=str, 
    help="distribution generation mode ('2D', '4D')"
)
parser.add_argument(
    "i_mo", 
    type=float, 
    help="octupole current in A"
)
parser.add_argument(
    "magnet_beta_pcent", 
    type=float, 
    help="percentage change in the beta function at the magnet with which to scale field map"
)
parser.add_argument(
    "num_turns", 
    type=int, 
    help="number of turns to track"
)


def track_single_batch(
    line_path, 
    colldb_path, 
    beam, 
    seed, 
    init_dist_path, 
    mqxf_fieldmap_file, 
    cliq_case, 
    batch, 
    halo_model, 
    dist_mode, 
    i_mo, 
    magnet_beta_pcent, 
    num_turns, 
    bunch_int
):
    '''
    Set up the line
    '''
    # Load line
    line = xt.load(line_path)
    line.vars[f"i_mo.b{beam}"] = i_mo

    # Initialise colldb
    if colldb_path.endswith(".yaml"):
        colldb = xc.CollimatorDatabase.from_yaml(colldb_path, beam=beam)
    else:
        colldb = xc.CollimatorDatabase.from_SixTrack(colldb_path, nemitt_x=2.5e-6, nemitt_y=2.5e-6)

    # Install collimators into line
    colldb.install_everest_collimators(line=line, verbose=True)

    # Assign the optics to collimators to deduce gaps
    tw = line.twiss()
    line.xcoll.collimators.assign_optics(twiss=tw)


    '''
    Generate particles
    '''
    # Load initial distribution
    with open(init_dist_path, "rb") as file:
        dist_data = pickle.load(file)
    nemitt_x = dist_data["nemitt_x"]
    nemitt_y = dist_data["nemitt_y"]
    if not np.isclose(dist_data["i_mo"], i_mo):
        raise ValueError(
            "Octupole current used to generate distribution does not match " /
            "the one requested for tracking!"
        )
    dist_data = dist_data[batch]["norm"]
    x_norm = dist_data["x_norm"]
    px_norm = dist_data["px_norm"]
    y_norm = dist_data["y_norm"]
    py_norm = dist_data["py_norm"]

    # Make a matched Gaussian to get proper longitudinal distribution
    tmp = xp.generate_matched_gaussian_bunch(
        num_particles=int(1e5), 
        total_intensity_particles=bunch_int, 
        nemitt_x=nemitt_x, 
        nemitt_y=nemitt_y, 
        sigma_z=7.55e-2, 
        line=line, 
        # co_guess=tw.particle_on_co
    )
    zeta = tmp.zeta
    delta = tmp.delta

    # Generate initial distribution
    part = line.build_particles(
        zeta=zeta, 
        delta=delta, 
        x_norm=x_norm, 
        px_norm=px_norm, 
        y_norm=y_norm, 
        py_norm=py_norm, 
        nemitt_x=nemitt_x, 
        nemitt_y=nemitt_y, 
        # co_guess=tw.particle_on_co
    )


    '''
    Set up CLIQ failure
    '''
    # Load file
    field_data = np.loadtxt(mqxf_fieldmap_file, skiprows=1, delimiter=',')
    if cliq_case in [
        "Q1R5a", "Q3R5a", "Q2L5a", "Q1R1a", "Q3R1a", "Q2L1a", 
        "Q1R5b", "Q3R5b", "Q2L5b", "Q1R1b", "Q3R1b", "Q2L1b"
    ]:
        for i in range(7):
            if (i % 2) == 0:
                field_data[:, i+2] *= -1
            if (i % 2) != 0:
                field_data[:, i+9] *= -1
    
    # Extend knl and ksl of relevant magnet
    if cliq_case == "Q1R5a":
        failing_names = [f"mqxfa.a1r5/b{beam}"]
    elif cliq_case == "Q1R5b":
        failing_names = [f"mqxfa.b1r5/b{beam}"]
    elif cliq_case == "Q2R5a":
        failing_names = [f"mqxfb.a2r5/b{beam}"]
    elif cliq_case == "Q2R5b":
        failing_names = [f"mqxfb.b2r5/b{beam}"]
    elif cliq_case == "Q3R5a":
        failing_names = [f"mqxfa.a3r5/b{beam}"]
    elif cliq_case == "Q3R5b":
        failing_names = [f"mqxfa.b3r5/b{beam}"]
    elif cliq_case == "Q1L5a":
        failing_names = [f"mqxfa.a1l5/b{beam}"]
    elif cliq_case == "Q1L5b":
        failing_names = [f"mqxfa.b1l5/b{beam}"]
    elif cliq_case == "Q2L5a":
        failing_names = [f"mqxfb.a2l5/b{beam}"]
    elif cliq_case == "Q2L5b":
        failing_names = [f"mqxfb.b2l5/b{beam}"]
    elif cliq_case == "Q3L5a":
        failing_names = [f"mqxfa.a3l5/b{beam}"]
    elif cliq_case == "Q3L5b":
        failing_names = [f"mqxfa.b3l5/b{beam}"]
    elif cliq_case == "Q1R1a":
        failing_names = [f"mqxfa.a1r1/b{beam}"]
    elif cliq_case == "Q1R1b":
        failing_names = [f"mqxfa.b1r1/b{beam}"]
    elif cliq_case == "Q2R1a":
        failing_names = [f"mqxfb.a2r1/b{beam}"]
    elif cliq_case == "Q2R1b":
        failing_names = [f"mqxfb.b2r1/b{beam}"]
    elif cliq_case == "Q3R1a":
        failing_names = [f"mqxfa.a3r1/b{beam}"]
    elif cliq_case == "Q3R1b":
        failing_names = [f"mqxfa.b3r1/b{beam}"]
    elif cliq_case == "Q1L1a":
        failing_names = [f"mqxfa.a1l1/b{beam}"]
    elif cliq_case == "Q1L1b":
        failing_names = [f"mqxfa.b1l1/b{beam}"]
    elif cliq_case == "Q2L1a":
        failing_names = [f"mqxfb.a2l1/b{beam}"]
    elif cliq_case == "Q2L1b":
        failing_names = [f"mqxfb.b2l1/b{beam}"]
    elif cliq_case == "Q3L1a":
        failing_names = [f"mqxfa.a3l1/b{beam}"]
    elif cliq_case == "Q3L1b":
        failing_names = [f"mqxfa.b3l1/b{beam}"]
    else:
        raise ValueError(f"Invalid CLIQ case: {cliq_case}!")
    print(failing_names)
    line.env.extend_knl_ksl(6, failing_names)
    
    # Define variables for CLIQ field components
    line.env.vars["cliq_Kn0"] = 0.0
    line.env.vars["cliq_Kn1"] = 0.0
    line.env.vars["cliq_Kn2"] = 0.0
    line.env.vars["cliq_Kn3"] = 0.0
    line.env.vars["cliq_Kn4"] = 0.0
    line.env.vars["cliq_Kn5"] = 0.0
    line.env.vars["cliq_Kn6"] = 0.0
    line.env.vars["cliq_Ks0"] = 0.0
    line.env.vars["cliq_Ks1"] = 0.0
    line.env.vars["cliq_Ks2"] = 0.0
    line.env.vars["cliq_Ks3"] = 0.0
    line.env.vars["cliq_Ks4"] = 0.0
    line.env.vars["cliq_Ks5"] = 0.0
    line.env.vars["cliq_Ks6"] = 0.0

    # Assign variables controlling strengths
    Brho = line.particle_ref.rigidity0[0]
    for name in failing_names:
        for i in range(7):
            line.element_refs[name].knl[i] += line.env.vars[f"cliq_Kn{i}"] / Brho * line.env[name].length * np.sqrt(1+magnet_beta_pcent/100)
            line.element_refs[name].ksl[i] += line.env.vars[f"cliq_Ks{i}"] / Brho * line.env[name].length * np.sqrt(1+magnet_beta_pcent/100)

    # Define their time-dependence
    if batch > 9:
        raise ValueError("Batch/CLIQ timestamp offset cannot exceed 9!")
    for i in range(7):
        line.functions[f"cliq_Kn{i}_func"] = xt.FunctionPieceWiseLinear(
            x=field_data[batch:, 1] * tw.t_rev0,      # time in s
            y=field_data[batch:, i+2]                 # on-off
        )
        line.env.vars[f"cliq_Kn{i}"] = line.functions[f"cliq_Kn{i}_func"](line.ref['t_turn_s'])
        line.functions[f"cliq_Ks{i}_func"] = xt.FunctionPieceWiseLinear(
            x=field_data[batch:, 1] * tw.t_rev0,      # time in s
            y=field_data[batch:, i+9]                 # on-off
        )
        line.env.vars[f"cliq_Ks{i}"] = line.functions[f"cliq_Ks{i}_func"](line.ref['t_turn_s'])

    # Allow identification of primary losses
    line.xcoll.scattering.identify_primary_losses()

    
    '''
    Track
    '''
    # Move the line to an OpenMP context to be able to use all cores
    # line.discard_tracker()
    # line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

    # Enable scattering
    line.xcoll.scattering.enable()

    # Enable time-dependent variables
    line.enable_time_dependent_vars = True
    
    # Track
    line.track(
        part, 
        num_turns=num_turns, 
        time=True, 
        with_progress=1
    )
    print(f"Done tracking in {line.time_last_track:.1f}s.")

    # Move the line back to the default context to be able to use all prebuilt kernels for the aperture interpolation
    # line.discard_tracker()
    # line.build_tracker(_context=xo.ContextCpu())


    '''
    Save data
    '''
    # Save final distribution
    with open(f'part_fin_batch{batch}_seed{seed}_{cliq_case}_{halo_model}_{dist_mode}_imo_{int(i_mo)}.pkl', 'wb') as file:
        pickle.dump(part.to_dict(), file)

    
    '''
    Simulation done
    '''
    print("Simulation ran successfully, all data saved!")


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

    
    '''
    Parse arguments
    '''
    args = parser.parse_args()
    line_path = args.line
    colldb_path = args.colldb
    beam = args.beam
    seed = args.seed
    init_dist_path = args.init_dist_path
    mqxf_fieldmap_file = args.mqxf_fieldmap_file
    cliq_case = args.cliq_case
    halo_model = args.halo_model
    dist_mode = args.dist_mode
    i_mo = args.i_mo
    magnet_beta_pcent = args.magnet_beta_pcent
    num_turns = args.num_turns
    bunch_int = 2.3e11


    '''
    Run 10 batches
    '''
    for batch in range(10):
        print(f"Tracking batch {batch}...")
        track_single_batch(
            line_path, 
            colldb_path, 
            beam, 
            seed, 
            init_dist_path, 
            mqxf_fieldmap_file, 
            cliq_case, 
            batch, 
            halo_model, 
            dist_mode, 
            i_mo, 
            magnet_beta_pcent, 
            num_turns, 
            bunch_int
        )

    print("All batches finished!")


if __name__ == "__main__":
    main()