import argparse
import re
import pickle
import numpy as np
from scipy.stats import chi
from scipy.special import gamma
from scipy.integrate import cumulative_trapezoid

import xobjects as xo
import xdeps as xd
import xtrack as xt
import xpart as xp
import xfields as xf
import xcoll as xc
import xsuite as xs


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
    "mqxf_fieldmap_file", 
    type=str, 
    help="path to csv file containing fit of MQXF (Q2 R5) magnetic field over time"
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
    "q", 
    type=float, 
    help="q value of distribution"
)
parser.add_argument(
    "beta", 
    type=float, 
    help="beta value of distribution"
)
parser.add_argument(
    "num_part", 
    type=int, 
    help="number of particles to track"
)
parser.add_argument(
    "num_turns", 
    type=int, 
    help="number of turns to track"
)


def f2D_r(r, q, beta):
    """
    2D density function obtained from a 1D q-Gaussian profile. Based on Eq 20 in
    https://cds.cern.ch/record/2912366/files/document.pdf

    Parameters
    ----------
    r : float | ndarray
        Radius in 2D space.
    q : float
        q parameter (1 < q < 3).
    beta : float
        Scale parameter.

    Returns
    -------
    float | ndarray
    """

    if not (1 < q < 3):
        raise ValueError("Require 1 < q < 3")

    A = - beta**1.5 * (q - 3) * np.sqrt(q - 1) * r / (2 * np.pi)
    B = (1 + 1/(beta*(q-1)*r**2))**((q+1)/(2-2*q))
    C = (beta*(q-1)*r**2)**(q/(1-q))

    return A * B * C


def f4D_m(m, q, beta):
    """
    4D density function obtained from a 1D q-Gaussian profile. Based on Eq 21 in
    https://cds.cern.ch/record/2912366/files/document.pdf

    Parameters
    ----------
    m : float | ndarray
        Radius in 4D space.
    q : float
        q parameter (1 < q < 3).
    beta : float
        Scale parameter.

    Returns
    -------
    float | ndarray
    """

    if not (1 < q < 3):
        raise ValueError("Require 1 < q < 3")

    pref = (
        -beta * (q-3) * (q+1)
        * (1/(beta*(q-1)))**(1/(1-q))
        / (4*np.pi**2 * m**3 * gamma(1/(q-1)))
        * (beta*(q-1))**((q+1)/(2-2*q))
        * gamma(q/(q-1))
    )

    power1 = (1 + 1/(beta*m**2*(q-1)))**(1/(1-q) - 1.5)
    power2 = (beta*m**2*(q-1))**(1/(1-q))

    return pref * power1 * power2


def sample_radius(pdf_func, q, beta, D, N, rmax=50, gridsize=200000):
    """
    Sampling function to sample radius from custom PDF.

    Parameters
    ----------
    pdf_func : function
        Custom PDF to sample from.
    q : float
        q parameter (1 < q < 3).
    beta : float
        Scale parameter.
    D : int
        Dimension (2 or 4).
    N : int
        Number of points to sample.
    rmax : float
        Maximum radius. Default is ``50`` sigma.
    gridsize : int
        Number of grid points. Default is ``200000``.
    

    Returns
    -------
    ndarray
        Array of samples.
    """

    r = np.linspace(1e-10, rmax, gridsize)

    # radial probability density p(r) proportional to r^(D-1) f_D(r)
    if D == 2:
        surface = 2*np.pi*r
    elif D == 4:
        surface = 2*np.pi**2 * r**3
    else:
        raise ValueError("Only D=2 or D=4 supported")

    pdf = surface * pdf_func(r, q, beta)
    pdf[pdf < 0] = 0

    cdf = cumulative_trapezoid(pdf, r, initial=0)
    cdf /= cdf[-1]

    u = np.random.rand(N)

    return np.interp(u, cdf, r)


def random_direction(D, N):
    v = np.random.normal(size=(N, D))
    v /= np.linalg.norm(v, axis=1)[:, None]
    
    return v


def generate_4D_nonfactorizable(N, q, beta):
    m = sample_radius(f4D_m, q, beta, D=4, N=N)
    directions = random_direction(4, N)
    res = directions * m[:, None]

    return res[:, 0], res[:, 1], res[:, 2], res[:, 3]


def generate_2D_factorizable(N, q, beta):

    r_x = sample_radius(f2D_r, q, beta, D=2, N=N)
    r_y = sample_radius(f2D_r, q, beta, D=2, N=N)

    theta_x = 2*np.pi*np.random.rand(N)
    theta_y = 2*np.pi*np.random.rand(N)

    x  = r_x * np.cos(theta_x)
    px = r_x * np.sin(theta_x)

    y  = r_y * np.cos(theta_y)
    py = r_y * np.sin(theta_y)

    return x, px, y, py


def generate_2D_factorizable_cut(N, q, beta, r_cut):

    x, px, y, py = generate_2D_factorizable(N, q, beta)

    r_x = np.sqrt(x**2 + px**2)
    r_y = np.sqrt(y**2 + py**2)

    mask = (r_x < r_cut) & (r_y < r_cut)

    x_final = x[mask]
    px_final = px[mask]
    y_final = y[mask]
    py_final = py[mask]

    while len(x_final) < N:
        x, px, y, py = generate_2D_factorizable(N, q, beta)

        r_x = np.sqrt(x**2 + px**2)
        r_y = np.sqrt(y**2 + py**2)

        mask = (r_x < r_cut) & (r_y < r_cut)

        x_final = np.hstack((x_final, x[mask]))
        px_final = np.hstack((px_final, px[mask]))
        y_final = np.hstack((y_final, y[mask]))
        py_final = np.hstack((py_final, py[mask]))

    return x_final[:N], px_final[:N], y_final[:N], py_final[:N]


def generate_4D_nonfactorizable_cut(N, q, beta, r_cut):

    x, px, y, py = generate_4D_nonfactorizable(N, q, beta)

    r_x = np.sqrt(x**2 + px**2)
    r_y = np.sqrt(y**2 + py**2)

    mask = (r_x < r_cut) & (r_y < r_cut)

    x_final = x[mask]
    px_final = px[mask]
    y_final = y[mask]
    py_final = py[mask]

    while len(x_final) < N:
        x, px, y, py = generate_4D_nonfactorizable(N, q, beta)

        r_x = np.sqrt(x**2 + px**2)
        r_y = np.sqrt(y**2 + py**2)

        mask = (r_x < r_cut) & (r_y < r_cut)

        x_final = np.hstack((x_final, x[mask]))
        px_final = np.hstack((px_final, px[mask]))
        y_final = np.hstack((y_final, y[mask]))
        py_final = np.hstack((py_final, py[mask]))

    return x_final[:N], px_final[:N], y_final[:N], py_final[:N]


def track_single_batch(
    line_path, 
    colldb_path, 
    beam, 
    seed, 
    mqxf_fieldmap_file, 
    cliq_case, 
    batch, 
    dist_mode, 
    q, 
    beta, 
    num_part, 
    num_turns, 
    bunch_int
):
    '''
    Set up the line
    '''
    # Load line
    line = xt.load(line_path)

    # Check apertures
    if beam == 1:
        apercheck = line.check_aperture()
        missingaper = apercheck[apercheck.misses_aperture_upstream].name.values
        missaper = xt.LimitRect(min_x=-0.1, max_x=0.1, min_y=-0.1, max_y=0.1)
        for ele in missingaper:
            line.insert_element(ele+'_aper', missaper, index=ele)

        line.insert_element('tcspm.6r7.b1_aper', missaper, index='tcspm.6r7.b1')
        line.insert_element('tcspm.e5r7.b1_aper', missaper, index='tcspm.e5r7.b1')
        line.insert_element('tcspm.b4l7.b1_aper', missaper, index='tcspm.b4l7.b1')
        line.insert_element('tdisa.a4l2.b1_aper', missaper, index='tdisa.a4l2.b1')
        line.insert_element('tdisb.a4l2.b1_aper', missaper, index='tdisb.a4l2.b1')
        line.insert_element('tdisc.a4l2.b1_aper', missaper, index='tdisc.a4l2.b1')
        line.insert_element('tcld.a11r2.b1_aper', missaper, index='tcld.a11r2.b1')
    else:
        apercheck = line.check_aperture()
        missingaper = apercheck[apercheck.misses_aperture_upstream].name.values
        missaper = xt.LimitRect(min_x=-0.1, max_x=0.1, min_y=-0.1, max_y=0.1)
        for ele in missingaper:
            line.insert_element(ele+'_aper', missaper, index=ele)

        line.insert_element('tcspm.6l7.b2_aper', missaper, index='tcspm.6l7.b2')
        line.insert_element('tcspm.d4r7.b2_aper', missaper, index='tcspm.d4r7.b2')
        line.insert_element('tcspm.e5l7.b2_aper', missaper, index='tcspm.e5l7.b2')
        line.insert_element('tcspm.b4r7.b2_aper', missaper, index='tcspm.b4r7.b2')

    # Initialise colldb
    if colldb_path.endswith(".yaml"):
        colldb = xc.CollimatorDatabase.from_yaml(colldb_path, beam=beam)
    else:
        colldb = xc.CollimatorDatabase.from_SixTrack(colldb_path, nemitt_x=2.5e-6, nemitt_y=2.5e-6)

    # Install collimators into line
    colldb.install_everest_collimators(line=line, verbose=True)

    # Assign the optics to collimators to deduce gaps
    tw = line.twiss()
    line.collimators.assign_optics(twiss=tw)

    # Aperture model check
    # print('\nAperture model check:')
    # df_with_coll = line.check_aperture()
    # assert not np.any(df_with_coll.has_aperture_problem)


    '''
    Generate particles
    '''
    # First make a matched Gaussian to get proper longitudinal distribution
    tmp = xp.generate_matched_gaussian_bunch(
        num_particles=num_part, 
        total_intensity_particles=bunch_int, 
        nemitt_x=colldb.nemitt_x, 
        nemitt_y=colldb.nemitt_y, 
        sigma_z=7.55e-2, 
        line=line, 
        co_guess=tw.particle_on_co
    )
    zeta = tmp.zeta
    delta = tmp.delta

    # Generate transverse distributions as Gaussians cut at 3sigma
    if dist_mode == "4D":
        x_norm, px_norm, y_norm, py_norm = generate_4D_nonfactorizable_cut(num_part, q, beta, r_cut=6.7)
    elif dist_mode == "2D":
        x_norm, px_norm, y_norm, py_norm = generate_2D_factorizable_cut(num_part, q, beta, r_cut=6.7)

    # Generate initial distribution
    part = line.build_particles(
        zeta=zeta, 
        delta=delta, 
        x_norm=x_norm, 
        px_norm=px_norm, 
        y_norm=y_norm, 
        py_norm=py_norm, 
        nemitt_x=colldb.nemitt_x, 
        nemitt_y=colldb.nemitt_y, 
        co_guess=tw.particle_on_co
    )
    part_lm = line.build_particles(
        zeta=zeta, 
        delta=delta, 
        x_norm=x_norm, 
        px_norm=px_norm, 
        y_norm=y_norm, 
        py_norm=py_norm, 
        nemitt_x=colldb.nemitt_x, 
        nemitt_y=colldb.nemitt_y, 
        co_guess=tw.particle_on_co
    )

    # Save initial distribution
    with open(f'part_init_batch{batch}_seed{seed}_{cliq_case}_{dist_mode}.pkl', 'wb') as file:
        pickle.dump(part.to_dict(), file)


    '''
    Set up CLIQ failure
    '''
    # Load file
    field_data = np.loadtxt(mqxf_fieldmap_file, skiprows=1, delimiter=',')
    if cliq_case in ["Q1R5", "Q3R5", "Q2L5", "Q1R1", "Q3R1", "Q2L1"]:
        for i in range(7):
            if (i % 2) == 0:
                field_data[:, i+2] *= -1
            if (i % 2) != 0:
                field_data[:, i+9] *= -1
    
    # Extend knl and ksl of relevant magnet
    failing_names = []
    for name in line.element_names:
        if cliq_case == "Q1R5":
            if name.startswith("mqxfa.") and "1r5.." in name and "_aper" not in name:
                failing_names.append(name)
        elif cliq_case == "Q2R5":
            if name.startswith("mqxfb.") and "2r5.." in name and "_aper" not in name:
                failing_names.append(name)
        elif cliq_case == "Q3R5":
            if name.startswith("mqxfa.") and "3r5.." in name and "_aper" not in name:
                failing_names.append(name)
        elif cliq_case == "Q1L5":
            if name.startswith("mqxfa.") and "1l5.." in name and "_aper" not in name:
                failing_names.append(name)
        elif cliq_case == "Q2L5":
            if name.startswith("mqxfb.") and "2l5.." in name and "_aper" not in name:
                failing_names.append(name)
        elif cliq_case == "Q3L5":
            if name.startswith("mqxfa.") and "3l5.." in name and "_aper" not in name:
                failing_names.append(name)
        elif cliq_case == "Q1R1":
            if name.startswith("mqxfa.") and "1r1.." in name and "_aper" not in name:
                failing_names.append(name)
        elif cliq_case == "Q2R1":
            if name.startswith("mqxfb.") and "2r1.." in name and "_aper" not in name:
                failing_names.append(name)
        elif cliq_case == "Q3R1":
            if name.startswith("mqxfa.") and "3r1.." in name and "_aper" not in name:
                failing_names.append(name)
        elif cliq_case == "Q1L1":
            if name.startswith("mqxfa.") and "1l1.." in name and "_aper" not in name:
                failing_names.append(name)
        elif cliq_case == "Q2L1":
            if name.startswith("mqxfb.") and "2l1.." in name and "_aper" not in name:
                failing_names.append(name)
        elif cliq_case == "Q3L1":
            if name.startswith("mqxfa.") and "3l1.." in name and "_aper" not in name:
                failing_names.append(name)
        else:
            raise ValueError(f"Invalid CLIQ case: {cliq_case}!")
    print(failing_names)
    line.extend_knl_ksl(6, failing_names)
    
    # Define variables for CLIQ field components
    line.vars["cliq_Kn0"] = 0.0
    line.vars["cliq_Kn1"] = 0.0
    line.vars["cliq_Kn2"] = 0.0
    line.vars["cliq_Kn3"] = 0.0
    line.vars["cliq_Kn4"] = 0.0
    line.vars["cliq_Kn5"] = 0.0
    line.vars["cliq_Kn6"] = 0.0
    line.vars["cliq_Ks0"] = 0.0
    line.vars["cliq_Ks1"] = 0.0
    line.vars["cliq_Ks2"] = 0.0
    line.vars["cliq_Ks3"] = 0.0
    line.vars["cliq_Ks4"] = 0.0
    line.vars["cliq_Ks5"] = 0.0
    line.vars["cliq_Ks6"] = 0.0

    # Assign variables controlling strengths
    Brho = line.particle_ref.rigidity0[0]
    for name in failing_names:
        for i in range(7):
            line.element_refs[name].knl[i] += line.vars[f"cliq_Kn{i}"] / Brho * line[name].length
            line.element_refs[name].ksl[i] += line.vars[f"cliq_Ks{i}"] / Brho * line[name].length

    # Define their time-dependence
    if batch > 9:
        raise ValueError("Batch/CLIQ timestamp offset cannot exceed 9!")
    for i in range(7):
        line.functions[f"cliq_Kn{i}_func"] = xt.FunctionPieceWiseLinear(
            x=field_data[batch:, 1] * tw.T_rev0,      # time in s
            y=field_data[batch:, i+2]                 # on-off
        )
        line.vars[f"cliq_Kn{i}"] = line.functions[f"cliq_Kn{i}_func"](line.ref['t_turn_s'])
        line.functions[f"cliq_Ks{i}_func"] = xt.FunctionPieceWiseLinear(
            x=field_data[batch:, 1] * tw.T_rev0,      # time in s
            y=field_data[batch:, i+9]                 # on-off
        )
        line.vars[f"cliq_Ks{i}"] = line.functions[f"cliq_Ks{i}_func"](line.ref['t_turn_s'])

    
    '''
    Track
    '''
    # Move the line to an OpenMP context to be able to use all cores
    # line.discard_tracker()
    # line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

    # Enable scattering
    line.scattering.enable()

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
    line.track(
        part_lm, 
        num_turns=5, 
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
    with open(f'part_fin_batch{batch}_seed{seed}_{cliq_case}_{dist_mode}.pkl', 'wb') as file:
        pickle.dump(part.to_dict(), file)

    # Save full loss map to json
    line_is_reversed = True if f'{beam}' == '2' else False
    ThisLM = xc.LossMap(line, line_is_reversed=line_is_reversed, part=part_lm)
    ThisLM.to_json(file=f'lossmap_batch{batch}_seed{seed}_{cliq_case}_{dist_mode}.json')

    print(ThisLM.summary)

    
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
    print(f"xsuite: {xs.__version__}")

    
    '''
    Parse arguments
    '''
    args = parser.parse_args()
    line_path = args.line
    colldb_path = args.colldb
    beam = args.beam
    seed = args.seed
    mqxf_fieldmap_file = args.mqxf_fieldmap_file
    cliq_case = args.cliq_case
    dist_mode = args.dist_mode
    q = args.q
    beta = args.beta
    num_part = args.num_part
    num_turns = args.num_turns
    bunch_int = 2.2e11

    if seed != 0 and line_path.endswith("no_errors.json"):
        line_path = line_path.split("no_errors.json")[0] + f"seed{seed}.json"
    elif seed != 0 and not line_path.endswith("no_errors.json") and "seed" in line_path:
        line_path_split = re.split("seed|.json")
        line_path_base = line_path_split[0]
        if seed != line_path_split[1]:
            line_path = line_path_base + f"seed{seed}.json"
    elif seed == 0:
        pass
    else:
        raise RuntimeError


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
            mqxf_fieldmap_file, 
            cliq_case, 
            batch, 
            dist_mode, 
            q, 
            beta, 
            num_part, 
            num_turns, 
            bunch_int
        )

    print("All batches finished!")


if __name__ == "__main__":
    main()