#%%
import sys
import argparse
from pathlib import Path
import h5py as h5
import lqcd.core as cr
from lqcd.io import set_backend, get_backend
from lqcd.gauge import Smear as gSmear
from lqcd.fermion import DiracOperator, Smear as qSmear
from lqcd.algorithms import Inverter, GFlow
import lqcd.measurements.contract_funcs as cf
import lqcd.measurements.analysis_funcs as af
import lqcd.utils as ut
from opt_einsum import contract
import numpy as np
import matplotlib.pyplot as plt
import tqdm



#%%
# 2025.09.09: Cross check gf.



#%%
def read_gauge_Ani(filename):
    U_Ani = np.genfromtxt(filename, delimiter=',', skip_header=1)
    U_Ani = U_Ani[:,7] + 1j * U_Ani[:,8]
    U_Ani = np.reshape(U_Ani, (X, Y, Z, T, 4, 3, 3))
    # Change form xyztmuab to txyzmuab. Important: the mu axis should also change to match
    # 1. xyzt to txyz
    U_Ani = np.moveaxis(U_Ani, 3, 0)
    # 2. permute mu also
    U_Ani = U_Ani[..., [3, 0, 1, 2], :, :]
    return U_Ani

def read_spinor_Ani(filename):
    chi_Ani = np.genfromtxt(filename, delimiter=',', skip_header=1)
    chi_Ani = chi_Ani[:,6] + 1j * chi_Ani[:,7]
    chi_Ani = np.reshape(chi_Ani, (X, Y, Z, T, 4, 3))
    chi_Ani = np.moveaxis(chi_Ani, 3, 0)
    chi_Ani = chi_Ani[..., [3, 0, 1, 2], :]
    return chi_Ani


#%%
if __name__ == "__main__":
    #%%
    if len(sys.argv) != 1:
        parser = argparse.ArgumentParser(description='PDF contractions.')
        parser.add_argument('-cd', '--confdir', type = str, required = True, metavar = '', help = 'peram dir')
        parser.add_argument('-e', '--ens', type = str, required = True, metavar = '', help = 'conf name')
        parser.add_argument('-cn', '--confnum', type = int, required = True, metavar = '', help = 'conf serial number')
        parser.add_argument('-d','--diagram', type = str, nargs='+', required = True, metavar = '', help='diagram, loop, BD, or W')
        args = parser.parse_args()
    else:
        class Args:
            confdir = "/Users/haobo/Documents/Lattice QCD/research/LittleQCD/LittleQCD/lqcd/algorithms"
            #confdir = "/lustre/home/2101110113/research/hpv"
            ens = "beta6.00"
            confnum = 1000
            diagram = ["loop", "bdv", "w", "mx", "ko"]
            diagram = ["bdv"]
        args = Args()
    print(args)

    #%%
    # Initialization
    set_backend("numpy")
    xp = get_backend()

    # Definitions
    Cg5 = 1j * cr.Gamma(2) * cr.Gamma(0) * cr.Gamma(5)

    # Settings
    geo_vec = [16, 8, 8, 8]

    [T, X, Y, Z] = geo_vec
    geometry = cr.QCD_geometry(geo_vec)

    # Setup
    # Path("../data/beta_%.2f_L%dx%d/data/%d"%(beta, X, T, args.confnum)).mkdir(parents=True, exist_ok=True)

    U = cr.Gauge(geometry)
    U.field = read_gauge_Ani("../WFlow_tests_Ani/gauge_in_8c16.dat")
    ut.check("Gauge field", U.field[3,0,3,2,0,1,0].real, -0.4700440480904416)
    ut.check("Plaquette", U.plaquette_measure(), 0.5820438913066585)
    # U = U.apply_boundary_condition_periodic_quark()

    chi = cr.Fermion(geometry)
    chi.point_source([0, 0, 0, 0, 0, 0])
    chi.field = read_spinor_Ani("../WFlow_tests_Ani/spinor_in_8c16.dat")

    gflow = GFlow(U, chi, {"dt": 0.125, "niter": 4})
    U_flowed, xi_flowed = gflow.forward()
    for i in range(1, len(gflow.U_list)):
        U_Ani_flowed = cr.Gauge(geometry)
        U_Ani_flowed.field = read_gauge_Ani("../WFlow_tests_Ani/gauge_out_8c16_epsilon0.125000_n_steps%d.dat" % i)
        ut.check("Gauge flowed n = %d:" % i, gflow.U_list[i].field[3,0,3,2,0,1,0].real, U_Ani_flowed.field[3,0,3,2,0,1,0].real)
        if i == 2:
            xi_Ani_flowed = cr.Fermion(geometry)
            xi_Ani_flowed.field = read_spinor_Ani("../WFlow_tests_Ani/spinor_out_8c16_epsilon0.125000_n_steps%d.dat" % i)
            ut.check("Spinor flowed n = %d:" % i, gflow.chi_list[i].field[3,0,3,2,0,1].real, xi_Ani_flowed.field[3,0,3,2,0,1].real)

