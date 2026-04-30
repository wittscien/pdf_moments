#%%
import sys
import argparse
from pathlib import Path
import h5py as h5
import lqcd.core as cr
from lqcd.io import set_backend, get_backend, set_gamma_convention
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
# 2026.04.30: More derivatives.
# 2026.03.01: Non-local 3pt for PDF.
# 2026.01.08: Pion 2pt and 3pt.
# 2025.10.30: Cross check fermion adj gf without bc.
# 2025.10.15: Cross check fermion gf without bc.
# 2025.09.09: Cross check gauge gf without bc.



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
        parser.add_argument('-d','--diagram', type = str, nargs='+', required = True, metavar = '', help='diagram, loop, BD, or W')
        args = parser.parse_args()
    else:
        class Args:
            confdir = "/Users/haobo/Documents/Lattice QCD/research/LittleQCD/LittleQCD/lqcd/algorithms"
            #confdir = "/lustre/home/2101110113/research/hpv"
            ens = "beta6.00"
            diagram = ["loop", "bdv", "w", "mx", "ko"]
            diagram = ["bdv"]
        args = Args()
    print(args)

    #%%
    # Initialization
    set_backend("numpy")
    set_gamma_convention("ukqcd")
    xp = get_backend()

    # Definitions
    Cg5 = 1j * cr.Gamma(2) * cr.Gamma(0) * cr.Gamma(5)

    # Settings
    # geo_vec = [16, 8, 8, 8]
    geo_vec = [8, 4, 4, 4]

    [T, X, Y, Z] = geo_vec
    geometry = cr.QCD_geometry(geo_vec)

    # Setup
    beta = 6
    Path("../data/test/").mkdir(parents=True, exist_ok=True)

    check = 1
    if check:
        confs = [0]
    else:
        confs = xp.arange(1000, 2020, 20, dtype=int)
    for iconf in range(len(confs)):
        conf = confs[iconf]
        print('conf: ', conf)
        if check:
            U = cr.Gauge(geometry)
            # U.field = read_gauge_Ani("../WFlow_tests_Ani/gauge_in_8c16.dat")
            U.field = np.load("../WFlow_tests_Ani/conf.0000.npy")
            # ut.check("Gauge field", U.field[3,0,3,2,0,1,0].real, -0.4700440480904416)
            # ut.check("Plaquette", U.plaquette_measure(), 0.5820438913066585)
        else:
            U = cr.Gauge(geometry)
            U.read("/Users/haobo/Documents/LatticeQCD/research/LittleQCD/LittleQCD/lqcd/algorithms/configurations/beta_6.00_L4x8/beta_6.00_L4x8_conf_%d.h5" % conf)

        U_with_phase = U.apply_boundary_condition_minus_one()
    
        #%%
        # GF tests
        if 0:
            chi = cr.Fermion(geometry)
            chi.field = read_spinor_Ani("../WFlow_tests_Ani/spinor_in_8c16.dat")
        
            gflow = GFlow(U_with_phase, chi, {"dt": 0.125, "niter": 2})
            gflow.forward()
            gflow.adjoint(chi)
        
            for i in range(1, len(gflow.U_list)):
                U_Ani_flowed = cr.Gauge(geometry)
                U_Ani_flowed.field = read_gauge_Ani("../WFlow_tests_Ani/gauge_out_8c16_epsilon0.125000_n_steps%d.dat" % i)
                print("Gauge flowed n = %d:" % i, np.allclose(gflow.U_list[i].field, U_Ani_flowed.field))
                if i == 2:
                    chi_Ani_flowed = cr.Fermion(geometry)
                    chi_Ani_flowed.field = read_spinor_Ani("../WFlow_tests_Ani/spinor_out_8c16_epsilon0.125000_n_steps%d.dat" % i)
                    print("Spinor forward flowed n = %d:" % i, np.allclose(gflow.chi_list[i].field, chi_Ani_flowed.field))
        
                    xi_Ani_adj_flowed = cr.Fermion(geometry)
                    xi_Ani_adj_flowed.field = read_spinor_Ani("../WFlow_tests_Ani/spinor_out_Adj_8c16_epsilon0.125000_n_steps%d.dat" % i)
                    print("Spinor adjoint flowed n = %d:" % i, np.allclose(gflow.xi_list[i].field, xi_Ani_adj_flowed.field))
            
            # Save the fields
            if 0:
                Path("../WFlow_tests_Ani/mine/").mkdir(parents=True, exist_ok=True)
                U.save('../WFlow_tests_Ani/mine/gauge')
                chi.save('../WFlow_tests_Ani/mine/spinor')
                gflow.U_list[2].save('../WFlow_tests_Ani/mine/gauge_flowed_2')
                gflow.chi_list[2].save('../WFlow_tests_Ani/mine/spinor_fwd_flowed_2')
                gflow.xi_list[2].save('../WFlow_tests_Ani/mine/spinor_adj_flowed_2')

        #%%
        # Preparation
        # Dirac operator
        Q = DiracOperator(U_with_phase, {'fermion_type':'twisted_mass_clover', 'm': 0.5, 'mu': 0.112994350282, 'csw': 1.74})
    
        # Inverter parameters
        inv_params = {"method": 'BiCGStab', "tol": 1e-9, "maxit": 500, "check_residual": False, "verbose": 0, "tm_rotation": True}
    
        # Source: point-to-all propagator
        quark_smr_params = {"tech": "Jacobi", "kappa": 0.2, "niter": 20}
        Smr = qSmear(U, quark_smr_params)
        src = cr.Fermion(geometry)
        srcfull = cr.Propagator(geometry)
        for s in range(4):
            for c in range(3):
                src.point_source([0, 0, 0, 0, s, c])
                # src = Smr.smear(src)
                srcfull.set_Fermion(src, s, c)
    
        # Intermediate check
        if 0:
            src_test = cr.Fermion(geometry)
            src_test.point_source([0, 0, 0, 0, 0, 0])
            test_fermion = Q.Dirac(src_test, 'u')
    
        # Propagator
        Su_ps = ut.propagator_parallelized(Q, inv_params, srcfull, 'u')
        Sd_ps = ut.propagator_parallelized(Q, inv_params, srcfull, 'd')
    
        # Sink smearing
        # Su_ss = ut.prop_smear(Smr, Su_ps)
        # Sd_ss = ut.prop_smear(Smr, Sd_ps)
        Su_ss = Su_ps
        Sd_ss = Sd_ps
    
        #%%
        # 2pt
        corr_2pt_pion = xp.zeros((T), dtype = complex)
        corr_2pt_pion_space = cf.meson(Su_ps, Su_ps, 5, 5) # Two minus signs: dagger sign and trace sign
        for t in range(T):
            corr_2pt_pion[t] = xp.sum(corr_2pt_pion_space[t])
        xp.save("../data/test/corr_2pt_pion_conf_%d.npy" % (conf), corr_2pt_pion)

        #%%
        # 3pt with gamma_t
        if check:
            gflow_niter = 2
            gflow_dt = 0.125
            corr_3pt_pion = xp.zeros((gflow_niter + 1, T, T), dtype = complex)
            for it in tqdm.tqdm(range(gflow_niter + 1), desc = "Flow"):
                gflow_params = {"dt": gflow_dt, "niter": it}
                gf_tau = it * gflow_dt
                Su_fs = ut.prop_fwd_flow(U_with_phase, gflow_params, Su_ps)
                Phi = cr.Propagator(geometry)
                Phi.field = contract("CB, txyzBAba -> txyzCAba", cr.Gamma(5).mat, Su_ps.field)
                for tsep in range(T):
                    Phi_t = Phi.keep_one_time_slice(tsep)
                    Seq_ps = ut.propagator_parallelized(Q, inv_params, Phi_t, 'd')
                    Seq_fs = ut.prop_fwd_flow(U_with_phase, gflow_params, Seq_ps)
                    if it == 0 and tsep == 5: Seq_fs.save('Seq_it_%d_tsep%d.npy' % (it, tsep))
                    if it == 1 and tsep == 5: Seq_fs.save('Seq_it_%d_tsep%d.npy' % (it, tsep))
                    corr_3pt_pion_space = contract('txyzBAba, BC, CD, txyzDAba -> txyz', xp.conjugate(Seq_fs.field), cr.Gamma(5).mat, cr.Gamma(0).mat, Su_fs.field)
                    for tins in range(tsep + 1):
                        corr_3pt_pion[it, tsep, tins] = xp.sum(corr_3pt_pion_space[tins])
            xp.save("../data/test/corr_3pt_pion_conf_%d_g4.npy" % (conf), corr_3pt_pion)

        #%%
        # 3pt with non-local insertions
        mu_num2st = {0: ['t', '-t'], 1: ['x', '-x'], 2: ['y', '-y'], 3: ['z', '-z']}
        gflow_niter = 1
        gflow_dt = 0.125
        N_der = 3
        corr_3pt_pdf_pion = []
        for d in range(N_der + 1):
            # Ngf x (mu x mu ...) x (d x d x ...) * tsep x tins
            # d = 0 means fwd (+1), d = 1 means bwd (-1)
            shape = (gflow_niter + 1,) + (4,) * (d + 1) + (2,) * d + (T, T)
            corr_3pt_pdf_pion.append(xp.zeros(shape, dtype = complex))

        color_idx = "bcdefg"
        def contract_pdf_term(Seq_fs, Su_fs, U_f, mu1, mu_list, fb_list):
            link_fields = []
            Su_shifted = Su_fs
            U_shifted = U_f
            for mu, fb in zip(mu_list, fb_list):
                direction = mu_num2st[mu][fb]
                link_fields.append(U_shifted.mu(direction).field)
                Su_shifted = Su_shifted.shift(direction)
                U_shifted = U_shifted.shift(direction)
            n_links = len(link_fields)
            color_chain = color_idx[:n_links + 1]
            link_subscripts = ["txyz%s%s" % (color_chain[i], color_chain[i + 1]) for i in range(n_links)]
            Su_subscript = "txyzDA%sa" % color_chain[n_links]
            subscript = ", ".join(["txyzBAba", "BC", "CD"] + link_subscripts + [Su_subscript]) + " -> txyz"
            sign = (-1) ** sum(fb_list)
            return sign * contract(subscript, xp.conjugate(Seq_fs.field), cr.Gamma(5).mat, cr.Gamma(mu1).mat, *link_fields, Su_shifted.field)

        # For flowed gauge
        gflow_params = {"dt": gflow_dt, "niter": gflow_niter}
        gflow = GFlow(U_with_phase, cr.Fermion(geometry), gflow_params)
        gflow.forward()
        for it in tqdm.tqdm(range(gflow_niter + 1), desc = "Flow"):
            gflow_params = {"dt": gflow_dt, "niter": it}
            gf_tau = it * gflow_dt
            U_f = gflow.U_list[it]
            Su_fs = ut.prop_fwd_flow(U_with_phase, gflow_params, Su_ps)
            Phi = cr.Propagator(geometry)
            Phi.field = contract("CB, txyzBAba -> txyzCAba", cr.Gamma(5).mat, Su_ps.field)
            for tsep in range(T):
                Phi_t = Phi.keep_one_time_slice(tsep)
                Seq_ps = ut.propagator_parallelized(Q, inv_params, Phi_t, 'd')
                Seq_fs = ut.prop_fwd_flow(U_with_phase, gflow_params, Seq_ps)
                # m = 0
                if N_der >= 0:
                    for mu1 in range(4):
                        corr_3pt_pdf_pion_space = contract('txyzBAba, BC, CD, txyzDAba -> txyz', xp.conjugate(Seq_fs.field), cr.Gamma(5).mat, cr.Gamma(mu1).mat, Su_fs.field)
                        for tins in range(tsep + 1): corr_3pt_pdf_pion[0][it, mu1, tsep, tins] = xp.sum(corr_3pt_pdf_pion_space[tins])
                # m = 1
                if N_der >= 1:
                    for mu1 in range(4):
                        for mu2 in range(4):
                            [fwdmu2, bwdmu2] = mu_num2st[mu2]
                            corr_3pt_pdf_pion_space = contract('txyzBAba, BC, CD, txyzbc, txyzDAca -> txyz', xp.conjugate(Seq_fs.field), cr.Gamma(5).mat, cr.Gamma(mu1).mat, U_f.mu(fwdmu2).field, Su_fs.shift(fwdmu2).field)
                            for tins in range(tsep + 1): corr_3pt_pdf_pion[1][it, mu1, mu2, 0, tsep, tins] = xp.sum(corr_3pt_pdf_pion_space[tins])
                            corr_3pt_pdf_pion_space = - contract('txyzBAba, BC, CD, txyzbc, txyzDAca -> txyz', xp.conjugate(Seq_fs.field), cr.Gamma(5).mat, cr.Gamma(mu1).mat, U_f.mu(bwdmu2).field, Su_fs.shift(bwdmu2).field)
                            for tins in range(tsep + 1): corr_3pt_pdf_pion[1][it, mu1, mu2, 1, tsep, tins] = xp.sum(corr_3pt_pdf_pion_space[tins])
                # m = 2
                if N_der >= 2:
                    for mu1 in range(4):
                        for mu2 in range(4):
                            for mu3 in range(4):
                                [fwdmu2, bwdmu2] = mu_num2st[mu2]
                                [fwdmu3, bwdmu3] = mu_num2st[mu3]
                                corr_3pt_pdf_pion_space = contract('txyzBAba, BC, CD, txyzbc, txyzcd, txyzDAda -> txyz', xp.conjugate(Seq_fs.field), cr.Gamma(5).mat, cr.Gamma(mu1).mat, U_f.mu(fwdmu2).field, U_f.shift(fwdmu2).mu(fwdmu3).field, Su_fs.shift(fwdmu2).shift(fwdmu3).field)
                                for tins in range(tsep + 1): corr_3pt_pdf_pion[2][it, mu1, mu2, mu3, 0, 0, tsep, tins] = xp.sum(corr_3pt_pdf_pion_space[tins])
                                corr_3pt_pdf_pion_space = - contract('txyzBAba, BC, CD, txyzbc, txyzcd, txyzDAda -> txyz', xp.conjugate(Seq_fs.field), cr.Gamma(5).mat, cr.Gamma(mu1).mat, U_f.mu(fwdmu2).field, U_f.shift(fwdmu2).mu(bwdmu3).field, Su_fs.shift(fwdmu2).shift(bwdmu3).field)
                                for tins in range(tsep + 1): corr_3pt_pdf_pion[2][it, mu1, mu2, mu3, 0, 1, tsep, tins] = xp.sum(corr_3pt_pdf_pion_space[tins])
                                corr_3pt_pdf_pion_space = - contract('txyzBAba, BC, CD, txyzbc, txyzcd, txyzDAda -> txyz', xp.conjugate(Seq_fs.field), cr.Gamma(5).mat, cr.Gamma(mu1).mat, U_f.mu(bwdmu2).field, U_f.shift(bwdmu2).mu(fwdmu3).field, Su_fs.shift(bwdmu2).shift(fwdmu3).field)
                                for tins in range(tsep + 1): corr_3pt_pdf_pion[2][it, mu1, mu2, mu3, 1, 0, tsep, tins] = xp.sum(corr_3pt_pdf_pion_space[tins])
                                corr_3pt_pdf_pion_space = contract('txyzBAba, BC, CD, txyzbc, txyzcd, txyzDAda -> txyz', xp.conjugate(Seq_fs.field), cr.Gamma(5).mat, cr.Gamma(mu1).mat, U_f.mu(bwdmu2).field, U_f.shift(bwdmu2).mu(bwdmu3).field, Su_fs.shift(bwdmu2).shift(bwdmu3).field)
                                for tins in range(tsep + 1): corr_3pt_pdf_pion[2][it, mu1, mu2, mu3, 1, 1, tsep, tins] = xp.sum(corr_3pt_pdf_pion_space[tins])
                # m = 3, 4, 5
                for d in range(3, N_der + 1):
                    for mu1 in range(4):
                        for mu_list in np.ndindex(*(4,) * d):
                            for fb_list in np.ndindex(*(2,) * d):
                                corr_3pt_pdf_pion_space = contract_pdf_term(Seq_fs, Su_fs, U_f, mu1, mu_list, fb_list)
                                for tins in range(tsep + 1):
                                    corr_3pt_pdf_pion[d][(it, mu1) + mu_list + fb_list + (tsep, tins)] = xp.sum(corr_3pt_pdf_pion_space[tins])
        for d in range(N_der + 1):
            xp.save("../data/test/corr_3pt_pion_conf_%d_Nder_%d.npy" % (conf, d), corr_3pt_pdf_pion[d])
