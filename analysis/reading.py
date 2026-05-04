import os
from multiprocessing import Pool
from pathlib import Path
import time
import numpy as np
import h5py as h5
import tqdm
import funcs as tp
import inputs
import math
import itertools


def combine_covariant_derivative(data, der):
    # data has shape:
    # [flow, mu1, ..., mu_{der+1}, d2, ..., d_{der+1}, tsep, tins].
    # The direction index convention is 0 -> d = +1 and 1 -> d = -1.
    if der == 0:
        return data.copy()

    # The covariant-derivative result keeps the flow, Lorentz, tsep, and tins axes,
    # but sums over all derivative direction axes.
    cov_shape = data.shape[:der + 2] + data.shape[-2:]
    cov_data = np.zeros(cov_shape, dtype=data.dtype)
    directions = tuple(np.ndindex(*(2,) * der))

    for mu_indices in np.ndindex(*(4,) * (der + 1)):
        derivative_mus = mu_indices[1:]
        wt = derivative_mus.count(0)
        cov_index = (slice(None),) + mu_indices + (slice(None), slice(None))

        for direction_indices in directions:
            # w_plus counts time derivatives whose direction is +1.
            wp = sum((mu == 0) and (direction_index == 0)
                     for mu, direction_index in zip(derivative_mus, direction_indices))
            data_index = (slice(None),) + mu_indices + direction_indices + (slice(None), slice(None))
            for k in range(wt + 1):
                # np.roll(a, wp-k)[tau] gives a[tau - wp + k], matching
                # t_y - w_plus + k in the formula.
                cov_data[cov_index] += math.comb(wt, k) * np.roll(data[data_index], wp - k, axis=-1) / 2 ** wt

    # der = n - 1, so this is the overall 1 / 2^(n-1) factor.
    cov_data /= 2 ** der
    return cov_data


def symmetrized_operator(data, indices):
    # Braces in the table mean normalized symmetrization over unique permutations.
    perms = set(itertools.permutations(indices))
    result = None
    for perm in perms:
        # Build data[:, perm[0], perm[1], ..., perm[-1], :, :].
        # The first ':' is flow time; the final two ':' axes are tsep and tins.
        term = data[(slice(None),) + tuple(perm) + (slice(None), slice(None))]
        if result is None:
            result = term.copy()
        else:
            result += term
    result /= len(perms)
    return result


def add_covariant_and_pdf_operators(data3_par, hadron, nder):
    # Build <hadron>-cov-nder_d from <hadron>-nder_d_diag_0 for every
    # derivative order that was read from disk.
    for d in range(nder + 1):
        data3_par['%s-cov-nder_%d' % (hadron, d)] = combine_covariant_derivative(data3_par['%s-nder_%d_diag_0' % (hadron, d)], d)

    # Combine into operators relevant to PDFs. The same traceless basis is used
    # for pion and kaon; only the hadron key prefix changes.
    # Need <hadron>-cov-nder_1 to construct n = 2.
    if nder >= 1:
        # n = 2: O44 - 1/3 sum_i Oii.
        data3_par['%s-PDF-n_2' % hadron] = data3_par['%s-cov-nder_1' % hadron][:,0,0,:,:].copy()
        for mu in [1,2,3]:
            data3_par['%s-PDF-n_2' % hadron] -= data3_par['%s-cov-nder_1' % hadron][:,mu,mu,:,:] / 3
    # Need <hadron>-cov-nder_2 to construct n = 3.
    if nder >= 2:
        # n = 3: O444 - sum_i O_{ii4}.
        data3_par['%s-PDF-n_3' % hadron] = data3_par['%s-cov-nder_2' % hadron][:,0,0,0,:,:].copy()
        for mu in [1,2,3]:
            data3_par['%s-PDF-n_3' % hadron] -= (data3_par['%s-cov-nder_2' % hadron][:,mu,mu,0,:,:] + data3_par['%s-cov-nder_2' % hadron][:,mu,0,mu,:,:] + data3_par['%s-cov-nder_2' % hadron][:,0,mu,mu,:,:]) / 3
    # Need <hadron>-cov-nder_3 to construct n = 4.
    if nder >= 3:
        # n = 4: O4444 - 2 sum_i O_{ii44} + 1/5 sum_i Oiiii
        #        + 2/5 sum_{i<j} O_{iijj}.
        cov = data3_par['%s-cov-nder_3' % hadron]
        data3_par['%s-PDF-n_4' % hadron] = cov[:,0,0,0,0,:,:].copy()
        for mu in [1,2,3]:
            data3_par['%s-PDF-n_4' % hadron] += cov[:,mu,mu,mu,mu,:,:] / 5
            data3_par['%s-PDF-n_4' % hadron] -= 2 * symmetrized_operator(cov, (mu, mu, 0, 0))
        for mu1 in [1,2,3]:
            for mu2 in range(mu1 + 1, 4):
                data3_par['%s-PDF-n_4' % hadron] += 2 * symmetrized_operator(cov, (mu1, mu1, mu2, mu2)) / 5
    # Need <hadron>-cov-nder_4 to construct n = 5.
    if nder >= 4:
        # n = 5: O44444 - 10/3 sum_i O_{ii444}
        #        + sum_i O_{iiii4} + 2 sum_{i<j} O_{iijj4}.
        cov = data3_par['%s-cov-nder_4' % hadron]
        data3_par['%s-PDF-n_5' % hadron] = cov[:,0,0,0,0,0,:,:].copy()
        for mu in [1,2,3]:
            data3_par['%s-PDF-n_5' % hadron] += symmetrized_operator(cov, (mu, mu, mu, mu, 0))
            data3_par['%s-PDF-n_5' % hadron] -= 10 * symmetrized_operator(cov, (mu, mu, 0, 0, 0)) / 3
        for mu1 in [1,2,3]:
            for mu2 in range(mu1 + 1, 4):
                data3_par['%s-PDF-n_5' % hadron] += 2 * symmetrized_operator(cov, (mu1, mu1, mu2, mu2, 0))
    # Need <hadron>-cov-nder_5 to construct n = 6.
    if nder >= 5:
        # n = 6: O444444 - 5 sum_i O_{ii4444}
        #        + 3 sum_i O_{iiii44} + 6 sum_{i<j} O_{iijj44}
        #        - 1/7 sum_i Oiiiiii - 3/7 sum_{i<j} O_{iiiijj}
        #        - 6/7 O_{112233}.
        cov = data3_par['%s-cov-nder_5' % hadron]
        data3_par['%s-PDF-n_6' % hadron] = cov[:,0,0,0,0,0,0,:,:].copy()
        data3_par['%s-PDF-n_6' % hadron] -= 6 * symmetrized_operator(cov, (1,1,2,2,3,3)) / 7
        for mu in [1,2,3]:
            data3_par['%s-PDF-n_6' % hadron] -= cov[:,mu,mu,mu,mu,mu,mu,:,:] / 7
            data3_par['%s-PDF-n_6' % hadron] += 3 * symmetrized_operator(cov, (mu, mu, mu, mu, 0, 0))
            data3_par['%s-PDF-n_6' % hadron] -= 5 * symmetrized_operator(cov, (mu, mu, 0, 0, 0, 0))
        for mu1 in [1,2,3]:
            for mu2 in range(mu1 + 1, 4):
                data3_par['%s-PDF-n_6' % hadron] -= 3 * symmetrized_operator(cov, (mu1, mu1, mu1, mu1, mu2, mu2)) / 7
                data3_par['%s-PDF-n_6' % hadron] += 6 * symmetrized_operator(cov, (mu1, mu1, mu2, mu2, 0, 0))



def reading_2_parallel(read2,conf,datadir,params):
    atime = time.time()
    nsrc = params['nsrc']
    L = params['L']
    T = params['T']
    V = L ** 3
    data2 = {}
    for hadron in ['pion', 'kaon']:
        if hadron not in params['key_2pt']:
            continue
        file_conf = '../%s/corr_conf/%s/data_2pt_%s_%s_%d.pckl'%(datadir['mydata'],params['ensemble'],params['ensemble'],hadron,conf)
        if read2 == 'fastconf' and Path(file_conf).is_file():
            print('fastconf: Loading %s'%(file_conf))
            data2_par = tp.fast_read(file_conf)
        else:
            data2_par = {}
            data2_par[hadron] = np.zeros(T, dtype=complex)
            data2_par['%s-direct' % hadron] = np.zeros(T, dtype=complex)
            for isrc in range(nsrc):
                file = "../%s/%s/corr_2pt_%s_conf_%d.npy" % (datadir['2pt'], params['ensemble'], hadron, conf)
                direct = np.load(file)
                data2_par['%s-direct' % hadron] += direct * V / nsrc
            data2_par[hadron] = data2_par['%s-direct' % hadron]
            tp.write_data(file_conf, data2_par)
        data2.update(data2_par)

    print('2pt time, conf = %d: %.0f s'%(conf, time.time() - atime), flush=True)

    return data2


def reading_3_parallel(read3,conf,datadir,params):
    atime = time.time()
    nsrc = params['nsrc']
    nder = params['nder']
    nflow = params['nflow']
    L = params['L']
    T = params['T']
    V = L ** 3
    data3 = {}
    # [src-snk-sep, indices of ins]
    for hadron in ['pion', 'kaon']:
        if hadron not in params['key_3pt']:
            continue
        file_conf = '../%s/corr_conf/%s/data_3pt_%s_%s_%d.pckl'%(datadir['mydata'],params['ensemble'],params['ensemble'],hadron,conf)
        if read3 == 'fastconf' and Path(file_conf).is_file():
            print('fastconf: Loading %s'%(file_conf))
            data3_par = tp.fast_read(file_conf)
        else:
            data3_par = {}
            tsnk_max_3pt = params['tsnk_max_3pt'][hadron]
            # Original data
            for d in range(nder + 1):
                shape = (nflow + 1,) + (4,) * (d + 1) + (2,) * d + (tsnk_max_3pt, tsnk_max_3pt)
                for i in range(1):
                    data3_par['%s-nder_%d_diag_%d' % (hadron, d, i)] = np.zeros(shape, dtype=complex)
                for isrc in range(nsrc):
                    file = "../%s/%s/corr_3pt_%s_conf_%d_Nder_%d.npy" % (datadir['3pt'], params['ensemble'], hadron, conf, d)
                    direct = np.load(file)
                    for i in range(1):
                        data3_par['%s-nder_%d_diag_%d' % (hadron, d, i)] += direct #* V / nsrc

            # Combine into operators with proper covariant derivatives, I call them cov
            # Old explicit implementation for m = 0, 1, 2:
            # # m = 0
            # data3_par['pion-cov-nder_0'] = data3_par['pion-nder_0_diag_0']
            # # m = 1
            # data3_par['pion-cov-nder_1'] = np.zeros((nflow + 1, 4, 4, tsnk_max_3pt, tsnk_max_3pt), dtype = complex)
            # for mu1 in range(4):
            #     for mu2 in range(4):
            #         wt = [mu2].count(0)
            #         for d2 in [-1, 1]:
            #             d2ind = 0 if d2 == 1 else 1
            #             wp = (mu2 == 0) * (d2 == 1)
            #             for k in range(wt + 1):
            #                 data3_par['pion-cov-nder_1'][:,mu1,mu2,:,:] += math.comb(wt, k) * np.roll(data3_par['pion-nder_1_diag_0'][:,mu1,mu2,d2ind,:,:], wp - k, axis = -1) / 2 ** wt
            # data3_par['pion-cov-nder_1'] /= 2 ** (2 - 1)
            # # m = 2
            # data3_par['pion-cov-nder_2'] = np.zeros((nflow + 1, 4, 4, 4, tsnk_max_3pt, tsnk_max_3pt), dtype = complex)
            # for mu1 in range(4):
            #     for mu2 in range(4):
            #         for mu3 in range(4):
            #             wt = [mu2, mu3].count(0)
            #             for d2 in [-1, 1]:
            #                 for d3 in [-1, 1]:
            #                     d2ind = 0 if d2 == 1 else 1
            #                     d3ind = 0 if d3 == 1 else 1
            #                     wp = (mu2 == 0) * (d2 == 1) + (mu3 == 0) * (d3 == 1)
            #                     for k in range(wt + 1):
            #                         data3_par['pion-cov-nder_2'][:,mu1,mu2,mu3,:,:] += math.comb(wt, k) * np.roll(data3_par['pion-nder_2_diag_0'][:,mu1,mu2,mu3,d2ind,d3ind,:,:], wp - k, axis = -1) / 2 ** wt
            # data3_par['pion-cov-nder_2'] /= 2 ** (3 - 1)
            add_covariant_and_pdf_operators(data3_par, hadron, nder)

            tp.write_data(file_conf, data3_par)
        data3.update(data3_par)

    print('3pt time, conf = %d: %.0f s'%(conf, time.time() - atime), flush=True)

    return data3


def reading(params, read2, read3):
    datadir = params['datadir']
    T = params['T']
    N = params['N']
    tau = params['tau']
    data2 = {}
    data3 = {}

    Path("../%s/corr_conf/%s/"%(params['datadir']['mydata'],params['ensemble'])).mkdir(parents=True, exist_ok=True)

    if read2 in ['direct', 'fastconf']:
        atime = time.time()
        with Pool(processes = os.cpu_count()) as pool:
            pool_result = pool.starmap(reading_2_parallel, [(read2,conf,datadir,params) for conf in params['confs']])
        for k in pool_result[0].keys():
            data2[k] = np.zeros([N, T],dtype=complex)
        for i in range(len(pool_result)):
            for k in pool_result[i].keys():
                data2[k][i] = pool_result[i][k]
        print('2pt all time:', time.time() - atime, flush=True)
        # Write data
        wfile = '../%s/corr/%s/data_2pt_%s.pckl'%(datadir['mydata'],params['ensemble'],params['ensemble'])
        tp.write_data(wfile,data2)

    elif read2 == 'fast':
        # Fast read
        rfile = '../%s/corr/%s/data_2pt_%s.pckl'%(datadir['mydata'],params['ensemble'],params['ensemble'])
        data2 = tp.fast_read(rfile)
        # Truancate the data to limited confs
        for k in data2.keys():
            data2[k] = data2[k][:len(params['confs'])]

    if read3 in ['direct', 'fastconf']:
        atime = time.time()
        with Pool(processes = os.cpu_count()) as pool:
            pool_result = pool.starmap(reading_3_parallel, [(read3,conf,datadir,params) for conf in params['confs']])
        keys = pool_result[0].keys()
        for k in keys:
            data3[k] = np.zeros((N,) + pool_result[0][k].shape, dtype = complex)
            for i in range(len(pool_result)):
                data3[k][i] = pool_result[i][k]
        print('3pt all time:', time.time() - atime, flush=True)
        # Write data
        wfile = '../%s/corr/%s/data_3pt_%s.pckl'%(datadir['mydata'],params['ensemble'],params['ensemble'])
        tp.write_data(wfile,data3)

    elif read3 == 'fast':
        # Fast read
        rfile = '../%s/corr/%s/data_3pt_%s.pckl'%(datadir['mydata'],params['ensemble'],params['ensemble'])
        data3 = tp.fast_read(rfile)
        # Truancate the data to limited confs
        for k in data3.keys():
            data3[k] = data3[k][:len(params['confs'])]

    return data2, data3
