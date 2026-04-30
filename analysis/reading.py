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



def reading_2_parallel(read2,conf,datadir,params):
    atime = time.time()
    nsrc = params['nsrc']
    L = params['L']
    T = params['T']
    V = L ** 3
    data2 = {}
    if 'pion' in params['key_2pt']:
        file_conf = '../%s/corr_conf/%s/data_2pt_%s_pion_%d.pckl'%(datadir['mydata'],params['ensemble'],params['ensemble'],conf)
        if read2 == 'fastconf' and Path(file_conf).is_file():
            print('fastconf: Loading %s'%(file_conf))
            data2_par = tp.fast_read(file_conf)
        else:
            data2_par = {}
            data2_par['pion'] = np.zeros(T, dtype=complex)
            data2_par['pion-direct'] = np.zeros(T, dtype=complex)
            for isrc in range(nsrc):
                file = "../%s/%s/corr_2pt_pion_conf_%d.npy" % (datadir['2pt'], params['ensemble'], conf)
                direct = np.load(file)
                data2_par['pion-direct'] += direct * V / nsrc
            data2_par['pion'] = data2_par['pion-direct']
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
    if 'pion' in params['key_3pt']:
        file_conf = '../%s/corr_conf/%s/data_3pt_%s_pion_%d.pckl'%(datadir['mydata'],params['ensemble'],params['ensemble'],conf)
        if read3 == 'fastconf' and Path(file_conf).is_file():
            print('fastconf: Loading %s'%(file_conf))
            data3_par = tp.fast_read(file_conf)
        else:
            data3_par = {}
            tsnk_max_3pt = params['tsnk_max_3pt']['pion']
            # Original data
            for d in range(nder + 1):
                shape = (nflow + 1,) + (4,) * (d + 1) + (2,) * d + (tsnk_max_3pt, tsnk_max_3pt)
                for i in range(1):
                    data3_par['pion-nder_%d_diag_%d' % (d, i)] = np.zeros(shape, dtype=complex)
                for isrc in range(nsrc):
                    file = "../%s/%s/corr_3pt_pion_conf_%d_Nder_%d.npy" % (datadir['3pt'], params['ensemble'], conf, d)
                    direct = np.load(file)
                    for i in range(1):
                        data3_par['pion-nder_%d_diag_%d' % (d, i)] += direct * V / nsrc

            # Combine into operators with proper covariant derivatives, I call them cov
            # m = 0
            data3_par['pion-cov-nder_0'] = data3_par['pion-nder_0_diag_0']
            # m = 1
            data3_par['pion-cov-nder_1'] = np.zeros((nflow + 1, 4, 4, tsnk_max_3pt, tsnk_max_3pt), dtype = complex)
            for mu1 in range(4):
                for mu2 in range(4):
                    wt = [mu2].count(0)
                    for d2 in [-1, 1]:
                        d2ind = 0 if d2 == 1 else 1
                        wp = (mu2 == 0) * (d2 == 1)
                        for k in range(wt + 1):
                            data3_par['pion-cov-nder_1'][:,mu1,mu2,:,:] += math.comb(wt, k) * np.roll(data3_par['pion-nder_1_diag_0'][:,mu1,mu2,d2ind,:,:], wp - k, axis = -1) / 2 ** wt
            data3_par['pion-cov-nder_1'] /= 2 ** (2 - 1)
            # m = 2
            data3_par['pion-cov-nder_2'] = np.zeros((nflow + 1, 4, 4, 4, tsnk_max_3pt, tsnk_max_3pt), dtype = complex)
            for mu1 in range(4):
                for mu2 in range(4):
                    for mu3 in range(4):
                        wt = [mu2, mu3].count(0)
                        for d2 in [-1, 1]:
                            for d3 in [-1, 1]:
                                d2ind = 0 if d2 == 1 else 1
                                d3ind = 0 if d3 == 1 else 1
                                wp = (mu2 == 0) * (d2 == 1) + (mu3 == 0) * (d3 == 1)
                                for k in range(wt + 1):
                                    data3_par['pion-cov-nder_2'][:,mu1,mu2,mu3,:,:] += math.comb(wt, k) * np.roll(data3_par['pion-nder_2_diag_0'][:,mu1,mu2,mu3,d2ind,d3ind,:,:], wp - k, axis = -1) / 2 ** wt
            data3_par['pion-cov-nder_2'] /= 2 ** (3 - 1)

            # Combine into operators relevant to PDFs
            data3_par['pion-PDF-n_2'] = data3_par['pion-cov-nder_1'][:,0,0,:,:]
            for mu in [1,2,3]:
                data3_par['pion-PDF-n_2'] -= data3_par['pion-cov-nder_1'][:,mu,mu,:,:] / 3
            data3_par['pion-PDF-n_3'] = data3_par['pion-cov-nder_2'][:,0,0,0,:,:]
            for mu in [1,2,3]:
                data3_par['pion-PDF-n_3'] -= (data3_par['pion-cov-nder_2'][:,mu,mu,0,:,:] + data3_par['pion-cov-nder_2'][:,mu,0,mu,:,:] + data3_par['pion-cov-nder_2'][:,0,mu,mu,:,:]) / 3

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
