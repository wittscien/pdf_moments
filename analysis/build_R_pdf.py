import numpy as np
import funcs as tp
from flow_matching import c_numeric



def build_R_pdf(params, data2, data3, metadata, result):
    T = params['T']
    relen = params['relen']

    nflow = len(metadata['tau_list']) - 1

    R = {}
    x = {}
    for tsep in params['tsep_list']:
        x[tsep] = np.arange(tsep + 1) - tsep // 2
    for k in params['key_3pt']:
        R[k] = {}
        for n in range(3, 7):
            R[k][n] = {}
            for tf in range(nflow + 1):
                R[k][n][tf] = {}
                for itsep, tsep in enumerate(params['tsep_list']):
                    R[k][n][tf][tsep] = np.zeros((relen, tsep+1))

    for k in params['key_3pt']:
        if k == 'pion':
            k_one = 'pion'
        elif k in ['kaon', 'kaon_s']:
            k_one = 'kaon'

        for n in range(3, 7):
            for tf in range(nflow + 1):
                tf_GeV2 = tf * metadata['flow_dt'] * params['hca'] ** 2

                for itsep, tsep in enumerate(params['tsep_list']):
                    x = np.arange(tsep+1) - tsep / 2
                    for ls in range(relen):
                        result_use = {k_one: {ls: [0, tp.cal_mass(data2[k_one][ls].real,mtype='cosh',tau=params['tau'])[T//2]]}} if result is None else result
                        m_one = result_use[k_one][ls][1]

                        data_3pt_up = data3['%s-PDF-n_%d' % (k, n)][tsep][:,tf,:]
                        data_3pt_down = data3['%s-PDF-n_2' % (k)][tsep][:,tf,:] * (- m_one) ** (n - 1)
                        if tf != 0:
                            data_3pt_up *= c_numeric(2, tf_GeV2, 2)
                            data_3pt_down *= c_numeric(n + 1, tf_GeV2, 2)

                        data_3pt_up_tsep = data_3pt_up[ls]
                        data_3pt_down_tsep = data_3pt_down[ls]
                        R[k][n][tf][tsep][ls] = data_3pt_up_tsep / data_3pt_down_tsep

    return x, R
