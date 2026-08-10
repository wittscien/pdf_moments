import numpy as np
import funcs as tp
from flow_matching import c_numeric



def build_R_pdf(params, data2, data3, metadata, result):
    T = params['T']
    relen = params['relen']

    nflow = len(metadata['tau_list']) - 1

    matching_coeffs = {}
    for tf in range(1, nflow + 1):
        tf_GeV2 = tf * metadata['flow_dt'] * params['hca'] ** 2
        matching_coeffs[tf] = {
            matching_order: c_numeric(matching_order, tf_GeV2, 2)
            for matching_order in [2, 4, 5, 6, 7]
        }

    R = {}
    x = {}
    for tsep in params['tsep_list']:
        x[tsep] = np.arange(tsep + 1) - tsep // 2
    for k in params['key_3pt']:
        R[k] = {}
        for moment in range(3, 7):
            R[k][moment] = {}
            for tf in range(nflow + 1):
                R[k][moment][tf] = {}
                for itsep, tsep in enumerate(params['tsep_list']):
                    R[k][moment][tf][tsep] = np.zeros((relen, tsep+1))

    for k in params['key_3pt']:
        if k == 'pion':
            k_one = 'pion'
        elif k in ['kaon', 'kaon_s']:
            k_one = 'kaon'

        for moment in range(3, 7):
            for tf in range(nflow + 1):
                tf_GeV2 = tf * metadata['flow_dt'] * params['hca'] ** 2

                for itsep, tsep in enumerate(params['tsep_list']):
                    for ls in range(relen):
                        m_one = tp.cal_mass(data2[k_one][ls].real,mtype='cosh',tau=params['tau'])[T//2] if result is None else result[k_one][ls][1]

                        data_3pt_up = np.array(data3['%s-PDF-n_%d' % (k, moment)][tsep][:,tf,:], copy=True)
                        data_3pt_down = np.array(data3['%s-PDF-n_2' % (k)][tsep][:,tf,:] * (- m_one) ** (moment - 2), copy=True)
                        if tf != 0:
                            data_3pt_up *= matching_coeffs[tf][2]
                            data_3pt_down *= matching_coeffs[tf][moment + 1]

                        data_3pt_up_tsep = data_3pt_up[ls]
                        data_3pt_down_tsep = data_3pt_down[ls]
                        R[k][moment][tf][tsep][ls] = data_3pt_up_tsep / data_3pt_down_tsep

    return x, R
