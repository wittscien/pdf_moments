import sys
from pathlib import Path
import numpy as np
import gvar as gv
import funcs as tp
import inputs

def quantities(params, relen, result):
    relen = params['relen']
    tech = params['tech']
    L = params['L']
    mone_matrix = np.zeros([relen, 2])
    for i in range(relen):
        mone_matrix[i] = [result['pion'][i][1], result['kaon'][i][1]]
    mone_mean = tp.cal_mean(mone_matrix)
    mone_err = tp.cal_err(mone_matrix,tech)
    mone = gv.gvar(mone_mean * params['hca'], mone_err * params['hca'])
    #print(2 * np.sqrt(mpi**2+(2*np.pi/L)**2) + np.sqrt(mpi**2+2*(2*np.pi/L)**2))

    # mpi L
    mpiL_matrix = np.zeros(relen)
    for ls in range(relen):
        mpiL_matrix[ls] = result['pion'][ls][1] * params['L']
    mpiL_mean = tp.cal_mean(mpiL_matrix)
    mpiL_err = tp.cal_err(mpiL_matrix,tech)
    mpiL = gv.gvar(mpiL_mean,mpiL_err)
    print('\nPhysical quantities')
    print('-------------------')
    print('m_pi   = %s MeV' % mone[0])
    print('m_K    = %s MeV' % mone[1])
    print('m_pi L = %s\n' % mpiL)

    # For others
    # Path('../%s/levels/'%(params['datadir']['mydata'])).mkdir(parents=True, exist_ok=True)
    # one_matrix = np.zeros((relen, 7))
    # for ls in range(relen):
    #     one_matrix[ls, :6] = np.array([result['pion_0'][ls][1], result['K_0'][ls][1], result['eta_0'][ls][0], result['D_0'][ls][1], result['Ds_0'][ls][1], result['Dst_0'][ls][1]])
    #     one_matrix[ls, 6] = result['pion_0'][ls][1] * params['L']
    # one_mean = tp.cal_mean(one_matrix)
    # one_err = tp.cal_err(one_matrix, tech)
    # # Writing to txt
    # with open('../%s/levels/results_one_%s_%s.txt' % (params['datadir']['mydata'],params['ensname'],params['tech']), 'w') as txt_file:
    #     txt_file.write("m_pi m_K m_eta m_D m_Ds m_Dst mpiL\n")
    #     for i in range(one_matrix.shape[1]):
    #         txt_file.write("%s " % (repr(gv.gvar(one_mean[i], one_err[i]))))
    #     txt_file.write("\n")
    #     for ls in range(relen):
    #         txt_file.write("%d " % (ls))
    #         for i in range(one_matrix.shape[1]):
    #             txt_file.write("%f " % (one_matrix[ls, i]))
    #         txt_file.write("\n")
