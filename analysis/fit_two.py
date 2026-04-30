from pathlib import Path
import numpy as np
import gvar as gv
import funcs as tp
import inputs


def fit_two(params, selected, data2, mtype, obj):
    # Parameter structure: A0, E0, A1, E1, ...
    tau = params['tau']
    tech = params['tech']
    fdata2 = {k:data2[k].real for k in data2.keys()}

    one_name = '%s'%(params['ensname'])
    one_tit = r'$Ens=%s$'%(params['ensname'])
    one_dir = '%s/%s'%(params['ensname'],tech)

    # Write priors to txt
    Path('priors/%s/'%(one_dir)).mkdir(parents=True, exist_ok=True)
    prior_txt = open("priors/%s/prior_%s.txt"%(one_dir,one_name), 'w')
    prior_txt.close()

    selected_lazy = {}
    result_para = {}
    result_chi2dof = {}
    result = {}
    result_one = {}
    ans = {}
    for k in fdata2.keys():
        # if k.split('_')[0] == 'eta': continue
        #if k != 'pion_0': continue
        #if k.split('_')[0] != 'pion': continue
        print(k)
        par = k.split('_')[0]
        mom = int(k.split('_')[1])
        prior = {}
        if params['ens'] == 1:
            if (par == 'D'):
                if mom == 0:
                    prior['A_0'] = 47.06
                    prior['A_1'] = 10
                    prior['E_0'] = 0.77156
                    prior['E_1'] = 0.992
                elif mom == 1:
                    prior['A_0'] = 26.73
                    prior['A_1'] = 3.42
                    prior['E_0'] = 0.79354
                    prior['E_1'] = 0.945
                elif mom == 2:
                    prior['A_0'] = 16.50
                    prior['A_1'] = 2.55
                    prior['E_0'] = 0.8148
                    prior['E_1'] = 0.942
                elif mom == 3:
                    prior['A_0'] = 10.21
                    prior['A_1'] = 2.25
                    prior['E_0'] = 0.8358
                    prior['E_1'] = 0.979
                elif mom == 4:
                    prior['A_0'] = 6.46
                    prior['A_1'] = 1.92
                    prior['E_0'] = 0.8573
                    prior['E_1'] = 1.046
            elif (par == 'K'):
                if mom == 0:
                    prior['A_0'] = 47.06
                    prior['A_1'] = 10
                    prior['E_0'] = 0.77156
                    prior['E_1'] = 0.992
                elif mom == 1:
                    prior['A_0'] = 26.73
                    prior['A_1'] = 3.42
                    prior['E_0'] = 0.79354
                    prior['E_1'] = 0.945
                elif mom == 2:
                    prior['A_0'] = 16.50
                    prior['A_1'] = 2.55
                    prior['E_0'] = 0.8148
                    prior['E_1'] = 0.942
                elif mom == 3:
                    prior['A_0'] = 10.21
                    prior['A_1'] = 2.25
                    prior['E_0'] = 0.8358
                    prior['E_1'] = 0.979
                elif mom == 4:
                    prior['A_0'] = 6.46
                    prior['A_1'] = 1.92
                    prior['E_0'] = 0.8573
                    prior['E_1'] = 1.046
            elif (par == 'Ds'):
                if mom == 0:
                    prior['A_0'] = 47.06
                    prior['A_1'] = 10
                    prior['E_0'] = 0.77156
                    prior['E_1'] = 0.992
                elif mom == 1:
                    prior['A_0'] = 26.73
                    prior['A_1'] = 3.42
                    prior['E_0'] = 0.79354
                    prior['E_1'] = 0.945
                elif mom == 2:
                    prior['A_0'] = 16.50
                    prior['A_1'] = 2.55
                    prior['E_0'] = 0.8148
                    prior['E_1'] = 0.942
                elif mom == 3:
                    prior['A_0'] = 10.21
                    prior['A_1'] = 2.25
                    prior['E_0'] = 0.8358
                    prior['E_1'] = 0.979
                elif mom == 4:
                    prior['A_0'] = 6.46
                    prior['A_1'] = 1.92
                    prior['E_0'] = 0.8573
                    prior['E_1'] = 1.046
            elif (par == 'Dst'):
                if mom == 0:
                    prior['A_0'] = 37.3
                    prior['A_1'] = 8.7
                    prior['E_0'] = 0.8137
                    prior['E_1'] = 0.959
                elif mom == 1:
                    prior['A_0'] = 21.7
                    prior['A_1'] = 5.3
                    prior['E_0'] = 0.8325
                    prior['E_1'] = 0.913
                elif mom == 2:
                    prior['A_0'] = 14.4
                    prior['A_1'] = 4.0
                    prior['E_0'] = 0.8529
                    prior['E_1'] = 0.940
                elif mom == 3:
                    prior['A_0'] = 8.7
                    prior['A_1'] = 3.8
                    prior['E_0'] = 0.8707
                    prior['E_1'] = 0.952
                elif mom == 4:
                    prior['A_0'] = 6.89
                    prior['A_1'] = 3.01
                    prior['E_0'] = 0.89
                    prior['E_1'] = 1.116
            elif (par == 'pion'):
                if mom == 0:
                    prior['A_0'] = 146
                    prior['A_1'] = 50
                    prior['E_0'] = 0.119
                    prior['E_1'] = 0.46
                elif mom == 1:
                    prior['A_0'] = 46.72
                    prior['A_1'] = 7.2
                    prior['E_0'] = 0.22930
                    prior['E_1'] = 0.424
                elif mom == 2:
                    prior['A_0'] = 24.18
                    prior['A_1'] = 6
                    prior['E_0'] = 0.3025
                    prior['E_1'] = 0.61
                elif mom == 3:
                    prior['A_0'] = 13.44
                    prior['A_1'] = 5
                    prior['E_0'] = 0.3601
                    prior['E_1'] = 0.72
                elif mom == 4:
                    prior['A_0'] = 7.88
                    prior['A_1'] = 4
                    prior['E_0'] = 0.3986
                    prior['E_1'] = 1
            elif (par == 'etac'):
                if mom == 0:
                    prior['A_0'] = 52.282
                    prior['A_1'] = 8.7
                    prior['E_0'] = 1.222160
                    prior['E_1'] = 1.546
                elif mom == 1:
                    prior['A_0'] = 27.606
                    prior['A_1'] = 3.65
                    prior['E_0'] = 1.235611
                    prior['E_1'] = 1.487
                elif mom == 2:
                    prior['A_0'] = 17.428
                    prior['A_1'] = 4.05
                    prior['E_0'] = 1.248874
                    prior['E_1'] = 1.502
                elif mom == 3:
                    prior['A_0'] = 11.213
                    prior['A_1'] = 3.49
                    prior['E_0'] = 1.261952
                    prior['E_1'] = 1.507
                elif mom == 4:
                    prior['A_0'] = 7.213
                    prior['A_1'] = 3.29
                    prior['E_0'] = 1.274548
                    prior['E_1'] = 1.5272
            elif (par == 'jpsi'):
                if mom == 0:
                    prior['A_0'] = 35
                    prior['A_1'] = 2.72
                    prior['E_0'] = 1.25
                    prior['E_1'] = 1.5
                elif mom == 1:
                    prior['A_0'] = 28.651
                    prior['A_1'] = 2.17
                    prior['E_0'] = 1.26496
                    prior['E_1'] = 1.501
                elif mom == 2:
                    prior['A_0'] = 18.563
                    prior['A_1'] = 2.69
                    prior['E_0'] = 1.27783
                    prior['E_1'] = 1.516
                elif mom == 3:
                    prior['A_0'] = 12.208
                    prior['A_1'] = 2.57
                    prior['E_0'] = 1.29053
                    prior['E_1'] = 1.523
                elif mom == 4:
                    prior['A_0'] = 8.018
                    prior['A_1'] = 2.76
                    prior['E_0'] = 1.30283
                    prior['E_1'] = 1.553
        elif params['ens'] == 2:
            if (par == 'D'):
                if mom == 0:
                    prior['A_0'] = 97.76
                    prior['A_1'] = 11.4
                    prior['E_0'] = 0.77172
                    prior['E_1'] = 1.018
                elif mom == 1:
                    prior['A_0'] = 64.31
                    prior['A_1'] = 6.0
                    prior['E_0'] = 0.78176
                    prior['E_1'] = 0.997
                elif mom == 2:
                    prior['A_0'] = 46.63
                    prior['A_1'] = 5.25
                    prior['E_0'] = 0.79171
                    prior['E_1'] = 0.997
                elif mom == 3:
                    prior['A_0'] = 34.25
                    prior['A_1'] = 5.23
                    prior['E_0'] = 0.80154
                    prior['E_1'] = 1.010
                elif mom == 4:
                    prior['A_0'] = 25.18
                    prior['A_1'] = 4.97
                    prior['E_0'] = 0.81110
                    prior['E_1'] = 1.017
            if (par == 'K'):
                if mom == 0:
                    prior['A_0'] = 97.76
                    prior['A_1'] = 11.4
                    prior['E_0'] = 0.77172
                    prior['E_1'] = 1.018
                elif mom == 1:
                    prior['A_0'] = 64.31
                    prior['A_1'] = 6.0
                    prior['E_0'] = 0.78176
                    prior['E_1'] = 0.997
                elif mom == 2:
                    prior['A_0'] = 46.63
                    prior['A_1'] = 5.25
                    prior['E_0'] = 0.79171
                    prior['E_1'] = 0.997
                elif mom == 3:
                    prior['A_0'] = 34.25
                    prior['A_1'] = 5.23
                    prior['E_0'] = 0.80154
                    prior['E_1'] = 1.010
                elif mom == 4:
                    prior['A_0'] = 25.18
                    prior['A_1'] = 4.97
                    prior['E_0'] = 0.81110
                    prior['E_1'] = 1.017
            elif (par == 'Dst'):
                if mom == 0:
                    prior['A_0'] = 86.60
                    prior['A_1'] = 13.6
                    prior['E_0'] = 0.81416
                    prior['E_1'] = 1.035
                elif mom == 1:
                    prior['A_0'] = 60.11
                    prior['A_1'] = 10.4
                    prior['E_0'] = 0.82376
                    prior['E_1'] = 1.067
                elif mom == 2:
                    prior['A_0'] = 44.89
                    prior['A_1'] = 6.5
                    prior['E_0'] = 0.83306
                    prior['E_1'] = 1.030
                elif mom == 3:
                    prior['A_0'] = 33.86
                    prior['A_1'] = 6.03
                    prior['E_0'] = 0.84230
                    prior['E_1'] = 1.034
                elif mom == 4:
                    prior['A_0'] = 25.50
                    prior['A_1'] = 5.45
                    prior['E_0'] = 0.85128
                    prior['E_1'] = 1.034
            elif (par == 'pion'):
                if mom == 0:
                    prior['A_0'] = 310.24
                    prior['A_1'] = 26.1
                    prior['E_0'] = 0.11977
                    prior['E_1'] = 0.419
                elif mom == 1:
                    prior['A_0'] = 144.61
                    prior['A_1'] = 42
                    prior['E_0'] = 0.17780
                    prior['E_1'] = 0.59
                elif mom == 2:
                    prior['A_0'] = 87.65
                    prior['A_1'] = 42
                    prior['E_0'] = 0.22083
                    prior['E_1'] = 0.67
                elif mom == 3:
                    prior['A_0'] = 57.64
                    prior['A_1'] = 41
                    prior['E_0'] = 0.25723
                    prior['E_1'] = 0.769
                elif mom == 4:
                    prior['A_0'] = 38.97
                    prior['A_1'] = 36
                    prior['E_0'] = 0.28853
                    prior['E_1'] = 0.82
            elif (par == 'etac'):
                if mom == 0:
                    prior['A_0'] = 89.138
                    prior['A_1'] = 16
                    prior['E_0'] = 1.222696
                    prior['E_1'] = 1.478
                elif mom == 1:
                    prior['A_0'] = 54.165
                    prior['A_1'] = 14.9
                    prior['E_0'] = 1.228295
                    prior['E_1'] = 1.4680
                elif mom == 2:
                    prior['A_0'] = 38.792
                    prior['A_1'] = 13.85
                    prior['E_0'] = 1.234253
                    prior['E_1'] = 1.4732
                elif mom == 3:
                    prior['A_0'] = 28.613
                    prior['A_1'] = 12.21
                    prior['E_0'] = 1.240174
                    prior['E_1'] = 1.4773
                elif mom == 4:
                    prior['A_0'] = 21.373
                    prior['A_1'] = 10.67
                    prior['E_0'] = 1.246016
                    prior['E_1'] = 1.4829
            elif (par == 'jpsi'):
                if mom == 0:
                    prior['A_0'] = 92.03
                    prior['A_1'] = 12.5
                    prior['E_0'] = 1.252056
                    prior['E_1'] = 1.524
                elif mom == 1:
                    prior['A_0'] = 58.047
                    prior['A_1'] = 11.8
                    prior['E_0'] = 1.257863
                    prior['E_1'] = 1.501
                elif mom == 2:
                    prior['A_0'] = 42.143
                    prior['A_1'] = 10.46
                    prior['E_0'] = 1.263641
                    prior['E_1'] = 1.4945
                elif mom == 3:
                    prior['A_0'] = 31.452
                    prior['A_1'] = 9.54
                    prior['E_0'] = 1.269395
                    prior['E_1'] = 1.4971
                elif mom == 4:
                    prior['A_0'] = 23.757
                    prior['A_1'] = 8.84
                    prior['E_0'] = 1.275091
                    prior['E_1'] = 1.5057
        elif params['ens'] == 3:
            if (par == 'D'):
                if mom == 0:
                    prior['A_0'] = 94.31871
                    prior['A_1'] = 22.78762
                    prior['E_0'] = 0.74616
                    prior['E_1'] = 1.06306
                elif mom == 1:
                    prior['A_0'] = 62.60990
                    prior['A_1'] = 16.01730
                    prior['E_0'] = 0.75674
                    prior['E_1'] = 1.09664
                elif mom == 2:
                    prior['A_0'] = 45.66557
                    prior['A_1'] = 12.32433
                    prior['E_0'] = 0.76725
                    prior['E_1'] = 1.09656
                elif mom == 3:
                    prior['A_0'] = 33.63449
                    prior['A_1'] = 10.57442
                    prior['E_0'] = 0.77750
                    prior['E_1'] = 1.09326
                elif mom == 4:
                    prior['A_0'] = 24.80548
                    prior['A_1'] = 8.49358
                    prior['E_0'] = 0.78754
                    prior['E_1'] = 1.07913
            elif (par == 'Dst'):
                if mom == 0:
                    prior['A_0'] = 81.27701
                    prior['A_1'] = 18.38532
                    prior['E_0'] = 0.78755
                    prior['E_1'] = 1.01581
                elif mom == 1:
                    prior['A_0'] = 57.11895
                    prior['A_1'] = 11.54816
                    prior['E_0'] = 0.79767
                    prior['E_1'] = 1.04622
                elif mom == 2:
                    prior['A_0'] = 42.94385
                    prior['A_1'] = 8.19395
                    prior['E_0'] = 0.80734
                    prior['E_1'] = 1.03187
                elif mom == 3:
                    prior['A_0'] = 32.42940
                    prior['A_1'] = 6.48837
                    prior['E_0'] = 0.81683
                    prior['E_1'] = 1.01636
                elif mom == 4:
                    prior['A_0'] = 24.58044
                    prior['A_1'] = 5.20009
                    prior['E_0'] = 0.82641
                    prior['E_1'] = 1.01021
            elif (par == 'pion'):
                if mom == 0:
                    prior['A_0'] = 399.61102
                    prior['A_1'] = 67.86990
                    prior['E_0'] = 0.08167
                    prior['E_1'] = 0.51374
                elif mom == 1:
                    prior['A_0'] = 146.59702
                    prior['A_1'] = 56.55332
                    prior['E_0'] = 0.15507
                    prior['E_1'] = 0.58870
                elif mom == 2:
                    prior['A_0'] = 84.76810
                    prior['A_1'] = 37.01024
                    prior['E_0'] = 0.20383
                    prior['E_1'] = 0.61797
                elif mom == 3:
                    prior['A_0'] = 54.25815
                    prior['A_1'] = 22.33347
                    prior['E_0'] = 0.24273
                    prior['E_1'] = 0.61518
                elif mom == 4:
                    prior['A_0'] = 36.35865
                    prior['A_1'] = 20.38338
                    prior['E_0'] = 0.27563
                    prior['E_1'] = 0.67460
        elif params['ens'] == 4:
            if (par == 'D'):
                if mom == 0:
                    prior['A_0'] = 94.31871
                    prior['A_1'] = 22.78762
                    prior['E_0'] = 0.74616
                    prior['E_1'] = 1.06306
                elif mom == 1:
                    prior['A_0'] = 62.60990
                    prior['A_1'] = 16.01730
                    prior['E_0'] = 0.75674
                    prior['E_1'] = 1.09664
                elif mom == 2:
                    prior['A_0'] = 45.66557
                    prior['A_1'] = 12.32433
                    prior['E_0'] = 0.76725
                    prior['E_1'] = 1.09656
                elif mom == 3:
                    prior['A_0'] = 33.63449
                    prior['A_1'] = 10.57442
                    prior['E_0'] = 0.77750
                    prior['E_1'] = 1.09326
                elif mom == 4:
                    prior['A_0'] = 24.80548
                    prior['A_1'] = 8.49358
                    prior['E_0'] = 0.78754
                    prior['E_1'] = 1.07913
            elif (par == 'Dst'):
                if mom == 0:
                    prior['A_0'] = 81.27701
                    prior['A_1'] = 18.38532
                    prior['E_0'] = 0.78755
                    prior['E_1'] = 1.01581
                elif mom == 1:
                    prior['A_0'] = 57.11895
                    prior['A_1'] = 11.54816
                    prior['E_0'] = 0.79767
                    prior['E_1'] = 1.04622
                elif mom == 2:
                    prior['A_0'] = 42.94385
                    prior['A_1'] = 8.19395
                    prior['E_0'] = 0.80734
                    prior['E_1'] = 1.03187
                elif mom == 3:
                    prior['A_0'] = 32.42940
                    prior['A_1'] = 6.48837
                    prior['E_0'] = 0.81683
                    prior['E_1'] = 1.01636
                elif mom == 4:
                    prior['A_0'] = 24.58044
                    prior['A_1'] = 5.20009
                    prior['E_0'] = 0.82641
                    prior['E_1'] = 1.01021
            elif (par == 'pion'):
                if mom == 0:
                    prior['A_0'] = 399.61102
                    prior['A_1'] = 67.86990
                    prior['E_0'] = 0.08167
                    prior['E_1'] = 0.51374
                elif mom == 1:
                    prior['A_0'] = 146.59702
                    prior['A_1'] = 56.55332
                    prior['E_0'] = 0.15507
                    prior['E_1'] = 0.58870
                elif mom == 2:
                    prior['A_0'] = 84.76810
                    prior['A_1'] = 37.01024
                    prior['E_0'] = 0.20383
                    prior['E_1'] = 0.61797
                elif mom == 3:
                    prior['A_0'] = 54.25815
                    prior['A_1'] = 22.33347
                    prior['E_0'] = 0.24273
                    prior['E_1'] = 0.61518
                elif mom == 4:
                    prior['A_0'] = 36.35865
                    prior['A_1'] = 20.38338
                    prior['E_0'] = 0.27563
                    prior['E_1'] = 0.67460
        elif params['ens'] == 5:
            if (par == 'D'):
                if mom == 0:
                    prior['A_0'] = 97.76
                    prior['A_1'] = 11.4
                    prior['E_0'] = 0.77172
                    prior['E_1'] = 1.018
                elif mom == 1:
                    prior['A_0'] = 64.31
                    prior['A_1'] = 6.0
                    prior['E_0'] = 0.78176
                    prior['E_1'] = 0.997
                elif mom == 2:
                    prior['A_0'] = 46.63
                    prior['A_1'] = 5.25
                    prior['E_0'] = 0.79171
                    prior['E_1'] = 0.997
                elif mom == 3:
                    prior['A_0'] = 34.25
                    prior['A_1'] = 5.23
                    prior['E_0'] = 0.80154
                    prior['E_1'] = 1.010
                elif mom == 4:
                    prior['A_0'] = 25.18
                    prior['A_1'] = 4.97
                    prior['E_0'] = 0.81110
                    prior['E_1'] = 1.017
            elif (par == 'Dst'):
                if mom == 0:
                    prior['A_0'] = 86.60
                    prior['A_1'] = 13.6
                    prior['E_0'] = 0.81416
                    prior['E_1'] = 1.035
                elif mom == 1:
                    prior['A_0'] = 60.11
                    prior['A_1'] = 10.4
                    prior['E_0'] = 0.82376
                    prior['E_1'] = 1.067
                elif mom == 2:
                    prior['A_0'] = 44.89
                    prior['A_1'] = 6.5
                    prior['E_0'] = 0.83306
                    prior['E_1'] = 1.030
                elif mom == 3:
                    prior['A_0'] = 33.86
                    prior['A_1'] = 6.03
                    prior['E_0'] = 0.84230
                    prior['E_1'] = 1.034
                elif mom == 4:
                    prior['A_0'] = 25.50
                    prior['A_1'] = 5.45
                    prior['E_0'] = 0.85128
                    prior['E_1'] = 1.034
            elif (par == 'pion'):
                if mom == 0:
                    prior['A_0'] = 310.24
                    prior['A_1'] = 26.1
                    prior['E_0'] = 0.11977
                    prior['E_1'] = 0.419
                elif mom == 1:
                    prior['A_0'] = 144.61
                    prior['A_1'] = 42
                    prior['E_0'] = 0.17780
                    prior['E_1'] = 0.59
                elif mom == 2:
                    prior['A_0'] = 87.65
                    prior['A_1'] = 42
                    prior['E_0'] = 0.22083
                    prior['E_1'] = 0.67
                elif mom == 3:
                    prior['A_0'] = 57.64
                    prior['A_1'] = 41
                    prior['E_0'] = 0.25723
                    prior['E_1'] = 0.769
                elif mom == 4:
                    prior['A_0'] = 38.97
                    prior['A_1'] = 36
                    prior['E_0'] = 0.28853
                    prior['E_1'] = 0.82
        elif params['ens'] == 6:
            if (par == 'D'):
                if mom == 0:
                    prior['A_0'] = 94.31871*2
                    prior['A_1'] = 22.78762
                    prior['E_0'] = 1.015
                    prior['E_1'] = 1.4
                elif mom == 1:
                    prior['A_0'] = 62.60990*2
                    prior['A_1'] = 16.01730
                    prior['E_0'] = 1.025
                    prior['E_1'] = 1.4
                elif mom == 2:
                    prior['A_0'] = 45.66557*2
                    prior['A_1'] = 12.32433
                    prior['E_0'] = 1.03
                    prior['E_1'] = 1.4
                elif mom == 3:
                    prior['A_0'] = 33.63449*2
                    prior['A_1'] = 10.57442
                    prior['E_0'] = 1.04
                    prior['E_1'] = 1.4
                elif mom == 4:
                    prior['A_0'] = 24.80548*2
                    prior['A_1'] = 8.49358
                    prior['E_0'] = 1.05
                    prior['E_1'] = 1.4
            elif (par == 'Dst'):
                if mom == 0:
                    prior['A_0'] = 81.27701*2
                    prior['A_1'] = 18.38532
                    prior['E_0'] = 1.06
                    prior['E_1'] = 1.01581
                elif mom == 1:
                    prior['A_0'] = 57.11895*2
                    prior['A_1'] = 11.54816
                    prior['E_0'] = 1.07
                    prior['E_1'] = 1.04622
                elif mom == 2:
                    prior['A_0'] = 42.94385*2
                    prior['A_1'] = 8.19395
                    prior['E_0'] = 1.07
                    prior['E_1'] = 1.03187
                elif mom == 3:
                    prior['A_0'] = 32.42940*2
                    prior['A_1'] = 6.48837
                    prior['E_0'] = 1.08
                    prior['E_1'] = 1.01636
                elif mom == 4:
                    prior['A_0'] = 24.58044*2
                    prior['A_1'] = 5.20009
                    prior['E_0'] = 1.09
                    prior['E_1'] = 1.01021
            elif (par == 'pion'):
                if mom == 0:
                    prior['A_0'] = 399.61102*2
                    prior['A_1'] = 67.86990
                    prior['E_0'] = 0.07
                    prior['E_1'] = 0.51374
                elif mom == 1:
                    prior['A_0'] = 146.59702*2
                    prior['A_1'] = 56.55332
                    prior['E_0'] = 0.15507
                    prior['E_1'] = 0.58870
                elif mom == 2:
                    prior['A_0'] = 84.76810*2
                    prior['A_1'] = 37.01024
                    prior['E_0'] = 0.20383
                    prior['E_1'] = 0.61797
                elif mom == 3:
                    prior['A_0'] = 54.25815*2
                    prior['A_1'] = 22.33347
                    prior['E_0'] = 0.24273
                    prior['E_1'] = 0.61518
                elif mom == 4:
                    prior['A_0'] = 36.35865*2
                    prior['A_1'] = 20.38338
                    prior['E_0'] = 0.27563
                    prior['E_1'] = 0.67460

        # Lazy priors

        prior = {}
        lazy_time = 20
        prior['E_0'] = tp.cal_mass(fdata2[k][0],tau=2,mtype=mtype)[lazy_time]
        prior['E_1'] = prior['E_0'] + 0.4
        prior['A_0'] = 100
        prior['A_1'] = 100

        params['mtype'] = mtype
        #params['prior'] = [prior['A_0'], prior['E_0'], prior['A_1'], prior['E_1']]

        if obj == 'corr':
            # corr fit
            fdata2_onlyone = {}
            fdata2_onlyone[k] = np.copy(fdata2[k])
            #[selected_lazy_tmin,result_para[k],result_chi2dof[k],result[k],ans[k]] = tp.fitting(prior,params,fdata2_onlyone,mtype,selected,correlated=False)
            [selected_lazy_tmin,result_para[k],result_chi2dof[k],result[k],ans[k]] = tp.fitting(prior,params,fdata2_onlyone,mtype,selected)
        elif obj == 'meff':
            # meff fit
            fmdata2_onlyone = {}
            fmdata2_onlyone[k] = tp.cal_mass(fdata2[k],mtype,tau)
            params['mtype'] = 'const'
            [selected_lazy_tmin,result_para[k],result_chi2dof[k],result[k],ans[k]] = tp.fitting(prior,params,fmdata2_onlyone,'const',selected)

        print('Single fit result for %s of Ensemble %s:'%(k,params['ensname']))
        if obj == 'corr':
            print('m = '+repr(gv.gvar(ans[k]['mean'][1],ans[k]['err'][1]))+'\tA0 = '+repr(gv.gvar(ans[k]['mean'][0],ans[k]['err'][0])))
            print('m = '+repr(gv.gvar(ans[k]['mean'][1]*params['hca'],ans[k]['err'][1]*params['hca']))+' MeV')
            if selected['n'] == 2:
                print('E1 = '+repr(gv.gvar(ans[k]['mean'][3],ans[k]['err'][3]))+'\tA1 = '+repr(gv.gvar(ans[k]['mean'][2],ans[k]['err'][2])))

                prior_txt = open("priors/%s/prior_%s.txt"%(one_dir,one_name), 'a')
                prior_txt.write("""            if (par == '%s'):
                        if mom == %d:
                            prior['A_0'] = %.5f
                            prior['A_1'] = %.5f
                            prior['E_0'] = %.5f
                            prior['E_1'] = %.5f\n"""%(par,mom,ans[k]['mean'][0],ans[k]['mean'][2],ans[k]['mean'][1],ans[k]['mean'][3]))
                prior_txt.close()
        elif obj == 'meff':
            print('m = '+repr(gv.gvar(ans[k]['mean'][0],ans[k]['err'][0])))
            print('m = '+repr(gv.gvar(ans[k]['mean'][0]*params['hca'],ans[k]['err'][0]*params['hca']))+' MeV')

        if params['lazy_tmin']:
            selected['tmin'][k] = selected_lazy_tmin

    # Plot stability
    #for k in fdata2.keys():
        tp.plot_stability(k,1,params,selected,result_para[k],result_chi2dof[k],tit=r'%s %s'%(one_tit,inputs.labels(k)),sv='%s_%s'%(one_name,k),figdir='fit_one/%s'%(one_dir))

    # Plot results
    #for k in fdata2.keys():
        fdata2_plot = {}
        fdata2_plot[k] = np.copy(fdata2[k])
        tp.plot_result(fdata2_plot,params,selected,result[k],ans[k],tau=tau,mtype=params['mtype'],tit=r'%s %s'%(one_tit,inputs.labels(k)),sv='%s_%s'%(one_name,k),figdir='fit_one/%s'%(one_dir))

    # Sort the single fit results
    Nstates = len(inputs.particles) * (params['mommax'] + 1)
    energy_non = np.zeros(Nstates)
    energy_non_labels = []
    '''
    n = 0
    for par in inputs.particles:
        for mom in range(params['mommax']+1):
            parkey = "%s_%d"%(par,mom)
            energy_mean = ans[parkey]['mean'][1]
            energy_non[n] = energy_mean
            energy_non_labels.append(parkey)
            n += 1
    energy_non_labels = [x for _,x in sorted(zip(energy_non,energy_non_labels))]
    '''
    
    return fdata2, result_para, result_chi2dof, result, ans, energy_non_labels
