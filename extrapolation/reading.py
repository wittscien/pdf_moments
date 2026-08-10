from pathlib import Path
import numpy as np
import funcs as tp
import inputs


def reading(ensembles, tech, result_root, metadata_root):
    result_root = Path(result_root)
    metadata_root = Path(metadata_root)
    boots_num = 200
    rng = np.random.default_rng(0)
    params = {}
    metadata = {}
    data = {}
    for ensemble in ensembles:
        class Args: pass
        Args.ensemble = ensemble
        Args.tech = tech
        Args.read2 = 'fast'
        Args.read3 = 'fast'
        params[ensemble] = inputs.cal_params(Args)
        metadata_file = metadata_root / ensemble / ('metadata_%s.pckl' % ensemble)
        result_file = result_root / ensemble / ('results_three_largest_tsep_%s.pckl' % tech)
        metadata[ensemble] = tp.fast_read(metadata_file)
        ratio = tp.fast_read(result_file)
        data[ensemble] = {}
        for k in ratio.keys():
            data[ensemble][k] = {}
            for moment in ratio[k].keys():
                data[ensemble][k][moment] = {}
                for tf in ratio[k][moment].keys():
                    jackknife = ratio[k][moment][tf]
                    mean = tp.cal_mean(jackknife)
                    err = tp.cal_err(jackknife,'jackknife')
                    data[ensemble][k][moment][tf] = np.concatenate(([mean],rng.normal(mean,err,boots_num)))
    return params, metadata, data
