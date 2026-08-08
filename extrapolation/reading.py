import pickle
from pathlib import Path

import inputs


def read_results(ensembles, tech, result_root, metadata_root):
    result_root = Path(result_root)
    metadata_root = Path(metadata_root)
    params_by_ensemble = {}
    metadata_by_ensemble = {}
    result_by_ensemble = {}
    for ensemble in ensembles:
        class AnalysisArgs: pass
        AnalysisArgs.ensemble = ensemble
        AnalysisArgs.tech = tech
        AnalysisArgs.read2 = 'fast'
        AnalysisArgs.read3 = 'fast'
        params = inputs.cal_params(AnalysisArgs)
        metadata_file = metadata_root / ensemble / ('metadata_%s.pckl' % ensemble)
        result_file = result_root / ensemble / ('results_three_%s_%s.pckl' % (ensemble, tech))
        with open(metadata_file, 'rb') as metadata_stream: metadata = pickle.load(metadata_stream)
        with open(result_file, 'rb') as result_stream: loaded = pickle.load(result_stream)
        params_by_ensemble[ensemble] = params
        metadata_by_ensemble[ensemble] = metadata
        result_by_ensemble[ensemble] = loaded[1]
    return params_by_ensemble, metadata_by_ensemble, result_by_ensemble
