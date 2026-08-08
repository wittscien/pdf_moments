"""Fit windows for continuum and flow-time extrapolations.

Each continuum range lists the ensemble names used in the fit.  Each flow
range is ``[first, last]`` and refers to the index in the available flow list.
"""


_MOMENTS = range(3, 7)
_HADRONS = ('pion', 'kaon', 'kaon_s')


ranges_continuum = {hadron: {moment: ['cB211', 'cC211'] for moment in _MOMENTS} for hadron in _HADRONS}


# Flow-time fits use the 12th flow point (t/t0=1.2) through the largest available point.
ranges_flow = {hadron: {moment: [12, 25] for moment in _MOMENTS} for hadron in _HADRONS}
