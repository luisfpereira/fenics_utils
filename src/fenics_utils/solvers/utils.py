import numpy as np


def collect_SLEPc_eigenpairs(solver):
    '''Returns only real part.
    '''

    w, v = [], []
    for i in range(solver.get_number_converged()):
        r, _, rv, _ = solver.get_eigenpair(i)
        w.append(r)
        v.append(rv)

    return np.array(w), np.array(v).T
