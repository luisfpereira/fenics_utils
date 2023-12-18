import os
from abc import ABCMeta
from abc import abstractmethod
import warnings
import time

from dolfin import MPI
from dolfin.cpp.io import XDMFFile

from fenics_utils.utils import load_last_checkpoint
from fenics_utils.utils import XDMFCheckpointWriter

# TODO: need to be rethought (has to be a class)


def solve_time_dependent(u_n, u, solver, dt, outputs_writer, timer,
                         stop_criterion):

    # TODO: return solution?
    # TODO: update objects which expression change with time (user has to tell?)
    # TODO: what if several variables are involved?

    t = 0
    outputs_writer.write(u_n, t)
    timer.start()

    it = 0
    perform_time_dependent_loop(it, t, u_n, u, solver, dt, outputs_writer, timer,
                                stop_criterion)

    timer.stop()


def solve_time_dependent_with_restart(xdmf_filename, V, var_name,
                                      u_n, u, solver, dt, timer, stop_criterion):
    '''
    Notes:
        When it returns True, it means the simulation is finalized.
    '''

    is_restart = os.path.exists(xdmf_filename)

    with XDMFFile(MPI.comm_self, xdmf_filename) as file:
        outputs_writer = XDMFCheckpointWriter(file, append_first=is_restart)

        if is_restart:
            u_load, t, it = load_last_checkpoint(xdmf_filename, V, var_name)
            u_n.assign(u_load)

            stop = stop_criterion.check(it=it, t=t, u=u_n)
            if stop:
                return True

        else:
            t = 0.
            it = 0.
            outputs_writer.write(u_n, t)

        timer.start()
        perform_time_dependent_loop(it, t, u_n, u, solver, dt, outputs_writer,
                                    timer, stop_criterion)
        timer.stop()

        # TODO: stop criteria should be checked again (kind of a main criterion? assign it)

    return False


def perform_time_dependent_loop(it, t, u_n, u, solver, dt, outputs_writer, timer,
                                stop_criterion):

    while True:
        timer.start_iter()

        it += 1

        # update current time
        t += dt

        # compute solution
        solver.solve()

        # save to vtk file
        outputs_writer.write(u, t)

        # update previous solution
        u_n.assign(u)

        # TODO: make timer also count time to check criteria
        stop = stop_criterion.check(it=it, t=t, u=u)
        timer.stop_iter()

        if stop:
            break


class Criterion(metaclass=ABCMeta):

    @abstractmethod
    def check(self, it, t, u):
        '''
        Receives number of iteration, time and current solution and provides
        stop information (True means the criterion was reached).
        '''
        pass


class StopByTime(Criterion):

    def __init__(self, t_max):
        self.t_max = t_max

    def check(self, t, **kwargs):
        if t >= self.t_max:
            return True

        return False


class StopByIterations(Criterion):

    def __init__(self, max_iters):
        super().__init__()
        self.max_iters = max_iters

    def check(self, it, **kwargs):
        if it == self.max_iters:
            return True

        return False


class StopByExtremaDiff(Criterion):

    def __init__(self, min_max_diff, max_iters=1000):
        self.min_max_diff = min_max_diff
        self.max_iters = max_iters  # for protection

    def check(self, it, u, **kwargs):
        # TODO: will it work properly in parallel?
        u_vec = u.vector()
        diff = u_vec.max() - u_vec.min()
        if diff <= self.min_max_diff or it == self.max_iters:
            if it == self.max_iters:
                warnings.warn('Maximum iterations reached')
            return True

        return False


class CompositeCriteria(Criterion, metaclass=ABCMeta):

    def __init__(self, criteria):
        self.criteria = criteria

    def check(self, it, t, u):
        for criterion in self.criteria:
            if criterion.check(it=it, t=t, u=u):
                return True

        return False


class StopByRuntime(Criterion):

    def __init__(self, max_runtime):
        '''
        Args:
            max_runtime (float): seconds

        Notes:
            By definition, it always passes the maximum allowed time.
        '''
        self.max_runtime = max_runtime
        self._start_time = time.perf_counter()

    def check(self, **kwargs):
        return (time.perf_counter() - self._start_time) > self.max_runtime


class StopCleverlyByRuntime(Criterion):

    def __init__(self, max_runtime, last_iter_time=None, safety_coeff=1.):
        self.max_runtime = max_runtime
        self.last_iter_time = last_iter_time
        self.safety_coeff = safety_coeff
        self._start_time = time.perf_counter()

    def check(self, **kwargs):
        # TODO: conthere
        pass


class StopByItersOrRuntime(CompositeCriteria):

    def __init__(self, max_iters, max_runtime):

        # set criteria
        stop_by_iter = StopByIterations(max_iters)
        stop_by_runtime = StopByRuntime(max_runtime)
        super().__init__([stop_by_iter, stop_by_runtime])


class StopByItersOrCleverRuntime:
    # TODO: make it intelligent
    pass
