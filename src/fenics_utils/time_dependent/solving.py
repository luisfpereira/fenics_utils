from abc import ABCMeta
from abc import abstractmethod
import warnings


def solve_time_dependent(u_n, u, solver, dt, outputs_writer, timer,
                         stop_criterion):

    # TODO: return solution?
    # TODO: update objects which expression change with time (user has to tell?)

    t = 0
    outputs_writer.write(u_n, t)
    timer.start()

    it = 0
    _loop_time_dependent(it, t, u_n, u, solver, dt, outputs_writer, timer,
                         stop_criterion)

    timer.stop()


def _loop_time_dependent(it, t, u_n, u, solver, dt, outputs_writer, timer,
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
        u_vec = u.vector()
        diff = u_vec.max() - u_vec.min()
        if diff <= self.min_max_diff or it == self.max_iters:
            if it == self.max_iters:
                warnings.warn('Maximum iterations reached')
            return True

        return False
