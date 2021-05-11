

# TODO: update to receive objects which expression change with time

def solve_time_dependent(u_n, u, solver, dt, num_steps, vtkfile, timer):

    # TODO: return solution

    t = 0
    vtkfile << (u_n, t)
    timer.start()

    for n in range(num_steps):
        timer.start_iter()

        # update curent time
        t += dt

        # compute solution
        solver.solve()

        # save to vtk file
        vtkfile << (u, t)

        # update previous solution
        u_n.assign(u)

        timer.stop_iter()

    timer.stop()
