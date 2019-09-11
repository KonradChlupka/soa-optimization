import random
import time
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from deap import base  # contains Toolbox and base Fitness
from deap import creator  # creating types
from deap import tools  # contains operators
from deap import algorithms  # contains ready genetic evolutionary loops
from scipy import signal

SIGNAL_TIMEBASE = 20e-9  # one period
SIGNAL_LEN = 240  # one period has 240 points
STEADY_NEGATIVE_LEN = 110  # each period starts with a const -1
FULL_RANGE_LEN = 30  # in the rising part, signal can be from -1 to 1
TOP_LEN = 100  # final part of the signal can be between 0.5 and 1
EVOLVING_LEN = 130  # only last 130 points are subject to evolution
EVOLVING_TIMEBASE = SIGNAL_TIMEBASE * EVOLVING_LEN / SIGNAL_LEN


def initial_state(trans_func):
    """Calculates system's steady-state response to a long -1 input.

    Args:
        trans_func (scipy.signal.ltisys.TransferFunctionContinuous)

    Returns:
        float: System's steady-state response to a -1 input.
    """
    U = np.array([-1.0] * 480)
    T = np.linspace(0, 40e-9, 480)
    (T, yout, xout) = signal.lsim2(trans_func, U=U, T=T, X0=None, atol=1e-22)
    return xout[-1]


def rise_time(T, yout):
    """Calculates 10% - 90% rise time.

    Args:
        T (np.ndarray[float])
        yout (np.ndarray[float]): System's response. Must be same length
            as T.

    Returns:
        float: Rise time.
    """
    ss = np.mean(yout[-24:])  # steady-state
    start = yout[0]
    start_to_ss = ss - start  # amplitude difference
    ss_90 = start + 0.9 * start_to_ss
    ss_10 = start + 0.1 * start_to_ss
    for i, t in enumerate(T):
        if yout[i] >= ss_90:
            t_90 = t
            break
    for i, t in enumerate(T):
        if yout[i] >= ss_10:
            t_10 = t
            break
    try:
        return t_90 - t_10
    except UnboundLocalError:
        return 1.0


def mean_squared_error(T, yout):
    """Calculates mean squared error against perfect square response.

    The perfect square response is a square wave made up of 130 points,
    where the first 10 are -1.0 and the following 120 have the
    amplitude of a steady state response (average of last 24 points).

    Args:
        T (np.ndarray[float])
        yout (np.ndarray[float]): System's response. Must be same length
            as T.

    Returns:
        float: Mean squared error. 1.0 if output is invalid.
    """
    # TODO: compare to a constant value
    ss = np.mean(yout[-24:])  # steady-state
    start = yout[0]
    perfect_response = np.array([start] * 10 + [ss] * 120)
    mse = np.mean((perfect_response - yout) ** 2)
    if 0 < mse < 1:
        return mse
    else:
        return 1.0


def valid_driver_signal(U):
    """Checks if the driving signal is valid.

    Args:
        U (np.ndarray[float]): Driving signal.

    Returns:
        bool: True is driving signal is valid, False otherwise.
    """
    return (
        all(i >= -1.0 for i in U[:30])
        or all(i <= 1.0 for i in U[:30])
        or all(i >= 0.5 for i in U[30:])
        or all(i <= 1.0 for i in U[30:])
    )


def fitness(U, T, X0, trans_func):
    """TODO: docu
    TODO: add rise time
    """
    if not valid_driver_signal(U):
        print(1.0)
        return 1.0
    else:
        # atol of 1e-21 usually sufficient
        (_, yout, _) = signal.lsim2(trans_func, U=U, T=T, X0=X0, atol=1e-23)
        # TODO: sanity check of yout
        print(mean_squared_error(T, yout))
        return mean_squared_error(T, yout),


if __name__ == "__main__":
    num = [2.01199757841099e115]
    den = [
        1.0,
        1.00000001648985e19,
        1.64898505756825e30,
        4.56217233166632e40,
        3.04864287973918e51,
        4.76302109455371e61,
        1.70110870487715e72,
        1.36694076792557e82,
        2.81558045148153e92,
        9.16930673102975e101,
        1.68628748250276e111,
        2.40236028415562e120,
    ]
    trans_func = signal.TransferFunction(num, den)

    T = np.linspace(0, EVOLVING_TIMEBASE, 130)
    X0 = initial_state(trans_func)

    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    # TODO: start with random range
    toolbox.register("ind", tools.initRepeat, creator.Individual, lambda: 0.75, n=130)
    toolbox.register("population", tools.initRepeat, list, toolbox.ind, n=100)
    toolbox.register("map", multiprocessing.Pool(processes=100).map)
    toolbox.register("evaluate", fitness, T=T, X0=X0, trans_func=trans_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.05)  # lower mutate probs.
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population()
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    import time
    tic = time.time()
    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.6, # higher
        mutpb=0.05,
        ngen=100,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    print(time.time() - tic)

    gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
    # plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    # plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()
