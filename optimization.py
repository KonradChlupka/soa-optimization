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
    (T, yout, xout) = signal.lsim2(trans_func, U=U, T=T, X0=None, atol=1e-13)
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
        and all(i <= 1.0 for i in U[:30])
        and all(i >= 0.5 for i in U[30:])
        and all(i <= 1.0 for i in U[30:])
    )


def fitness(U, T, X0, trans_func):
    """Calculates fitness of a match.

    Args:
        U (np.ndarray[float]): Driving signal.
        T (np.ndarray[float])
        X0 (float): System's steady-state response to a -1 input.
        trans_func (scipy.signal.ltisys.TransferFunctionContinuous)

    """
    if not valid_driver_signal(U):
        print(1.0, U, end="\n\n")
        return (1.0,)
    else:
        # atol of 1e-21 is sufficient for a step func, original trans. func.
        (_, yout, _) = signal.lsim2(trans_func, U=U, T=T, X0=X0, atol=1e-12)
        mse = mean_squared_error(T, yout)
        print(mse, U, end="\n\n")
        return (mse,)


def run_simulation(trans_func, T, X0):
    pass


def soa_optimization():
    """TODO: cleanup and docu
    """

    from main import Lightwave7900B
    from main import Lightwave3220
    from main import AnritsuMS9740A
    from main import TektronixAWG7122B
    from main import Agilent8156A
    from main import Agilent86100C

    def waveform_delay(original, delayed):
        """Calculates index delay between signals.

        Requires that 'delayed' is the same signal as 'original' (or
        slightly changed due to ringing etc.), 'delayed' is inverted,
        falling edge of both signals is visible, at least one full
        period of each is visible, a centered square wave from -1 to 1
        is sent, and 'original' on the left side of the screen is high
        and close to the falling edge.

        Args:
            original (List or np.array)
            delayed (List or np.array)

        Returns:
            int: Index offset between delayed and original.
        """
        input(
            "Make sure the signals are positioned correctly for "
            "measurement. Refer to documentation of waveform_delay(). "
            "Press Enter..."
        )

        on_top = False
        for idx, el in enumerate(original):
            if el > 0:
                on_top = True
            if el < 0 and on_top is True:
                orig_crossover_idx = idx
                break

        on_top = False
        delayed = -1 * np.array(delayed)
        for idx, el in enumerate(delayed):
            if el > 0:
                on_top = True
            if el < 0 and on_top is True:
                delayed_crossover_idx = idx
                break

        if orig_crossover_idx > delayed_crossover_idx:
            raise ValueError(
                "Delayed signal seems to be before original, check if "
                "the signal is compliant with requirements in "
                "waveform_delay"
            )
        delay = delayed_crossover_idx - orig_crossover_idx
        print("Detected delay of {} points.".format(delay))
        return delay

    awg = TektronixAWG7122B("GPIB1::1::INSTR")
    osc = Agilent86100C("GPIB1::7::INSTR")

    # setup oscilloscope for measurement
    osc.set_acquire(average=True, count=50, points=1350)
    osc.set_timebase(position=2.4e-8, range_=30e-9)

    # get delay between signals
    awg.send_waveform(np.array([-1.0] * 120 + [1.0] * 120), suppress_messages=True)
    time.sleep(4)
    orig = osc.measurement(4)
    delayed = np.array(osc.measurement(1))
    delayed = delayed - np.mean(delayed)
    idx_delay = waveform_delay(orig, delayed)

    def valid_U(U):
        return (
            all(i >= -1.0 for i in U[:30])
            and all(i <= 1.0 for i in U[:30])
            and all(i >= 0.5 for i in U[30:])
            and all(i <= 1.0 for i in U[30:])
        )

    def soa_fitness(U):
        if not valid_U(U):
            return 1.0,
        awg_signal = [-1.0] * 90
        awg_signal.extend(U)
        awg.send_waveform(awg_signal, suppress_messages=True)
        time.sleep(4)
        orig = osc.measurement(4)
        delayed = np.array(osc.measurement(1))
        delayed = delayed - np.mean(delayed)
        del orig[-idx_delay]
        delayed = delayed[idx_delay:]
        orig = np.array(orig)
        delayed = -1 * np.array(delayed)
        rms_orig = np.sqrt(np.mean(orig ** 2))
        rms_delayed = np.sqrt(np.mean(delayed ** 2))
        orig_norm = orig / rms_orig
        delayed_norm = delayed / rms_delayed

        mse = np.mean((orig_norm - delayed_norm) ** 2)
        print(mse, U)

        return (mse,)

    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    initial = [[0.75] * 130]
    toolbox.register("ind", tools.initIterate, creator.Individual, lambda: initial)
    toolbox.register("population", tools.initRepeat, list, toolbox.ind, n=100)
    toolbox.register("evaluate", soa_fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.05
    )  # lower mutate probs.
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population()
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.6,  # higher
        mutpb=0.05,
        ngen=100,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))

    gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
    # plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    # plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()


# if __name__ == "__main__":
#     # num = [2.01199757841099e115]
#     # den = [
#     #     1.0,
#     #     1.00000001648985e19,
#     #     1.64898505756825e30,
#     #     4.56217233166632e40,
#     #     3.04864287973918e51,
#     #     4.76302109455371e61,
#     #     1.70110870487715e72,
#     #     1.36694076792557e82,
#     #     2.81558045148153e92,
#     #     9.16930673102975e101,
#     #     1.68628748250276e111,
#     #     2.40236028415562e120,
#     # ]
#     # trans_func = signal.TransferFunction(num, den)

#     # simulation parameters

#     # simplified tf
#     num = [2.01199757841099e85]
#     den = [
#         1.64898505756825e0,
#         4.56217233166632e10,
#         3.04864287973918e21,
#         4.76302109455371e31,
#         1.70110870487715e42,
#         1.36694076792557e52,
#         2.81558045148153e62,
#         9.16930673102975e71,
#         1.68628748250276e81,
#         2.40236028415562e90,
#     ]
#     trans_func = signal.TransferFunction(num, den)

#     T = np.linspace(0, EVOLVING_TIMEBASE, 130)
#     X0 = initial_state(trans_func)

#     creator.create("Fitness", base.Fitness, weights=(-1.0,))
#     creator.create("Individual", list, fitness=creator.Fitness)

#     toolbox = base.Toolbox()
#     initial = [random.uniform(-1, 1) for _ in range(30)] + [
#         random.uniform(0.5, 1) for _ in range(100)
#     ]
#     toolbox.register("ind", tools.initIterate, creator.Individual, lambda: initial)
#     toolbox.register("population", tools.initRepeat, list, toolbox.ind, n=100)
#     toolbox.register("map", multiprocessing.Pool(processes=100).map)
#     toolbox.register("evaluate", fitness, T=T, X0=X0, trans_func=trans_func)
#     toolbox.register("mate", tools.cxTwoPoint)
#     toolbox.register(
#         "mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.05
#     )  # lower mutate probs.
#     toolbox.register("select", tools.selTournament, tournsize=3)

#     pop = toolbox.population()
#     hof = tools.HallOfFame(1)
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean)
#     stats.register("min", np.min)
#     stats.register("max", np.max)

#     pop, logbook = algorithms.eaSimple(
#         pop,
#         toolbox,
#         cxpb=0.6,  # higher
#         mutpb=0.05,
#         ngen=100,
#         stats=stats,
#         halloffame=hof,
#         verbose=False,
#     )

#     print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))

#     gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
#     # plt.plot(gen, avg, label="average")
#     plt.plot(gen, min_, label="minimum")
#     # plt.plot(gen, max_, label="maximum")
#     plt.xlabel("Generation")
#     plt.ylabel("Fitness")
#     plt.legend(loc="lower right")
#     plt.show()
