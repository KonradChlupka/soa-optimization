import random
import numpy as np
import time
import matplotlib.pyplot as plt
from deap import base  # contains Toolbox and base Fitness
from deap import creator  # creating types
from deap import tools  # contains operators
from deap import algorithms  # contains ready genetic evolutionary loops

creator.create(
    "FitnessMax", base.Fitness, weights=(1.0,)
)  # positive weight means maximizing, only one means it's one objective
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()  # stores functions with arguments for usage
toolbox.register("attr_bool", lambda: random.random() * 10. - 5.)  # register such a function
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):  # fitness function
    return (sum(individual),)


def rastrigin(x):
    """Rastrigin test objective function.
    """
    x = np.copy(x)
    x -= 5.
    N = len(x)
    return (-(10 * N + sum(x ** 2 - 10 * np.cos(2 * np.pi * x))),)


# implementing necessary evolution steps
toolbox.register("evaluate", rastrigin)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.1,
        ngen=500,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    return pop, logbook, hof


if __name__ == "__main__":
    tic = time.time()
    pop, logbook, hof = main()
    toc = time.time()
    print(toc - tic)
    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))

    gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
    plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()
    input()


# signal code
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

# system
num = [2.01199757841099e115]
den = [
    1,
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

SIGNAL_TIMEBASE = 20e-9  # one period
SIGNAL_LEN = 240  # one period has 240 points
STEADY_NEGATIVE_LEN = 110  # each period starts with a const -1
FULL_RANGE_LEN = 30  # in the rising part, signal can be from -1 to 1
TOP_LEN = 100  # final part of the signal can be between 0.5 and 1
EVOLVING_LEN = 130  # only last 130 points are subject to evolution


def initial_state(trans_func):
    """TODO: docu
    """
    U = np.array([-1.0] * 240)
    T = np.linspace(0, 20e-9, 240)
    (T, yout, xout) = signal.lsim2(
        trans_func, U=np.array([-1.0] * 240), T=T, X0=None, atol=1e-21
    )
    return xout[-1]


def rise_time(T, yout):
    """TODO: docu
    """
    ss = np.mean(yout[-24:])  # steady-state
    start = yout[0]
    min_to_ss = ss - start
    ss_90 = start + 0.9 * min_to_ss
    ss_10 = start + 0.1 * min_to_ss
    return (
        T[next(i for i in range(len(yout) - 1) if yout[i] > ss_90)]
        - T[next(i for i in range(len(yout) - 1) if yout[i] > ss_10)]
    )


def mean_squared_error(T, yout):
    """TODO: docu
    """
    ss = np.mean(yout[-24:])  # steady-state
    start = yout[0]
    perfect_response = np.array([start] * 120 + [ss] * 120)
    return np.mean((perfect_response - yout) ** 2)


def illegal_driver_signal(U):
    """TODO: docu
    """
    if any(i != -1.0 for i in U[:STEADY_NEGATIVE_LEN]):
        raise AssertionError("First part of signal cannot change")
    # fmt: off
    return (
        any(i < -1.0 for i in U[STEADY_NEGATIVE_LEN:STEADY_NEGATIVE_LEN + FULL_RANGE_LEN])
        or any(i > 1.0 for i in U[STEADY_NEGATIVE_LEN:STEADY_NEGATIVE_LEN + FULL_RANGE_LEN])
        or any(i < 0.5 for i in U[STEADY_NEGATIVE_LEN + FULL_RANGE_LEN:])
        or any(i > 1.0 for i in U[STEADY_NEGATIVE_LEN + FULL_RANGE_LEN:])
    )
    # fmt: on


U = np.array([-1.0] * 120 + [1.0] * 120)
T = np.linspace(0, 20e-9, 240)
X0 = initial_state(trans_func)

# default rtol value is fine, but atol needs to be below 1e-21 or lower
(_, yout, _) = signal.lsim2(trans_func, U=U, T=T, X0=X0, atol=1e-21)
plt.plot(T, yout)
input()
