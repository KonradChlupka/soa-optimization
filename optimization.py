import random
import time
import multiprocessing
import sys
import threading

import numpy as np
import matplotlib.pyplot as plt
from deap import base  # contains Toolbox and base Fitness
from deap import creator  # creating types
from deap import tools  # contains operators
from deap import algorithms  # contains ready genetic evolutionary loops
from scipy import signal

import devices

# random.seed(0)

def find_x_init(trans_func):
    """Calculates the state-vector resulting from long -1 input.

    Args:
        trans_func (scipy.signal.ltisys.TransferFunctionContinuous)

    Returns:
        np.ndarray[float]: System's response.
    """
    U = np.array([-1.0] * 480)
    T = np.linspace(0, 40e-9, 480)
    (_, _, xout) = signal.lsim2(trans_func, U=U, T=T, X0=None, atol=1e-13)
    return xout[-1]


def rise_time(T, yout):
    """Calculates 10% - 90% rise time.

    The supplied signal must contain only the rising edge, and the
    rise time is calculated by comparing to the average of the last
    24 points of the signal.

    Args:
        T (np.ndarray[float])
        yout (np.ndarray[float]): System's response. Must be same
            length as T.

    Returns:
        float: Rise time. 1000.0 if cannot be found.
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
        return 1000.0


def mean_squared_error(yout, i_start, i_stop):
    """Calculates mean squared error against perfect square.

    The perfect square is a square wave made up of 480 points, which
    are 2 periods of a square wave:
    [-1.0] * 120 + [1.0] * 120 + [-1.0] * 120 + [1.0] * 120. The
    comparison is made between i_start and i_stop (exclusive).
    yout is normalized before the comparison.

    Args:
        yout (np.ndarray[float]): System's response. Must be same
        length as T.

    Returns:
        float: Mean squared error. 1000.0 if output is invalid.
    """
    square = np.array([-1.0] * 120 + [1.0] * 120 + [-1.0] * 120 + [1.0] * 120)
    square = square[i_start:i_stop]

    yout = np.array(yout)
    y_mean = np.mean(yout)
    y_centered = yout - y_mean
    y_centered_rms = np.mean(y_centered ** 2) ** 0.5
    y_norm = y_centered / y_centered_rms
    mse = np.mean((square - y_norm) ** 2)
    if 0 < mse < 1000:
        return mse
    else:
        return 1000.0


def valid_driver_signal(U):
    """Checks if the driving signal is valid.

    Args:
        U (np.ndarray[float]): Driving signal. If shorter than 240
        points, then assumed that it was clipped from the front.

    Returns:
        bool: True is driving signal is valid, False otherwise.
    """
    length = len(U)
    if length < 240:
        U = (240 - length) * [-0.75] + list(U)
    return (
        all(i > -1.0 for i in U)
        and all(i < 1.0 for i in U)
        and all(i < -0.5 for i in U[10:110])
        and all(i > 0.5 for i in U[130:230])
    )


def simulation_fitness(U, T, X0, trans_func):
    """Calculates fitness of a match.

    Args:
        U (np.ndarray[float]): Driving signal.
        T (np.ndarray[float])
        X0 (float): System's steady-state response to a -1 input.
        trans_func (scipy.signal.ltisys.TransferFunctionContinuous)

    """
    if not valid_driver_signal(U):
        return (1000.0,)
    else:
        # atol of 1e-21 is sufficient for a step func, original trans. func.
        (_, yout, _) = signal.lsim2(trans_func, U=U, T=T, X0=X0, atol=1e-12)
        t = rise_time(T, yout)
        mse = mean_squared_error(yout, 110, 240)
        # print(mse)
        return (t,)


def best_of_population(population):
    """Finds the best individual in a population.

    Assumes that lower fitness equals better.

    Args:
        population (Tuple[List[float]]):
    
    Returns:
        List[float]: Best individual in a population.
    """
    best_fitness = float("inf")
    for ind in population:
        if ind.fitness.values[0] < best_fitness:
            best_ind = ind
    return best_ind


def quitting_thread(newstdin, flag):
    """Used as a thread scanning for "q" from sys.stdin.
    TODO: improve docu
    """
    while not flag.is_set():
        if newstdin.readline().strip() == "q":
            print(
                "Evolution will end after the calculations for "
                "the current population are finished..."
            )
            break
        time.sleep(0.2)


def eaSimple(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    The algorithm is copied from algorithms.py, with slight
    modifications.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    plt.figure()
    plt.ylim((-10, -10 + 1e-10), auto=True)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    flag = threading.Event()
    thread = threading.Thread(target=quitting_thread, args=(sys.stdin, flag))
    thread.start()
    # Begin the generational process
    print("Begin the generational process. Input 'q' to finish early.")
    for gen in range(1, ngen + 1):
        print("Generation {}".format(gen))
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        plt.scatter(gen, logbook.select("min_fitness")[-1], c="blue")
        plt.pause(0.05)
        if not thread.is_alive():
            print("Evolution finished early. {} out of {} done.".format(gen, ngen))
            break
    flag.set()
    thread.join()
    return population, logbook


class SimulationOptimization:
    def __init__(
        self,
        pop_size=100,
        mu=0,
        sigma=0.1,
        indpb=0.05,
        tournsize=3,
        cxpb=0.6,
        mutpb=0.05,
        ngen=50,
    ):
        """Implements optimization for simulated SOA.

        Args:
            pop_size (int): Populaiton size (number of individuals in
                each generation).
            mu (number): Mean for the gaussian addition mutation.
            sigma (number): Standard deviation for the gaussian addition
                mutation.
            indpb (number): Independent probability for each attribute
                to be mutated.
            tournsize (int): The number of individuals participating in
                each tournament.
            cxpb (number): The probability of mating two individuals.
            mutpb (number): The probability of mutating an individual.
            ngen (int): The number of generation.
        """
        # simplified tf
        num = [2.01199757841099e85]
        den = [
            1.64898505756825e0,
            4.56217233166632e10,
            3.04864287973918e21,
            4.76302109455371e31,
            1.70110870487715e42,
            1.36694076792557e52,
            2.81558045148153e62,
            9.16930673102975e71,
            1.68628748250276e81,
            2.40236028415562e90,
        ]
        self.trans_func = signal.TransferFunction(num, den)
        self.T = np.linspace(0, 20e-9 * 130 / 240, 130)
        self.X0 = find_x_init(self.trans_func)

        creator.create("Fitness", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.Fitness)

        self.toolbox = base.Toolbox()
        initial = [random.uniform(-1, 1) for _ in range(20)] + [
            random.uniform(0.5, 1) for _ in range(110)
        ]
        # fmt: off
        self.toolbox.register("ind", tools.initIterate, creator.Individual, lambda: initial)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.ind, n=pop_size)
        self.toolbox.register("map", multiprocessing.Pool(processes=pop_size).map)
        self.toolbox.register("evaluate", simulation_fitness, T=self.T, X0=self.X0, trans_func=self.trans_func)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=indpb)
        self.toolbox.register("select", tools.selTournament, tournsize=tournsize)
        self.toolbox.register("eaSimple", eaSimple, cxpb=cxpb, mutpb=mutpb, ngen=ngen)
        # fmt: on

    def run(self):
        self.pop = self.toolbox.population()
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics()
        self.stats.register("min_per_population", best_of_population)
        self.stats.register(
            "min_fitness", lambda pop: np.min([ind.fitness.values for ind in pop])
        )

        self.pop, self.logbook = self.toolbox.eaSimple(
            self.pop, self.toolbox, stats=self.stats, halloffame=self.hof, verbose=False
        )

        print(
            "Best individual is: {}\nwith fitness: {}".format(
                self.hof[0], self.hof[0].fitness
            )
        )

        gen, min_, = self.logbook.select("gen", "min_fitness")
        plt.figure()
        plt.plot(gen, min_, label="minimum")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend(loc="lower right")
        plt.show()


if __name__ == "__main__":
    x = SimulationOptimization(pop_size=60, ngen=20, mutpb=0.05, indpb=0.05, sigma=0.01)
    x.run()
    # input()


def soa_optimization():
    """TODO: cleanup and docu
    """

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
            return (1.0,)
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
    initial = [0.75] * 130
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
