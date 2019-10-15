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


def rise_time(T, yout, n_steady_state=24, rise_start=None, rise_end=None):
    """Calculates 10% - 90% rise time.

    The supplied signal must contain only the rising edge, and the
    rise time is calculated by comparing to the average of the last
    n_steady_state points of the signal, or if rise_start and rise_end
    are supplied, to these points.

    Args:
        T (np.ndarray[float])
        yout (np.ndarray[float]): System's response. Must be same
            length as T.
        n_steady_state (int): Number of points taken from the end of
            the signal used to calculate steady-state.
        rise_start (float): Value to be used as low level.
        rise_end (float): Value to be used as high level.

    Returns:
        float: Rise time. 1000.0 if cannot be found.
    """
    ss = np.mean(yout[-n_steady_state:])  # steady-state
    start = yout[0]
    if rise_start is not None:
        start = rise_start
    if rise_end is not None:
        ss = rise_end
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


def settling_time(T, yout, ss_low, ss_high, settling_fraction=0.05):
    """Calculates settling time from 10% of rising edge.

    Args:
        T (Any[float])
        yout (Any[float]): System's response. Must be the same length
            as T.
        ss_low (float): Value of the output well before the rising
            edge. 10% of the rising edge will be calculated against
            this.
        ss_high (float): Value of the output well after the rising
            edge. 10% of the rising edge will be calculated against
            this.
        settling_fraction (float): Range within which the signal must
            be contained to count as settled.
    """
    ss_to_ss = ss_high - ss_low  # steady-state to steady-state
    ss_10 = ss_low + 0.1 * ss_to_ss
    settled_low = ss_high - settling_fraction * ss_to_ss
    settled_high = ss_high + settling_fraction * ss_to_ss

    # find time when signal crosses 10%
    for i, t in enumerate(T):
        if yout[i] >= ss_10:
            t_10 = t
            break

    # find time when signal settles
    for i, t in enumerate(T):
        if settled_low > yout[i] > settled_high:
            last_not_settled = t

    try:
        return last_not_settled - t_10
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
        (_, yout, _) = signal.lsim2(trans_func, U=U, T=T, X0=X0, atol=1e-12)
        t = rise_time(T, yout)
        mse = mean_squared_error(yout, 110, 240)
        return (mse,)


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
            best_fitness = ind.fitness.values[0]
            best_ind = ind
    return best_ind


def quitting_thread(newstdin, flag):
    """Used as a thread scanning for "q" from sys.stdin.
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
    interactive=True,
    show_plotting=True,
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
    :param interactive: Allows to press 'q' to stop execution, requires
                        confirmation after finished run.
    :param show_plotting: Shows a live plot of progress.
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

    if show_plotting:
        plt.figure()
        plt.ylim((-10, -10 + 1e-10), auto=True)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")

    if interactive:
        flag = threading.Event()
        thread = threading.Thread(target=quitting_thread, args=(sys.stdin, flag))
        thread.start()

    print(
        "Begin the generational process. "
        "Input 'q' to finish early if you're in interactive mode."
    )
    for gen in range(1, ngen + 1):
        print("Starting generation {}".format(gen))
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

        if show_plotting:
            plt.scatter(gen, logbook.select("min_fitness")[-1], c="blue")
            plt.pause(0.05)

        if interactive and not thread.is_alive():
            print("Evolution finished early. {} out of {} done.".format(gen, ngen))
            break
    if interactive:
        flag.set()
        print("Evolution finished. Press enter to continue.")
        thread.join()
    return population, logbook


def main_optimizer(optimizing, range_):
    default_args = {
        "pop_size": 60,
        "mu": 0,
        "sigma": 0.15,
        "indpb": 0.06,
        "tournsize": 4,
        "cxpb": 0.9,
        "mutpb": 0.3,
    }
    tot_evals = 80000

    pop_size = default_args["pop_size"]
    mu = default_args["mu"]
    sigma = default_args["sigma"]
    indpb = default_args["indpb"]
    tournsize = default_args["tournsize"]
    cxpb = default_args["cxpb"]
    mutpb = default_args["mutpb"]

    for i in range_:
        random.seed(0)
        if optimizing == "pop_size":
            pop_size = i
        elif optimizing == "mu":
            mu = i
        elif optimizing == "sigma":
            sigma = i
        elif optimizing == "indpb":
            indpb = i
        elif optimizing == "tournsize":
            tournsize = i
        elif optimizing == "cxpb":
            cxpb = i
        elif optimizing == "mutpb":
            mutpb = i

        ngen = tot_evals // pop_size
        print(
            "optimizing {}, pop_size {}, mu {}, sigma {}, indpb {}, tournsize {}, cxpb {}, mutpb {}, ngen {}.png".format(
                optimizing, pop_size, mu, sigma, indpb, tournsize, cxpb, mutpb, ngen
            )
        )

        x = SimulationOptimization(
            pop_size=pop_size,
            mu=mu,
            sigma=sigma,
            indpb=indpb,
            tournsize=tournsize,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=ngen,
            interactive=False,
            show_plotting=False,
        )
        x.run(show_final_plot=True)
        plt.ylim((0.9, 1.11))
        plt.pause(0.5)
        plt.savefig(
            "optimizing {}, pop_size {}, mu {}, sigma {}, indpb {}, tournsize {}, cxpb {}, mutpb {}, ngen {}.png".format(
                optimizing, pop_size, mu, sigma, indpb, tournsize, cxpb, mutpb, ngen
            )
        )
        plt.pause(0.5)
        plt.close()
        plt.pause(0.5)
        del x


class SimulationOptimization:
    def __init__(
        self,
        pop_size=60,
        mu=0,
        sigma=0.15,
        indpb=0.06,
        tournsize=4,
        cxpb=0.9,
        mutpb=0.3,
        ngen=50,
        interactive=True,
        show_plotting=True,
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
            interactive (bool): Allows to press 'q' to stop execution,
                requires confirmation after finished run.
            show_plotting (bool): Shows a live plot of progress.
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
        self.toolbox.register("map", multiprocessing.Pool().map)
        self.toolbox.register("evaluate", simulation_fitness, T=self.T, X0=self.X0, trans_func=self.trans_func)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=indpb)
        self.toolbox.register("select", tools.selTournament, tournsize=tournsize)
        self.toolbox.register("eaSimple", eaSimple, cxpb=cxpb, mutpb=mutpb, ngen=ngen, interactive=interactive, show_plotting=show_plotting)
        # fmt: on

    def run(self, show_final_plot=True):
        """Runs the optimization.

        Args:
            show_final_plot (bool): If True, will show a plot with
                the fitness over the generations.
        """
        self.pop = self.toolbox.population()
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics()
        self.stats.register("min_per_population", best_of_population)
        self.stats.register(
            "min_fitness", lambda pop: np.min([ind.fitness.values for ind in pop])
        )

        self.pop, self.logbook = self.toolbox.eaSimple(
            self.pop, self.toolbox, stats=self.stats, halloffame=self.hof
        )

        print(
            "Best individual is: {}\nwith fitness: {}".format(
                self.hof[0], self.hof[0].fitness
            )
        )

        gen, min_, = self.logbook.select("gen", "min_fitness")
        if show_final_plot:
            plt.figure()
            plt.plot(gen, min_, label="minimum")
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.legend(loc="lower right")
            plt.show()
            plt.pause(0.05)


class SOAOptimization:
    def __init__(
        self,
        pop_size=120,
        mu=0,
        sigma=0.15,
        indpb=0.06,
        tournsize=4,
        cxpb=0.9,
        mutpb=0.45,
        ngen=2000,
        interactive=True,
        show_plotting=True,
    ):
        """Implements optimization for real SOA.

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
            interactive (bool): Allows to press 'q' to stop execution,
                requires confirmation after finished run.
            show_plotting (bool): Shows a live plot of progress.
        """
        self.awg = devices.TektronixAWG7122B("GPIB1::1::INSTR")
        self.osc = devices.Agilent86100C("GPIB1::7::INSTR")

        # setup oscilloscope for measurement
        self.osc.set_acquire(average=True, count=30, points=1350)
        self.osc.set_timebase(position=2.4e-8, range_=12e-9)
        self.T = np.linspace(start=0, stop=12e-9, num=1350)

        # find rise-time ref values
        self.awg.send_waveform([-0.5] * 120 + [0.5] * 120)
        time.sleep(5)
        res = self.osc.measurement(1)
        self.ss_low = res[0]
        self.ss_high = res[-1]

        creator.create("Fitness", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.Fitness)

        self.toolbox = base.Toolbox()
        initial = lambda: [random.uniform(-1, 0) for _ in range(30)] + [
            random.uniform(0, 1) for _ in range(90)
        ]
        # fmt: off
        self.toolbox.register("ind", tools.initIterate, creator.Individual, initial)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.ind, n=pop_size)
        self.toolbox.register("evaluate", self.SOA_fitness)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=indpb)
        self.toolbox.register("select", tools.selTournament, tournsize=tournsize)
        self.toolbox.register("eaSimple", eaSimple, cxpb=cxpb, mutpb=mutpb, ngen=ngen, interactive=interactive, show_plotting=show_plotting)
        # fmt: on

    def valid_U(self, U):
        """Checks if the driving signal is valid.

        Args:
            U (List[float]): Driving signal. Length is assumed to be 40,
                and each point must be strongly between -1.0 and 1.0.
                The rising edge must be in the middle.

        Returns:
            bool: True is driving signal is valid, False otherwise.
        """
        return (
            all(i > -1.0 for i in U)
            and all(i < 1.0 for i in U)
            and all(i < 0.0 for i in U[0:30])
            and all(i > 0.0 for i in U[30:])
        )

    def SOA_fitness(self, U):
        if not self.valid_U(U):
            return (1000.0,)
        else:
            expanded_U = [-0.5] * 90 + list(U) + [0.5] * 30
            self.awg.send_waveform(expanded_U, suppress_messages=True)
            time.sleep(5)
            result = self.osc.measurement(channel=1)
            return settling_time(self.T, result, self.ss_low, self.ss_high, 0.05),

    def run(self, show_final_plot=True):
        """Runs the optimization.

        Args:
            show_final_plot (bool): If True, will show a plot with
                the fitness over the generations.
        """
        self.pop = self.toolbox.population()
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics()
        self.stats.register("min_per_population", best_of_population)
        self.stats.register(
            "min_fitness", lambda pop: np.min([ind.fitness.values for ind in pop])
        )

        self.pop, self.logbook = self.toolbox.eaSimple(
            self.pop, self.toolbox, stats=self.stats, halloffame=self.hof
        )

        print(
            "Best individual is: {}\nwith fitness: {}".format(
                self.hof[0], self.hof[0].fitness
            )
        )

        gen, min_, = self.logbook.select("gen", "min_fitness")
        if show_final_plot:
            plt.figure()
            plt.plot(gen, min_, label="minimum")
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.legend(loc="lower right")
            plt.show()
            plt.pause(0.05)


def rising_edge_optimization():
    """Finds optimal rising edge.

    The function tests out different rising edges, as follows:
    each full wave is 240 points, first 80 are -0.75, last 80 are 0.75,
    and the points in between are either 0.75 or 1.0 (negative before
    the rising edge). Then rise times can be compared.
    """
    awg = devices.TektronixAWG7122B("GPIB1::1::INSTR")
    osc = devices.Agilent86100C("GPIB1::7::INSTR")

    # setup oscilloscope for measurement
    osc.set_acquire(average=True, count=30, points=1350)
    osc.set_timebase(position=4e-8, range_=12e-9)
    T = np.linspace(start=0, stop=12e-9, num=1350)

    # find rise-time ref values
    awg.send_waveform([-0.75] * 120 + [0.75] * 120, suppress_messages=True)
    time.sleep(5)
    res = osc.measurement(channel=1)
    rise_start = res[0]
    rise_end = res[-1]

    results = []

    for n_high in range(9):
        for n_low in range(9):
            rising_edge = ([-0.75] * 5 * (8 - n_low) + [-1.0] * 5 * n_low) + (
                [1.0] * 5 * n_high + [0.75] * 5 * (8 - n_high)
            )
            awg.send_waveform(
                [-0.75] * 80 + rising_edge + [0.75] * 80, suppress_messages=True
            )
            time.sleep(5)
            result = osc.measurement(channel=1)
            my_rise_time = rise_time(
                T, result, rise_start=rise_start, rise_end=rise_end
            )
            print(my_rise_time)
            results.append((rising_edge, result, my_rise_time))

    return results


def steady_state_optimization():
    """Compares the effect of different steady-state on the rise time.
    """
    awg = devices.TektronixAWG7122B("GPIB1::1::INSTR")
    osc = devices.Agilent86100C("GPIB1::7::INSTR")

    # setup oscilloscope for measurement
    osc.set_acquire(average=True, count=30, points=1350)
    osc.set_timebase(position=4e-8, range_=12e-9)
    T = np.linspace(start=0, stop=12e-9, num=1350)

    results = []

    for level in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        awg.send_waveform([-level] * 120 + [level] * 120, suppress_messages=True)
        time.sleep(5)
        res = osc.measurement(channel=1)
        rise_start = res[0]
        rise_end = res[-1]
        rise_time_pure = rise_time(T, res, rise_start=rise_start, rise_end=rise_end)

        awg.send_waveform(
            [-level] * 110 + [-1.0] * 10 + [1.0] * 10 + [level] * 110,
            suppress_messages=True,
        )
        time.sleep(5)
        res = osc.measurement(channel=1)
        rise_time_optimized = rise_time(
            T, res, rise_start=rise_start, rise_end=rise_end
        )

        results.append((level, rise_time_pure, rise_time_optimized))

    return results


if __name__ == "__main__":
    # main_optimizer("mutpb", [0.3])
    x = SOAOptimization()
    x.run()
