import multiprocessing
import pickle
import random
import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms  # contains ready genetic evolutionary loops
from deap import base  # contains Toolbox and base Fitness
from deap import creator  # creating types
from deap import tools  # contains operators
from scipy import signal

from step_info import StepInfo

# List[List[float]], each el. of global logbook is one optimization
# which holds the best MSE per generation
global_logbook = []


def find_x_init(trans_func):
    """Calculates the state-vector resulting from long -0.5 input.

    Args:
        trans_func (scipy.signal.ltisys.TransferFunctionContinuous)

    Returns:
        np.ndarray[float]: System's response.
    """
    U = np.array([-0.5] * 480)
    T = np.linspace(0, 40e-9, 480)
    (_, _, xout) = signal.lsim2(trans_func, U=U, T=T, X0=None, atol=1e-13)
    return xout[-1]


def find_y_ss(trans_func, input_ss):
    """Calculates the output resulting from long input.

    Args:
        trans_func (scipy.signal.ltisys.TransferFunctionContinuous)
        input_ss (float)

    Returns:
        float: The output signal level.
    """
    U = np.array([input_ss] * 480)
    T = np.linspace(0, 40e-9, 480)
    (_, yout, _) = signal.lsim2(trans_func, U=U, T=T, X0=None, atol=1e-13)
    return yout[-1]


def valid_driver_signal(U):
    """Checks if the driving signal is valid.

    Args:
        U (np.ndarray[float]): Driving signal.

    Returns:
        bool: True is driving signal is valid, False otherwise.
    """
    return all(i > -1.0 for i in U) and all(i < 1.0 for i in U)


def simulation_fitness(U, T, X0, trans_func, ss_low, ss_high):
    """Calculates fitness of a match.

    Args:
        U (np.ndarray[float]): Driving signal of length 1000.
        T (np.ndarray[float])
        X0 (float): System's steady-state response to a -1 input.
        trans_func (scipy.signal.ltisys.TransferFunctionContinuous)
        ss_low (float): Low steady-state for MSE calculation.
        ss_high (float): High steady_state for MSE calculation.

    Returns:
        Tuple[Float]: MSE.
    """
    if not valid_driver_signal(U):
        return (1000.0,)
    else:
        sp = [ss_low] * 120 + [ss_high] * 120
        (_, yout, _) = signal.lsim2(trans_func, U=U, T=T, X0=X0, atol=1e-12)
        sp_mse = np.mean((np.array(yout) - np.array(sp)) ** 2)
        return (sp_mse,)


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


class SimulationOptimization:
    def __init__(
        self,
        pop_size=100,
        mu=0,
        sigma=0.15,
        indpb=0.06,
        tournsize=4,
        cxpb=0.9,
        mutpb=0.45,
        ngen=200,
        interactive=True,
        show_plotting=True,
    ):
        """Implements optimizat. for simulated SOA with optimal hypers.

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
        self.T = np.linspace(0, 20e-9, 240, endpoint=False)
        self.X0 = find_x_init(self.trans_func)
        self.ss_low = find_y_ss(self.trans_func, -0.5)
        self.ss_high = find_y_ss(self.trans_func, 0.5)

        creator.create("Fitness", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.Fitness)

        self.toolbox = base.Toolbox()
        initial = lambda: [random.uniform(-1, 1) for _ in range(240)]
        # fmt: off
        self.toolbox.register("ind", tools.initIterate, creator.Individual, initial)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.ind, n=pop_size)
        self.toolbox.register("map", multiprocessing.Pool().map)
        self.toolbox.register("evaluate", simulation_fitness, T=self.T, X0=self.X0, trans_func=self.trans_func, ss_low=self.ss_low, ss_high=self.ss_high)
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
        global_logbook.append(min_)
        if show_final_plot:
            plt.figure()
            plt.plot(gen, min_, label="minimum")
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.legend(loc="lower right")
            plt.show()
            plt.pause(0.05)


if __name__ == "__main__":
    x = SimulationOptimization()
    x.run(show_final_plot=False)
    pickle.dump(global_logbook, open("simulation_mse.pickle", "wb"))
