import functools
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

import devices
from step_info import StepInfo

# List[List[float]], each el. of global logbook is one optimization
# which holds the best MSE per generation
ss_amplitude = 0.5
global_logbook = []


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
    fitnesses = toolbox.map(functools.partial(toolbox.evaluate, gen=0), invalid_ind)
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
        fitnesses = toolbox.map(
            functools.partial(toolbox.evaluate, gen=gen), invalid_ind
        )
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


class SOAOptimization:
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
        """Implements optimization for real SOA.
        The hypers used are optimized by hyperparameter optimization on
        the simulation.
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
        self.osc.set_timebase(position=2.4e-8, range_=30e-9)
        self.T = np.linspace(start=0, stop=30e-9, num=1350, endpoint=False)

        # find rise-time ref values
        self.osc.set_timebase(position=2.4e-8, range_=15e-9)
        time.sleep(1)
        self.awg.send_waveform([-ss_amplitude] * 120 + [ss_amplitude] * 120)
        time.sleep(6)
        res = self.osc.measurement(1)
        self.osc.set_timebase(position=2.4e-8, range_=30e-9)
        time.sleep(1)
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
            U (List[float]): Driving signal. Length is assumed to be 60,
                and each point must be strongly between -1.0 and 1.0.
                The rising edge must be in the middle.
        Returns:
            bool: True is driving signal is valid, False otherwise.
        """
        return (
            all(i > -1.0 for i in U)
            and all(i < 1.0 for i in U)
            and all(i < 0.0 for i in U[28:30])
            and all(i > 0.0 for i in U[30:32])
        )

    def SOA_fitness(self, U, gen=None):
        if not self.valid_U(U):
            return (1000.0,)
        else:
            global global_logbook
            expanded_U = [-ss_amplitude] * 90 + list(U) + [ss_amplitude] * 30
            self.awg.send_waveform(expanded_U, suppress_messages=True)
            time.sleep(5)
            result = self.osc.measurement(channel=1)
            si = StepInfo(result, self.T, self.ss_low, self.ss_high, step_length=675)
            si.U = expanded_U
            si.gen = gen
            global_logbook.append(si)
            return (si.mse,)

    def run(self, show_final_plot=True):
        """Runs the optimization.
        Args:
            show_final_plot (bool): If True, will show a plot with the
                fitness over the generations.
        """
        self.pop = self.toolbox.population()
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics()
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


if __name__ == "__main__":
    x = SOAOptimization()
    x.run(show_final_plot=False)
    pickle.dump(global_logbook, open("soa_mse.pickle", "wb"))
