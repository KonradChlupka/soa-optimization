import random
import numpy as np
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
toolbox.register("attr_bool", random.randint, -5, 5)  # register such a function
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=10
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):  # fitness function
    return (sum(individual),)


def rastrigin(x):
    """Rastrigin test objective function.
    """
    x = np.copy(x)
    x -= 5
    N = len(x)
    return (-(10 * N + sum(x ** 2 - 10 * np.cos(2 * np.pi * x))),)


# implementing necessary evolution steps
toolbox.register("evaluate", rastrigin)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=-10, up=10, indpb=0.1)
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
        ngen=100,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    return pop, logbook, hof


if __name__ == "__main__":
    pop, logbook, hof = main()
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
# from scipy import signal

# num = [2.01199757841099e115]
# den = [
#     1,
#     1.00000001648985e19,
#     1.64898505756825e30,
#     4.56217233166632e40,
#     3.04864287973918e51,
#     4.76302109455371e61,
#     1.70110870487715e72,
#     1.36694076792557e82,
#     2.81558045148153e92,
#     9.16930673102975e101,
#     1.68628748250276e111,
#     2.40236028415562e120,
# ]


# trans = signal.TransferFunction(num, den)
# # default rtol value is fine, but atol needs to be below 1e-21 or lower
# (T, yout) = signal.step2(trans, T=np.linspace(0, 1e-8, 2000), atol=1e-21)
# plt.plot(T, yout)
