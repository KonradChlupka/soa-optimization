import random
from deap import base  # contains Toolbox and base Fitness
from deap import creator  # creating types
from deap import tools  # contains operators
from deap import algorithms  # contains ready genetic evolutionary loops

creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # type creation, maximizing function, weights must be iterable
creator.create("Individual", list, fitness=creator.FitnessMax)

ind = creator.Individual([1, 0, 1, 1, 0])  # using the created class

print(ind)
print(type(ind))
print(type(ind.fitness))

toolbox = base.Toolbox()  # stores functions with arguments for usage
toolbox.register("attr_bool", random.randint, 0, 1)  # register such a function
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=10
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

bit = toolbox.attr_bool()  # call function
ind = toolbox.individual()
pop = toolbox.population(n=3)

print("bit is of type %s and has value\n%s" % (type(bit), bit))
print("ind is of type %s and contains %d bits\n%s" % (type(ind), len(ind), ind))
print("pop is of type %s and contains %d individuals\n%s" % (type(pop), len(pop), pop))


def evalOneMax(individual):  # fitness function
    return (sum(individual),)


toolbox.register("evaluate", evalOneMax)  # registering operations and their default args
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)
toolbox.register("select", tools.selTournament, tournsize=3)

ind = toolbox.individual()
print(ind)
toolbox.mutate(ind)
print(ind)

mutant = toolbox.clone(ind)
print(mutant is ind)
print(mutant == ind)


def main():
    import numpy

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.1,
        ngen=10,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    return pop, logbook, hof


if __name__ == "__main__":
    pop, log, hof = main()
    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))

    import matplotlib.pyplot as plt

    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()
    input()
