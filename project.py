import numpy as np
from ioh import get_problem, ProblemClass, logger, Experiment, OptimizationType

BUDGET = 10000

class RandomSearch:
    def __init__(self, budget):
        #Note that we should re-initialize all dynamic variables if we want to run the same algorithm multiple times
        self.budget = budget

        # A parameter static over the course of an optimization run of an algorithm
        self.algorithm_id = np.random.randint(100)

        # A dynamic parameter updated by the algorithm
        self.a_tracked_parameter = None

    def __call__(self, func):
        for _ in range(self.budget):
            # We can use the problems meta information to see the number of variables needed
            x = np.random.randint(0, 2, size=func.meta_data.n_variables)
            func(x)

    @property
    def a_property(self):
        return np.random.randint(100)

    def reset(self):
        self.algorithm_id = np.random.randint(100)


    @property
    def a_property(self):
        return np.random.randint(100)

    def reset(self):
        self.algorithm_id = np.random.randint(100)

class MutationByProgressSequential:
    def __init__(self, budget):
        #Note that we should re-initialize all dynamic variables if we want to run the same algorithm multiple times
        self.budget = budget

        # A parameter static over the course of an optimization run of an algorithm
        self.algorithm_id = np.random.randint(100)

    def __call__(self, func):
        n = func.meta_data.n_variables
        best = np.random.randint(0, 2, size=n)
        best_fitness = func(best)

        for i in range(self.budget -1):
            progress = i * n // self.budget 
            x = np.random.randint(0, 2, size=(n-progress))
            x = np.concatenate((best[:progress], x))
            score = func(x)
            if score > best_fitness:
                best = x
                best_fitness = score

class MutationByProgress:
    def __init__(self, budget):
        #Note that we should re-initialize all dynamic variables if we want to run the same algorithm multiple times
        self.budget = budget

        # A parameter static over the course of an optimization run of an algorithm
        self.algorithm_id = np.random.randint(100)

    def __call__(self, func):
        n = func.meta_data.n_variables
        best = np.random.randint(0, 2, size=n)
        best_fitness = func(best)

        for i in range(self.budget -1):
            progress = i * n // self.budget 
            child = best.copy()
            for _ in range(n-progress):
                singleton = np.random.randint(0, n)
                child[singleton] = 0 if child[singleton] == 1 else 1

            score = func(child)
            if score > best_fitness:
                best = child
                best_fitness = score

class RandomFromSmallSpeedup:
    def __init__(self, budget, bucket_size=100):
        #Note that we should re-initialize all dynamic variables if we want to run the same algorithm multiple times
        self.budget = budget
        self.bucket_size = bucket_size

        # A parameter static over the course of an optimization run of an algorithm
        self.algorithm_id = np.random.randint(100)

    def __call__(self, func):
        n = func.meta_data.n_variables
        bucket = np.random.randint(0, n, size=self.bucket_size)
        best = [0] * n
        for singleton in bucket:
            best[singleton] = 1
        best_fitness = func(best)

        for i in range(self.budget -1):
            bucket = np.random.randint(0, n, size=self.bucket_size)
            for singleton in bucket:
                best[singleton] = 1
            score = func(best)
            if score <= best_fitness:
                best[singleton] = 0

class WeightedImpact:
    def __init__(self, budget):
        #Note that we should re-initialize all dynamic variables if we want to run the same algorithm multiple times
        self.budget = budget

        # A parameter static over the course of an optimization run of an algorithm
        self.algorithm_id = np.random.randint(100)

    def __call__(self, func):
        n = func.meta_data.n_variables
        best = np.random.randint(0, 2, size=n)
        best_fitness = func(best)
        factor = 1.1
        max = 0.95
        min = 0.05
        weights = [0.5] * n
        bucket_size = int(n * self.bucket_percent)

        for _ in range(self.budget -1):
            child = best.copy()
            bucket = np.random.choice(range(n), replace=False, size=bucket_size)
            for c in bucket:
                child[c] = np.random.choice([0, 1], p=[1-weights[c], weights[c]])
            score = func(child)
            if score > best_fitness:
                best = child
                best_fitness = score
                for i in bucket:
                    if child[i] == 1:
                        weights[i] = weights[i] * factor if weights[i] * factor < max else max
                    else:
                        weights[i] = weights[i] / factor if weights[i] / factor > min else min
            else:
                for i in bucket:
                    if child[i] == 1:
                        weights[i] = weights[i] / factor if weights[i] / factor > min else min
                    else:
                        weights[i] = weights[i] * factor if weights[i] * factor < max else max

class WeightedImpactMutation:
    def __init__(self, budget, max_bucket_percent=0.1):
        #Note that we should re-initialize all dynamic variables if we want to run the same algorithm multiple times
        self.budget = budget
        self.max_bucket_percent = max_bucket_percent

        # A parameter static over the course of an optimization run of an algorithm
        self.algorithm_id = np.random.randint(100)

    def __call__(self, func):
        n = func.meta_data.n_variables
        best = np.random.randint(0, 2, size=n)
        best_fitness = func(best)
        factor = 1.1
        max = 0.95
        min = 0.05
        weights = [0.5] * n
        max_bucket_size = int(n * self.max_bucket_percent)
        mutation_rate = 0.1
        stuck_count = 0

        for e in range(self.budget -1):
            child = best.copy()
            mutation_rate = 0.01 * stuck_count if 0.01 * stuck_count < 0.5 else 0.5
            weights = [w if np.random.rand() > mutation_rate else 0.5 for w in weights]
            bucket_size = int(e * max_bucket_size //self.budget)
            bucket = np.random.choice(range(n), replace=False, size=bucket_size)
            for c in bucket:
                child[c] = np.random.choice([0, 1], p=[1-weights[c], weights[c]])
            score = func(child)
            if score > best_fitness:
                best = child
                stuck_count = 0
                best_fitness = score
                for i in bucket:
                    if child[i] == 1:
                        weights[i] = weights[i] * factor if weights[i] * factor < max else max
                    else:
                        weights[i] = weights[i] / factor if weights[i] / factor > min else min
            else:
                stuck_count += 1
                for i in bucket:
                    if child[i] == 1:
                        weights[i] = weights[i] / factor if weights[i] / factor > min else min
                    else:
                        weights[i] = weights[i] * factor if weights[i] * factor < max else max

class SmallToLargeWithCleanup:
    def __init__(self, budget, max_bucket_percent=0.1):
        #Note that we should re-initialize all dynamic variables if we want to run the same algorithm multiple times
        self.budget = budget
        self.max_bucket_percent = max_bucket_percent

        # A parameter static over the course of an optimization run of an algorithm
        self.algorithm_id = np.random.randint(100)

    def __call__(self, func):
        n = func.meta_data.n_variables
        best = np.random.randint(0, 2, size=n)
        best_fitness = func(best)
        max_bucket_size = int(n * self.max_bucket_percent)
        max_cleanup_size = int(n * 0.05)

        for e in range(self.budget -1):
            child = best.copy()
            bucket_size = int(e * max_bucket_size //self.budget)
            cleanup_size = np.random.randint(1, int(e * max_cleanup_size //self.budget) +2)
            cleanup = np.random.choice(range(n), replace=False, size=cleanup_size)
            bucket = np.random.choice(range(n), replace=False, size=bucket_size)
            for c in bucket:
                    child[c] = 1 - child[c]
            for c in cleanup:
                child[c] = 0

            score = func(child)
            if score > best_fitness:
                best = child
                best_fitness = score

class ParallelSearch:
    def __init__(self, budget):
        #Note that we should re-initialize all dynamic variables if we want to run the same algorithm multiple times
        self.budget = budget

        # A parameter static over the course of an optimization run of an algorithm
        self.algorithm_id = np.random.randint(100)

        # A dynamic parameter updated by the algorithm
        self.a_tracked_parameter = None

    def __call__(self, func):
        n = func.meta_data.n_variables
        x = np.random.randint(0, 2, size=func.meta_data.n_variables)
        alpha = 1       # Parameter in [0..n]
        prob1 = 1/n     # Parameter in [0..1]
        for _ in range(self.budget):
            nb1 = np.count_nonzero(x)
            nb0 = n - nb1
            if(nb0 == 0):
                prob0 = 0.5
            else:
                prob0 = ((alpha + nb1*prob1))/nb0
            # The idea is that on average, we only add alpha new bits
            for i in range(n):
                if x[i] == 0:
                    r = np.random.uniform(0, 1)
                    if r <= prob0:
                        x[i] = 1
                else:
                    r = np.random.uniform(0, 1)
                    if r <= prob1:
                        x[i] = 0
            func(x)




class OnePOneWithCleanup:
    def __init__(self, budget, slope, p, c, alpha=0, y=10):
        #Note that we should re-initialize all dynamic variables if we want to run the same algorithm multiple times
        self.budget = budget

        # A parameter static over the course of an optimization run of an algorithm
        self.algorithm_id = np.random.randint(100)

        self.slope = slope

        self.p = p

        self.alpha = alpha # expected number of added 1s during the second part of the algorithm

        self.y = y #number of offsprings in the second part of the algorithm

        self.c = c #crossover parametre

    def parallelOffspring(self, n, x, p, q):
        for i in range(n):
            r = np.random.uniform(0, 1)
            if(x[i] == 1):
                if(r < p):
                    x[i] = 0
            else:
                if(r < q):
                    x[i] = 1
        return x

    def sol2str(self, x):
        res = ""
        for e in x:
            res = res + str(e)
        return res

    def str2sol(self, s):
        res = np.random.randint(0, 2, size=n)
        for i in range(len(s)):
            res[i] = s[i]
        return res

    def __call__(self, func):
        print("start of a run")
        n = func.meta_data.n_variables
        best = np.random.randint(0, 2, size=n)
        best_fitness = func(best)
        check = best_fitness
        change_strat = False
        chech_frequency = 166 # steps of checking
        N = 0 # Targer number of 1s
        budget = 1
        memory = dict()
        memory[self.sol2str(best)] = best_fitness
        while(budget < self.budget):
            # We check the ompovement slope with bigsteps
            if(budget > 500 and (not change_strat) and budget%166 == 0):
                print(budget)
                if(best_fitness/check < self.slope):
                    change_strat = True
                check = best_fitness

            if(not change_strat):
                child = best.copy()
                for c in range(n):
                    if child[c] == 1:
                        if np.random.rand() < 2.0 / n:
                            child[c] = 1 - child[c]
                    else:
                        if np.random.rand() < 1.0 / n:
                            child[c] = 1 - child[c]
                str_child = self.sol2str(child)
                if(not str_child in memory.keys()):
                    score = func(child)
                    budget = budget + 1
                    memory[str_child] = score
                    if score > best_fitness:
                        best = child
                        best_fitness = score

            else:
                nb1 = np.count_nonzero(child)
                nb0 = n - nb1
                if(nb1 == n):
                    q = 0.5
                else:
                    alpha = 0
                    q = (alpha + nb1*self.p)/nb0
                # generation of lambda offsprings, we keep the best

                best_child = best.copy()
                best_child = self.parallelOffspring(n, best_child, self.p, q)
                best_child_score = func(best_child)
                budget = budget + 1

                for _ in range(self.y - 1):
                    child = best.copy()
                    child = self.parallelOffspring(n, child, self.p, q)
                    str_child = self.sol2str(child)
                    if(str_child in memory.keys()):
                        score = memory[str_child]
                    else:
                        score = func(child)
                        budget = budget + 1
                        memory[str_child] = score
                    if(score > best_child_score):
                        best_child_score = score
                        best_child = child.copy()

                # Crossover phase
                crossover = best.copy()
                for i in range(n):
                    r = np.random.uniform(0, 1)
                    if(r < c):
                        crossover[i] = best_child[i]
                str_crossover = self.sol2str(crossover)
                if(str_crossover in memory.keys()):
                    crossover_score = memory[str_crossover]
                else:
                    crossover_score = func(crossover)
                    budget = budget + 1
                    memory[str_crossover] = crossover_score

                # We keep the best generated solution
                if(crossover_score > best_fitness):
                    best = crossover.copy()
                    best_fitness = crossover_score
                if(best_child_score > best_fitness):
                    best = best_child.copy()
                    best_fitness = best_child_score


# 1.000, 1/800, 0.5, alpha=0, y=10 :  11281
# 1.0005, 1/800, 0.5, alpha=0, y=10 : 11335

Algs = [RandomSearch,                   #0
        MutationByProgressSequential,   #1
        MutationByProgress,             #2
        RandomFromSmallSpeedup,         #3
        WeightedImpact,                 #4
        WeightedImpactMutation(BUDGET, max_bucket_percent=0.1),            #5
        ParallelSearch(BUDGET),         #6
        SmallToLargeWithCleanup(BUDGET, max_bucket_percent=0.1),           #7
        OnePOneWithCleanup(BUDGET, 1.0005, 1/800, 0.5, alpha=0, y=10),     #8
        ]                 

Alg = Algs[8]      


# MaxCut
functionsList = [2000, 2001, 2002, 2003, 2004] 

# MaxCoverage
#functionsList = [2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127]


_ = list(ProblemClass.GRAPH.problems.keys())
exp = Experiment(
    Alg,                                            # instance of optimization algorithm
    functionsList,   # list of problem id's
    [1],                                            # list of problem instances
    [1],                                            # list of problem dimensions
    problem_class = ProblemClass.GRAPH,             # the problem type, function ids should correspond to problems of this type
    njobs = 1,                                      # the number of parallel jobs for running this experiment
    reps = 10,                                      # the number of repetitions for each (id x instance x dim)
    logged_attributes = [                           # list of the tracked variables, must be available on the algorithm instance
    ],
)

tmp = exp.run()

print(tmp)
