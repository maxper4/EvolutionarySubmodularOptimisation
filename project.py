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
        factor = 1.01
        max = 0.8
        min = 0.2
        weights = [0.5] * n

        for _ in range(self.budget -1):
            child = [np.random.choice([0, 1], p=[1-w, w]) for w in weights]
            score = func(child)
            for i in range(n):
                if child[i] == 1:
                    weights[i] = weights[i] * factor if weights[i] * factor < max else max
                else:
                    weights[i] = weights[i] / factor if weights[i] * factor > min else min

class WeightedImpactMutation:
    def __init__(self, budget, bucket_percent=0.1):
        #Note that we should re-initialize all dynamic variables if we want to run the same algorithm multiple times
        self.budget = budget
        self.bucket_percent = bucket_percent

        # A parameter static over the course of an optimization run of an algorithm
        self.algorithm_id = np.random.randint(100)

    def __call__(self, func):
        n = func.meta_data.n_variables
        best = np.random.randint(0, 2, size=n)
        best_fitness = func(best)
        factor = 1.05
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
            for i in bucket:
                if child[i] == 1:
                    weights[i] = weights[i] * factor if weights[i] * factor < max else max
                else:
                    weights[i] = weights[i] / factor if weights[i] * factor > min else min
            if score > best_fitness:
                best = child
                best_fitness = score

exp = Experiment(
    WeightedImpactMutation(BUDGET, bucket_percent=0.1),                 # instance of optimization algorithm
    list(ProblemClass.GRAPH.problems.keys())[:3],             # list of problem id's
    [1],                             # list of problem instances
    [1],                                # list of problem dimensions
    problem_class = ProblemClass.GRAPH,  # the problem type, function ids should correspond to problems of this type
    njobs = 1,                          # the number of parallel jobs for running this experiment
    reps = 3,                           # the number of repetitions for each (id x instance x dim)
    logged_attributes = [               # list of the tracked variables, must be available on the algorithm instance
    ],
)

exp.run()
