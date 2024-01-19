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

exp = Experiment(
    RandomSearch(BUDGET),                 # instance of optimization algorithm
    list(ProblemClass.GRAPH.problems.keys())[:3],             # list of problem id's
    [1],                             # list of problem instances
    [1],                                # list of problem dimensions
    problem_class = ProblemClass.GRAPH,  # the problem type, function ids should correspond to problems of this type
    njobs = 1,                          # the number of parallel jobs for running this experiment
    reps = 3,                           # the number of repetitions for each (id x instance x dim)
    logged_attributes = [               # list of the tracked variables, must be available on the algorithm instance
        "a_property"
    ],
)

exp.run()
