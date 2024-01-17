import numpy as np
from ioh import get_problem, ProblemClass, logger, Experiment

np.random.seed(42)

class RandomSearch:
    def __init__(self, budget):
        #Note that we should re-initialize all dynamic variables if we want to run the same algorithm multiple times
        self.budget = budget

        # A parameter static over the course of an optimization run of an algorithm
        self.algorithm_id = np.random.randint(100)

        # A dynamic parameter updated by the algorithm
        self.a_tracked_parameter = None

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)

            # Update the tracked parameter
            self.a_tracked_parameter = i ** 10

            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt

    @property
    def a_property(self):
        return np.random.randint(100)

    def reset(self):
        self.algorithm_id = np.random.randint(100)

exp = Experiment(
    RandomSearch(10),                   # instance of optimization algorithm
    [1],                                # list of problem id's
    [1, 2],                             # list of problem instances
    [5],                                # list of problem dimensions
    problem_class = ProblemClass.BBOB,  # the problem type, function ids should correspond to problems of this type
    njobs = 1,                          # the number of parrellel jobs for running this experiment
    reps = 2,                           # the number of repetitions for each (id x instance x dim)
    logged_attributes = [               # list of the tracked variables, must be available on the algorithm instance (RandomSearch)
        "a_property",
        "a_tracked_parameter"
    ]
)

exp.run()
