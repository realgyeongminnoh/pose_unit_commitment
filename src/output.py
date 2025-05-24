import numpy as np


class Output:
    def __init__(
        self,
    ):
        # uc
        # total
        self.total_cost_system: float = -1
        self.total_cost_generation: float = -1
        self.total_cost_startup: float = -1
        self.total_cost_reserve: float = -1
        self.u: np.ndarray = np.ndarray([])
        self.p: np.ndarray = np.ndarray([])
        self.r: np.ndarray = np.ndarray([])