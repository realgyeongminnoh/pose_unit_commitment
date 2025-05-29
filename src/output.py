import numpy as np

from .parameter import Parameter


class Output:
    def __init__(
        self,
    ):
        # UC
        #
        self.total_cost_system: float = -1
        self.total_cost_generation: float = -1
        self.total_cost_startup: float = -1
        self.total_cost_reserve: float = -1
        # 
        self.u: np.ndarray = np.ndarray([])
        self.p: np.ndarray = np.ndarray([])
        self.p_bar: np.ndarray = np.ndarray([])
        self.cost_startup: np.ndarray = np.ndarray([])
        #
        self.r: np.ndarray = np.ndarray([])
        self.p_max_t: np.ndarray = np.ndarray([])
        self.p_min_t: np.ndarray = np.ndarray([])

    def compute_auxiliary(self, parameter: Parameter):
        # reserve[i, t]
        self.r = self.p_bar - self.p
        # total_cost_reserve
        self.total_cost_reserve = (
            self.r ** 2 * np.array(parameter.cost_quad)[:, None]
            + self.r * np.array(parameter.cost_lin)[:, None]
            + self.u * np.array(parameter.cost_const)[:, None]
        ).sum()
            
        # p_max_t = moving p_max
        self.p_max_t = (self.u * np.array(parameter.p_max)[:, None]).sum(axis=0)
        # p_min_t = moving p_min
        self.p_min_t = (self.u * np.array(parameter.p_min)[:, None]).sum(axis=0)

