import numpy as np

from .input import Input_uc


class Output_uc:
    def __init__(
        self,
    ):
        pass
        # UC
        #
        # self.total_cost_system: float = -1
        # self.total_cost_generation: float = -1
        # # self.total_cost_startup: float = -1
        # self.total_cost_reserve: float = -1
        # # 
        # self.u: np.ndarray = np.ndarray([])
        # self.p: np.ndarray = np.ndarray([])
        # self.p_bar: np.ndarray = np.ndarray([])
        # self.r: np.ndarray = np.ndarray([])
        # self.p_max_true: np.ndarray = np.ndarray([])
        # self.p_min_true: np.ndarray = np.ndarray([])

    # def compute_auxiliary(self, parameter: Parameter):
    #     # reserve[i, t]
    #     self.r = self.p_bar - self.p
    #     # p_max_true[i, t]
    #     self.p_max_true = (self.u * np.array(parameter.p_max)[:, None])
    #     # p_min_true[i, t]
    #     self.p_min_true = (self.u * np.array(parameter.p_min)[:, None])





        # total_cost_reserve
        # ??????????????????????????????????????????????????????????
        # i have no idea; reserve isn't even real
        # p.224 6.4.2
        # EXAMPLE 6.7
        # "to determine the price of reserve, we must figure out where an 
        # additional megawatt of serve would come from and how much it woudl cost"
        # ??????????????????????????????????????????????????????????
        # self.total_cost_reserve = (
        #     self.r ** 2 * np.array(parameter.cost_quad)[:, None]
        #     + self.r * np.array(parameter.cost_lin)[:, None]
        #     + self.u * np.array(parameter.cost_const)[:, None]
        # ).sum()
            

