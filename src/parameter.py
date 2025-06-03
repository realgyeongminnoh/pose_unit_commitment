import numpy as np 


class Parameter:
    def __init__(
        self,
        load,
        reserve,
        p_min,
        p_max,
        ramp_up,
        ramp_down,
        startup_ramp,
        shutdown_ramp,
        min_up,
        min_down,
        p_prev,
        u_prev,
        min_up_prev,
        min_down_prev,
        cost_quad,
        cost_lin,
        cost_const,
        cost_startup_step,
    ):
        to_list = self._to_list

        # system
        self.load = to_list(load)
        self.reserve = to_list(reserve)
        # power
        self.p_min = to_list(p_min)
        self.p_max = to_list(p_max)
        self.ramp_up = to_list(ramp_up)
        self.ramp_down = to_list(ramp_down)
        self.startup_ramp = to_list(startup_ramp)
        self.shutdown_ramp = to_list(shutdown_ramp)
        # time
        self.min_up = to_list(min_up)
        self.min_down = to_list(min_down)
        # generator cost function
        self.cost_quad = to_list(cost_quad)
        self.cost_lin = to_list(cost_lin)
        self.cost_const = to_list(cost_const)
        # previous horizon
        self.min_up_prev = to_list(min_up_prev)
        self.min_down_prev = to_list(min_down_prev)
        self.p_prev = [to_list(p_prev_i) for p_prev_i in p_prev]
        self.u_prev = [to_list(u_prev_i) for u_prev_i in u_prev]
        # startup cost piecewise function
        self.cost_startup_step = [to_list(cost_startup_step_i) for cost_startup_step_i in cost_startup_step]
        # numbers
        self.num_units = len(self.p_min)
        self.num_periods = len(self.load)
        self.num_cooling_steps = [len(cost_startup_step_i) for cost_startup_step_i in self.cost_startup_step]

    def _to_list(self, x):
        return x.tolist() if hasattr(x, 'tolist') else x

    # def _to_nparray(self):
    #     pass
    # convert from list to np.ndarray
    # but parameter attributes must stay in list for ED