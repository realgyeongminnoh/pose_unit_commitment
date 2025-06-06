import numpy as np 


class Parameter:
    def __init__(
        self,
        # meta
        num_units, num_periods, num_buses,
        # renewable
        solar_p_max, solar_p_min, wind_p, hydro_p,
        # system
        load, reserve_up, reserve_down,
        # operational constraint
        p_min, p_max,
        ramp_up, ramp_down,
        startup_ramp, shutdown_ramp,
        min_up, min_down,
        # generation cost function
        cost_quad, cost_lin, cost_const,
        # previous horizon
        p_prev, u_prev, min_up_prev, min_down_prev,
        # startup cost function
        cost_startup_step,
    ):
        to_list = self._to_list

        # meta
        self.num_units = int(num_units)
        self.num_periods = int(num_periods)
        self.num_buses = int(num_buses)
        # renewable
        self.solar_p_max = solar_p_max                                              # shape = (num_periods,)
        self.solar_p_min = solar_p_min                                              # shape = (num_periods,)
        self.wind_p = wind_p                                                        # shape = (num_periods,)
        self.hydro_p = hydro_p                                                      # shape = (num_periods,)
        # system
        self.load = [to_list(load_t) for load_t in load]                            # shape = (num_periods, num_buses)
        self.reserve_up = to_list(reserve_up)                                       # shape = (num_periods,)
        self.reserve_down = to_list(reserve_down)                                   # shape = (num_periods,)
        # operational constraint
        self.p_min = to_list(p_min)                                                 # shape = (num_units,)
        self.p_max = to_list(p_max)                                                 # shape = (num_units,)
        self.ramp_up = to_list(ramp_up)                                             # shape = (num_units,)
        self.ramp_down = to_list(ramp_down)                                         # shape = (num_units,)
        self.startup_ramp = to_list(startup_ramp)                                   # shape = (num_units,)
        self.shutdown_ramp = to_list(shutdown_ramp)                                 # shape = (num_units,)
        self.min_up = to_list(min_up)                                               # shape = (num_units,)
        self.min_down = to_list(min_down)                                           # shape = (num_units,)
        # generation cost function
        self.cost_quad = to_list(cost_quad)                                         # shape = (num_units,)
        self.cost_lin = to_list(cost_lin)                                           # shape = (num_units,)
        self.cost_const = to_list(cost_const)                                       # shape = (num_units,)
        # previous horizon
        self.min_up_prev = to_list(min_up_prev)                                     # shape = (num_units,)
        self.min_down_prev = to_list(min_down_prev)                                 # shape = (num_units,)
        self.p_prev = to_list(p_prev)                                               # shape = (num_units,)
        self.u_prev = [to_list(u_prev_i) for u_prev_i in u_prev]                    # shape = (num_units, \bar\tau_i for each i)
        # startup cost function
        self.cost_startup_step = [to_list(csc_i) for csc_i in cost_startup_step]    # shape = (num_units, \bar\tau_i for each i)
        self.num_cooling_steps = [len(csc_i) for csc_i in self.cost_startup_step]   # shape = (num_units,)

        self._validate_input()

    def _to_list(self, x):
        return x.tolist() if hasattr(x, 'tolist') else x
    
    def _validate_input(self):
        for name in [
            'solar_p_max', 'wind_p', 'hydro_p', 'reserve_up', 'reserve_down',
            'p_min', 'p_max', 'ramp_up', 'ramp_down', 'startup_ramp', 'shutdown_ramp',
            'min_up', 'min_down', 'cost_quad', 'cost_lin', 'cost_const',
            'min_up_prev', 'min_down_prev', 'p_prev', 'u_prev', 'cost_startup_step'
        ]:
            val = getattr(self, name)
            expected_len = self.num_periods if name in ['reserve_up', 'reserve_down'] and not isinstance(val[0], list) else self.num_units
            if isinstance(val, list) and len(val) != expected_len:
                raise ValueError(f"[Parameter | {name}] got {len(val)}, expected {expected_len}")

        if not np.array(self.load).shape == (self.num_periods, self.num_buses):
            raise ValueError(f"[Parameter | load] got {np.array(self.load).shape}, expected {(self.num_periods, self.num_buses)}")

        if not all(len(u) == n for u, n in zip(self.u_prev, self.num_cooling_steps)):
            raise ValueError("[Parameter | u_prev] mistmatch in shape between u_prev and num_cooling_steps")

    def convert_to_ndarray(self):
        for attr in vars(self):
            if attr in ['u_prev', 'cost_startup_step', 'num_cooling_steps']:
                continue
            val = getattr(self, attr)
            if isinstance(val, list):
                setattr(self, attr, np.array(val))