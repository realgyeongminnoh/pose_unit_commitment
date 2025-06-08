import numpy as np 

from src.output import Output_uc


class Input_uc:
    def __init__(
        self,
        # meta
        num_units, num_periods, num_buses, voll, let_blackout, curtail_penalty, let_curtail, exact_reserve,
        # renewable
        solar_p_max, solar_p_min, wind_p, hydro_p,
        # system
        load, system_reserve_up, system_reserve_down,
        # operational constraint
        p_min, p_max,
        ramp_up, ramp_down,
        startup_ramp, shutdown_ramp,
        min_up, min_down,
        # generation cost function
        cost_quad, cost_lin, cost_const,
        # previous horizon
        p_prev, u_prev,
        # startup cost function
        cost_startup_step,
        # mustoff
        mustoff,
    ):
        to_list = self._to_list
        
        # meta
        self.num_units = int(num_units)
        self.num_periods = int(num_periods)
        self.num_buses = int(num_buses)
        self.voll = float(voll) if let_blackout else 0.0
        self.let_blackout = bool(let_blackout)
        self.curtail_penalty = float(curtail_penalty) if let_curtail else 0.0
        self.let_curtail = bool(let_curtail)
        self.exact_reserve = bool(exact_reserve)
        # renewable
        self.solar_p_max = solar_p_max                                              # shape = (num_periods,)
        self.solar_p_min = solar_p_min                                              # shape = (num_periods,)
        self.wind_p = wind_p                                                        # shape = (num_periods,)
        self.hydro_p = hydro_p                                                      # shape = (num_periods,)
        # system
        self.load = [                                                               # shape = (num_periods, num_buses)
            to_list(load_t) for load_t in 
            np.array(load).reshape(num_periods, num_buses)
        ]                                                                           
        self.system_reserve_up = to_list(system_reserve_up)                         # shape = (num_periods,)
        self.system_reserve_down = to_list(system_reserve_down)                     # shape = (num_periods,)
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
        self.p_prev = to_list(p_prev)                                               # shape = (num_units,)
        self.u_prev = [to_list(u_prev_i) for u_prev_i in u_prev]                    # shape = (num_units, \bar\tau_i for each i)
        self._calc_min_prev()
        self.min_up_prev = to_list(self.min_up_prev)                                # shape = (num_units,)
        self.min_down_prev = to_list(self.min_down_prev)                            # shape = (num_units,)
        # startup cost function
        self.cost_startup_step = [to_list(csc_i) for csc_i in cost_startup_step]    # shape = (num_units, \bar\tau_i for each i)
        self.num_cooling_steps = [len(csc_i) for csc_i in self.cost_startup_step]   # shape = (num_units,)
        # mustoff
        self.mustoff = mustoff
        self._validate_input()

    def _to_list(self, x):
        return x.tolist() if hasattr(x, 'tolist') else x

    def _calc_min_prev(self):

        def _tail_count(seq, value):
            cnt = 0
            for x in reversed(seq):
                if x == value:
                    cnt += 1
                else:
                    break
            return cnt

        min_up_prev   = []
        min_down_prev = []
        for hist, mu, md in zip(self.u_prev, self.min_up, self.min_down):
            on_tail  = _tail_count(hist, 1)
            off_tail = _tail_count(hist, 0)

            if on_tail:          # unit is ON heading into t=0
                min_up_prev.append(max(0, mu - on_tail))
                min_down_prev.append(0)
            else:                # unit is OFF heading into t=0
                min_up_prev.append(0)
                min_down_prev.append(max(0, md - off_tail))
        self.min_up_prev, self.min_down_prev = np.array(min_up_prev).astype(int).tolist(), np.array(min_down_prev).astype(int).tolist()

    def _validate_input(self):
        period_based = {
            'solar_p_max', 'solar_p_min', 'wind_p', 'hydro_p',
            'system_reserve_up', 'system_reserve_down'
        }
        unit_based = {
            'p_min', 'p_max', 'ramp_up', 'ramp_down',
            'startup_ramp', 'shutdown_ramp', 'min_up', 'min_down',
            'cost_quad', 'cost_lin', 'cost_const',
            'min_up_prev', 'min_down_prev', 'p_prev',
            'u_prev', 'cost_startup_step'
        }

        for name in period_based:
            val = getattr(self, name)
            if len(val) != self.num_periods:
                raise ValueError(f"[Input_uc | {name}] got {len(val)}, expected {self.num_periods}.")

        for name in unit_based:
            val = getattr(self, name)
            if len(val) != self.num_units:
                raise ValueError(f"[Input_uc | {name}] got {len(val)}, expected {self.num_units}.")

        load_shape = np.array(self.load).shape
        if load_shape != (self.num_periods, self.num_buses):
            raise ValueError(f"[Input_uc | load] got shape {load_shape}, expected {(self.num_periods, self.num_buses)}.")

        for i, (u, csc) in enumerate(zip(self.u_prev, self.cost_startup_step)):
            if len(u) != len(csc):
                raise ValueError(f"[Input_uc | u_prev[{i}]] length {len(u)} ≠ cost_startup_step[{i}] length {len(csc)}.")

    def convert_to_ndarray(self):
        for attr in vars(self):
            if attr in ['u_prev', 'cost_startup_step']:
                continue
            val = getattr(self, attr)
            if isinstance(val, list):
                setattr(self, attr, np.array(val))


class Input_ed:
    def __init__(
        self,
        # meta
        time_period, num_units, num_buses, voll, let_blackout, curtail_penalty, let_curtail, exact_reserve,
        # renewable
        solar_p_max, solar_p_min,
        # uc
        input_uc: Input_uc, 
        output_uc: Output_uc,        
    ):
        to_list = self._to_list

        # meta
        self.time_period = time_period
        self.num_units = num_units
        self.num_buses = num_buses
        self.voll = voll
        self.let_blackout = let_blackout
        self.curtail_penalty = curtail_penalty
        self.let_curtail = let_curtail
        self.exact_reserve = exact_reserve
        # renewable
        self.solar_p_max = float(solar_p_max[time_period])
        self.solar_p_min = float(solar_p_min[time_period])
        self.wind_p = float(input_uc.wind_p[time_period])
        self.hydro_p = float(input_uc.hydro_p[time_period])
        # system
        self.load = input_uc.load[time_period]
        self.system_reserve_up = float(output_uc.system_reserve_up[time_period])
        self.system_reserve_down = float(output_uc.system_reserve_down[time_period])
        # operational constraint
        self.u_uc = to_list(output_uc.u[:, time_period])
        self.p_min = input_uc.p_min
        self.p_max = input_uc.p_max
        # generation cost function
        self.cost_quad = input_uc.cost_quad
        self.cost_lin = input_uc.cost_lin
        self.cost_const = input_uc.cost_const
    
    def _to_list(self, x):
        return x.tolist() if hasattr(x, 'tolist') else x
    

class Input_ed_prev:
    def __init__(
        self,
        # meta
        num_units, num_buses, voll, let_blackout, curtail_penalty, let_curtail, exact_reserve,
        # renewable
        solar_p_max, solar_p_min, wind_p, hydro_p,
        # system
        load, system_reserve_up, system_reserve_down,
        # u_prev
        u_prev,
        # gen
        p_min, p_max, cost_quad, cost_lin, cost_const,
    ):
        to_list = self._to_list

        # meta
        self.num_units = num_units
        self.num_buses = num_buses
        self.voll = voll
        self.let_blackout = let_blackout
        self.curtail_penalty = curtail_penalty
        self.let_curtail = let_curtail
        self.exact_reserve = exact_reserve
        # renewable
        self.solar_p_max = float(solar_p_max)
        self.solar_p_min = float(solar_p_min)
        self.wind_p = float(wind_p)
        self.hydro_p = float(hydro_p)
        # system
        self.load = [to_list(load_b) for load_b in load]
        self.system_reserve_up = float(system_reserve_up)
        self.system_reserve_down = float(system_reserve_down)
        # u_prev
        self.u_prev = to_list(u_prev)
        # operational
        self.p_min = p_min
        self.p_max = p_max
        self.cost_quad = cost_quad
        self.cost_lin = cost_lin
        self.cost_const = cost_const
    
    def _to_list(self, x):
        return x.tolist() if hasattr(x, 'tolist') else x