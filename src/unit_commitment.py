import gc
import numpy as np
import gurobipy as gp

from .input import Input_uc
from .output import Output_uc
from .utils import GurobiModelStatus


def solve_uc(
    input_uc: Input_uc,
    output_uc: Output_uc,
):
    #################### INPUT ATTRIBUTE LOCALIZATION ####################
    # meta
    num_units = input_uc.num_units
    num_periods = input_uc.num_periods
    num_buses = input_uc.num_buses
    voll = input_uc.voll
    let_blackout = input_uc.let_blackout
    curtail_penalty = input_uc.curtail_penalty
    let_curtail = input_uc.let_curtail
    exact_reserve = input_uc.exact_reserve
    # renewable
    solar_p_max = input_uc.solar_p_max                 # solar_p_max [t]
    solar_p_min = input_uc.solar_p_min                 # solar_p_min [t]
    wind_p = input_uc.wind_p                           # wind_p [t]
    hydro_p = input_uc.hydro_p                         # hydro_p [t]
    # system
    load = input_uc.load                               # load [t] [b]
    system_reserve_up = input_uc.system_reserve_up     # system_reserve_up [t]
    system_reserve_down = input_uc.system_reserve_down # system_reserve_down [t]
    # operational constraint
    p_min = input_uc.p_min                             # p_min [i]
    p_max = input_uc.p_max                             # p_max [i]
    ramp_up = input_uc.ramp_up                         # ramp_up [i]
    ramp_down = input_uc.ramp_down                     # ramp_down [i]
    startup_ramp = input_uc.startup_ramp               # startup_ramp [i]
    shutdown_ramp = input_uc.shutdown_ramp             # shutdown_ramp [i]
    min_up = input_uc.min_up                           # min_up [i]
    min_down = input_uc.min_down                       # min_down [i]
    # generation cost function
    cost_quad = input_uc.cost_quad                     # cost_quad [i]
    cost_lin = input_uc.cost_lin                       # cost_lin [i]
    cost_const = input_uc.cost_const                   # cost_const [i]
    # previous horizon
    min_up_prev = input_uc.min_up_prev                 # min_up_prev [i]
    min_down_prev = input_uc.min_down_prev             # min_down_prev [i]
    p_prev = input_uc.p_prev                           # p_prev [i]
    u_prev = input_uc.u_prev                           # u_prev [i] [tau]
    # startup cost function
    cost_startup_step = input_uc.cost_startup_step     # cost_startup_step [i] [tau]
    num_cooling_steps = input_uc.num_cooling_steps     # num_cooling_steps [i]

    #################### MODEL ####################
    model = gp.Model()
    model.setParam("OutputFlag", 0)

    # helper for variable setup
    p_ub = [[p_max_i] * num_periods for p_max_i in p_max]
    cost_startup_step_ub = [[ub] * num_periods for ub in [max(cost_i) for cost_i in cost_startup_step]]

    # variables and auxiliary variables
    u = model.addVars(range(num_units), range(num_periods), vtype=gp.GRB.BINARY)
    p = model.addVars(range(num_units), range(num_periods), lb=0, ub=p_ub)
    p_up = model.addVars(range(num_units), range(num_periods), lb=0, ub=p_ub)
    p_down = model.addVars(range(num_units), range(num_periods), lb=0, ub=p_ub)
    cost_startup = model.addVars(range(num_units), range(num_periods), lb=0, ub=cost_startup_step_ub)

    if let_blackout:
        z = model.addVars(range(num_periods), range(num_buses), vtype=gp.GRB.BINARY)
    else:
        z = gp.tupledict({
            (t, b): 1
            for t in range(num_periods)
            for b in range(num_buses)
        })

    if let_curtail:
        solar_p = model.addVars(range(num_periods), lb=solar_p_min, ub=solar_p_max)
    else:
        solar_p = solar_p_max

    # helper cleanup
    del p_ub, cost_startup_step_ub
    gc.collect()

    # helper functions for minus index proof previous horizon lookup
    def p_minus_proof(i, t_):
        return p[i, t_] if t_ >= 0 else p_prev[i]
    
    def u_minus_proof(i, t_):
        return u[i, t_] if t_ >= 0 else u_prev[i][t_]
    
    #################### CONSTRAINTS ####################
    # LOAD GENERATION BALANCE + ROLLING BLACKOUT
    model.addConstrs(
        gp.quicksum(
            p[i, t]
            for i in range(num_units)
        )
        +
        solar_p[t]
        ==
        gp.quicksum(
            z[t, b] * load[t][b]
            for b in range(num_buses)
        )
        - wind_p[t] - hydro_p[t]
        for t in range(num_periods)
    )

    # SYSTEM RESERVE UP & DOWN REQUIREMENT
    if exact_reserve:
        model.addConstrs(
            gp.quicksum(
                p_up[i, t] - p[i, t]
                for i in range(num_units)
            )
            ==
            system_reserve_up[t]
            for t in range(num_periods)
        )
        model.addConstrs(
            gp.quicksum(
                p[i, t] - p_down[i, t]
                for i in range(num_units)
            )
            ==
            system_reserve_down[t]
            for t in range(num_periods)
        )
    else:
        model.addConstrs(
            gp.quicksum(
                p_up[i, t] - p[i, t]
                for i in range(num_units)
            )
            >=
            system_reserve_up[t]
            for t in range(num_periods)
        )
        model.addConstrs(
            gp.quicksum(
                p[i, t] - p_down[i, t]
                for i in range(num_units)
            )
            >=
            system_reserve_down[t]
            for t in range(num_periods)
        )

    # P_DOWN, P, P_UP BOUNDED BY EACH OTHER & STATUS RESPECTING P_MIN, P_MAX
    model.addConstrs(
        p_down[i, t]
        >=
        u[i, t] * p_min[i]
        for i in range(num_units)
        for t in range(num_periods)
    )
    model.addConstrs(
        p[i, t]
        >=
        p_down[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )
    model.addConstrs(
        p_up[i, t]
        >=
        p[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )
    model.addConstrs(
        u[i, t] * p_max[i]
        >=
        p_up[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )

    # BACKWARD; RAMPUP/STARTUP RAMP-AWARE P+R_UP
    model.addConstrs(
        p_up[i, t]
        <=
        p_minus_proof(i, t - 1)
        + ramp_up[i] * u_minus_proof(i, t - 1)
        + startup_ramp[i] * (u[i, t] - u_minus_proof(i, t - 1))
        + p_max[i] * (1 - u[i, t])
        for i in range(num_units)
        for t in range(num_periods)
    )

    # FORWARD; TO SHUTDOWN NEXT HOUR P+R_UP SHOULD NOT EXCEED SHUTDOWN RAMP
    model.addConstrs(
        p_up[i, t]
        <=
        p_max[i] * u[i, t + 1]
        + shutdown_ramp[i] * (u[i, t] - u[i, t + 1])
        for i in range(num_units)
        for t in range(num_periods - 1)
    )

    # BACKWARD; RAMPDOWN/SHUTDOWN RAMP-AWARE P+R_DOWN
    model.addConstrs(
        p_down[i, t]
        >=
        p_minus_proof(i, t - 1)
        - ramp_down[i] * u[i, t]
        - shutdown_ramp[i] * (u_minus_proof(i, t - 1) - u[i, t])
        - p_max[i] * (1 - u_minus_proof(i, t - 1))
        for i in range(num_units)
        for t in range(num_periods)
    )

    # FORWARD; im going crazy because above backward didn't respect ramps; this works
    model.addConstrs(
        p[i, t] - p_down[i, t] # R_down explicit
        <=
        ramp_down[i] * u[i, t + 1]
        + shutdown_ramp[i] * (u[i, t] - u[i, t + 1])
        + p_max[i] * (1 - u[i, t])
        for i in range(num_units)
        for t in range(num_periods - 1)
    )

    # TRIPLE MIN UP TIME CONSTRAINTS
    model.addConstrs(
        gp.quicksum(
            1 - u[i, t]
            for t in range(min_up_prev[i])
        )
        ==
        0
        for i in range(num_units)
    )
    model.addConstrs(
        gp.quicksum(
            u[i, t_delta]
            for t_delta in range(t, t + min_up[i])
        )
        >=
        min_up[i] * (
            u[i, t] - u_minus_proof(i, t - 1)
        )
        for i in range(num_units)
        for t in range(min_up_prev[i], num_periods - min_up[i] + 1)
    )
    model.addConstrs(
        gp.quicksum(
            u[i, t_delta]
            for t_delta in range(t, num_periods)
        )
        >=
        (num_periods - t) * (
            u[i, t] - u_minus_proof(i, t - 1)
        )
        for i in range(num_units)
        for t in range(num_periods - min_up[i] + 1, num_periods)
    )

    # TRIPLE MIN DOWN TIME CONSTRAINTS
    model.addConstrs(
        gp.quicksum(
            u[i, t]
            for t in range(min_down_prev[i])
        )
        ==
        0
        for i in range(num_units)
    )
    model.addConstrs(
        gp.quicksum(
            1 - u[i, t_delta]
            for t_delta in range(t, t + min_down[i])
        )
        >=
        min_down[i] * (
            u_minus_proof(i, t - 1) - u[i, t]
        )
        for i in range(num_units)
        for t in range(min_down_prev[i], num_periods - min_down[i] + 1)
    )
    model.addConstrs(
        gp.quicksum(
            1 - u[i, t_delta]
            for t_delta in range(t, num_periods)
        )
        >=
        (num_periods - t) * (
            u_minus_proof(i, t - 1) - u[i, t]
        )
        for i in range(num_units)
        for t in range(num_periods - min_down[i] + 1, num_periods)
    )

    # STARTUP COST
    model.addConstrs(
        cost_startup[i, t]
        >=
        cost_startup_step[i][tau - 1] * (
            u[i, t]
            -
            gp.quicksum(
                u_minus_proof(i, t - t_delta)
                for t_delta in range(1, tau + 1)
            )
        )
        for i in range(num_units)
        for t in range(num_periods)
        for tau in range(1, num_cooling_steps[i] + 1)
    )

    #################### OBJECTIVE ####################
    # SYSTEM GENERATION COST
    total_cost_generation = gp.quicksum(
        cost_quad[i] * p[i, t] * p[i, t]
        + cost_lin[i] * p[i, t]
        + cost_const[i] * u[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )

    # SYSTEM STARTUP COST
    total_cost_startup = gp.quicksum(
        cost_startup[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )

    # SYSTEM VALUE OF LOST LOST RESERVE
    total_cost_voll = voll * gp.quicksum(
        (1 - z[t, b]) * load[t][b]
        for t in range(num_periods)
        for b in range(num_buses)
    )

    # SYSTEM CURTAIL PENALTY
    total_cost_curtail_penalty = curtail_penalty * gp.quicksum(
        solar_p_max[t] - solar_p[t]
        for t in range(num_periods)
    )

    total_cost_system = total_cost_generation + total_cost_startup + total_cost_voll + total_cost_curtail_penalty
    model.setObjective(total_cost_system, gp.GRB.MINIMIZE)
    model.optimize()

    # NON-OPTIMALITY
    if model.Status != gp.GRB.OPTIMAL:
        # if True then probably should let_curtail True then incrementally decrease solar_p_min from solar_p_max recursively to get OPTIMALITY
        raise GurobiModelStatus(model.Status)
    
    #################### OUTPUT_UC REGISTER ####################
    output_uc.total_cost_system = total_cost_system.getValue()
    output_uc.total_cost_generation = total_cost_generation.getValue()
    output_uc.total_cost_startup = total_cost_startup.getValue()
    output_uc.total_cost_voll = total_cost_voll.getValue() if let_blackout else 0.0
    output_uc.total_cost_curtail_penalty = total_cost_curtail_penalty.getValue() if let_curtail else 0.0

    output_uc.u = np.array(model.getAttr("X", u).select()).reshape(num_units, num_periods)
    output_uc.z = np.array(model.getAttr("X", z).select()).reshape(num_periods, num_buses) if let_blackout else np.ones((num_periods, num_buses))
    output_uc.p = np.array(model.getAttr("X", p).select()).reshape(num_units, num_periods)
    output_uc.r_up = np.array(model.getAttr("X", p_up).select()).reshape(num_units, num_periods) - output_uc.p
    output_uc.r_down = output_uc.p - np.array(model.getAttr("X", p_down).select()).reshape(num_units, num_periods)

    output_uc.system_reserve_up = output_uc.r_up.sum(axis=0)
    output_uc.system_reserve_down = output_uc.r_down.sum(axis=0)

    output_uc.blackout = (1 - output_uc.z) * np.array(load)
    output_uc.solar_p = np.array(model.getAttr("X", solar_p).select()).reshape(num_periods,) if let_curtail else np.array(solar_p_max)
    output_uc.solar_curtail = np.array(solar_p_max) - output_uc.solar_p
    
    output_uc.cost_generation = (
        np.array(cost_quad)[:, None] * output_uc.p * output_uc.p
        + np.array(cost_lin)[:, None] * output_uc.p
        + np.array(cost_const)[:, None] * output_uc.u
    ).sum(axis=0)
    output_uc.cost_startup = np.array(model.getAttr("X", cost_startup).select()).reshape(num_units, num_periods).sum(axis=0)
    output_uc.cost_voll = voll * output_uc.blackout.sum(axis=1)
    output_uc.cost_curtail_penalty = curtail_penalty * output_uc.solar_curtail
    output_uc.cost_system = output_uc.cost_generation + output_uc.cost_startup + output_uc.cost_voll + output_uc.cost_curtail_penalty

    if not np.all([
        output_uc.cost_generation.sum() == output_uc.total_cost_generation,
        output_uc.cost_startup.sum() == output_uc.total_cost_startup,
        output_uc.cost_voll.sum() == output_uc.total_cost_voll,
        output_uc.cost_curtail_penalty.sum() == output_uc.total_cost_curtail_penalty,
    ]):
        raise ValueError("something wrong")