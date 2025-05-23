import gc
import gurobipy as gp

from .parameter import Parameter
from .utils import GurobiModelStatus


def solve_uc(
    parameter: Parameter
):
    # attribute localization
    num_units = parameter.num_units
    num_periods = parameter.num_periods
    num_cooling_steps = parameter.num_cooling_steps
    load = parameter.load
    reserve = parameter.reserve
    p_min = parameter.p_min
    p_max = parameter.p_max
    ramp_up = parameter.ramp_up
    ramp_down = parameter.ramp_down
    startup_ramp = parameter.startup_ramp
    shutdown_ramp = parameter.shutdown_ramp
    min_up = parameter.min_up
    min_down = parameter.min_down
    p_prev = parameter.p_prev
    u_prev = parameter.u_prev
    min_up_prev = parameter.min_up_prev
    min_down_prev = parameter.min_down_prev
    cost_quad = parameter.cost_quad
    cost_lin = parameter.cost_lin
    cost_const = parameter.cost_const
    cost_startup_step = parameter.cost_startup_step

    # model
    model = gp.Model()
    model.setParam("OutputFlag", 0)

    # helper ub
    p_ub = [[p_max_i] * num_periods for p_max_i in p_max]
    cost_startup_step_ub = [[ub] * num_periods for ub in [max(cost_i) for cost_i in cost_startup_step]]

    # variables
    p = model.addVars(range(num_units), range(num_periods), lb=0, ub=p_ub)
    u = model.addVars(range(num_units), range(num_periods), vtype=gp.GRB.BINARY)

    # auxiliary variables
    p_bar = model.addVars(range(num_units), range(num_periods), lb=0, ub=p_ub)
    cost_startup = model.addVars(range(num_units), range(num_periods), lb=0, ub=cost_startup_step_ub)

    # cleanup
    del p_ub, cost_startup_step_ub
    gc.collect()

    # helper functions for minus index proof previous horizon lookup
    def p_minus_proof(i, t_):
        return p[i, t_] if t_ >= 0 else p_prev[i][t_]
    
    def u_minus_proof(i, t_):
        return u[i, t_] if t_ >= 0 else u_prev[i][t_]
        
    # 
    model.addConstrs(
        gp.quicksum(
            p[i, t] 
            for i in range(num_units)
        )
        == 
        load[t]
        for t in range(num_periods)
    )

    #
    model.addConstrs(
        gp.quicksum(
            p_bar[i, t] 
            for i in range(num_units)
        )
        >=
        load[t] + reserve[t]
        for t in range(num_periods)
    )

    #
    model.addConstrs(
        p[i, t]
        >=
        u[i, t] * p_min[i]
        for i in range(num_units)
        for t in range(num_periods)
    )

    #
    model.addConstrs(
        p_bar[i, t]
        >=
        p[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )

    #
    model.addConstrs(
        p_bar[i, t]
        <=
        u[i, t] * p_max[i]
        for i in range(num_units)
        for t in range(num_periods)
    )

    #
    model.addConstrs(
        p_bar[i, t]
        <=
        p_minus_proof(i, t - 1)
        + ramp_up[i] * u_minus_proof(i, t - 1)
        + startup_ramp[i] * (u[i, t] - u_minus_proof(i, t - 1))
        + p_max[i] * (1 - u[i, t])
        for i in range(num_units)
        for t in range(num_periods)
    )

    #
    model.addConstrs(
        p_bar[i, t]
        <=
        p_max[i] * u[i, t + 1]
        + shutdown_ramp[i] * (u[i, t] - u[i, t + 1])
        for i in range(num_units)
        for t in range(num_periods - 1)
    )

    #
    model.addConstrs(
        p_minus_proof(i, t - 1) - p[i, t]
        <=
        ramp_down[i] * u[i, t]
        + shutdown_ramp[i] * (u_minus_proof(i, t - 1) - u[i, t])
        + p_max[i] * (1 - u_minus_proof(i, t - 1))
        for i in range(num_units)
        for t in range(num_periods)
    )

    #
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

    # 
    model.addConstrs(
        gp.quicksum(
            1 - u[i, t]
            for t in range(min_up_prev[i])
        )
        ==
        0
        for i in range(num_units)
    )

    #
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

    # 
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

    #
    model.addConstrs(
        gp.quicksum(
            u[i, t]
            for t in range(min_down_prev[i])
        )
        ==
        0
        for i in range(num_units)
    )

    #
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

    #
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
    
    #
    total_cost_generation = gp.quicksum(
        cost_quad[i] * p[i, t] * p[i, t]
        + cost_lin[i] * p[i, t]
        + cost_const[i] * u[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )

    #
    total_cost_startup = gp.quicksum(
        cost_startup[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )

    #
    total_cost_system = total_cost_generation + total_cost_startup

    #
    model.setObjective(total_cost_system, gp.GRB.MINIMIZE)

    #
    model.optimize()

    if model.Status != gp.GRB.OPTIMAL:
        raise GurobiModelStatus(model.Status)
    return model