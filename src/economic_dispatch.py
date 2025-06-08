import gc
import numpy as np
import gurobipy as gp

from .input import Input_ed, Input_ed_prev
from .output import Output_ed, Output_ed_prev
from .utils import GurobiModelStatus


def solve_ed(
    input_ed: Input_ed,
    output_ed: Output_ed,
):
    #################### INPUT ATTRIBUTE LOCALIZATION #################### # name convention messed up with uc but whatever
    # meta
    time_period = input_ed.time_period
    num_units = input_ed.num_units
    num_buses = input_ed.num_buses
    voll = input_ed.voll
    let_blackout = input_ed.let_blackout
    curtail_penalty = input_ed.curtail_penalty
    let_curtail = input_ed.let_curtail
    exact_reserve = input_ed.exact_reserve
    # renewable
    solar_p_max = input_ed.solar_p_max
    solar_p_min = input_ed.solar_p_min
    wind_p = input_ed.wind_p
    hydro_p = input_ed.hydro_p
    # system
    load = input_ed.load
    system_reserve_up = input_ed.system_reserve_up
    system_reserve_down = input_ed.system_reserve_down
    # operational
    u_uc = input_ed.u_uc
    p_min = input_ed.p_min
    p_max = input_ed.p_max
    # generation
    cost_quad = input_ed.cost_quad
    cost_lin = input_ed.cost_lin
    cost_const = input_ed.cost_const

    #################### MODEL ####################
    model = gp.Model()
    model.setParam("OutputFlag", 0)

    p = model.addVars(num_units, lb=0, ub=p_max)

    if let_blackout:
        z = model.addVars(range(num_buses), vtype=gp.GRB.BINARY)
    else:
        z = [1] * num_buses

    if let_curtail:
        solar_p = model.addVar(lb=solar_p_min, ub=solar_p_max)
    else:
        solar_p = solar_p_max

    #################### CONSTRAINTS ####################
    # POWER DISPATCH CONSTRAINT
    model.addConstrs(
        p[i]
        >=
        u_uc[i] * p_min[i]
        for i in range(num_units)
    )
    model.addConstrs(
        p[i]
        <=
        u_uc[i] * p_max[i]
        for i in range(num_units)
    )

    # BALANCE CONSTRAINT
    CONSTR_LOAD = model.addConstr(
        gp.quicksum(
            p[i]
            for i in range(num_units)
        )
        +
        solar_p
        ==
        gp.quicksum(
            z[b] * load[b]
            for b in range(num_buses)
        )
        - wind_p - hydro_p,
    )

    # RESERVE CONSTRAINTS
    if exact_reserve:
        CONSTR_RESERVEUP = model.addConstr(
            gp.quicksum(
                u_uc[i] * p_max[i]
                -
                p[i]
                for i in range(num_units)
            )
            ==
            system_reserve_up,
        )
        CONSTR_RESERVEDOWN = model.addConstr(
            gp.quicksum(
                p[i]
                -
                u_uc[i] * p_min[i]
                for i in range(num_units)
            )
            >=
            system_reserve_down,
        )
    else: # i literally have no idea now because == gives infeasible while >= gives optimal
        CONSTR_RESERVEUP = model.addConstr(
            gp.quicksum(
                u_uc[i] * p_max[i]
                -
                p[i]
                for i in range(num_units)
            )
            >=
            system_reserve_up,
        )
        CONSTR_RESERVEDOWN = model.addConstr(
            gp.quicksum(
                p[i]
                -
                u_uc[i] * p_min[i]
                for i in range(num_units)
            )
            >=
            system_reserve_down,
        )

    #################### OBJECTIVE ####################
    # SYSTEM GENERATION COST
    total_cost_generation = gp.quicksum(
        cost_quad[i] * p[i] * p[i]
        + cost_lin[i] * p[i]
        + cost_const[i] * u_uc[i]
        for i in range(num_units)
    )

    # SYSTEM VALUE OF LOST LOST RESERVE
    total_cost_voll = voll * gp.quicksum(
        (1 - z[b]) * load[b]
        for b in range(num_buses)
    )

    # SYSTEM CURTAIL PENALTY
    total_cost_curtail_penalty = curtail_penalty * (solar_p_max - solar_p)

    total_cost_system = total_cost_generation + total_cost_voll + total_cost_curtail_penalty
    model.setObjective(total_cost_system, gp.GRB.MINIMIZE)
    model.optimize()

    # NON-OPTIMALITY
    if model.Status != gp.GRB.OPTIMAL:
        # if True then probably should let_curtail True then incrementally decrease solar_p_min from solar_p_max recursively to get OPTIMALITY
        raise GurobiModelStatus(f"[solve_ed | {time_period}] {model.Status}")
    
    #################### OUTPUT_ED REGISTER ####################
    output_ed.cost_system[time_period] = total_cost_system.getValue()
    output_ed.cost_generation[time_period] = total_cost_generation.getValue()
    output_ed.cost_voll[time_period] = total_cost_voll.getValue() if let_blackout else 0.0
    output_ed.cost_curtail_penalty[time_period] = total_cost_curtail_penalty.getValue() if let_curtail else 0.0

    output_ed.z[time_period] = np.array(model.getAttr("X", z).select()).reshape(num_buses,) if let_blackout else np.array(z)
    output_ed.p[:, time_period] = np.array(model.getAttr("X", p).select()).reshape(num_units,)

    output_ed.blackout[time_period] = (1- output_ed.z[time_period]) * np.array(load)
    output_ed.solar_p[time_period] = solar_p.X if let_curtail else solar_p_max
    output_ed.solar_curtail[time_period] = solar_p_max - solar_p.X if let_curtail else solar_p_max

    output_ed.marginal_smp[time_period] = CONSTR_LOAD.Pi
    output_ed.marginal_reserve_up[time_period] = CONSTR_RESERVEUP.Pi
    output_ed.marginal_reserve_down[time_period] = CONSTR_RESERVEDOWN.Pi



def solve_ed_prev(
    input_ed_prev: Input_ed_prev,
    output_ed_prev: Output_ed_prev,
    only_p_prev: bool,
):
    #################### INPUT ATTRIBUTE LOCALIZATION #################### # name convention messed up with uc but whatever
    # meta
    num_units = input_ed_prev.num_units
    num_buses = input_ed_prev.num_buses
    voll = input_ed_prev.voll
    let_blackout = input_ed_prev.let_blackout
    curtail_penalty = input_ed_prev.curtail_penalty
    let_curtail = input_ed_prev.let_curtail
    exact_reserve = input_ed_prev.exact_reserve
    # renewable
    solar_p_max = input_ed_prev.solar_p_max
    solar_p_min = input_ed_prev.solar_p_min
    wind_p = input_ed_prev.wind_p
    hydro_p = input_ed_prev.hydro_p
    # system
    load = input_ed_prev.load
    system_reserve_up = input_ed_prev.system_reserve_up
    system_reserve_down = input_ed_prev.system_reserve_down
    # u_prev
    u_prev = input_ed_prev.u_prev
    # operational
    p_min = input_ed_prev.p_min
    p_max = input_ed_prev.p_max
    cost_quad = input_ed_prev.cost_quad
    cost_lin = input_ed_prev.cost_lin
    cost_const = input_ed_prev.cost_const

    #################### MODEL ####################
    model = gp.Model()
    model.setParam("OutputFlag", 0)

    p = model.addVars(num_units, lb=0, ub=p_max)

    if let_blackout:
        z = model.addVars(range(num_buses), vtype=gp.GRB.BINARY)
    else:
        z = [1] * num_buses

    if let_curtail:
        solar_p = model.addVar(lb=solar_p_min, ub=solar_p_max)
    else:
        solar_p = solar_p_max

    #################### CONSTRAINTS ####################
    # POWER DISPATCH CONSTRAINT
    model.addConstrs(
        p[i]
        >=
        u_prev[i] * p_min[i]
        for i in range(num_units)
    )
    model.addConstrs(
        p[i]
        <=
        u_prev[i] * p_max[i]
        for i in range(num_units)
    )

    # BALANCE CONSTRAINT
    CONSTR_LOAD = model.addConstr(
        gp.quicksum(
            p[i]
            for i in range(num_units)
        )
        +
        solar_p
        ==
        gp.quicksum(
            z[b] * load[b]
            for b in range(num_buses)
        )
        - wind_p - hydro_p,
    )

    # RESERVE CONSTRAINTS
    if exact_reserve:
        CONSTR_RESERVEUP = model.addConstr(
            gp.quicksum(
                u_prev[i] * p_max[i]
                -
                p[i]
                for i in range(num_units)
            )
            ==
            system_reserve_up,
        )
        CONSTR_RESERVEDOWN = model.addConstr(
            gp.quicksum(
                p[i]
                -
                u_prev[i] * p_min[i]
                for i in range(num_units)
            )
            >=
            system_reserve_down,
        )
    else: # i literally have no idea now because == gives infeasible while >= gives optimal
        CONSTR_RESERVEUP = model.addConstr(
            gp.quicksum(
                u_prev[i] * p_max[i]
                -
                p[i]
                for i in range(num_units)
            )
            >=
            system_reserve_up,
        )
        CONSTR_RESERVEDOWN = model.addConstr(
            gp.quicksum(
                p[i]
                -
                u_prev[i] * p_min[i]
                for i in range(num_units)
            )
            >=
            system_reserve_down,
        )

    #################### OBJECTIVE ####################
    # SYSTEM GENERATION COST
    total_cost_generation = gp.quicksum(
        cost_quad[i] * p[i] * p[i]
        + cost_lin[i] * p[i]
        + cost_const[i] * u_prev[i]
        for i in range(num_units)
    )

    # SYSTEM VALUE OF LOST LOST RESERVE
    total_cost_voll = voll * gp.quicksum(
        (1 - z[b]) * load[b]
        for b in range(num_buses)
    )

    # SYSTEM CURTAIL PENALTY
    total_cost_curtail_penalty = curtail_penalty * (solar_p_max - solar_p)

    total_cost_system = total_cost_generation + total_cost_voll + total_cost_curtail_penalty
    model.setObjective(total_cost_system, gp.GRB.MINIMIZE)
    model.optimize()

    # NON-OPTIMALITY
    if model.Status != gp.GRB.OPTIMAL:
        # if True then probably should let_curtail True then incrementally decrease solar_p_min from solar_p_max recursively to get OPTIMALITY
        raise GurobiModelStatus(f"[solve_ed_prev] {model.Status}")

    #################### OUTPUT_ED REGISTER ####################
    if only_p_prev:
        return np.array(model.getAttr("X", p).select()).reshape(num_units,).tolist()
    
    output_ed_prev.cost_system = total_cost_system.getValue()
    output_ed_prev.cost_generation = total_cost_generation.getValue()
    output_ed_prev.cost_voll = total_cost_voll.getValue() if let_blackout else 0.0
    output_ed_prev.cost_curtail_penalty = total_cost_curtail_penalty.getValue() if let_curtail else 0.0

    output_ed_prev.z = np.array(model.getAttr("X", z).select()).reshape(num_buses,) if let_blackout else np.array(z)
    output_ed_prev.p = np.array(model.getAttr("X", p).select()).reshape(num_units,)

    output_ed_prev.blackout = (1- output_ed_prev.z) * np.array(load)
    output_ed_prev.solar_p = solar_p.X if let_curtail else solar_p_max
    output_ed_prev.solar_curtail = solar_p_max - solar_p.X if let_curtail else solar_p_max

    output_ed_prev.marginal_smp = CONSTR_LOAD.Pi
    output_ed_prev.marginal_reserve_up = CONSTR_RESERVEUP.Pi
    output_ed_prev.marginal_reserve_down = CONSTR_RESERVEDOWN.Pi

    output_ed_prev.cost_retailor = float(output_ed_prev.p.sum(axis=0) * output_ed_prev.marginal_smp)