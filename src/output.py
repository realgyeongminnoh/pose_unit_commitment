import numpy as np


class Output_ed_prev:
    def __init__(self):
        self.cost_retailor = None
        self.cost_system = None
        self.cost_generation = None
        self.cost_voll = None
        self.cost_curtail_penalty = None

        self.z = None
        self.p = None
        
        self.blackout = None
        self.solar_p = None
        self.solar_curtail = None

        self.marginal_smp = None
        self.marginal_reserve_up = None
        self.marginal_reserve_down = None


class Output_ed:
    def __init__(self, num_periods, num_units, num_buses):
        self.total_cost_retailor = None ###
        self.total_cost_system = None
        self.total_cost_generation = None
        self.total_cost_voll = None
        self.total_cost_curtail_penalty = None

        self.cost_retailor = None ###
        self.cost_system = np.zeros(num_periods)
        self.cost_generation = np.zeros(num_periods)
        self.cost_voll = np.zeros(num_periods)
        self.cost_curtail_penalty = np.zeros(num_periods)

        self.z = np.ones((num_periods, num_buses))
        self.p = np.zeros((num_units, num_periods))
        
        self.blackout = np.zeros((num_periods, num_buses))
        self.solar_p = np.zeros(num_periods)
        self.solar_curtail = np.zeros(num_periods)

        self.marginal_smp = np.zeros(num_periods) ###
        self.marginal_reserve_up = np.zeros(num_periods) ###
        self.marginal_reserve_down = np.zeros(num_periods) ###


    def compute_auxiliary_results(self):
        self.total_cost_system = float(self.cost_system.sum())
        self.total_cost_generation = float(self.cost_generation.sum())
        self.total_cost_voll = float(self.cost_voll.sum())
        self.total_cost_curtail_penalty = float(self.cost_curtail_penalty.sum())

        self.cost_retailor = self.p.sum(axis=0) * self.marginal_smp
        self.total_cost_retailor = float(self.cost_retailor.sum())


class Output_uc:
    def __init__(self):
        self.total_cost_system = None
        self.total_cost_generation = None
        self.total_cost_startup = None ###
        self.total_cost_voll = None
        self.total_cost_curtail_penalty = None
        self.total_cost_reserve_up = None ###
        self.total_cost_reserve_down = None ###

        self.cost_system = None
        self.cost_generation = None
        self.cost_startup = None ###
        self.cost_voll = None
        self.cost_curtail_penalty = None
        self.cost_reserve_up = None ###
        self.cost_reserve_down = None ###

        self.u = None ###
        self.z = None

        self.p = None
        self.r_up = None ###
        self.r_down = None ###
        self.system_reserve_up = None ###
        self.system_reserve_down = None ###
        
        self.blackout = None
        self.solar_p = None
        self.solar_curtail = None

    def compute_auxiliary_results(self, output_ed: Output_ed):
        self.cost_reserve_up = (self.r_up * output_ed.marginal_reserve_up).sum(axis=0)
        self.cost_reserve_down = (self.r_down * output_ed.marginal_reserve_down).sum(axis=0)
        self.total_cost_reserve_up = float(self.cost_reserve_up.sum())
        self.total_cost_reserve_down = float(self.cost_reserve_down.sum())




class Output_uc_snapshot:
    def __init__(self, num_periods, num_units):
        self.total_cost_generation = None
        self.cost_generation = np.ones(num_periods)

        self.u = np.ones((num_periods, num_units)) ###
        self.p = np.zeros((num_periods, num_units))


    def compute_auxiliary_results(self):
        self.total_cost_generation = float(self.cost_generation.sum())
        self.u = self.u.transpose()
        self.p = self.p.transpose()