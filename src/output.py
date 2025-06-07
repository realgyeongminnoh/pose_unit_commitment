import numpy as np


class Output_uc:
    def __init__(
        self,
    ):
        pass


class Output_ed:
    def __init__(self, num_periods, num_units, num_buses):
        self.total_cost_retailor = None
        self.total_cost_system = None
        self.total_cost_generation = None
        self.total_cost_voll = None
        self.total_cost_curtail_penalty = None

        self.cost_retailor = None
        self.cost_system = np.zeros(num_periods)
        self.cost_generation = np.zeros(num_periods)
        self.cost_voll = np.zeros(num_periods)
        self.cost_curtail_penalty = np.zeros(num_periods)

        self.z = np.ones((num_periods, num_buses))
        self.p = np.zeros((num_units, num_periods))
        
        self.blackout = np.zeros((num_periods, num_buses))
        self.solar_p = np.zeros(num_periods)
        self.solar_curtail = np.zeros(num_periods)

        self.smp = np.zeros(num_periods)
        self.cost_reserve_up = np.zeros(num_periods)
        self.cost_reserve_down = np.zeros(num_periods)


    def compute_auxiliary_results(self):
        self.total_cost_system = float(self.cost_system.sum())
        self.total_cost_generation = float(self.cost_generation.sum())
        self.total_cost_voll = float(self.cost_voll.sum())
        self.total_cost_curtail_penalty = float(self.cost_curtail_penalty.sum())

        self.cost_retailor = self.p.sum(axis=0) * self.smp
        self.total_cost_retailor = float(self.cost_retailor.sum())