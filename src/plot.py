import matplotlib.pyplot as plt

from  .parameter import Parameter
from .output import Output


class Plot:
    def __init__(
        self,
        parameter: Parameter,
        output: Output,
    ):
        self.num_units = self.parameter.num_units
        self.num_periods = self.parameter.num_periods
        self.num_cooling_steps = self.parameter.num_cooling_steps        
        self.parameter = parameter
        self.output = output

    