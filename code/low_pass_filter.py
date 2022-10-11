#!/usr/bin/self.env python
import numpy as np
import pdb

class LowPassFilter:
    def __init__(self, low_pass_filter_coeff = 2):
        self.coefficient_ = low_pass_filter_coeff
        self.previous_measurements_ = np.array([0.,0.])
        self.previous_filtered_measurement_ = 0.
        self.scale_term_ = (1./(1.+self.coefficient_))
        self.feedback_term_ = (1.-low_pass_filter_coeff)
    
    def reset(self, data):
        self.previous_measurements_[0] = data
        self.previous_measurements_[1] = data 
        self.previous_filtered_measurement_ = data 
    

    def filter(self,new_measurement):
        self.previous_measurements_[1] = self.previous_measurements_[0]
        self.previous_measurements_[0] = new_measurement
        new_filter_mesaurement = self.scale_term_ * (self.previous_measurements_[1] + self.previous_measurements_[0] - self.feedback_term_ * self.previous_filtered_measurement_)

        self.previous_filtered_measurement_ = new_filter_mesaurement
        return self.previous_filtered_measurement_