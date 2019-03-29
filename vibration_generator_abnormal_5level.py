# -*- coding: utf-8 -*-

import random
import numpy as np
import math

class generator():
    def __init__(self, num, max_amp = 10, diff_amp = 0.1, measure_term = 50, count_per_second = 1000, frequency_min = 20, frequency_max = 40):
        self.object_num = num
        self.amplitude_matrix = np.zeros((num, 2))
        self.frequency_matrix = np.zeros((num))
        self.measure_term = measure_term
        self.count_per_second = count_per_second
        for i in range(num):
            center = max_amp * random.random()
            diff = center * diff_amp * random.random()
            self.amplitude_matrix[i, 0] = center - diff
            self.amplitude_matrix[i, 1] = center + diff
            self.frequency_matrix[i] = random.randint(frequency_min, frequency_max) * 2 * math.pi
        
    def generate_normal(self):
        ret= np.zeros((self.object_num, self.measure_term * self.count_per_second))
        for i in range(self.object_num):
            for k in range(self.measure_term * self.count_per_second):
                ret[i, k] = random.uniform(self.amplitude_matrix[i, 0], self.amplitude_matrix[i, 1]) * math.sin((self.frequency_matrix[i] * k / self.count_per_second))
        
        return ret
    
    def generate_abnormal(self, abnormal, ab_rate = 1.01):
        if self.object_num <= abnormal:
            return None
        ret= np.zeros((self.object_num, self.measure_term * self.count_per_second))
        for i in range(self.object_num):
            for k in range(self.measure_term * self.count_per_second):
                if i == abnormal:
                    if k / 10000 < 1:
                        abnormal_rate = ab_rate
                    elif k / 10000 < 2:
                        abnormal_rate = ab_rate + 0.01
                    elif k / 10000 < 3:
                        abnormal_rate = ab_rate + 0.02
                    elif k / 10000 < 4:
                        abnormal_rate = ab_rate + 0.03
                    else:
                        abnormal_rate = ab_rate + 0.04
                else:
                    abnormal_rate = 1
                ret[i, k] = abnormal_rate * random.uniform(self.amplitude_matrix[i, 0], self.amplitude_matrix[i, 1]) * math.sin((self.frequency_matrix[i] * k / self.count_per_second))
        
        return ret
    
# test
if __name__ == '__main__':
    generator = generator(5)
    hoge = generator.generate_normal()