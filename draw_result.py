import matplotlib.pyplot as plt
import numpy as np


class draw_result():
    
    def __init__(self):
        pass

    def draw_test(self,test_file_1,test_file_2):
        fh_1 = open(test_file_1)
        all__ret = fh_1.readlines()
        count = 1
        for line in all_ret:
            if (count < 40):
                print(line)
            count += 1

            
