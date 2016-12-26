import matplotlib.pyplot as plt
import numpy as np


class draw_result():
    
    def __init__(self):
        pass

    def draw_test(self,test_file_1,test_file_2,iftest):
        fh_1 = open(test_file_1)
        fh_2 = open(test_file_2)
        fh_1_cont = fh_1.readlines()
        fh_2_cont = fh_2.readlines()

        count = 1
        f1_rmse = []
        f2_rmse = []
        for line in fh_1_cont:
            if (count < 6 and iftest==True):
                pass
            else:
                line = line.strip('\n')
                line = float(line)
                f1_rmse.append(line)
            count += 1
        f1_rmse_array = np.array(f1_rmse)
        _num = f1_rmse_array.shape[0]
        
        count = 1
        
        for line in fh_2_cont:
            if(count < 6):
                pass
            else:
                line = line.strip('\n')
                line = float(line)
                f2_rmse.append(line)
            count += 1
        f2_rmse_array = np.array(f2_rmse)
        _num_2 = f2_rmse_array.shape[0]

        my_plt = plt.plot(range(_num),f1_rmse_array,'r--')
        my_plt_2 = plt.plot(range(_num_2),f2_rmse_array,'b')
        plt.ylabel('RMSE')
        plt.xlabel('Iteration')
        #plt.axis([0,50,0.3,5])
        plt.show()


            
