# _*_coding:utf-8_*_
import time
import threading
import numpy as np


def mainfun():
    reg_set=[1,2,3];
    ret = np.zeros((3,3))
    threads=[]

    for i in range(100):
        tk = threading.Thread(target=main_thread, args=(ret,reg_set,i))
        #tk.start()
        threads.append(tk)
    #for tk in threads:
    #    tk.join()
    print("end main thread")
    print(ret)

def main_thread(ret,reg_set,k):
    lock = threading.Lock()
    for i in range(3):
        for j in range(3):
            lock.acquire()
            try:
                ret[i,j] = ret[i,j] + 1
            finally:
                lock.release()
    print('end one thread')

mainfun()
