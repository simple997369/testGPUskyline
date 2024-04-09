import os, sys
sys.path.append(os.path.abspath(os.pardir))
here = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from numba import cuda
import time

@cuda.jit
def skyline_query_gpu(data, skyline_result, wsize):
    i, j = cuda.grid(2)
    # data.shape[0] = object數量
    # data.shape[1]-3 = dimension
    if i < data.shape[0] and j < data.shape[0]:
        # 获取当前线程处理的两个点
        point1 = data[i]
        point2 = data[j]
        if i != j:
            dominated = True
            for k in range(data.shape[1]-3):
                if point1[k+3] <= point2[k+3]:
                    dominated = False
                    break
            if dominated:
                skyline_result[i] = 0

if __name__ == '__main__':
    
    """
    test code
    """
    wsize = 300
    # ps = 2
    # path ='anticor_2d_10_' + str(ps) + '.txt'
    path ='anticor_10d_10000_5.txt' 
    
    host_data = np.loadtxt(here+'/data/dataset/'+path)
    device_data = cuda.to_device(host_data)

    block_size = (16, 16)
    # block_size = (32, 32)  
    grid_size = (host_data.shape[0] // block_size[0] + 1, host_data.shape[0] // block_size[1] + 1)
    
    start_time = time.time()
    avgsk1 = 0
    for wcount in range(device_data.shape[0]-wsize+1):
        skyline_result = np.ones(wsize, dtype=np.int32)
        skyline_query_gpu[grid_size, block_size](device_data[wcount:wcount+wsize], skyline_result, wsize)
        skyline_result = skyline_result.nonzero()[0]
        avgsk1 += skyline_result.size
    
    print("--- %s seconds ---" % (time.time() - start_time))

    # skyline_result = skyline_result.nonzero()[0]
    # for idx in skyline_result:
    #     print(host_data[idx])
    
    print("avgsk1 = ", avgsk1)
