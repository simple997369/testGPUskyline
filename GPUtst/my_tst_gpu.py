import os, sys
sys.path.append(os.path.abspath(os.pardir))
here = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from numba import cuda
import time

# 定义GPU核函数
@cuda.jit
def skyline_query_gpu(data, skyline_result):
    i, j = cuda.grid(2)
    # data.shape[0] = object數量
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

# 读取数据并将其复制到GPU
host_data = np.loadtxt(here+'/data/dataset/'+'anticor_5d_5_2.txt')
device_data = cuda.to_device(host_data)

# 创建结果数组
skyline_result = np.ones(host_data.shape[0], dtype=np.int32)

# 定义CUDA网格和线程块大小
# block_size = (16, 16)
block_size = (32, 32)  
grid_size = (host_data.shape[0] // block_size[0] + 1, host_data.shape[0] // block_size[1] + 1)

start_time = time.time()
# 执行GPU核函数
skyline_query_gpu[grid_size, block_size](device_data, skyline_result)
# os._exit(0)
print("--- %s seconds ---" % (time.time() - start_time))

# 检索GPU计算结果
skyline_result = skyline_result.nonzero()[0]

# 打印结果
# for idx in skyline_result:
#     print(host_data[idx])
print(skyline_result.size)
