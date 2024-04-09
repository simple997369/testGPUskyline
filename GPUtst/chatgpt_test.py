import numpy as np
from numba import cuda

# 定义GPU核函数
@cuda.jit
def skyline_query_gpu(data, skyline_result):
    i, j = cuda.grid(2)
    if i < data.shape[0] and j < data.shape[0]:
        # 获取当前线程处理的两个点
        point1 = data[i]
        point2 = data[j]

        if i != j:
            dominated = True
            for k in range(data.shape[1]):
                if point1[k] <= point2[k]:
                    dominated = False
                    break
            if dominated:
                skyline_result[i] = 0

# 示例数据集
data = np.array([[3, 5], [2, 6], [8, 1], [4, 3], [7, 2]])

# 创建GPU上的数据副本
data_gpu = cuda.to_device(data)

# 创建结果数组
skyline_result = np.ones(data.shape[0], dtype=np.int32)

# 定义CUDA网格和线程块大小
block_size = (16, 16)
grid_size = (data.shape[0] // block_size[0] + 1, data.shape[0] // block_size[1] + 1)

# 执行GPU核函数
skyline_query_gpu[grid_size, block_size](data_gpu, skyline_result)

# 检索GPU计算结果
skyline_result = skyline_result.nonzero()[0]

# 打印结果
for idx in skyline_result:
    print(data[idx])
