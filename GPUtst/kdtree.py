"""

import numpy as np
from numba import cuda

@cuda.jit
def my_kernel(data, result):
    tx = cuda.threadIdx.x  # 线程在块内的 x 方向上的索引
    ty = cuda.threadIdx.y  # 线程在块内的 y 方向上的索引
    bw = cuda.blockDim.x  # 块的大小（x 方向上的大小）
    bx = cuda.blockIdx.x  # 线程块在 grid 中的 x 方向上的索引
    by = cuda.blockIdx.y  # 线程块在 grid 中的 y 方向上的索引
    
    i, j = cuda.grid(2)
    
    global_thread_id = tx + ty * bw + (bx * bw * cuda.gridDim.x) + by * bw * cuda.gridDim.x
    
    if global_thread_id == 0:
        value = data[0, 0]
        print("Value at global_thread_id 0:", value)

    # 在这里处理数据，将结果存储到result数组中
    result[global_thread_id] = data[ty, tx]   # 以示例方式将数据乘以2存储到result中

# 示例数据
host_data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
# 创建 GPU 上的数据
device_data = cuda.to_device(host_data)

# 创建保存线程计算结果的数组
device_result = cuda.device_array_like(host_data)

block_size = (8, 8)
grid_size = (host_data.shape[0] // block_size[0] + 1, host_data.shape[0] // block_size[1] + 1)


# 调用核函数时传递两个参数
my_kernel[grid_size, block_size](device_data, device_result)

# 将结果从 GPU 复制回主机
result_host = device_result.copy_to_host()

# 输出结果
print(result_host)

"""
"""
import numpy as np
from numba import cuda

# CUDA kernel实现KD树构建
@cuda.jit
def build_kdtree(points, tree):
    num_points, dim = points.shape

    # 每个线程负责处理一个点
    idx = cuda.grid(1)

    if idx < num_points:
        # 在所选轴上对点进行排序
        for i in range(dim):
            for j in range(i + 1, dim):
                if points[idx, i] > points[idx, j]:
                    temp = points[idx, i]
                    points[idx, i] = points[idx, j]
                    points[idx, j] = temp

        # 将中间点插入到树中
        for i in range(dim):
            tree[idx, i] = points[idx, i]

# CPU函数来调用CUDA核函数并初始化KD树
def construct_kdtree(points):
    num_points, dim = points.shape

    # 在GPU上分配内存来存储KD树
    tree = np.zeros((num_points, dim), dtype=np.float32)

    # 将数据传输到GPU内存
    d_points = cuda.to_device(points)
    d_tree = cuda.to_device(tree)

    # 调用CUDA核函数来构建KD树
    threads_per_block = 64
    blocks_per_grid = (num_points + threads_per_block - 1) // threads_per_block
    build_kdtree[blocks_per_grid, threads_per_block](d_points, d_tree)

    # 从GPU上取回构建好的KD树
    cuda.synchronize()
    tree = d_tree.copy_to_host()

    return tree

# 测试数据
points = np.array([[2,3], [5,4], [9,6], [4,7], [8,1], [7,2]], dtype=np.float32)

# 构建KD树
kdtree = construct_kdtree(points)
print("Constructed KD-Tree:")
print(kdtree)

"""


import numpy as np

class Node:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

def build_kdtree(points, depth=0):
    if len(points) == 0:
        return None

    dim = len(points[0])
    axis = depth % dim

    points = points[points[:, axis].argsort()]
    median = len(points) // 2

    return Node(
        point=points[median],
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1)
    )

def print_kdtree(node, depth=0):
    if node is not None:
        print("  " * depth + f"Node: {node.point}")
        print_kdtree(node.left, depth + 1)
        print_kdtree(node.right, depth + 1)


# 测试数据
points = np.array([[2,3], [5,4], [9,6], [4,7], [8,1], [7,2]], dtype=np.float32)

# 构建KD树
kdtree = build_kdtree(points)

# 打印KD树
print("Constructed KD-Tree:")
print_kdtree(kdtree)