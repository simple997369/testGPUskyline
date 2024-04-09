

"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

# 创建一个密集矩阵
dense_matrix = np.array([[1, 0, 0],
                         [0, 0, 2],
                         [0, 3, 0]])

# 将密集矩阵转换为稀疏矩阵（压缩稀疏行格式）
sparse_csr = csr_matrix(dense_matrix)

# 将密集矩阵转换为稀疏矩阵（列表列表格式）
sparse_lil = lil_matrix(dense_matrix)

# 访问稀疏矩阵的元素
print("CSR格式中 (1, 2) 处的元素:", sparse_csr[1, 2])  # 打印第1行第2列的值
print("LIL格式中 (1, 2) 处的元素:", sparse_lil[1, 2])

# 更改稀疏矩阵的元素
sparse_lil[1, 2] = 5  # 修改第1行第2列的值
print("修改后的 LIL 格式中 (1, 2) 处的元素:", sparse_lil[1, 2])

# 转换回密集矩阵
dense_matrix_from_csr = sparse_csr.toarray()
print("从CSR格式得到的密集矩阵:\n", dense_matrix_from_csr)

dense_matrix_from_lil = sparse_lil.toarray()
print("从LIL格式得到的密集矩阵:\n", dense_matrix_from_lil)


import numpy as np

x = np.array([[1,2],[3,4]])
print(np.repeat(x, 2))


print(np.repeat(x, 2, axis=1))  #沿着纵轴方向重复，增加列数


print(np.repeat(x, [1, 2], axis=0))  #沿着横轴方向重复，增加行数，分别为一次和两次

y = [1,2,3,4,5,6,7,8,9,10]
print(y[2:])
"""

import numpy as np
import time

# 定义一个函数，使用条件分支来判断数组元素的正负，并计算元素的平方根
def sqrt_with_branch(x):
    result = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] >= 0:
            result[i] = np.sqrt(x[i])
        else:
            result[i] = 0
    return result

# 定义一个函数，直接计算数组元素的平方根，不使用条件分支
def sqrt_without_branch(x):
    return np.sqrt(np.maximum(x, 0))

# 生成一个包含随机数的大数组
n = 1000000
x = np.random.randn(n)

# 测量使用条件分支的函数执行时间
start_time = time.time()
result_with_branch = sqrt_with_branch(x)
end_time = time.time()
time_with_branch = end_time - start_time
print("Execution time with branch divergence:", time_with_branch)

# 测量不使用条件分支的函数执行时间
start_time = time.time()
result_without_branch = sqrt_without_branch(x)
end_time = time.time()
time_without_branch = end_time - start_time
print("Execution time without branch divergence:", time_without_branch)

# 检查结果是否一致
print("Results are equal:", np.allclose(result_with_branch, result_without_branch))

