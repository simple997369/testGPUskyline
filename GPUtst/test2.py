import numpy as np
"""
# 生成三维数据
data = np.random.randn(5, 3)  # 生成100个样本，每个样本有3个特征

print(data)

# 计算相关系数矩阵
corr_matrix = np.corrcoef(data, rowvar=False)  # 设置rowvar=False表示每一列是一个变量

print("Correlation matrix:")
print(corr_matrix)
"""
x = np.zeros(10)
v = 0.5
x[:] = v

# print(x)

h = np.random.uniform(-0.1, 0.1)
# print(h)

# print((0+1)%10)


# 定义一个3x3的矩阵
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

# print("Eigenvalues:")
# print(eigenvalues)
# print("\nEigenvectors:")
# print(eigenvectors)


import matplotlib.pyplot as plt

# 定义起点和方向
X = [1, 2, 3]
Y = [1, 2, 3]
U = [1, 0, -1]
V = [0, 1, -1]

# 创建一个简单的图形
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')

# 绘制箭头
plt.quiver(X, Y, U, V)
# 在指定位置添加文本标注
plt.text(3, 9, 'Example Text', fontsize=20, color='blue')

# 显示图形
# plt.show()
plt.savefig('test2')