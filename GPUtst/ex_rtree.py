from rtree import index

# 创建一个 R-tree 索引
idx = index.Index()

# 添加一些点到索引中
points = [(0, 0), (1, 1), (2, 2), (3, 3)]
for i, point in enumerate(points):
    idx.insert(i, point)

# 搜索与给定范围相交的点
bbox = (0.0, 0.0, 2.5, 2.5)  # 查询范围：(min_x, min_y, max_x, max_y)
result = list(idx.intersection(bbox))
print("Points within the bounding box:", result)

# 搜索最近的点
query_point = (1.5, 1.5)  # 查询点
nearest = list(idx.nearest(query_point, 2))  # 返回最近的两个点
print("Nearest points to the query point:", nearest)
