import os, sys
sys.path.append(os.path.abspath(os.pardir))
here = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import time

def skyline_query(data):
    data = data[:,3:]
    skyline_points = []
    for i, point in enumerate(data):
        is_skyline = True
        for j, other_point in enumerate(data):
            if i != j:
                if all(point >= other_point):
                    is_skyline = False
                    break
        if is_skyline:
            skyline_points.append(point)
    return skyline_points

# 示例数据集
host_data = np.loadtxt(here+'/data/dataset/'+'ind_2d_10_1.txt')

start_time = time.time()
# 执行天际线查询
result = skyline_query(host_data)

print("--- %s seconds ---" % (time.time() - start_time))
# 打印结果
for point in result:
    print(point)
print("skyline_result.size = ", len(result))
