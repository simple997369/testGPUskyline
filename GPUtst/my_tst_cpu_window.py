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

if __name__ == '__main__':
    
    """
    test code
    """
    wsize = 300
    
    host_data = np.loadtxt(here+'/data/dataset/'+'anticor_10d_10000_5.txt')

    start_time = time.time()
    avgsk1 = 0
    for wcount in range(host_data.shape[0]-wsize+1):
        result = skyline_query(host_data[wcount:wcount+wsize])
        avgsk1 += len(result)

    print("--- %s seconds ---" % (time.time() - start_time))
    # 打印结果
    # for point in result:
    #     print(point)
    print("avgsk1 = ", avgsk1)
