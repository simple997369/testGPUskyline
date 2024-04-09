import os, sys
sys.path.append(os.path.abspath(os.pardir))

import numpy as np
import time
from data.dataClass import batchImport
from tqdm import tqdm 
from grid import GridIndex

# lemma_1
def lemma_1(instance, Max):
    # 以instance的角度來看 是否該保留
    instance_reserved = np.empty((0,dim+3))
    for point in tqdm(instance):
        is_reserved = True
        for Max_point in Max:
            if point[0]!=Max_point[0]:
                if all(point[3:] >= Max_point[3:]):
                    is_reserved = False
                    break
        if is_reserved:
            instance_reserved = np.append(instance_reserved, [point], axis=0)
    return instance_reserved


def lemma_2(instance_reserved, threshold, grid_index_all, grid_result_result):
    obj_prob = []
    ins_prob = []

    for i in tqdm(range(len(grid_index_all))):
        point = grid_index_all[i]
        total_prob = {}
        object_reserved = np.empty((0,dim+3))

        for other_point in grid_result_result:
            temp = 0
            flag = 0
            for d in range(dim):
                if point[d] > other_point[:dim][d]:
                    temp += 1
                elif point[d] == other_point[d]:
                    temp += 1
                    flag = 1
            if temp == dim:
                # print(other_point)
                for index in other_point[dim:]:
                    if index < 0 :
                        # print(123456)
                        pass
                    else:
                        if flag == 0:
                            object_reserved = np.append(object_reserved, [host_data[index]], axis=0)
                        elif host_data[index,0] == instance_reserved[i,0]:
                            pass
                        elif index == instance_reserved[i,1]:
                            pass
                        else:
                            temp = 0
                            for l in range(dim):
                                if instance_reserved[i][l+3] >= host_data[index][l+3]:
                                    temp += 1
                            if temp == dim:
                                object_reserved = np.append(object_reserved, [host_data[index]], axis=0)
        # print("#######",object_reserved)
        for item in object_reserved:
            # print("item",item)
            number = item[0]
            ins_p = item[2]
            if number in total_prob:
                total_prob[number] += ins_p
            else:
                total_prob[number] = ins_p
        probability = 1
        for key, value in total_prob.items():
            probability = probability * (1 - value)
        lst = [instance_reserved[i][0], instance_reserved[i][2], probability]
        ins_prob.append(lst)
    # print("#######",ins_prob)
    for item in ins_prob:
        number = item[0]
        obj_p = item[1] * item[2]
        found = False
        for entry in obj_prob:
            if entry[0] == number:
                entry[1] += obj_p
                found = True
                break
        if not found:
            obj_prob.append([number, obj_p])
    remove = []
    # print(obj_prob)
    prob_skyline = obj_prob
    for i in range(len(obj_prob)):
        if obj_prob[i][1] < threshold:
            remove.insert(0,i)
    for i in remove:
        del prob_skyline[i]
    return prob_skyline
                            





if __name__ == '__main__':
    
    """
    test code
    """
    dim = 50
    count = 10000
    ps = 5
    # 以object的角度來看windowsize
    wsize = count * ps
    threshold = 0.6
    grid = 1
    
    host_data, Max = batchImport('cor_'+str(dim)+'d_'+str(count)+'_'+str(ps)+'_five.txt', count, dim, ps)
    
    # dim = 4
    # count = 3
    # ps = 3
    # # 以object的角度來看windowsize
    # wsize = count * ps
    # threshold = 0.6
    
    # host_data, Max = batchImport('test_dataset.txt', count, dim, ps)

    
    start_time = time.time()
    avgsk1 = 0

    instance_reserved = lemma_1(host_data, Max)
    # print(instance_reserved)

    grid_index = GridIndex(grid, dim, count, ps)
    grid_index.calculate_grid_index(*instance_reserved)
    # grid_index.print_grid()

    grididx_result = []
    for key, value in grid_index.grid.items():
        # 将键和值合并为一个列表
        merged = list(key) + value
        # 将合并后的列表添加到结果列表中
        grididx_result.append(merged)
    max_len = max(len(sublist) for sublist in grididx_result)
    grid_result = [sublist + [-1] * (max_len - len(sublist)) for sublist in grididx_result]
    grid_index_all =grid_index.all

    # print(grid_index_all, grid_result)


    prob_skyline = lemma_2(instance_reserved, threshold, grid_index_all, grid_result)
    # print(prob_skyline)
    avgsk1 += len(prob_skyline)

    print("--- %s seconds ---" % (time.time() - start_time))
    print("avgsk1 = ", avgsk1)
