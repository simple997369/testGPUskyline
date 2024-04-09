import os, sys
sys.path.append(os.path.abspath(os.pardir))

import numpy as np
import time
from numba import cuda
from data.dataClass import batchImport
from grid import GridIndex
# from quads import QuadTree


# import cupy as cp
                                                                                              
# print(cuda.is_available())

# max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK
# max_block_size = cuda.get_current_device().MAX_THREADS_PER_BLOCK
# print(max_block_size)

@cuda.jit
def lemma_1(instance, Max, instance_reserved):
    # 以instance的角度來看 是否該保留
    i, j = cuda.grid(2)
    # print(i,j)
    # if i == 50015 and j == 50015:
    #     print("bingo")
    if i < instance.shape[0] and j < Max.shape[0]:
        # if i == 49999 and j == 9999 :
        #     print("bingo")
        point1 = instance[i]
        point2 = Max[j]
        
        if i//ps != j:
            temp = 0
            for k in range(dim):
                if point1[k+3] >= point2[k+3]:
                    temp += 1
            if temp == dim:
                instance_reserved[i] = 0
            
@cuda.jit
def new_lemma_2(device_instance_reserved, new_lemma2_reserved):
    i, j = cuda.grid(2)
    # if i == 31 and j == 31:
    #     print("2bingo")
    if i < device_instance_reserved.shape[0] and j < device_instance_reserved.shape[0]:
        point1 = device_instance_reserved[i]
        point2 = device_instance_reserved[j]
        if point1[0] != point2[0]:
            temp = 0
            for k in range(dim):
                if point1[k+3] >= point2[k+3]:
                    temp += 1
            if temp == dim:
                new_lemma2_reserved[ i, int(point2[1])] = point2[2]
            
  
@cuda.jit                
def new_lemma_3(new_lemma2_reserved, ins_prob):
    i, j = cuda.grid(2)
    # if i == 511 and j == 511:
    #     print("3bingo")
    if i < new_lemma2_reserved.shape[0] and j < count:
        point = new_lemma2_reserved[i]
        # print(i,j)
        total = 0
        for k in range(j*ps, j*ps+ps):
            total += point[k]
        ins_prob[i,j] = total
        
        
@cuda.jit
def new_lemma_4(new_ins_prob, new_result):
    i,j = cuda.grid(2)
    # if i == 511 and j == 511:
    #     print("4bingo")
    if i < new_ins_prob.shape[0] and j == 0:
        # print(i,j)
        # new_result[i] *= 1 - new_ins_prob[i, j]
        for k in range(count):
            new_result[i] *= 1 - new_ins_prob[i, k]
         
            
@cuda.jit         
def new_lemma_5(device_instance_reserved, new_result):
    i,j = cuda.grid(2)
    # if i == 511 and j == 511:
    #     print("5bingo")
    if i < new_ins_prob.shape[0] and j == 0:
        new_result[i] = new_result[i] * device_instance_reserved[i, 2]

@cuda.jit
def new_lemma_6(device_instance_reserved, new_result,result):
    i, j = cuda.grid(2)
    if i < new_ins_prob.shape[0]:  
        # print(i)
        idx = int(device_instance_reserved[i, 0])
        result[idx] += new_result[i]

@cuda.jit
def tst(grid_all, grid_array_result, tst_reserved, test, host_data):
    i, j = cuda.grid(2)
    # print(i)
    if i < grid_all.shape[0] and j < grid_array_result.shape[0]:
        # print(i,j)
        point1 = grid_all[i]
        point2 = grid_array_result[j]
        temp = 0
        flag = 0
        for k in range(dim):
            if point1[k] >= point2[k]:
                temp += 1
                if point1[k] == point2[k]:
                    flag = 1
        if temp == dim:
            for index in point2[dim:]: 
                if index < 0 :
                    break
                if host_data[index,0] == test[i,0]:
                    continue
                # elif index == test[i,1]:
                #     pass
                else:
                    if flag == 0:
                        tst_reserved[i][index] = host_data[index][2]
                    else:
                        temp = 0
                        for l in range(dim):
                            if test[i][l+3] >= host_data[index][l+3]:
                                temp += 1
                        if temp == dim:
                            tst_reserved[i][index] = host_data[index][2]
                            # print("3###",j,index, test[j][2])
        
# @cuda.jit
# def tst2(tst_reserved, tst2_reserved, test):
#     i, j = cuda.grid(2)
#     # print(i,j)
#     if i < tst_reserved.shape[0] and j < count*ps:
#         # print(i,j)
#         point1 = test[i]
#         point2 = test[j]




if __name__ == '__main__':
    
    """
    test code
    """
    dim = 7
    count = 10000
    ps = 5
    # 以object的角度來看windowsize
    # wsize = 10000 * ps
    threshold = 0.6
    grid = 1
    host_data, Max = batchImport('anticor_'+str(dim)+'d_'+str(count)+'_'+str(ps)+'.txt', count, dim, ps)
    
    # dim = 4
    # count = 3
    # ps = 3
    # grid = 1
    # # 以object的角度來看windowsize
    # # wsize = count * ps
    # threshold = 0.6
    # host_data, Max = batchImport('test_dataset.txt', count, dim, ps)
    
    device_data  = cuda.to_device(host_data)
    device_Max = cuda.to_device(Max)
    
    block_size = (32,32)  
    grid_size = (host_data.shape[0] // block_size[0] + 1, host_data.shape[0] // block_size[1] + 1)
    
    # print(block_size,grid_size)
    
    
    
    
    start_time = time.time()
    
    instance_reserved = np.ones(count * ps, dtype=bool)
    lemma_1[grid_size, block_size](device_data, device_Max, instance_reserved)
    test = host_data[instance_reserved]
    # print(instance_reserved.nonzero()[0].size)
    # print("#####",test.size//(dim+3))
    # print(test[:,1])
    del device_data
    del device_Max
    
    # device = cuda.get_current_device()
    # device.reset()

    device_instance_reserved = cuda.to_device(test)

    
    # print(type(test))
    # sourceFile = open('demo2.txt', 'w')
    # np.set_printoptions(threshold=np.inf)
    # print(test, file = sourceFile)
    # sourceFile.close()

    # print( test)
    
    grid_index = GridIndex(grid, dim, count, ps)
    grid_index.calculate_grid_index(*test)
    # grid_index.sort_grid()
    # grid_index.print_grid()
    # print(grid_index.grid.keys())
    # print(len(grid_index.grid.keys()))



    grid_result = []
    for key, value in grid_index.grid.items():
        # 将键和值合并为一个列表
        merged = list(key) + value
        # 将合并后的列表添加到结果列表中
        grid_result.append(merged)
    # average = (len(min(grid_result, key=len)) + len(max(grid_result, key=len))) / len(grid_result)
    # print(len(max(grid_result, key=len)) - dim)
    # print(len(min(grid_result, key=len)) - dim)
    print(test.shape[0], len(grid_result), test.shape[0] / len(grid_result))
    # print(average)

    max_len = max(len(sublist) for sublist in grid_result)
    grid_result_adjusted = [sublist + [-1] * (max_len - len(sublist)) for sublist in grid_result]
    grid_array_result = np.array(grid_result_adjusted)
    print(grid_array_result.shape[0])
    # grid_array_result = np.nan_to_num(grid_array_result, None=-1)

    grid_index_all = np.array(grid_index.all)

    # print("grid_array_result",grid_array_result)
    # print("grid_index.all",grid_index.all)

    # print(np.array(list(grid_index.grid.items())))
    # numero_DT = []


    tst_reserved = np.zeros((test.shape[0], count * ps), dtype=np.float32)
    tst[grid_size, block_size](grid_index_all, grid_array_result, tst_reserved, device_instance_reserved, host_data)

    # print('*******************************')
    # print(f'Size={sys.getsizeof(tst_reserved)}')
    # print('*******************************')

    # sourceFile = open('demo.txt', 'w')
    # np.set_printoptions(threshold=np.inf)
    # print(tst_reserved, file = sourceFile)
    # sourceFile.close()

    del instance_reserved
    del host_data
    del grid_index_all
    del grid_array_result
    del device_instance_reserved

    # print(tst_reserved)

    # tst2_reserved = np.zeros((test.shape[0], count * ps), dtype=np.float64)
    # tst2[grid_size, block_size](tst_reserved, tst2_reserved, test)

    # for i in range(len(test)):
    #     print(i)
    #     i_grid = [int(x * (2**grid)) for x in test[i][3:]]
    #     # print(i_grid)
    #     numero = test[:,1].tolist()
    #     for j in range(dim):
    #         # print(grid_index.sorted_axis[j])
    #         # print("#####",grid_index.grid.items())
    #         filtered_dict = {key: value for key, value in grid_index.grid.items() if key[j] > i_grid[j]}
    #         for sublist in filtered_dict.values():
    #             for value in sublist:
    #                 if value in numero:
    #                     numero.remove(value)

        # for j in range(dim):
        #     print(grid_index.sorted_axis[j])
        #     filtered = [item for item in grid_index.sorted_axis[j] if item[j] == 0]
        #     _x = grid_index.sorted_axis[j].index(filtered[0])
        #     print(_x)
        #     exit()

        # print(numero)
        # exit()
    
    # print(test)
    # sourceFile = open('demo.txt', 'w')
    # np.set_printoptions(threshold=np.inf)
    # print(test, file = sourceFile)
    # sourceFile.close()




    # new_lemma2_reserved = np.zeros((test.shape[0], count * ps), dtype=np.float64)
    # new_lemma_2[grid_size, block_size](device_instance_reserved, new_lemma2_reserved)
    # print(new_lemma2_reserved)
    # print(new_lemma2_reserved.sum())
    # sourceFile = open('demo.txt', 'w')
    # np.set_printoptions(threshold=np.inf)
    # print(new_lemma2_reserved, file = sourceFile)
    # sourceFile.close()
    # cuda.close()
    
    # exit()

    # cuda.select_device(1)
    # cuda.close()

    new_ins_prob = np.zeros((test.shape[0], count), dtype=np.float32)
    new_lemma_3[grid_size, block_size](tst_reserved, new_ins_prob)
    # print(new_ins_prob)

    del tst_reserved
    
    new_result = np.ones(test.shape[0] , dtype=np.float32)
    new_lemma_4[grid_size, block_size](new_ins_prob, new_result)
    # print(new_result)

    del new_ins_prob
    
    
    # skyline_result = np.zeros(count, dtype=np.int32)
    # new_lemma_5[grid_size, block_size](device_instance_reserved, new_result)
    new_result = new_result * test[:,2]
    # print(new_result)
    # sourceFile = open('demo.txt', 'w')
    # np.set_printoptions(threshold=np.inf)
    # print(new_result, file = sourceFile)
    # sourceFile.close()
    
    result = np.zeros(count, dtype=np.float32)
    # new_lemma_6[grid_size, block_size](device_instance_reserved, new_result,result)
    for i in range(test.shape[0]):
        idx = int(test[i, 0])
        result[idx] += new_result[i]
        
    print("skyline size = ", result[result >= threshold].size)
    
    
    
    # x = lemma2_reserved.copy_to_host()
    # print(x)
    
    # print(skyline_result)
    
    # skyline_result = skyline_result.nonzero()[0]
    print("--- %s seconds ---" % (time.time() - start_time))
    # print("skyline size = ", skyline_result.size)
