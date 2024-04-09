import os, sys
sys.path.append(os.path.abspath(os.pardir))

import numpy as np
import time
from numba import cuda
from data.dataClass import batchImport

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



if __name__ == '__main__':
    
    """
    test code
    """
    dim = 10
    count = 10000
    ps = 5
    # 以object的角度來看windowsize
    # wsize = 10000 * ps
    threshold = 0.6
    host_data, Max = batchImport('anticor_'+str(dim)+'d_'+str(count)+'_'+str(ps)+'.txt', count, dim, ps)
    
    # dim = 4
    # count = 3
    # ps = 3
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
    # print(test[:,2])
    del device_data
    del device_Max
    device_instance_reserved = cuda.to_device(test)
    
    new_lemma2_reserved = np.zeros((test.shape[0], count * ps), dtype=np.float64)
    new_lemma_2[grid_size, block_size](device_instance_reserved, new_lemma2_reserved)
    # print(new_lemma2_reserved)
    # print(new_lemma2_reserved.sum())
    # sourceFile = open('demo.txt', 'w')
    # np.set_printoptions(threshold=np.inf)
    # print(new_lemma2_reserved, file = sourceFile)
    # sourceFile.close()
    # cuda.close()
    
    new_ins_prob = np.zeros((test.shape[0], count), dtype=np.float64)
    new_lemma_3[grid_size, block_size](new_lemma2_reserved, new_ins_prob)
    # print(new_ins_prob)
    
    
    new_result = np.ones(test.shape[0] , dtype=np.float64)
    new_lemma_4[grid_size, block_size](new_ins_prob, new_result)
    # print(new_result)
    
    
    # skyline_result = np.zeros(count, dtype=np.int32)
    # new_lemma_5[grid_size, block_size](device_instance_reserved, new_result)
    new_result = new_result * test[:,2]
    # print(new_result)
    # sourceFile = open('demo.txt', 'w')
    # np.set_printoptions(threshold=np.inf)
    # print(new_result, file = sourceFile)
    # sourceFile.close()
    
    result = np.zeros(count, dtype=np.float64)
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
