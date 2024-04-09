import os, sys
sys.path.append(os.path.abspath(os.pardir))

import numpy as np
import time
from numba import cuda
from data.dataClass import batchImport
                                                                                              
# print(cuda.is_available())

# max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK
# print(max_threads_per_block)

@cuda.jit
def lemma_1(instance, Max, instance_reserved):
    # 以instance的角度來看 是否該保留
    i, j = cuda.grid(2)
    # print(i,j)
    
    if i < instance.shape[0] and j < Max.shape[0]:
        point1 = instance[i]
        point2 = Max[j]
        
        if i//ps != j:
            is_dominated = True
            for k in range(instance.shape[1]-3):
                if point1[k+3] < point2[k+3]:
                    is_dominated = False
                    break
            if is_dominated:
                instance_reserved[i] = 0
            
'''
@cuda.jit
def lemma_2(instance, instance_reserved, threshold, lemma2_reserved, skyline_result):
    i, j = cuda.grid(2)
    if i < instance.shape[0] and j < instance.shape[0]:
        point1 = instance[i]
        point2 = instance[j]
        if instance_reserved[i] == 0 or instance_reserved[j] == 0 or point1[0] == point2[0]:
            pass
        else:
            is_dominated = True
            for k in range(instance.shape[1]-3):
                if point1[k+3] < point2[k+3]:
                    is_dominated = False
                    break
            if is_dominated:
                lemma2_reserved[i * 2] = instance[i][0]
                lemma2_reserved[i * 2 + 1] = instance[i][2]
                print(i,j)
        # print(instance[i][0],instance[i][2])
'''           

@cuda.jit
def lemma_2(label, instance, instance_reserved, lemma2_reserved):
    i, j = cuda.grid(2)
    if i < instance.shape[0] and j < instance.shape[0]:
        if int(instance[i,0]) != label:
            pass
        else:
            # print(i,j)
            point1 = instance[i]
            point2 = instance[j]
            if instance_reserved[i] == 0 or instance_reserved[j] == 0 or point1[0] == point2[0]:
                pass
            else:
                
                
                is_dominated = True
                for k in range(instance.shape[1]-3):
                    if point1[k+3] < point2[k+3]:
                        is_dominated = False
                        break
                # 如果i被j征服
                if is_dominated:
                    lemma2_reserved[ i % ps , j * 2] = point2[0]
                    lemma2_reserved[ i % ps , j * 2 + 1] = point2[2]
                    
@cuda.jit
def lemma_3(lemma2_reserved, ins_prob):
    i, j = cuda.grid(2)
    if i < ps and j < count*ps:
        labels = lemma2_reserved[i][::2]
        probs = lemma2_reserved[i][1::2]
        cuda.atomic.add(ins_prob[i], labels[j], probs[j])

        
@cuda.jit
def lemma_4(ins_prob, result_device): 
    i = cuda.grid(1)
    if i < ins_prob.shape[0]:
        for j in range(ins_prob.shape[1]):
            result_device[i] *= 1 - ins_prob[i, j]
            
            

if __name__ == '__main__':
    
    """
    test code
    """
    dim = 10
    count = 10000
    ps = 5
    # 以object的角度來看windowsize
    wsize = 10000 * ps
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
    
    # max_threads_per_block = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    # print(max_threads_per_block)
    # print(block_size)
    # print(grid_size)
    # exit()
    
    instance_reserved = cuda.to_device(np.ones(count * ps, dtype=np.int32))
    
    skyline_result = np.ones(count, dtype=np.int32)
    
    start_time = time.time()
    
    lemma_1[grid_size, block_size](device_data, device_Max, instance_reserved)

    # exit()
    # lemma_2[grid_size, block_size](device_data, instance_reserved, threshold, lemma2_reserved, skyline_result)
    for label in range(count):
        # print(label)
        lemma2_reserved = cuda.to_device(np.zeros((ps, count * ps * 2), dtype=np.float64))
        lemma_2[grid_size, block_size](label, device_data, instance_reserved, lemma2_reserved)
        x = lemma2_reserved.copy_to_host()
        # print(x)
        # if label == 3:
        #     exit()
        ins_prob = cuda.to_device(np.zeros((ps, count), dtype=np.float64))
        lemma_3[grid_size, block_size](lemma2_reserved, ins_prob)
        result_device = cuda.to_device(np.ones(ps , dtype=np.float64))  # 初始值为 1.0
        lemma_4[grid_size, block_size](ins_prob, result_device)
        total = cuda.to_device(np.zeros(1 , dtype=np.float64))
        for x in range(ps):
            total[0] += result_device[x] * device_data[label*ps : label*ps + ps][x , 2]
        if total[0] < threshold:
            skyline_result[label] = 0
    
    # x = lemma2_reserved.copy_to_host()
    # print(x)
    
    print(skyline_result)
    skyline_result = skyline_result.nonzero()[0]
    print("--- %s seconds ---" % (time.time() - start_time))
    print("skyline size = ", skyline_result.size)
