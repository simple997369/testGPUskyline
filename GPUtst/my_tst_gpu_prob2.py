import os, sys
sys.path.append(os.path.abspath(os.pardir))

import numpy as np
import time
from numba import cuda
from data.dataClass import batchImport
import cupy as cp
from tqdm import tqdm 

def lemma_1(lemma1_result):
    instance_reserved = np.empty((0,dim+3))
    for i in tqdm(range(count*ps)):
        is_reserved = True
        # print("i=",i)
        for j in range(count):
            # print(j)
            if device_instance[i, 0] != device_Max[j, 0]:
                if cp.all(device_instance[i][3:] > device_Max[j][3:]):
                    is_reserved = False
                    break
        if is_reserved:
                lemma1_result[i] = 1
                instance_reserved = np.append(instance_reserved, [host_data[i]], axis=0)
    return instance_reserved            
                
# def lemma_2():
#     obj_prob = []
#     ins_prob = []
#     for i in range(count*ps):
#         total_prob = {}
#         object_reserved = np.empty((0,dim+3))
#         for j in range(count*ps):
#             if lemma1_result[i] == 0 or lemma1_result[j] == 0 or host_data[i,0] == host_data[j,0]:
#                 continue
#             else:
#                 is_reserved = False
#                 # if cp.all(device_instance[i][3:] >= device_instance[j][3:]):
#                 if all(host_data[i][3:]>=host_data[j][3:]):
#                     is_reserved = True
#                 if is_reserved:
#                     object_reserved = np.append(object_reserved, [host_data[j]], axis=0)
#         for item in object_reserved:
#             number = item[0]
#             ins_p = item[2]
#             if number in total_prob:
#                 total_prob[number] += ins_p
#             else:
#                 total_prob[number] = ins_p
#         print(total_prob)
#         probability = 1
#         for key, value in total_prob.items():
#             probability = probability * (1 - value)
#         lst = [host_data[i][0], host_data[i][2], probability]
#         ins_prob.append(lst)
#     for item in ins_prob:
#         number = item[0]
#         obj_p = item[1] * item[2]
#         found = False
#         for entry in obj_prob:
#             if entry[0] == number:
#                 entry[1] += obj_p
#                 found = True
#                 break
#         if not found:
#             obj_prob.append([number, obj_p])
#     remove = []
#     # print(obj_prob)
#     prob_skyline = obj_prob
#     for i in range(len(obj_prob)):
#         if obj_prob[i][1] < threshold:
#             remove.insert(0,i)
#     for i in remove:
#         del prob_skyline[i]
#     return prob_skyline
                
def lemma_2(device_instance_reserved, threshold):
    obj_prob = []
    ins_prob = []
    for i in range(len(instance_reserved)):
        total_prob = {}
        object_reserved = np.empty((0,dim+3))
        for j in range(len(instance_reserved)):
            is_reserved = False
            if device_instance_reserved[i][0]!=device_instance_reserved[j][0]:
                if cp.all(device_instance_reserved[i][3:] >= device_instance_reserved[j][3:]):
                    is_reserved = True
            if is_reserved:
                # print(other_point)
                object_reserved = np.append(object_reserved, [instance_reserved[j]], axis=0)
        # 可以跟上面合併，item = other_point
        for item in object_reserved:
            number = item[0]
            ins_p = item[2]
            if number in total_prob:
                total_prob[number] += ins_p
            else:
                total_prob[number] = ins_p
        
        # print(total_prob)
        probability = 1
        for key, value in total_prob.items():
            probability = probability * (1 - value)
        lst = [device_instance_reserved[i][0], device_instance_reserved[i][2], probability]
        ins_prob.append(lst)
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
        
    start_time = time.time()
    
    device_instance = cp.asarray(host_data)
    device_Max = cp.asarray(Max)
    
    avgsk1 = 0
    lemma1_result = np.zeros(host_data.shape[0], dtype=np.int32)
    
    
    
    instance_reserved = lemma_1(lemma1_result)
    
    # print(len(instance_reserved))
    device_instance_reserved = cp.asarray(instance_reserved)
    
    # print(len(lemma1_result))
    # os._exit(0)
    prob_skyline = lemma_2(device_instance_reserved, threshold)
    # print(prob_skyline)
    # x = lemma2_reserved.copy_to_host()
    # print(x)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    print("skyline size = ", len(prob_skyline))
