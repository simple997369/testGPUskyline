import os, sys
sys.path.append(os.path.abspath(os.pardir))

import numpy as np
import time
from data.dataClass import batchImport
from tqdm import tqdm 

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


def lemma_2(instance_reserved, threshold):
    obj_prob = []
    ins_prob = []
    for point in tqdm(instance_reserved):
        total_prob = {}
        object_reserved = np.empty((0,dim+3))
        for other_point in instance_reserved:
            is_reserved = False
            if point[0]!=other_point[0]:
                if all(point[3:] >= other_point[3:]):
                    is_reserved = True
            if is_reserved:
                # print(other_point)
                object_reserved = np.append(object_reserved, [other_point], axis=0)
        # print("#######",object_reserved)
        # 可以跟上面合併，item = other_point
        for item in object_reserved:
            number = item[0]
            ins_p = item[2]
            if number in total_prob:
                total_prob[number] += ins_p
            else:
                total_prob[number] = ins_p
                
        # sourceFile = open('demo.txt', 'w')
        # print(total_prob) #lemma_3
        # exit()
        # sourceFile.close()
        
        probability = 1
        for key, value in total_prob.items():
            probability = probability * (1 - value)
        lst = [point[0], point[2], probability]
        ins_prob.append(lst)
    # print(ins_prob) #lemma_4
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
'''
def lemma_2(instance_reserved, threshold):
    obj_prob = []
    ins_prob = []

    for point in instance_reserved:
        object_reserved = np.array([other_point for other_point in instance_reserved if point[0] != other_point[0] and all(point[3:] >= other_point[3:])])
        
        total_prob = np.sum(object_reserved[:, [0, 2]], axis=0)
        print(total_prob)
        probability = np.prod(1 - total_prob[:, 1])

        ins_prob.append([point[0], point[2], probability])

    for item in ins_prob:
        number, ins_p = item[0], item[1]
        obj_p = np.sum([entry[1] for entry in obj_prob if entry[0] == number], default=0)
        obj_prob.append([number, obj_p + item[1] * item[2]])

    prob_skyline = [entry for entry in obj_prob if entry[1] >= threshold]

    return prob_skyline
'''

def skyline_query(data, count, ps):
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
    dim = 7
    count = 10000
    ps = 5
    # 以object的角度來看windowsize
    wsize = count * ps
    threshold = 0.6
    
    host_data, Max = batchImport('anticor_'+str(dim)+'d_'+str(count)+'_'+str(ps)+'.txt', count, dim, ps)
    
    # dim = 4
    # count = 3
    # ps = 3
    # # 以object的角度來看windowsize
    # wsize = count * ps
    # threshold = 0.6
    
    # host_data, Max = batchImport('test_dataset.txt', count, dim, ps)
    
    start_time = time.time()
    avgsk1 = 0
    for wcount in range(0, host_data.shape[0]-wsize+1, ps):
        instance_reserved = lemma_1(host_data[ wcount : wcount + wsize], Max[ wcount//ps : wcount//ps + wsize//ps ])
        print("#####",len(instance_reserved))
        prob_skyline = lemma_2(instance_reserved, threshold)
        # print(prob_skyline)
        # result = skyline_query(host_data[wcount:wcount+wsize], count, ps)
        avgsk1 += len(prob_skyline)

    print("--- %s seconds ---" % (time.time() - start_time))
    print("avgsk1 = ", avgsk1)
