import numpy as np

def diference_in_both(list_a,list_b):
    dic_result = {}
    intersect = np.intersect1d(list_a,list_b)
    dic_result['Not_in_a'] = (set(list_b)-set(intersect))
    dic_result['Not_in_b'] = (set(list_a) - set(intersect))
    return dic_result

