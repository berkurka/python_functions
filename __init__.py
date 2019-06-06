import numpy as np
import pandas as pd

def diference_in_both(list_a,list_b):
    dic_result = {}
    intersect = np.intersect1d(list_a,list_b)
    dic_result['Not_in_a'] = (set(list_b)-set(intersect))
    dic_result['Not_in_b'] = (set(list_a) - set(intersect))
    return dic_result

# compare_op = lambda a,b,op: a>b if op==">" else a<b  if op=="<" else a<=b  if op=="<=" else a<=b  if op==">=" else  a==b


bola = {"folds":[1,2,3], "mean":[50,60,70],"std":[2,4,5]}
# bola[3] = [5,7]
final_df = pd.DataFrame(bola)
print(final_df)


stringbk = [1,2,3,4]
print(stringbk.__reversed__()[1])


def list_to_single_string(list1):
    return ''.join(str(e) for e in list1)

def list_to_int(list1):
    return list(map(int, list1))

def list_to_string(list1):
    return list(map(str, list1))