import time
from datetime import datetime
import numpy as np
import itertools
from numpy.core.fromnumeric import _diagonal_dispatcher
import seaborn as sns
import pandas as pd
from itertools import cycle, permutations, combinations


"""
Compute cycle types for n states (2 inputs)
"""
# https://stackoverflow.com/questions/400794/generating-the-partitions-of-a-number
q = { 1: [[1]] }
def decompose(n):
    try:
        return q[n]
    except:
        pass

    result = [[n]]

    for i in range(1, n):
        a = n-i
        R = decompose(i)
        for r in R:
            if r[0] <= a:
                result.append([a] + r)

    q[n] = result
    return result

def cycle_types(N):
    r = np.arange(N)
    decomposition = decompose(N)
    ds = [all(x>1 for x in d) for d in decomposition]
    d = np.array(decomposition,dtype=object)
    return d[[i for i, x in enumerate(ds) if x]]

def get_whole_cycles(p):
    cycles = []
    others = list(itertools.permutations(np.arange(p-1)))
    for o in others:
        cycles.append([p-1]+list(o))
    return cycles

def tuples_to_list(row,r):
    cur = list(set([i for sub in row['comb'] for i in sub]))
    row['totcomb'] = cur
    row['sortedcomb'] = tuple(sorted(row['comb'], key=lambda tup: tup[1]))
    if cur == r:
        row['check'] = True
    else:
        row['check'] = False
    row['comb'] = [list(i) for i in row['comb']]
    return row

def get_all_cycles(N):
    types = cycle_types(N)
    cycles = []
    r = list(range(N))
    for cur, typ in enumerate(types):
        if len(typ) == 1:
            cycles = cycles + get_whole_cycles(N)
        else:
            combs = []
            for i in typ:
                combs.append(list(itertools.combinations(r,i)))

            # get all possible combination of this type where each number is permuted
            data = {'comb':[]}
            data['comb'] = list(itertools.product(*combs))
            d = pd.DataFrame(data=data)
            d = d.apply(tuples_to_list,args=(r,),axis=1)
            d = d[d['check']]
            d = d.drop_duplicates(subset=['sortedcomb'], keep="last")
            cycles = cycles + list(d['comb'])

    return cycles

"""
Compute sudoku SRVs, i.e. SRVs for two input variables
"""
        
def all_sudokus(states):
    cycles = get_all_cycles(states)
    print("ALL CYCLES",cycles)
    r = np.arange(states)
    srvs = []
    empt_mat = np.zeros((states,states))-1
    empt_mat[0] = r
    for c in cycles:
        temp = empt_mat.copy()
        temp = next_row(cycles,[c],[get_indexes(c,states)],temp,states)
        srvs = srvs + temp
    return srvs
   
def next_row(cycles,prev,prev_ids,temp,states,k=1):
    new_row = get_row(list(temp[k-1]),prev[-1])
    temp[k] = new_row
    if k+1 == len(temp):
        return [temp]
    
    old_ids = prev_ids[-1]  # dict value: index in cycle
    new_perms = []
    for c in cycles:
        new_ids = get_indexes(c,states)  # dict value: index in cycle
        
        # TO DO: replace with compute_cycle in cycle_check notebook?
        if check_cycles(prev,c,prev_ids,new_ids,states):
#             new = get_cycle(prev,c,old_ids,new_ids,states)
            prev_copy = prev.copy()
            prev_copy.append(c)
            prev_ids_copy = prev_ids.copy()
            prev_ids_copy.append(new_ids)
            temp_copy = temp.copy()
            new_perms = new_perms + next_row(cycles,prev_copy,prev_ids_copy,temp_copy,states,k+1)
    return new_perms

def get_indexes(cycle,states):
    r = np.arange(states)
    d = {}

    for c in r:
        if isinstance(cycle[0],list):
            d[c] = [[i, cyc.index(c)] for i, cyc in enumerate(cycle) if c in cyc][0]
        else:
            d[c] = [cycle.index(c)]
    return d

def get_row(x, cycle):
    if isinstance(cycle[0],int):
        states = len(x)
        xnew = x.copy()
        for val in cycle:
            xnew[x.index(val)] = cycle[(cycle.index(val)+1)%len(cycle)]
        return xnew
    else:
        for c in cycle:
            x = get_row(x,c)
        return x

def get_val(new_val, val_ids):
    ids = val_ids.copy()
    while len(ids) > 0:
        prev_val = new_val
        cur = ids[0]
        ids.pop(0)    
        new_val = prev_val[cur]
    val = prev_val[(cur+1)%len(prev_val)]
    return val

"Calculate cycle of combination of two cycles"
def get_cycle(old,new,old_ids,new_ids,states,cur=0):
    cycles = []
    cycle = [0]
    cur = 0
    r = list(range(1,states))
    while len(r)>0:
        old_val = get_val(old,old_ids[cur])
        new_val = get_val(new,new_ids[old_val])
        if new_val not in cycle:
            cycle.append(new_val)
            cur = new_val
            r.remove(new_val)
        else:
            cycles.append(cycle)
            cur = r[0]
            cycle = [cur]
            r.remove(cur)
    
    if len(cycle) != states:
        cycles.append(cycle)
        cycle = cycles
    return cycle

def check_cycles(prev,c,prev_ids,new_ids,states):
    for i,cur_prev in enumerate(reversed(prev)):
        if i > 0:
            prev_cycle = get_cycle(cur_prev, prev_cycle,prev_ids[len(prev)-1-i],ids,states)
            ids = get_indexes(prev_cycle,states)
        else:
            prev_cycle = prev[-1]
            ids = prev_ids[-1]
        
        if not check_cycle(prev_cycle, c,ids,new_ids,states):
            return False
    return True

def check_cycle(old,new,old_ids,new_ids,states):
    r = np.arange(states)
    
    for i in r:
        old_val = get_val(old,old_ids[i])
        new_val = get_val(new,new_ids[old_val])
        if i == new_val:
            return False
    return True