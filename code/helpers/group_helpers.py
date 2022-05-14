import numpy as np
import itertools
import pandas as pd
import time
import pickle
import os
import glob
import dit
from datetime import datetime
from jointpdfpython3.JointProbabilityMatrix import JointProbabilityMatrix, ConditionalProbabilityMatrix
lenX = 2

#############################
# Calculates all possible modulo random variables for a given number of states
#############################
def exhaustive_search(args):
    first_rows = list(itertools.permutations(np.arange(args.states)))

    rows = []
    empt_mat = np.zeros((args.states,args.states))-1
    for f in first_rows:
        temp = empt_mat.copy()
        temp[0] = f
        rows.append(temp)

    return find_sudokus(rows)

# TO DO optimize by not transferring whole list of sudokus the whole time
def find_sudokus(perms,k=0):
    new_perms = []
    print(len(perms))
    for p in perms:
        new_perms = new_perms + append_rows(p,k+1)

    if k == len(perms[0])-2:
        return new_perms
    else:
        return find_sudokus(new_perms,k+1)

def append_rows(srv,k=1):
    rows = all_rows(srv, k)
    new_mods = []
    for r in rows:
        temp = srv.copy()
        temp[k] = r
        new_mods.append(temp)
    return new_mods

# TODO optimize getting 'prevs' list srv[:k]?
def all_rows(srv, k):
    s = len(srv)
    states = np.arange(s)
    prevs = srv[:k]

    # only select valid rows where all values occur once
    others = [np.setdiff1d(states,prevs[:,i]) for i in range(s)]
    arr = np.array(list(itertools.product(*others)))
    valids = np.array([np.count_nonzero(np.bincount(a))==s for a in arr]) 
    return arr[np.where(valids)]

###############
# Compute cycle types for n states (2 inputs)
###############
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

############################################
# Compute sudoku SRVs, i.e. SRVs for two input variables
############################################
def all_sudokus(args):
    states=args.states
    cycles = get_all_cycles(states)
    r = np.arange(states)
    srvs = []
    empt_mat = np.zeros((states,states))-1
    empt_mat[0] = r
    print(len(cycles))
    for i,c in enumerate(cycles):
        print(i)
        temp = empt_mat.copy()
        temp = next_row(cycles,[c],[get_indexes(c,states)],temp,states)
        srvs = srvs + temp
    return srvs
   
def next_row(cycles,prev,prev_ids,temp,states,k=1):
    new_row = get_row(list(temp[k-1]),prev[-1])
    temp[k] = new_row
    if k+1 == len(temp):
        return [temp]
    
    new_perms = []
    for c in cycles:
        new_ids = get_indexes(c,states)  # dict value: index in cycle
        
        if check_cycles(prev,c,prev_ids,new_ids,states):
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

###################################################
# Transform sudokus to SRVs and compute conditional entropy matrix between sudokus
###################################################
def cond_entropies(mods, args,cond=False):
    cond_mat = np.zeros((len(mods),len(mods)))
    subjects = np.arange(lenX)
    pX = JointProbabilityMatrix(lenX,args.states,joint_probs=args.dist_type)
    print("TOT RVs",len(mods))
    for i in range(len(mods)):
        print(i, datetime.fromtimestamp(time.time()))

        pXSi = append_srv(pX.copy(),mods[i],args, subjects, cond)
        entSi = pXSi.entropy(variables=[lenX])
        for j in range(i+1,len(mods)):
            pXS = append_srv(pXSi.copy(),mods[j],args, subjects, cond)

            entSj = pXS.entropy(variables=[lenX+1])     
            entS = pXS.entropy(variables=[lenX,lenX+1])
            H_Si_gSj = entS - entSj
            H_Sj_gSi = entS - entSi
            # H_Si_gSj = pXS.conditional_entropy([lenX],[lenX+1])
            # H_Sj_gSi = pXS.conditional_entropy([lenX+1],[lenX])

            cond_mat[i][j] = H_Si_gSj
            cond_mat[j][i] = H_Sj_gSi
                        
    return cond_mat

def append_srv(pX,srv,args,subjects,cond=False):
    if cond:
        condmatrix = mat_to_cond(srv,args.states,args.lenX)
    else:
        matrix = srv_to_mat(srv, args.states,args.lenX)
        condmatrix = mat_to_cond(matrix,args.states,args.lenX)

    # cond to Pr(XS)
    pX.append_variables_using_conditional_distributions(condmatrix, subjects)
    return pX

def mat_to_cond(matrix, states,lenX):
    # get list of all possible input states
    input_values = list(itertools.product(*[np.arange(states) for _ in range(lenX)]))

    cmatrix = ConditionalProbabilityMatrix()
    for i,row in enumerate(matrix):
        pdummy = JointProbabilityMatrix(1, states)
        pdummy.joint_probabilities.joint_probabilities = row
        cmatrix.cond_pdf[input_values[i]] = pdummy
    return cmatrix

def srv_to_mat(srv, states, lenX, p=0):
    input_states = states**lenX
    code_flat = srv.flatten()
    cond_mat = np.zeros((input_states,states))

    prob_cur = 1 - (p*((states-1)/states))
    prob_others = (1-prob_cur)/(states-1)
    for i in range(len(cond_mat)):
        cond_mat[i, int(code_flat[i])] = prob_cur
        
        others = list(range(states))
        del others[int(code_flat[i])]
        cond_mat[i,others] = prob_others

    return cond_mat

def group_by_cond(mat):
    labels = np.arange(len(mat))
    groups = {l:[l] for l in labels}
    for i,m in enumerate(mat):
        for j in range(i+1, len(mat)):
            if labels[i] != labels[j]:
                if m[j] == 0 and mat[j][i] == 0:
                    new_group = groups[labels[i]] + groups[labels[j]]
                    del groups[labels[i]]
                    del groups[labels[j]]
                    groups[labels[i]] = new_group
                    labels[[i,j]] = labels[i]
    return groups

###############
# COMPUTE LOWER-ORDER & OVERSIZED SUDOKUS
###############
def all_lowerorder_sudokus(args):
    # load all MSRVs with M < args.states
    sudokus, combs = sudokus_combs(args.states,args.folder)
    noisies = []
    M = args.states
    for i,c in enumerate(combs):
        # print(i)
        curlen = len(c)
        cursudos = sudokus[curlen]
        permutations = list(itertools.permutations(c))
        for c in cursudos:
            for p in permutations:
                unimat = np.full((M,M,M),1/M)
                for i in range(curlen):
                    for j in range(curlen):
                        cur = unimat[p[i]][p[j]]
                        cur[list(p)] = 0
                        cur[p[int(c[i][j])]] = curlen/M
                        unimat[p[i]][p[j]] = cur
                noisies.append(unimat.reshape(M*M,M))
    return np.array(noisies)

def sudokus_combs(states,folder):
    allfiles = np.array([name for name in list(glob.iglob(folder+"*.pkl")) if 'permutation' in name])
    allsudokus = {}
    combs = []
    for a in allfiles:
        cur = int(a[-5])
        if cur < states:
            with open(a, 'rb') as f:
                allsudokus[cur] = pickle.load(f)
                combs = combs + list(itertools.combinations(np.arange(states), cur))
    return allsudokus, combs

def exhaustive_noisy(args,Msudo=0):
    with open(args.folder+'permutation_sudokus_states'+str(args.states)+'.pkl', 'rb') as f:
        sudokus = pickle.load(f)
    M = len(sudokus[0])
    if Msudo == 0:
        Msudo = M
    adjusted = []
    for s in sudokus:
        condsym,reshaped = reshapesym(s,M,Msudo)
        adjusted = adjusted + get_noisies(reshaped,M,Msudo)
    return np.array(adjusted)
def reshapesym(s,M,Msudo=0):
    shape = list(s.shape)
    condsym = np.zeros(tuple(shape+[Msudo]))
    for i in range(M):
        for j in range(M):
            condsym[i,j][int(s[i,j])] += 1
    reshaped = condsym.reshape((M*M,Msudo))
    return condsym,reshaped

def get_noisies(reshaped,M,Msudo):
    adjusted = []
    r = np.arange(M*M)
    uni = np.zeros(Msudo)+(1/Msudo)
    for k in range(1,M*M):
        c = list(itertools.combinations(r,k))
        for combs in c:
            cur = reshaped.copy()
            for comb in combs:
                cur[comb] = uni
            adjusted.append(cur)
    return adjusted

def getcond(nsym):
    symstates = len(nsym[0])
    b = np.not_equal([1/symstates], nsym)
    yesrows = np.unique(np.where(b)[0])
    return nsym[yesrows]

# COMPUTE OVERSIZED SYMS
def oversizedsyms(gs,syms):
    ns = []
    for k,noisies in syms.items():
        uni = [1/k for _ in range(k)]
        totlen = k*k
        for n in noisies:
            if totlen-len(np.where(np.all(n==uni,axis=1))[0])==(gs*gs):
                ns.append(n)
    nsym = [getcond(i) for i in ns]
    return nsym

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)   

def oversizedtodit(s,goalstate,lenX=2):
    args = Namespace(lenX=2,states=goalstate) 
    pX = JointProbabilityMatrix(2,goalstate,joint_probs='iid')
    prX = pX.joint_probabilities.joint_probabilities
    
    ls = np.arange(args.states)
    lensym = len(s[0])
    inpstates=list(itertools.product(*[ls,ls]))
    s = s.reshape((args.states,args.states,lensym))
    z = []
    for i in inpstates:
        z.append(s[i]*prX[i])
    z = np.array(z).reshape((goalstate,goalstate,lensym))
    return dit.Distribution.from_ndarray(z)

# LOAD & SORT SYMS
def load_sudokus(states=2):
    os.chdir("../results/sudokus")
    allfiles = glob.glob("*.npy")
    filename = None
    for file in allfiles:
        if 'constructedSRVstates'+str(states)+'.npy' in file:
            filename = file
            # load symSRVs for number of states
            with open(file, 'rb') as f:
                syms = np.load(f,allow_pickle=True)

    os.chdir("../../code")
    return syms

def classify_syms(syms,states):
    classes = {}
    for i in range(len(syms)):
        cur = syms[i]
        unqs = np.unique(cur)
        lc = len(cur[0])
        if lc == states:
            try:
                b = (unqs==np.arange(states)).all()
                if 1/states not in unqs:
                    classes[i] = 'full sym'
                elif 1-(1/states) in unqs:
                    classes[i] = 'lower order sym'
                else:
                    classes[i] = 'noisy sym'
            except AttributeError as err:
                if 1/states not in unqs:
                    classes[i] = 'full sym'
                elif 1 not in unqs:
                    classes[i] = 'noisy sym'
                else:
                    classes[i] = 'lower order sym'
        elif lc>states:
            classes[i] = 'oversized states '+str(lc)
    classvals = list(classes.values())
    invclasses = {k:[] for k in np.unique(classvals)}
    for i, c in enumerate(classvals):
        invclasses[c].append(i)
    return classvals, invclasses
