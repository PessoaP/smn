import numpy as np
from numba import njit,typed,types
from numba.experimental import jitclass

@njit
def sm_asarray(lines,columns,values,shape):
    res = np.zeros((shape,shape))
    for i in range(lines.size):
        res[lines[i],columns[i]] += values[i]
    return res
@njit
def sm_sum(lines,columns,values,shape):
    res = np.zeros(shape)
    ind=lines
    for i in range(lines.size):
        res[ind[i]] += values[i]
    return res

@njit
def multiply_sparse_matrices(lines1,columns1,values1,lines2,columns2,values2):    
    res_lines = []
    res_cols = []
    res_vals = []

    for line1, col1, val1 in zip(lines1,columns1, values1):
        for line2,col2, val2 in zip(lines2,columns2,values2):
            if col1 == line2:
                res_lines.append(line1)
                res_cols.append(col2)
                res_vals.append(val1*val2)

    return np.array(res_lines),np.array(res_cols),np.array(res_vals)

@njit
def simplify_sparse_matrix(lines,columns, values):
    seen_indices={} #dictionary to be filled at the first apparence of a line and colums pairing

    for lin,col,val in zip(lines,columns, values):
        if (lin,col) in seen_indices:
            seen_indices[(lin,col)] += val
        else:
            seen_indices[(lin,col)] = val
    
    simplified_lines  = []
    simplified_cols = []
    simplified_values = []

    for (l,c),v in zip(seen_indices.keys(),seen_indices.values()):
        if v!=0:
            simplified_lines.append(l)
            simplified_cols.append(c)
            simplified_values.append(v)
    return np.array(simplified_lines), np.array(simplified_cols), np.array(simplified_values)

@jitclass([('lines', types.int64[:]),
           ('columns', types.int64[:]),
           ('values', types.float64[:]),
           ('shape',types.int64)])    
class sparse_matrix:
    def __init__ (self,lins,cols,values,shap=0):
        self.lines=lins
        self.columns=cols
        self.values=values
        self.shape=shap
        if shap == 0:
            self.shape=1+lins.max()#assume always square
    
    def __add__(self,other):
        return sparse_matrix(np.concatenate((self.lines,other.lines)),
                             np.concatenate((self.columns,other.columns)),
                             np.concatenate((self.values,other.values)),
                             max(self.shape,other.shape))
            
    def __sub__(self,other):
        return sparse_matrix(np.concatenate((self.lines,other.lines)),
                             np.concatenate((self.columns,other.columns)),
                             np.concatenate((self.values,-other.values)),
                             max(self.shape,other.shape))
    
    def to_dense(self):
        return sm_asarray(self.lines,self.columns,self.values,self.shape)

    def line_sum(self):
        return sm_sum(self.lines,self.columns,self.values,self.shape)
    
    def column_sum(self):
        return sm_sum(self.columns,self.lines,self.values,self.shape)

    def simplify(self):
        self.lines,self.columns,self.values = simplify_sparse_matrix(self.lines,self.columns, self.values)
        ind1 = np.argsort(self.columns)
        self.lines,self.columns,self.values = self.lines[ind1],self.columns[ind1],self.values[ind1]
        ind2 = np.argsort(self.lines)
        self.lines,self.columns,self.values = self.lines[ind2],self.columns[ind2],self.values[ind2]

    def __mul__(self,other):
        if other.shape != self.shape:
            print('Warning: you are trying to multiply square matrices of different shapes')
        prod = sparse_matrix(*multiply_sparse_matrices(self.lines,self.columns,self.values,other.lines,other.columns,other.values), self.shape)  
        prod.simplify()
        return prod

@njit
def array_times_sm(arr,sm):
    res=np.zeros(sm.shape,types.float64)
    for (line,col,val) in zip(sm.lines,sm.columns,sm.values):
        res[col] += val*arr[line]
    return res

@njit
def sm_times_array(sm,arr):
    res=np.zeros(sm.shape,types.float64)
    for (line,col,val) in zip(sm.lines,sm.columns,sm.values):
        res[line] += val*arr[col]
    return res

def sum(sm,axis=-1):
    if axis ==0:
        return sm.column_sum()
    elif axis ==1:
        return sm.line_sum()
    else:
        return sm.line_sum().sum()
    
def dot(A,B):
    if isinstance(A,sparse_matrix):
        if isinstance(B,sparse_matrix):
            return A*B
        if isinstance(B,np.ndarray):
            return sm_times_array(A,B)
    elif isinstance(A,np.ndarray):
        if isinstance(B,sparse_matrix):
            return array_times_sm(A,B)
        if isinstance(B,np.ndarray):
            return np.dot(A,B)
