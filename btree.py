import numpy as np

# Implementation of _BTree from the package lifelines.utils.btree



class _BTree:

    def __init__(self, values):
        self._tree=self._treeify(values)
        self._counts = np.zeros_like(self._tree, dtype=int)

    @staticmethod

    def _treeify(values):
        if len(values)==1:
            return values
        tree = np.empty_like(values)
        last_full_row = int(np.log2(len(values)+1)-1)
        len_ragged_row = len(values) - (2**(last_full_row+1)-1)
        if len_ragged_row>0:
            bottom_row_ix  =np.s_[:2*len_ragged_row : 2]
            tree[-len_ragged_row:]  = values[bottom_row_ix]
            values = np.delete(values, bottom_row_ix)

        values_start = 0
        values_space = 2
        values_len = 2**last_full_row
        while values_start<len(values):
            tree[values_len-1:2*values_len-1] = values[values_start::values_space]
            values_start+=int(values_space/2)
            values_space*=2
            values_len = int(values_len/2)
        return tree

    def insert(self, value):
        i=0
        n=len(self._tree)
        while i<n:
            cur = self._tree[i]
            self._counts[i]+=1
            if value<cur:
                i =2*i+1
            elif value>cur:
                i=2*i+2
            else:
                return
        raise  ValueError("Value %s not contained in tree." "Also, the counts are messed up." % value)

    def __len__(self):
        return self._counts[0]


    def rank(self, value):
        i=0
        n= len(self._tree)
        rank=0
        count=0
        while i<n:
            cur = self._tree[i]
            if value<cur:
                i=2*i+1
                continue
            elif value>cur:
                rank+=self._counts[i]
                nexti = 2*i+2
                if nexti<n:
                    rank -= self._counts[nexti]
                    i=nexti
                    continue
                else:
                    return (rank,count)
            else:
                count=self._counts[i]
                lefti = 2*i +1
                if lefti<n:
                    nleft = self._counts[lefti]
                    count -= nleft
                    rank+=nleft
                    righti = lefti+1
                    if righti<n:
                        count -= self._counts[righti]
                return (rank,count)
        return (rank, count)