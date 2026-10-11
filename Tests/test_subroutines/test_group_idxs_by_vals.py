import numpy as np
import pandas as pd
import time
from pyEFPEHM.utils import my_cgroup_idxs_by_vals

###########################################################################

#function to compute the index of each unique value in arr
#arr is an array_like of non-negative indices
def python_group_idxs_by_vals(arr):

	#compute the maximum value of arr
	a_max = np.amax(arr)

	#initialize lists to store indices
	grouped_idxs = [[] for _ in range(a_max + 1)]

	#loop over array, storing the idx of each element in the correct list
	for i, a in enumerate(arr.tolist()): grouped_idxs[a].append(i)

	return grouped_idxs
	
#function to compute the index of each unique value in arr
#arr is an array_like of non-negative indices
def naive_group_idxs_by_vals(arr):

	#compute the maximum value of arr
	arr = np.asarray(arr)
	a_max = np.amax(arr)

	#loop over values of arr
	grouped_idxs = []
	for a in range(a_max + 1):
		#save the indices where arr == a
		grouped_idxs.append(np.where(arr==a)[0])

	return grouped_idxs

#function to compute the index of each unique value in arr
#arr is an array_like of non-negative indices
def numpy_group_idxs_by_vals(arr):

	#make sure arr is a numpy array
	arr = np.asarray(arr)

	#compute how many times each index appears in arr
	a_counts = np.bincount(arr)

	#now generate arrays to store indices and where each index will be located
	grouped_idxs = np.empty_like(arr)
	a_idxs = np.zeros_like(a_counts)
	a_idxs[1:] = np.cumsum(a_counts[:-1])

	slice_idxs = np.concatenate((a_idxs, [len(arr)]))
	
	#now loop over arr
	for i, a in enumerate(arr):
		grouped_idxs[a_idxs[a]] = i
		a_idxs[a] += 1
		
	return grouped_idxs, slice_idxs

#function to compute the index of each unique value in arr
#arr is an array_like of non-negative indices
def pandas_group_idxs_by_vals(arr):

	#create a pandas series from array
	s = pd.Series(arr)

	#group s by its own values
	return s.groupby(s).indices

###########################################################################

#Input
N = 10000000
M = 100

###########################################################################

#create random array
arr_test = np.random.randint(0, M, size=N)

#Time the O(N) cython method
start_time = time.time()
c_grouped_idxs = my_cgroup_idxs_by_vals(arr_test)
end_time = time.time()
print("O(N) cython implementation: %s seconds"%(end_time - start_time))


#Time the O(N) python method
start_time = time.time()
python_grouped_idxs = python_group_idxs_by_vals(arr_test)
end_time = time.time()
print("O(N) python implementation: %s seconds"%(end_time - start_time))

# Time the O(M*N) NumPy method
start_time = time.time()
naive_grouped_idxs = naive_group_idxs_by_vals(arr_test)
end_time = time.time()
print("O(M*N) numpy implementation: %s seconds"%(end_time - start_time))

# Time the O(N) NumPy-accelerated method
start_time = time.time()
np_grouped_idxs_all, np_slice_idxs = numpy_group_idxs_by_vals(arr_test)
end_time = time.time()
print("O(N) numpy implementation: %s seconds"%(end_time - start_time))

np_grouped_idxs = np.split(np_grouped_idxs_all, np_slice_idxs[1:])

# Time the O(?) NumPy-accelerated method
start_time = time.time()
pd_grouped_idxs = pandas_group_idxs_by_vals(arr_test)
end_time = time.time()
print("O(?) pandas implementation: %s seconds"%(end_time - start_time))


#Make sure all methods give the same indices for every value
n_groups = len(python_grouped_idxs)
for val in range(n_groups):
	g0 = c_grouped_idxs[val]
	assert np.array_equal(g0, python_grouped_idxs[val])
	assert np.array_equal(g0, naive_grouped_idxs[val])
	assert np.array_equal(g0, np_grouped_idxs[val])
	assert np.array_equal(g0, pd_grouped_idxs[val])

#the numpy split can include a trailing empty group
assert all(len(g)==0 for g in np_grouped_idxs[n_groups:])
