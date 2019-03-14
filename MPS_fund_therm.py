import numpy as np
import scipy as sc
from scipy.linalg import svd
import time
import sys
from decimal import Decimal
from math import factorial

#************************************#
#***------------------------------***#
#***| USEFUL BLOCKS AS FUNCTIONS |***#
#***------------------------------***#
#************************************#

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

#################################################
### SVD with only significant singular values ###
#################################################
def SVD_sig(block,cutoff):
	"""Performing SVD with singular values above a certain threshold
	INPUT: the block to perform SVD on, threshold for singular values
	OUTPUT: SVD elements and the number of significant singular values"""

	# Predefining the list for the output
	svd_final = [0]*3

	# Performing the SVD
	svd_init  = svd(block,full_matrices=False)
	# Storing the singular values in an array
	sing      = np.array(svd_init[1])
	# Storing only the significant singular values
#	print(sing)

	d   = 0
	eps = 100.
	while eps>cutoff:
		d = d+1
		eps = np.sqrt(np.sum(sing[d:]**2))

	# Determining the number of significant singular values and resizing the svd matrices accordingly
	#print("link_dim",sing_num)
#	if d==1:
#		link_dim     = 0
#		svd_final[0] = svd_init[0][:,0]
#		svd_final[1] = sing[0]
#		svd_final[2] = svd_init[2][0,:]
#	else:
	link_dim = d
	svd_final[0] = svd_init[0][:,:link_dim]
	svd_final[1] = np.diag(sing[:d])
	svd_final[2] = svd_init[2][:link_dim,:]

	# Clear unnecessary variables
	svd_init  = None
	sing      = None
	sign_sing = None

	return svd_final,link_dim

######################################################################################################################################################################################################
#######################
### Merging indices ###
#######################
#S2F2 = MERGE(S2F2,[1,2])
def MERGE(what,where):
	"""Merging indices to provide a lower-rank tensor from a block
	INPUT: block for index-merging and the number of indices on the left and right hand side
	OUTPUT: block with the merged indices and the original dimensions on the left and right hand side"""

	number = len(where)
	orig_shape = np.array(what.shape)
	merge_dims = orig_shape[where]
	if where[0]>0 and where[-1]<len(orig_shape-1):
		new_block = np.zeros(np.concatenate((orig_shape[:where[0]],np.array([np.prod(merge_dims),]),orig_shape[where[-1]+1:]),axis=0),complex)
	elif where[0]>0 and where[-1]==len(orig_shape-1):
		new_block = np.zeros(np.concatenate((orig_shape[:where[0]],np.array([np.prod(merge_dims),])),axis=0),complex)
	elif where[0]==0 and where[-1]<len(orig_shape-1):
		new_block = np.zeros(np.concatenate((np.array([np.prod(merge_dims),]),orig_shape[where[-1]+1:]),axis=0),complex)
	else:
		print("Problem with merging location")
		
	i1 = np.arange(0,merge_dims[-1])
	idx_o = [slice(None)]*what.ndim
	idx_o[where[-1]] = i1
	idx_n = [slice(None)]*(what.ndim-(number-1))
	
	if number == 2:
		for i2 in range(0,merge_dims[0]):
			idx_o[where[0]] = i2
			idx_n[where[0]] = i1 + merge_dims[1]*i2
			
			new_block[tuple(idx_n)] = what[tuple(idx_o)]
			
	elif number == 3:
		for i3 in range(0,merge_dims[0]):
			for i2 in range(0,merge_dims[1]):
				idx_o[where[1]] = i2
				idx_o[where[0]] = i3
				idx_n[where[0]] = i1 + merge_dims[2]*i2 + merge_dims[1]*merge_dims[2]*i3
				
				new_block[tuple(idx_n)] = what[tuple(idx_o)]

	elif number == 4:
		for i4 in range(0,merge_dims[0]):
			for i3 in range(0,merge_dims[1]):
				for i2 in range(0,merge_dims[2]):
					idx_o[where[2]] = i2
					idx_o[where[1]] = i3
					idx_o[where[0]] = i4
					idx_n[where[0]] = (i1 + merge_dims[3]*i2 + merge_dims[2]*merge_dims[3]*i3 + 
									   merge_dims[1]*merge_dims[2]*merge_dims[3]*i4)
				
					new_block[tuple(idx_n)] = what[tuple(idx_o)]

	elif number == 5:
		for i5 in range(0,merge_dims[0]):
			for i4 in range(0,merge_dims[1]):
				for i3 in range(0,merge_dims[2]):
					for i2 in range(0,merge_dims[3]):
						idx_o[where[3]] = i2
						idx_o[where[2]] = i3
						idx_o[where[1]] = i4
						idx_o[where[0]] = i5
						idx_n[where[0]] = (i1 + merge_dims[4]*i2 + merge_dims[3]*merge_dims[4]*i3 + 
										   merge_dims[2]*merge_dims[3]*merge_dims[4]*i4 +
										   merge_dims[1]*merge_dims[2]*merge_dims[3]*merge_dims[4]*i5)
					
						new_block[tuple(idx_n)] = what[tuple(idx_o)]

	return new_block, merge_dims

######################################################################################################################################################################################################
###############################
### Undoing the index-merge ###
###############################
def UNMERGE(block,new_shape,merge_dims,where):
	"""Unmerging indices to provide a higher-rank tensor from block
	INPUT: the block for which the index-merging should be undone, the dimensions of the final block indices
	on the left and the right.
	OUTPUT: the obtained higher-rank tensor"""

	new_block  = np.zeros(new_shape,complex)
	merged     = block.shape[where[0]]
	number = len(where)
	idx_o = [slice(None)]*block.ndim
	idx_n = [slice(None)]*(len(new_shape))
		
	if number == 2:
		for J in range(0,merged):
			idx_o[where[0]] = J
			idx_n[where[1]] = J%merge_dims[1]
			idx_n[where[0]] = int(J/merge_dims[1])
			
			new_block[tuple(idx_n)] = block[tuple(idx_o)]
			
	elif number == 3:
		for J in range(0,merged):
			idx_o[where[0]] = J
			idx_n[where[2]] = J%merge_dims[2]
			idx_n[where[1]] = int(J/merge_dims[2])%merge_dims[1]
			idx_n[where[0]] = int(J/(merge_dims[1]*merge_dims[2]))
			
			new_block[tuple(idx_n)] = block[tuple(idx_o)]

	elif number == 4:
		for J in range(0,merged):
			idx_o[where[0]] = J
			idx_n[where[3]] = J%merge_dims[3]
			idx_n[where[2]] = int(J/merge_dims[3])%merge_dims[2]
			idx_n[where[1]] = int(J/(merge_dims[3]*merge_dims[2]))%merge_dims[1]
			idx_n[where[0]] = int(J/(merge_dims[1]*merge_dims[2]*merge_dims[3]))
		
			new_block[tuple(idx_n)] = block[tuple(idx_o)]
			
	elif number == 5:
		for J in range(0,merged):
			idx_o[where[0]] = J
			idx_n[where[4]] = J%merge_dims[4]
			idx_n[where[3]] = int(J/merge_dims[4])%merge_dims[3]
			idx_n[where[2]] = int(J/(merge_dims[4]*merge_dims[3]))%merge_dims[2]
			idx_n[where[1]] = int(J/(merge_dims[4]*merge_dims[2]*merge_dims[3]))%merge_dims[1]
			idx_n[where[0]] = int(J/(merge_dims[1]*merge_dims[2]*merge_dims[3]*merge_dims[4]))
		
			new_block[tuple(idx_n)] = block[tuple(idx_o)]
			
	return new_block


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

##########################################
### Relocating the orthognality centre ###
##########################################
def OC_reloc(state_left,state_right,where,tolerance):
	"""Relocating the orthogonality centre from either of the two adjacent timebins
	INPUT: time bin state on the left, on the right, where the orthogonality centre should go and the allowed
	tolerance of the SVD singular values
	OUTPUT: the obtained time bins on the left and right"""
#	print(state_left)
#	print(state_left.shape)
	# Number of indices for each state besides the link index between them
	left_ind = len(state_left.shape)-1
	right_ind = len(state_right.shape)-1
	# Number of indices left after contraction
	num_ind = left_ind+right_ind

	if num_ind==3:
		# Tensor after the contraction at the link index between the states
		Combo = np.tensordot(state_left,state_right,1)
		if left_ind ==2: # link index is located on the left
			# Merging the indices on the left to be able to perform svd
			Combo_merged,merge_dims = MERGE(Combo,[0,1])
			# number of singular values that matter is in link_dim
			Combo_merged_svd,link_dim = SVD_sig(Combo_merged,tolerance)
			Combo_merged=None
			# Dimensions of the indices after the OC relocation and unmerge of indices
			dims = np.concatenate((merge_dims,np.array([link_dim])),axis=0)
			if where=="left":
				# OC relocation to the left
				new_left_merged = np.dot(Combo_merged_svd[0],Combo_merged_svd[1])
				#def UNMERGE(block,new_shape,merge_dims,where):
				new_left = UNMERGE(new_left_merged,dims,merge_dims,[0,1])
				new_left_merged=None
				new_right = Combo_merged_svd[2]
				Combo_merged_svd=None
			elif where=="right":
				# OC relocation to the right
				new_right = np.dot(Combo_merged_svd[1],Combo_merged_svd[2])
				new_left = UNMERGE(Combo_merged_svd[0],dims,merge_dims,[0,1])
				Combo_merged_svd=None
		elif left_ind ==1: # link index is located on the right
			# Merging the indices on the right to be able to perform svd
			Combo_merged,merge_dims = MERGE(Combo,[1,2])
			# number of singular values that matter is in link_dim
			Combo_merged_svd, link_dim = SVD_sig(Combo_merged,tolerance)
			Combo_merged=None
			# Dimensions of the indices after the OC relocation and unmerge of indices
			dims = np.concatenate((np.array([link_dim]),merge_dims),axis=0)
			if where=="left":
				# OC relocation to the left
				new_left = np.dot(Combo_merged_svd[0],Combo_merged_svd[1])
				new_right = UNMERGE(Combo_merged_svd[2],dims,merge_dims,[1,2])
				Combo_merged_svd=None
			elif where=="right":
				# OC relocation to the right
				new_right_merged = np.dot(Combo_merged_svd[1],Combo_merged_svd[2])
				new_right = UNMERGE(new_right_merged,dims,merge_dims,[1,2])
				new_right_merged=None
				new_left = Combo_merged_svd[0]            
				Combo_merged_svd=None
                
	elif num_ind==4:
		# Tensor after the contraction at the link index between the states
		Combo = np.tensordot(state_left,state_right,1)
		# Merging the indices on both sides to be able to perform svd
		Combo_merged_left,merge_dims_left = MERGE(Combo,[0,1])
		Combo_merged,merge_dims_right = MERGE(Combo_merged_left,[1,2])
		# number of singular values that matter is in link_dim
		Combo_merged_svd,link_dim = SVD_sig(Combo_merged,tolerance)
		Combo_merged=None
		# Dimensions of the indices on the left after the OC relocation and unmerge of indices
		dims_left = np.array([merge_dims_left[0],merge_dims_left[1],link_dim])
		# Dimensions of the indices on the right after the OC relocation and unmerge of indices
		dims_right = np.array([link_dim,merge_dims_right[0],merge_dims_right[1]])
		if where=="left":
			# OC relocation to the left
			new_left_merged = np.dot(Combo_merged_svd[0],Combo_merged_svd[1])
			# unmerging the indices on the left
			new_left = UNMERGE(new_left_merged,dims_left,merge_dims_left,[0,1])
			new_left_merged=None
			# unmerging the indices on the right
			new_right = UNMERGE(Combo_merged_svd[2],dims_right,merge_dims_right,[1,2])
		elif where=="right":
			# OC relocation to the right
			new_right_merged = np.dot(Combo_merged_svd[1],Combo_merged_svd[2])
			# unmerging the indices on the right
			new_right = UNMERGE(new_right_merged,dims_right,merge_dims_right,[1,2])
			new_right_merged=None
			# unmerging the indices on the left
			new_left = UNMERGE(Combo_merged_svd[0],dims_left,merge_dims_left,[0,1])
		Combo_merged_svd=None
	return new_left,new_right

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

####################################
### Separate two physical blocks ###
####################################
def cut(block,tolerance,how,OC=None):
	"""cutting up a block to different time bins
	INPUT: block to be separated, the tolerance of the SVD singular values, where the link indices initially
	are and if on both sides, then which time bin should incorporate the orthogonality centre (optional)
	OUTPUT: the obtained time bins on the left and right"""

	if how=="left":
		# merging the link indices into the physical indices
		block_merged, dims = MERGE(block,[0,1])
		# separate the block
		block_merged_svd,link_dim = SVD_sig(block_merged,tolerance)
		#print(block.shape, block_merged.shape)
		block_merged=None
		# specifying the final indices of the left block
		left_dims = np.concatenate((dims,np.array([link_dim])),axis=0)
		# unmerging the link index
		new_left = UNMERGE(block_merged_svd[0],left_dims,dims,[0,1])
		if len(block_merged_svd[2].shape)>1:
			# position the orthogonality centre to the right
			new_right = np.einsum("ij,jk->ik",block_merged_svd[1],block_merged_svd[2])
		elif len(block_merged_svd[2].shape)==1 & len(block_merged_svd[0].shape)==1:  # for separate bins
			# position the orthogonality centre to the right
			new_right = block_merged_svd[1]*block_merged_svd[2]
		else:
			print("something's wrong with the ranks of the tensors after svd in cut left")
            
	elif how=="right":
		# merging the link indices into the physical indices
		block_merged, dims = MERGE(block,[1,2])
		# separate the block
		block_merged_svd,link_dim = SVD_sig(block_merged,tolerance)
		#print(block.shape, block_merged.shape)
		block_merged=None
		# specifying the final indices of the right block
		right_dims = np.concatenate((np.array([link_dim]),dims),axis=0)
		new_left = block_merged_svd[0]
		if len(block_merged_svd[2].shape)>1:
			# position the orthogonality centre to the right and unmerging the link index
			new_right = UNMERGE(np.einsum("ij,jk->ik",block_merged_svd[1],block_merged_svd[2]),right_dims,dims,[1,2])
		elif len(block_merged_svd[2].shape)==1 & len(block_merged_svd[0].shape)==1: # for separate bins
			# position the orthogonality centre to the right and unmerging the link index
			new_right = UNMERGE(block_merged_svd[1]*block_merged_svd[2],right_dims,dims,[1,2])
		else:
			print("something's wrong with the ranks of the tensors after svd in cut right")

	elif how=="both":
		# merging the link indices into the physical indices
		block_left_merged, left_dim = MERGE(block,[0,1])
		block_merged, right_dim = MERGE(block_left_merged,[1,2])
		# separate the block
		block_merged_svd,link_dim = SVD_sig(block_merged,tolerance)
		#print(block.shape, block_merged.shape)
		block_merged=None

		left_dims = np.concatenate((left_dim,np.array([link_dim])),axis=0)
		right_dims = np.concatenate((np.array([link_dim]),right_dim),axis=0)
		if OC=="left":
			# positioning the orthognality centre to the left
			new_right = UNMERGE(block_merged_svd[2],right_dims,right_dim,[1,2])
			new_left = UNMERGE(np.einsum("jk,kl->jl",block_merged_svd[0],block_merged_svd[1]),left_dims,left_dim,[0,1])
		elif OC=="right":
			# positioning the orthognality centre to the right
			new_right = UNMERGE(np.einsum("ij,jk->ik",block_merged_svd[1],block_merged_svd[2]),right_dims,right_dim,[1,2])
			new_left = UNMERGE(block_merged_svd[0],left_dims,left_dim,[0,1])
		elif OC== None:
			print("Please specify where is the orthogonality centre after operation with the keywords 'left' or 'right'")
	left_dims = None
	right_dims = None
	block_merged_svd = None
	return new_left,new_right

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

#########################
### Swapping timebins ###
#########################
def SWAP(states,from_,direction,M,L,tol):
	"""Moving the interacting past bin over the length of the whole delay towards future or past bins.
	INPUT: list of timebins, index of start, direction of SWAPs ("future" or "past") and the location of
	the orthogonality centre (optional)
	OUTPUT: list of states that changed"""
	
	if M==0:
		for i in range(L-1):
#			print(states[from_-i].shape,states[from_-i-1].shape)
			block = np.einsum("ij,k->ikj",states[from_-i],states[from_-1-i])
			merged,merge_dims = MERGE(block,[0,1])
			merged_svd,link_dim = SVD_sig(merged,tol)
			states[from_-i] = UNMERGE(merged_svd[0],np.array([merge_dims[0],merge_dims[1],link_dim]),merge_dims,[0,1])
			states[from_-i-1] = np.dot(merged_svd[1],merged_svd[2])
#			print(states[from_-i].shape,states[from_-i-1].shape)
#		print("0 shape",states[0].shape)

		return states[0:L]

	else:
		if direction=="past":
			for i in range(L-1):
				block = np.einsum("ijk,klm->iljm",states[from_-i],states[from_-1-i])
				left_merged,merge_left_dims = MERGE(block,[0,1])
				merged,merge_right_dims = MERGE(left_merged,[1,2])
				merged_svd,link_dim = SVD_sig(merged,tol)
				states[from_-i] = UNMERGE(merged_svd[0],np.array([merge_left_dims[0],merge_left_dims[1],link_dim]),
							  merge_left_dims,[0,1])
				states[from_-i-1] = UNMERGE(np.dot(merged_svd[1],merged_svd[2]),
							    np.array([link_dim,merge_right_dims[0],merge_right_dims[1]]),
							    merge_right_dims,[1,2])
			return states[(from_-L+1):(from_+1)]
			
		elif direction=="future":
#			print(states[from_-1].shape,from_)
			for i in range(L-1):
#				print(states[L+1].shape)
#				print(states[from_+i+1].shape,states[from_+i].shape)
				block = np.einsum("ijk,klm->iljm",states[from_+i+1],states[from_+i])
				left_merged,merge_left_dims = MERGE(block,[0,1])
				merged,merge_right_dims = MERGE(left_merged,[1,2])
				merged_svd,link_dim = SVD_sig(merged,tol)
				states[from_+i+1] = UNMERGE(np.dot(merged_svd[0],merged_svd[1]),
							    np.array([merge_left_dims[0],merge_left_dims[1],link_dim]),
							    merge_left_dims,[0,1])
				states[from_+i] = UNMERGE(merged_svd[2],np.array([link_dim,merge_right_dims[0],merge_right_dims[1]]),
							  merge_right_dims,[1,2])
#				print(states[from_+i+1].shape,states[from_+i].shape)
#				print(states[L+1].shape,from_,from_+L-2)
			return states[(from_):(from_+L)]

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

def coherentE(alpha,array):
    for i in range(len(array)):
        array[i] = np.exp(-np.abs(alpha)**2/2)*(alpha)**i/np.sqrt(float(factorial(i)))
    return array


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

def coherentS(alpha,array):
    for i in range(len(array)):
        array[i] = np.exp(-np.abs(alpha)**2/2)*(alpha)**i/np.sqrt(float(factorial(i)))
    return array


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

