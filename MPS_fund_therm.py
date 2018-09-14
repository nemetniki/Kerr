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

#####################################################################################################################################################################################################
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
	if d==1:
		link_dim     = 0
		svd_final[0] = svd_init[0][:,0]
		svd_final[1] = sing[0]
		svd_final[2] = svd_init[2][0,:]
	else:
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
def merge(block,NL,NR):
	"""Merging indices to provide a lower-rank tensor from a block
	INPUT: block for index-merging and the number of indices on the left and right hand side
	OUTPUT: block with the merged indices and the original dimensions on the left and right hand side"""

	#Dimensions associated with the left and right hand side of the tensor
	dL = block.shape[:NL]
	dR = block.shape[NL:]
	
	#Tensor initialization
	left_block = np.zeros(np.concatenate((np.array([np.prod(dL)]),dR)),dtype = np.complex128)
	merged_block = np.zeros((np.prod(dL),np.prod(dR)),dtype = np.complex128)

	#Index initialization
	i1 = np.arange(0,dL[0])
	j1 = np.arange(0,dR[0])
    

	#%%%%%%#
	# LEFT #
	#%%%%%%#
	if NL == 1: #No merging needed
		left_block += block

	elif NL == 2: 
		for i2 in range(0,dL[1]):
			idx_o = [slice(None)]*block.ndim
			idx_o[0] = i1
			idx_o[1] = i2
			idx_n = [slice(None)]*(block.ndim-1)
			idx_n[0] = i1+dL[0]*i2
			
			left_block[tuple(idx_n)] = block[tuple(idx_o)]

	elif NL == 3: 
		for i2 in range(0,dL[1]):
			for i3 in range(0,dL[2]):
				idx_o = [slice(None)]*block.ndim
				idx_o[0] = i1
				idx_o[1] = i2
				idx_o[2] = i3
				idx_n = [slice(None)]*(block.ndim-2)
				idx_n[0] = i1+dL[0]*i2+dL[1]*dL[0]*i3
				
				left_block[tuple(idx_n)] = block[tuple(idx_o)]

	elif NL == 4: 
		for i2 in range(0,dL[1]):
			for i3 in range(0,dL[2]):
				for i4 in range(0,dL[3]):
					idx_o = [slice(None)]*block.ndim
					idx_o[0] = i1
					idx_o[1] = i2
					idx_o[2] = i3
					idx_o[3] = i4
					idx_n = [slice(None)]*(block.ndim-3)
					idx_n[0] = i1+dL[0]*i2+dL[1]*dL[0]*i3+dL[2]*dL[1]*dL[0]*i4
					
					left_block[tuple(idx_n)] = block[tuple(idx_o)]

	elif NL == 5: 
		for i2 in range(0,dL[1]):
			for i3 in range(0,dL[2]):
				for i4 in range(0,dL[3]):
					for i5 in range(0,dL[4]):
						idx_o = [slice(None)]*block.ndim
						idx_o[0] = i1
						idx_o[1] = i2
						idx_o[2] = i3
						idx_o[3] = i4
						idx_o[4] = i5
						idx_n = [slice(None)]*(block.ndim-4)
						idx_n[0] = i1+dL[0]*i2+dL[1]*dL[0]*i3+dL[2]*dL[1]*dL[0]*i4+dL[3]*dL[2]*dL[1]*dL[0]*i5
						
						left_block[tuple(idx_n)] = block[tuple(idx_o)]

	#%%%%%%%#
	# RIGHT #
	#%%%%%%%#

	if NR == 1: #No merging needed
		merged_block += left_block

	elif NR == 2:
		for j2 in range(0,dR[1]):
			merged_block[:,j1+dR[0]*j2] = left_block[:,j1,j2]

	elif NR == 3:
		for j2 in range(0,dR[1]):
			for j3 in range(0,dR[2]):
				merged_block[:,j1+dR[0]*j2+dR[1]*dR[0]*j3] = left_block[:,j1,j2,j3]
	elif NR == 4:
		for j2 in range(0,dR[1]):
			for j3 in range(0,dR[2]):
				for j4 in range(0,dR[3]):
					merged_block[:,j1+dR[0]*j2+dR[1]*dR[0]*j3+dR[2]*dR[1]*dR[0]*j4] = left_block[:,j1,j2,j3,j4]

	elif NR == 5:
		for j2 in range(0,dR[1]):
			for j3 in range(0,dR[2]):
				for j4 in range(0,dR[3]):
					for j5 in range(0,dR[4]):
						merged_block[:,j1+dR[0]*j2+dR[1]*dR[0]*j3+dR[2]*dR[1]*dR[0]*j4+dR[3]*dR[2]*dR[1]*dR[0]*j5] = left_block[:,j1,j2,j3,j4,j5]

	#Clearing the original left_block
	left_block = None
	
	return merged_block, dL, dR

######################################################################################################################################################################################################
###############################
### Undoing the index-merge ###
###############################
def unmerge(block,dL,dR):
	"""Unmerging indices to provide a higher-rank tensor from block
	INPUT: the block for which the index-merging should be undone, the dimensions of the final block indices
	on the left and the right.
	OUTPUT: the obtained higher-rank tensor"""

	#Dimensions of the merged array
	DL = block.shape[1]
	DR = block.shape[2]
	
	#Number of unmerged dimensions on the left and the right
	NL = len(dL)
	NR = len(dR)
	
	#Initializing the left-merged and the unmerged tensor
	right_block = np.zeros(np.concatenate((np.array([DL]),dR)),dtype=np.complex128)
	unmerged_block = np.zeros(np.concatenate((dL,dR)),dtype=np.complex128)

	#Initializing the indices	
	j1 = np.arange(0,dR[0])
	i1 = np.arange(0,dL[0])

	#%%%%%%%#
	# RIGHT #
	#%%%%%%%#

	if NR == 1: #No unmerge needed
		right_block = block

	elif NR == 2:
		for J in range(0,DR):
			right_block[:,J%dR[0],int(J/dR[0])] = block[:,J]

	elif NR == 3:
		for J in range(0,DR):
			right_block[:,J%dR[0],int(J/dR[0])%dR[1],int(J/(dR[0]*dR[1]))] = block[:,J]
		
	elif NR == 4:
		for J in range(0,DR):
			right_block[:,J%dR[0],int(J/dR[0])%dR[1],int(J/(dR[0]*dR[1]))%dR[2],int(J/(dR[0]*dR[1]*dR[2]))] = block[:,J]
		
	elif NR == 5:
		for J in range(0,DR):
			right_block[:,J%dR[0],int(J/dR[0])%dR[1],int(J/(dR[0]*dR[1]))%dR[2],int(J/(dR[0]*dR[1]*dR[2]))%dR[3],int(J/(dR[0]*dR[1]*dR[2]*dR[3]))] = block[:,J]

	#%%%%%%#
	# LEFT #
	#%%%%%%#

	if NL == 1: #No unmerge needed
		unmerged_block = right_block

	elif NL == 2:
		for I in range(0,DL):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[0] = I
			idx_n = [slice(None)]*(right_block.ndim+1)
			idx_n[0] = I%dL[0]
			idx_n[1] = int(I/dL[0])
			
			unmerged_block[tuple(idx_n)] = right_block[tuple(idx_o)]

	elif NL == 3:
		for I in range(0,DL):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[0] = I
			idx_n = [slice(None)]*(right_block.ndim+2)
			idx_n[0] = I%dL[0]
			idx_n[1] = int(I/dL[0])%dL[1]
			idx_n[2] = int(I/(dL[0]*dL[1]))
			
			unmerged_block[tuple(idx_n)] = right_block[tuple(idx_o)]
		
	elif NL == 4:
		for I in range(0,DL):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[0] = I
			idx_n = [slice(None)]*(right_block.ndim+3)
			idx_n[0] = I%dL[0]
			idx_n[1] = int(I/dL[0])%dL[1]
			idx_n[2] = int(I/(dL[0]*dL[1]))%dL[2]
			idx_n[3] = int(I/(dL[0]*dL[1]*dL[2]))
			
			unmerged_block[tuple(idx_n)] = right_block[tuple(idx_o)]

	elif NL == 5:
		for I in range(0,DL):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[0] = I
			idx_n = [slice(None)]*(right_block.ndim+4)
			idx_n[0] = I%dL[0]
			idx_n[1] = int(I/dL[0])%dL[1]
			idx_n[2] = int(I/(dL[0]*dL[1]))%dL[2]
			idx_n[3] = int(I/(dL[0]*dL[1]*dL[2]))%dL[3]
			idx_n[4] = int(I/(dL[0]*dL[1]*dL[2]*dL[3]))
			
			unmerged_block[tuple(idx_n)] = right_block[tuple(idx_o)]
	
	right_block = None

	return unmerged_block

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

	# Number of indices for each state besides the link index between them
	left_ind = len(state_left.shape)-1
	right_ind = len(state_right.shape)-1
	# Number of indices left after contraction
	num_ind = left_ind+right_ind

	if left_ind==0 or right_ind==0:
		new_left = state_left
		new_right = state_right
	elif num_ind==2:
		if state_left.shape[1]>state_left.shape[0]:
			new_left = state_left
			new_right = state_right
		else:
			# Tensor after the contraction at the link index between the states
			Combo = np.tensordot(state_left,state_right,1)
			Combo_svd,link_dim = SVD_sig(Combo,tolerance)
			if where=="left":
				new_left = np.dot(Combo_svd[0],Combo_svd[1])
				new_right = Combo_svd[2]
			elif where=="right":
				new_right = np.dot(Combo_svd[1],Combo_svd[2])
				new_left = Combo_svd[0]
			Combo_svd = None
        
	elif num_ind==3:
		# Tensor after the contraction at the link index between the states
		Combo = np.tensordot(state_left,state_right,1)
		if left_ind ==2: # link index is located on the left
			# Merging the indices on the left to be able to perform svd
			Combo_merged,merge_dims = merge(Combo,"left")
			# number of singular values that matter is in link_dim
			Combo_merged_svd,link_dim = SVD_sig(Combo_merged,tolerance)
			Combo_merged=None
			# Dimensions of the indices after the OC relocation and unmerge of indices
			dims = np.concatenate((merge_dims,np.array([link_dim])),axis=0)
			if where=="left":
				# OC relocation to the left
				new_left_merged = np.dot(Combo_merged_svd[0],Combo_merged_svd[1])
				new_left = unmerge(new_left_merged,dims,"left")
				new_left_merged=None
				new_right = Combo_merged_svd[2]
				Combo_merged_svd=None
			elif where=="right":
				# OC relocation to the right
				new_right = np.dot(Combo_merged_svd[1],Combo_merged_svd[2])
				new_left = unmerge(Combo_merged_svd[0],dims,"left")
				Combo_merged_svd=None
		elif left_ind ==1: # link index is located on the right
			# Merging the indices on the right to be able to perform svd
			Combo_merged,merge_dims = merge(Combo,"right")
			# number of singular values that matter is in link_dim
			Combo_merged_svd, link_dim = SVD_sig(Combo_merged,tolerance)
			Combo_merged=None
			# Dimensions of the indices after the OC relocation and unmerge of indices
			dims = np.concatenate((np.array([link_dim]),merge_dims),axis=0)
			if where=="left":
				# OC relocation to the left
				new_left = np.dot(Combo_merged_svd[0],Combo_merged_svd[1])
				new_right = unmerge(Combo_merged_svd[2],dims,"right")
				Combo_merged_svd=None
			elif where=="right":
				# OC relocation to the right
				new_right_merged = np.dot(Combo_merged_svd[1],Combo_merged_svd[2])
				new_right = unmerge(new_right_merged,dims,"right")
				new_right_merged=None
				new_left = Combo_merged_svd[0]            
				Combo_merged_svd=None
                
	elif num_ind==4:
		# Tensor after the contraction at the link index between the states
		Combo = np.tensordot(state_left,state_right,1)
		# Merging the indices on both sides to be able to perform svd
		Combo_merged,merge_dims = merge(Combo,"both")
		# number of singular values that matter is in link_dim
		Combo_merged_svd,link_dim = SVD_sig(Combo_merged,tolerance)
		Combo_merged=None
		# Dimensions of the indices on the left after the OC relocation and unmerge of indices
		dims_left = np.array([merge_dims[0],merge_dims[1],link_dim])
		# Dimensions of the indices on the right after the OC relocation and unmerge of indices
		dims_right = np.array([link_dim,merge_dims[2],merge_dims[3]])
		if where=="left":
			# OC relocation to the left
			new_left_merged = np.dot(Combo_merged_svd[0],Combo_merged_svd[1])
			# unmerging the indices on the left
			new_left = unmerge(new_left_merged,dims_left,"left")
			new_left_merged=None
			# unmerging the indices on the right
			new_right = unmerge(Combo_merged_svd[2],dims_right,"right")
		elif where=="right":
			# OC relocation to the right
			new_right_merged = np.dot(Combo_merged_svd[1],Combo_merged_svd[2])
			# unmerging the indices on the right
			new_right = unmerge(new_right_merged,dims_right,"right")
			new_right_merged=None
			# unmerging the indices on the left
			new_left = unmerge(Combo_merged_svd[0],dims_left,"left")
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

	# merging the link indices into the physical indices
	block_merged, dims = merge(block,how)
	# separate the block
	block_merged_svd,link_dim = SVD_sig(block_merged,tolerance)
	#print(block.shape, block_merged.shape)
	block_merged=None
	# specifying the link dimension
	if how=="left":
		# specifying the final indices of the left block
		if link_dim==0: # for separate bins
			left_dims = dims
		else:
			left_dims = np.concatenate((dims,np.array([link_dim])),axis=0)
		# unmerging the link index
		new_left = unmerge(block_merged_svd[0],left_dims,"left")
		if len(block_merged_svd[2].shape)>1:
			# position the orthogonality centre to the right
			new_right = np.einsum("ij,jk->ik",block_merged_svd[1],block_merged_svd[2])
		elif len(block_merged_svd[2].shape)==1 & len(block_merged_svd[0].shape)==1:  # for separate bins
			# position the orthogonality centre to the right
			new_right = block_merged_svd[1]*block_merged_svd[2]
		else:
			print("something's wrong with the ranks of the tensors after svd in cut left")
            
	elif how=="right":
		# specifying the final indices of the right block
		if link_dim==0: # for separate bins
			right_dims = dims
		else:
			right_dims = np.concatenate((np.array([link_dim]),dims),axis=0)
		new_left = block_merged_svd[0]
		if len(block_merged_svd[2].shape)>1:
			# position the orthogonality centre to the right and unmerging the link index
			new_right = unmerge(np.einsum("ij,jk->ik",block_merged_svd[1],block_merged_svd[2]),right_dims,"right")
		elif len(block_merged_svd[2].shape)==1 & len(block_merged_svd[0].shape)==1: # for separate bins
			# position the orthogonality centre to the right and unmerging the link index
			new_right = unmerge(block_merged_svd[1]*block_merged_svd[2],right_dims,"right")
		else:
			print("something's wrong with the ranks of the tensors after svd in cut right")

	elif how=="both":
		if link_dim==0: # for separate bins
			left_dims = dims[:2]
			right_dims = dims[2:]
			new_right = unmerge(block_merged_svd[2],right_dims,"right")
			new_left = unmerge(block_merged_svd[0],left_dims,"left")
		# specifying the final indices of the blocks
		else: # for separate bins
			left_dims = np.concatenate((dims[:2],np.array([link_dim])),axis=0)
			right_dims = np.concatenate((np.array([link_dim]),dims[2:]),axis=0)
			if OC=="left":
				# positioning the orthognality centre to the left
				new_right = unmerge(block_merged_svd[2],right_dims,"right")
				new_left = unmerge(np.einsum("jk,kl->jl",block_merged_svd[0],block_merged_svd[1]),left_dims,"left")
			elif OC=="right":
				# positioning the orthognality centre to the right
				new_right = unmerge(np.einsum("ij,jk->ik",block_merged_svd[1],block_merged_svd[2]),right_dims,"right")
				new_left = unmerge(block_merged_svd[0],left_dims,"left")
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
def SWAP(states,base_ind,direction,L,tol):
	"""Moving the interacting past bin over the length of the whole delay towards future or past bins.
	INPUT: list of timebins, index of start, direction of SWAPs ("future" or "past") and the location of
	the orthogonality centre (optional)
	OUTPUT: list of states that changed"""

	# Determining the direction of SWAPs
	if direction=="future":
		c = 1
	elif direction=="past":
		c = -1
        
	for s in range(0,L-1):
		leg_r = len(states[base_ind+c*s].shape)
		leg_l = len(states[base_ind+c*s+1].shape)
#		print("before: ",len(states[base_ind+c*s+1].shape),states[base_ind+c*s+1].shape,len(states[base_ind+c*s].shape),states[base_ind+c*s].shape)
		# if the past bin is independent, tensor product should be performed
		if leg_r==1:
			# if both bins are independent, swap does not need SVD, just a "shelf"
			if leg_l==1:
				right = states[base_ind+1+c*s]
				states[base_ind+c*s+1] = states[base_ind+c*s]
				states[base_ind+c*s] = right
				right=None
			# otherwise the cut function can be used to SWAP the bins
			elif leg_l==2:
				swap_block = np.tensordot(states[base_ind+c*s+1],states[base_ind+c*s],0)
				# the physical indices should be swapped
				states[base_ind+c*s+1],states[base_ind+c*s] = cut(np.einsum("ijk->ikj",swap_block),tol,"left")
				# the cut function puts the OC to the right by default. In order to move the interacting past
				# bin next to the system bin, the orthogonality centre should also shift towards the left (future).
			else:
				print("problem with tensor rank in swap")
			if direction =="future":
#				if leg_l>1 and leg_r>1:
				states[base_ind+c*s+1],states[base_ind+c*s] = OC_reloc(states[base_ind+c*s+1],
																		   states[base_ind+c*s],"left",tol)
                
		elif leg_r==2 :            
			if leg_l==1:
				swap_block = np.tensordot(states[base_ind+1+c*s],states[base_ind+c*s],0)
				states[base_ind+1+c*s],states[base_ind+c*s] = cut(np.einsum("ijk->jik",swap_block),tol,"right")     
				if direction =="future":
	#				if leg_l>1 and leg_r>1:
					states[base_ind+c*s+1],states[base_ind+c*s] = OC_reloc(states[base_ind+c*s+1],
																		   states[base_ind+c*s],"left",tol)
			elif leg_l==2:
				if states[base_ind+1+c*s].shape[0]>states[base_ind+c*s+1].shape[1]:
					swap_block = np.tensordot(states[base_ind+1+c*s],states[base_ind+c*s],1)
					svd_block = svd_sig(swap_block,tol)
					if direction=="future":
						states[base_ind+1+c*s] = np.einsum("ij,jk",svd_block[0],svd_block[1])
						states[base_ind+c*s] = svd_block[2]
					elif direction=="past":
						states[base_ind+c*s] = np.einsum("ij,jk",svd_block[1],svd_block[2])
						states[base_ind+1+c*s] = svd_block[0]
				else:
					swap_block = np.einsum("ij,kl->ikjl",states[base_ind+1+c*s],states[base_ind+c*s])
					if direction=="future":
						states[base_ind+1+c*s],states[base_ind+c*s] = cut(swap_block,tol,"both","left")     
					elif direction=="past":
						states[base_ind+1+c*s],states[base_ind+c*s] = cut(swap_block,tol,"both","right")     
			elif leg_l==3:
			# the physical indices should be swapped while contracting the link indices
				swap_block = np.einsum("ijk,kl->ilj",states[base_ind+1+c*s],states[base_ind+c*s])
				states[base_ind+1+c*s],states[base_ind+c*s] = cut(swap_block,tol,"left")     
			# the cut function puts the OC to the right by default. In order to move the interacting past
			# bin next to the system bin, the orthogonality centre should also shift towards the left (future).
				if direction =="future":
	#				if leg_l>1 and leg_r>1:
					states[base_ind+c*s+1],states[base_ind+c*s] = OC_reloc(states[base_ind+c*s+1],
																		   states[base_ind+c*s],"left",tol)
#				print("after: ",states[base_ind+1+c*s].shape,states[base_ind+c*s].shape)
		else:            
			# the physical indices should be swapped while contracting the link indices
			if leg_l==2:
				swap_block = np.einsum("ij,jlk->lik",states[base_ind+1+c*s],states[base_ind+c*s])
				states[base_ind+1+c*s],states[base_ind+c*s] = cut(swap_block,tol,"right")     
			# the physical indices should be swapped while contracting the link indices
				if direction =="future":
	#				if leg_l>1 and leg_r>1:
					states[base_ind+c*s+1],states[base_ind+c*s] = OC_reloc(states[base_ind+c*s+1],
																		   states[base_ind+c*s],"left",tol)
			else:
				swap_block = np.einsum("ijk,klm->iljm",states[base_ind+1+c*s],states[base_ind+c*s])
				if direction=="future":
					states[base_ind+1+c*s],states[base_ind+c*s] = cut(swap_block,tol,"both","left")     
				elif direction=="past":
					states[base_ind+1+c*s],states[base_ind+c*s] = cut(swap_block,tol,"both","right")     
	# because of the different directions of operations the position of the output states are either to the left
	# or to the right from the index of start
	if direction =="future":
		return states[base_ind:(base_ind+L)]
	elif direction =="past":
		return states[(base_ind-L+2):(base_ind+2)]

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

def coherent(n,phi,array):
    for i in range(len(array)):
        array[i] = np.exp(-n/2)*(np.sqrt(n)*np.exp(phi))**i/np.sqrt(float(factorial(i)))
    return array


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

