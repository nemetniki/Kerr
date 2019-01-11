import numpy as np
import scipy as sc
from numpy.linalg import svd
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
def SVD(block,cutoff):
	"""Performing SVD with singular values above a certain threshold
	INPUT: the block to perform SVD on, threshold for singular values
	OUTPUT: SVD elements and the number of significant singular values"""

	# Predefining the list for the output
	svd_final = [0]*3
	block = np.nan_to_num(block,copy=False)
	#print(block.shape)
	# Performing the SVD
	try:
		svd_init  = svd(block,full_matrices=False)
	except:
		np.savetxt("failed_array.txt",block)
		raise
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
		new_block = np.zeros(np.concatenate((orig_shape[:where[0]],np.array([np.prod(merge_dims),]),orig_shape[where[-1]+1:]),axis=0))
	elif where[0]>0 and where[-1]==len(orig_shape-1):
		new_block = np.zeros(np.concatenate((orig_shape[:where[0]],np.array([np.prod(merge_dims),])),axis=0))
	elif where[0]==0 and where[-1]<len(orig_shape-1):
		new_block = np.zeros(np.concatenate((np.array([np.prod(merge_dims),]),orig_shape[where[-1]+1:]),axis=0))
	else:
		print("Problem with merging location")
		
	i1 = np.arange(0,merge_dims[-1]])
	idx_o = [slice(None)]*what.ndim
	idx_o[where[-1]]] = i1
	idx_n = [slice(None)]*(what.ndim-(number-1))
	
	if number == 2:
		for i2 in range(0,merge_dims[0]):
			idx_o[where[0]] = i2
			idx_n[where[0]] = i1 + merge_dims[1]*i2
			
			new_block[tuple(idx_n)] = block[tuple(idx_o)]
			
	elif number == 3:
		for i3 in range(0,merge_dims[0]):
			for i2 in range(0,merge_dims[1]):
				idx_o[where[1]] = i2
				idx_o[where[0]] = i3
				idx_n[where[0]] = i1 + merge_dims[2]*i2 + merge_dims[1]*merge_dims[2]*i3
				
				new_block[tuple(idx_n)] = block[tuple(idx_o)]

	elif number == 4:
		for i4 in range(0,merge_dims[0]):
			for i3 in range(0,merge_dims[1]):
				for i2 in range(0,merge_dims[2]):
					idx_o[where[2]] = i2
					idx_o[where[1]] = i3
					idx_o[where[0]] = i4
					idx_n[where[0]] = (i1 + merge_dims[3]*i2 + merge_dims[2]*merge_dims[3]*i3 + 
									   merge_dims[1]*merge_dims[2]*merge_dims[3]*i4
				
					new_block[tuple(idx_n)] = block[tuple(idx_o)]

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
										   merge_dims[1]*merge_dims[2]*merge_dims[3]*merge_dims[4]*i5
					
						new_block[tuple(idx_n)] = block[tuple(idx_o)]

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

	new_block  = np.zeros(new_shape)
	merged     = block.shape[where[0]]
	number = len(where)
	idx_o = [slice(None)]*block.ndim
	idx_n = [slice(None)]*(len(orig_shape))
		
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

######################################################################################################################################################################################################
#############################
### Swapping to perform U ###
#############################
#S2F2,statesF = SWAP(S2F2,statesF,test_var+L-1,test_var+1)
def SWAP(what,statesF,from_,to_):
	"""Moving the interacting fibre and system bins next to each other.
	INPUT: time, delay, list of fibre bins, tolerance of SVD
	OUTPUT: list of fibre states"""

	length = to_-from_+1
	context = ["aI,J->aJI","aIb,bJc->aJIc","bJc,aIb->aJIc"]
	
	if from_>to_:
		start = from_
		end   = to_-1
		step  = -1
	elif from_<to_:
		start = from_
		end   = to_+1
		step  = 1
	else:

	def swapping(what,with_):
		what_size = what.ndim
		with_size = with_.ndim
		if what_size!=1:
			combined = es(context[what.ndim-int((step+3)/2)],what,with_)
			if what_size>2:
				left,left_merge  = MERGE(combined,[0,1])
				right,right_merge = MERGE(left,[2,3])
				svded,link_dim   = SVD(right,tol)
				right,left       = None,None
				right_merge      = None
				what             = UNMERGE(np.dot(svded[1],svded[2]),np.concatenate((np.array([link_dim,]),right_merge),axis=0),
											right_merge,[1,2])
				
			if what_size==2:
				merged,left_merge = MERGE(combined,[0,1])
				svded,link_dim    = SVD(merged,tol)
				merged            = None
				what              = np.dot(svded[1],svded[2])
								
		elif what_size==1 and with_size==1:
			return what,with_
		else:
			print("Problem with size of arrays to swap")
			
		with_ = UNMERGE(svded[0],np.concatenate((left_merge,np.array([link_dim,])),axis=0),left_merge,[0,1])

		left_merge,svded = None
		
		return what,with_
	
	
	for i in range(start,end,step):
		#upper branch
		what,statesF[start] = swapping(what,statesF[start])

	return what,statesF

#####################################################################################################################################################################################################
###################################
### SVD creating 2 link-indices ###
###################################
##splist = SPLIT(S2F2,2,"left",2)
##statesS[1],statesF[test_var+L] = splist[0],splist[1]
def SPLIT(what,number,where,link,blocksize=0):
	"""Performing SVD with singular values above a certain threshold
	INPUT: the block to split after U by performing SVD, threshold for singular values
	OUTPUT: block 1 and block 2"""
	splist = [None]*number
	
	if blocksize>0:
		if link==1 or link==2:
			first,block_merge = MERGE(what,list(what.ndim-np.arange(blocksize+1,1,-1)))
		else:
			first,block_merge = MERGE(what,list(what.ndim-np.arange(blocksize,0,-1)))
	elif blocksize<0:
		if link==-1 or link==2:
			first,block_merge = MERGE(what,list(np.arange(1,-blocksize,1)))
		else:
			first,block_merge = MERGE(what,list(np.arange(-blocksize)))
	else:
		first = np.zeros(what.shape)
		first += what

	if link==0:
		first = np.tensordot(np.ones(1),np.tensordot(first,np.ones(1),axes=0),axes=0)
	elif link==1:
		first = np.tensordot(np.ones(1),first,axes=0)
	elif link==-1:
		first = np.tensordot(first,np.ones(1),axes=0)

	if where == "right":
			
		for i in range(number-1):
			second,second_merge = MERGE(first,[0,1])
			third,third_merge   = MERGE(second,list(np.arange(1,second.ndim,1))) 
			svded,link_dim      = SVD(third,tol)
			splist[i]           = UNMERGE(svded[0],np.concatenate((second_merge,np.array([link_dim,])),axis=0),second_merge,[0,1])
			first              = UNMERGE(np.dot(svded[1],svded[2]),
											np.concatenate((np.array([link_dim,]),third_merge),axis=0), third_merge,
											list(np.arange(1,third.ndim,1)))
		splist[-1] = second

	elif where == "left":
			
		for i in range(number-1,0,-1):
			second,second_merge = MERGE(first,[first.ndim-2,second.ndim-1])
			third,third_merge   = MERGE(second,list(np.arange(0,second.ndim-1,1))) 
			svded,link_dim      = SVD(third,tol)
			splist[i]           = UNMERGE(svded[2],np.concatenate(np.array([link_dim,]),second_merge),
										second_merge,[second.ndim-2,second.ndim-1])
			second 	            = UNMERGE(np.dot(svded[0],svded[1]),
										np.concatenate((third_merge,np.array([link_dim,])),axis=0),
										third_mege,list(np.arange(0,third.ndim-1,1)))
		splist[0] = second

	if link==1 or link==0:
		splist[0]  = splist[0][0,:,:]
	if link==-1 or link==0:
		splist[-1] = splist[-1][:,:,0]
	
	return splist


#####################################################################################################################################################################################################
###################################
### SVD creating 2 link-indices ###
###################################
##statesB1[M],statesF,block = OC_RELOC(statesB1[M],block,statesF,0,test_var)

def OC_RELOC(from_op,to_op,statesF,from_,to_):
	if from_<to_:
		if from_op.ndim==2:
			first               = es("Ia,aJb->IJb",from_op,statesF[from_])
			second,second_merge = MERGE(first,[1,2])
			svded,link_dim      = SVD(second,tol)
			from_op             = svded[0]
			statesF[from_]      = UNMERGE(np.dot(svded[1],svded[2]),
										np.concatenate((np.array([link_dim,]),second_merge),axis=0),second_merge,[1,2])
		elif from_op.ndim==3:
			first               = es("cIa,aJb->cIJb",from_op,statesF[from_])
			second,second_merge = MERGE(first,[2,3])
			third,third_merge   = MERGE(second,[0,1])
			svded,link_dim      = SVD(second,tol)
			from_op             = UNMERGE(svded[0],np.concatenate((third_merge,np.array([link_dim,])),axis=0),third_merge,[0,1])
			statesF[from_]      = UNMERGE(np.dot(svded[1],svded[2]),
										np.concatenate((np.array([link_dim,]),first_shape[2:]),axis=0),second_merge,[2,3])
		
		for i in range(from_,to_,1):
			first               = es("aIb,bJc->aIJc",statesF[i],statesF[i+1])
			second,second_merge = MERGE(first,[2,3])
			third,third_merge   = MERGE(second,[0,1])
			svded,link_dim      = SVD(second,tol)
			statesF[i]          = UNMERGE(svded[0],np.concatenate((third_merge,np.array([link_dim,])),axis=0),third_merge,[0,1])
			statesF[i+1]        = UNMERGE(np.dot(svded[1],svded[2]),
										np.concatenate((np.array([link_dim,]),second_merge),axis=0),second_merge,[2,3])
		
		first               = es("aIb,bJc->aIJc",statesF[to_],to_op)
		second,second_merge = MERGE(first,[2,3])
		third,third_merge   = MERGE(second,[0,1])
		svded,link_dim      = SVD(second,tol)
		statesF[to_]        = UNMERGE(svded[0],np.concatenate((third_merge,np.array([link_dim,])),axis=0),
									third_merge,[0,1])
		to_op               = UNMERGE(np.dot(svded[1],svded[2]),
									np.concatenate((np.array([link_dim,]),second_merge),axis=0),second_merge,[2,3])
		
	elif from_>to_:
		first               = es("cIa,aJb->cIJb",statesF[from_],from_op)
		second,second_merge = MERGE(first,[2,3])
		third,third_merge   = MERGE(second,[0,1])
		svded,link_dim      = SVD(second,tol)
		statesF[from_]      = UNMERGE(np.dot(svded[0],svded[1]),np.concatenate((third_merge,np.array([link_dim,])),axis=0),third_merge,[0,1])
		from_op             = UNMERGE(svded[2],np.concatenate((np.array([link_dim,]),second_merge),axis=0),second_merge,[2,3])
		
		for i in range(from_,to_,-1):
			first               = es("aIb,bJc->aIJc",statesF[i-1],statesF[i])
			second,second_merge = MERGE(first,[2,3])
			third,third_merge   = MERGE(second,[0,1])
			svded,link_dim      = SVD(second,tol)
			statesF[i-1]        = UNMERGE(np.dot(svded[0],svded[1]),np.concatenate((third_merge,np.array([link_dim,])),axis=0),third_merge,[0,1])
			statesF[i]          = UNMERGE(svded[2],np.concatenate((np.array([link_dim,]),second_merge),axis=0),second_merge,[2,3])
		
		first               = es("aIb,bJc->aIJc",to_op,statesF[to_])
		second,second_merge = MERGE(first,[2,3])
		third,third_merge   = MERGE(second,[0,1])
		svded,link_dim      = SVD(second,tol)
		to_op               = UNMERGE(np.dot(svded[0],svded[1]),np.concatenate((third_merge,np.array([link_dim,])),axis=0),third_merge,[0,1])
		statesF[to_]        = UNMERGE(svded[2],np.concatenate((np.array([link_dim,]),second_merge),axis=0),second_merge,[2,3])
	
	return from_op,statesF,to_op
	
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

