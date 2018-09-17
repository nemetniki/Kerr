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

#####################################################################################################################################################################################################
####################################
### SVD splitting with 2 indices ###
####################################
def SVD_split(block,cutoff):
	"""Performing SVD with singular values above a certain threshold
	INPUT: the block to split after U by performing SVD, threshold for singular values
	OUTPUT: block 1 and block 2"""

	# Predefining the list for the output
	svd_final = [0]*3

	# Performing the SVD
	svd_init  = svd(block,full_matrices=False)

	# Omitting the insignificant singular values
	d   = 0
	eps = 100.
	while eps>cutoff:
		d = d+1
		eps = np.sqrt(np.sum(svd_init[1][d:]**2))

	# Dimension of the two inner indices
	dim = int(np.sqrt(d))+1
	
	# Final matrices
	svd_final = [np.zeros((svd_init[0].shape[0],dim,dim)),np.zeros((dim,dim)),np.zeros((dim,dim,svd_init[2].shape[1]))]
	for J in range(0,dim**2):
		svd_final[0][:,j%dim,int(j/dim)] = svd_init[0][:,J]
		svd_final[1][j%dim,int(j/dim)]   = svd_init[1][J]
		svd_final[2][j%dim,int(j/dim),:] = svd_init[2][J,:]
	
	# Calculating block 1 and block 2 after U and the SVD
	block_1 = np.einsum("ijk,jk->ijk",svd_final[0],svd_final[1])
	block_2 = svd_final[2]
	
	# Clear unnecessary variables
	svd_init  = None
	svd_final = None

	return block_1,block_2

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
	i1 = np.arange(0,dL[-1])
	j1 = np.arange(0,dR[-1])
    

	#%%%%%%#
	# LEFT #
	#%%%%%%#
	if NL == 1: #No merging needed
		left_block += block

	elif NL == 2: 
		for i2 in range(0,dL[0]):
			idx_o = [slice(None)]*block.ndim
			idx_o[0] = i1
			idx_o[1] = i2
			idx_n = [slice(None)]*(block.ndim-1)
			idx_n[0] = i1+dL[1]*i2
			
			left_block[tuple(idx_n)] = block[tuple(idx_o)]

	elif NL == 3: 
		for i3 in range(0,dL[0]):
			for i2 in range(0,dL[1]):
				idx_o = [slice(None)]*block.ndim
				idx_o[2] = i1
				idx_o[1] = i2
				idx_o[0] = i3
				idx_n = [slice(None)]*(block.ndim-2)
				idx_n[0] = i1+dL[2]*i2+dL[1]*dL[2]*i3
				
				left_block[tuple(idx_n)] = block[tuple(idx_o)]

	elif NL == 4: 
		for i4 in range(0,dL[0]):
			for i3 in range(0,dL[1]):
				for i2 in range(0,dL[2]):
					idx_o = [slice(None)]*block.ndim
					idx_o[3] = i1
					idx_o[2] = i2
					idx_o[1] = i3
					idx_o[0] = i4
					idx_n = [slice(None)]*(block.ndim-3)
					idx_n[0] = i1+dL[3]*i2+dL[2]*dL[3]*i3+dL[1]*dL[2]*dL[3]*i4
					
					left_block[tuple(idx_n)] = block[tuple(idx_o)]

	elif NL == 5: 
		for i5 in range(0,dL[0]):
			for i4 in range(0,dL[1]):
				for i3 in range(0,dL[2]):
					for i2 in range(0,dL[3]):
						idx_o = [slice(None)]*block.ndim
						idx_o[4] = i1
						idx_o[3] = i2
						idx_o[2] = i3
						idx_o[1] = i4
						idx_o[0] = i5
						idx_n = [slice(None)]*(block.ndim-4)
						idx_n[0] = i1+dL[4]*i2+dL[3]*dL[4]*i3+dL[2]*dL[3]*dL[4]*i4+dL[1]*dL[2]*dL[3]*dL[4]*i5
						
						left_block[tuple(idx_n)] = block[tuple(idx_o)]

	#%%%%%%%#
	# RIGHT #
	#%%%%%%%#

	if NR == 1: #No merging needed
		merged_block += left_block

	elif NR == 2:
		for j2 in range(0,dR[0]):
			merged_block[:,j1+dR[1]*j2] = left_block[:,j2,j1]

	elif NR == 3:
		for j3 in range(0,dR[0]):
			for j2 in range(0,dR[1]):
				merged_block[:,j1+dR[2]*j2+dR[1]*dR[2]*j3] = left_block[:,j3,j2,j1]
	elif NR == 4:
		for j4 in range(0,dR[0]):
			for j3 in range(0,dR[1]):
				for j2 in range(0,dR[2]):
					merged_block[:,j1+dR[3]*j2+dR[2]*dR[3]*j3+dR[1]*dR[2]*dR[3]*j4] = left_block[:,j4,j3,j2,j1]

	elif NR == 5:
		for j5 in range(0,dR[0]):
			for j4 in range(0,dR[1]):
				for j3 in range(0,dR[2]):
					for j2 in range(0,dR[3]):
						merged_block[:,j1+dR[4]*j2+dR[3]*dR[4]*j3+dR[2]*dR[3]*dR[4]*j4+dR[1]*dR[2]*dR[3]*dR[4]*j5] = left_block[:,j5,j4,j3,j2,j1]

	#Clearing the original left_block
	left_block = None
	
	return merged_block, dL, dR

######################################################################################################################################################################################################
###############################
### Undoing the index-merge ###
###############################
def unmerge(block,dL,dR,which=0):
	"""Unmerging indices to provide a higher-rank tensor from block
	INPUT: the block for which the index-merging should be undone, the dimensions of the final block indices
	on the left and the right.
	OUTPUT: the obtained higher-rank tensor"""

	#Number of unmerged dimensions on the left and the right
	NL = len(dL)
	NR = len(dR)
	
	spec = int(np.any(which))
	#Dimensions of the merged array
	DL = block.shape[:which]*spec+block.shape[0]*(1-spec)*np.array([1])
	DR = block.shape[-3+which:]*spec+block.shape[1]*(1-spec)*np.array([1])
	
	dL = dL*int(np.any(which-2))+DL*(1-int(np.any(which-2))) #dL unless unmerge on the right
	dR = dR*int(np.any(which-1))+DR*(1-int(np.any(which-1))) #dR unless unmerge on the left

	#Initializing the left-merged and the unmerged tensor
	right_block = np.zeros(np.concatenate((DL,dR)),dtype=np.complex128)
	unmerged_block = np.zeros(np.concatenate((dL,dR)),dtype=np.complex128)
		
	#%%%%%%%#
	# RIGHT #
	#%%%%%%%#

	if NR == 1: #No unmerge needed
		right_block = block

	elif NR == 2:
		for J in np.arange(0,DR):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[-1] = J
			idx_n = [slice(None)]*(right_block.ndim+1)
			idx_n[-1] = J%dR[1]
			idx_n[-2] = int(J/dR[1])

			right_block[idx_n] = block[idx_o]

	elif NR == 3:
		for J in np.arange(0,DR):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[-1] = J
			idx_n = [slice(None)]*(right_block.ndim+1)
			idx_n[-1] = J%dR[2]
			idx_n[-2] = int(J/dR[2])%dR[1]
			idx_n[-3] = int(J/(dR[2]*dR[1]))

			right_block[idx_n] = block[idx_o]
		
	elif NR == 4:
		for J in np.arange(0,DR):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[-1] = J
			idx_n = [slice(None)]*(right_block.ndim+1)
			idx_n[-1] = J%dR[3]
			idx_n[-2] = int(J/dR[3])%dR[2]
			idx_n[-3] = int(J/(dR[3]*dR[2]))%dR[1]
			idx_n[-4] = int(J/(dR[3]*dR[2]*dR[1]))

			right_block[idx_n] = block[idx_o]
		
	elif NR == 5:
		for J in np.arange(0,DR):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[-1] = J
			idx_n = [slice(None)]*(right_block.ndim+1)
			idx_n[-1] = J%dR[4]
			idx_n[-2] = int(J/dR[4])%dR[3]
			idx_n[-3] = int(J/(dR[4]*dR[3]))%dR[2]
			idx_n[-4] = int(J/(dR[4]*dR[3]*dR[2]))%dR[1]
			idx_n[-5] = int(J/(dR[4]*dR[3]*dR[2]*dR[1]))

			right_block[idx_n] = block[idx_o]

	#%%%%%%#
	# LEFT #
	#%%%%%%#

	if NL == 1: #No unmerge needed
		unmerged_block = right_block

	elif NL == 2:
		for I in np.arange(0,DL):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[0] = I
			idx_n = [slice(None)]*(right_block.ndim+1)
			idx_n[1] = I%dL[1]
			idx_n[0] = int(I/dL[1])
			
			unmerged_block[tuple(idx_n)] = right_block[tuple(idx_o)]

	elif NL == 3:
		for I in np.arange(0,DL):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[0] = I
			idx_n = [slice(None)]*(right_block.ndim+2)
			idx_n[2] = I%dL[2]
			idx_n[1] = int(I/dL[2])%dL[1]
			idx_n[0] = int(I/(dL[2]*dL[1]))
			
			unmerged_block[tuple(idx_n)] = right_block[tuple(idx_o)]
		
	elif NL == 4:
		for I in np.arange(0,DL):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[0] = I
			idx_n = [slice(None)]*(right_block.ndim+3)
			idx_n[3] = I%dL[3]
			idx_n[2] = int(I/dL[3])%dL[2]
			idx_n[1] = int(I/(dL[3]*dL[2]))%dL[1]
			idx_n[0] = int(I/(dL[3]*dL[2]*dL[1]))
			
			unmerged_block[tuple(idx_n)] = right_block[tuple(idx_o)]

	elif NL == 5:
		for I in np.arange(0,DL):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[0] = I
			idx_n = [slice(None)]*(right_block.ndim+4)
			idx_n[4] = I%dL[4]
			idx_n[3] = int(I/dL[4])%dL[3]
			idx_n[2] = int(I/(dL[4]*dL[3]))%dL[2]
			idx_n[1] = int(I/(dL[4]*dL[3]*dL[2]))%dL[1]
			idx_n[0] = int(I/(dL[4]*dL[3]*dL[2]*dL[1]))
			
			unmerged_block[tuple(idx_n)] = right_block[tuple(idx_o)]
	
	right_block = None

	return unmerged_block

######################################################################################################################################################################################################
############################
### Splitting the tensor ###
############################
def split(M,block,which,tol):
	context = ["IJKab->IbJaK","abLMN->LaMbN","gIeJKab->IbJgaKe","abhLfMN->LaMhbNf"]
	spec    = int(np.any(M))

	#%%%%%%%%%%%%%%%%#
	# INDEX ORDERING #
	#%%%%%%%%%%%%%%%%#
	block = np.einsum(context[spec*2+which-1],block)
	
	#%%%%%%%%%%%%%%%%%%%%%%#
	# SPLITTING F FROM B&S #
	#%%%%%%%%%%%%%%%%%%%%%%#
	# Merging the indices in block to F+SB
	merged_FSB,dimF,dimSB = merge(block,2,3+2*spec) 
	#indices -> AB (A=Ib/La, B=JaK/MbN or B=JgaKe/MhbNf)		
	# Performing the SVD to split the F from SB
	svd_FSB    = svd_sig(merged_FSB,tol) #indices->Ac/Ad,cc/dd,cB/dB
	merged_FSB = None
	# Reordering the indices of F
	F = np.einsum("Ibc->cIb",unmerge(svd_FSB[0],dimF,np.array([1]))) #indices->Ibc/Lad->cIb/dLa
	
	#%%%%%%%%%%%%%%%%%%%%%#
	# SEPARATING B FROM S #
	#%%%%%%%%%%%%%%%%%%%%%#
	# Keeping the orthognality centre in S. Unmerging the indices of S from B, but merging the indices of S and B separately
	SB,dimS,dimB = merge(unmerge(np.dot(svd_FSB[1],svd_FSB[2]),np.array([1]),np.array([np.prod(dimSB[:spec+2]),np.prod(dimSB[spec+2:]))),2,1)
	#indices->cB/dB->cCK/dCN or cCE/dCE (C=Ja/Mb or C=Jga/Mhb,E=Ke/Nf)->DK/DN or DE (D=cJa/dMb or D=cJga/dMhb)
	svd_FSB = None
	# Performing SVD to separate S from B
	svd_SB = svd_sig(SB,tol) #indices->De/Df,ee/ff,eK/fN or eE/fE
	# Unmerging the potential merged indices in B
	B    = unmerge(svd_SB[2],np.array([1]),dimSB[3:]*spec+np.array([1])*(1-spec)) #indices->eK/fN or eKe/fNf
	Sstr = ["cJae->aeJc","cJgae->aegJc"]
	# Keeping the orthoginality centre in S, unmerging and reordering the indices of S
	S = np.einsum(Sstr[spec],unmerge(np.dot(svd_SB[0],svd_SB[1]),
								np.concatenate((np.array([dimS[0]]),dimSB[:2+spec])),np.array([1])))
	#De/Df->cJae/dMbf or cJgae/dMhbf->aeJc/bfMd or aegJc/bfhMd
	return F,B,S	

######################################################################################################################################################################################################
#######################################
### Swapping back to original place ###
#######################################
def SWAP_back(M,L,F,S,tol):
	"""Moving the interacting past bin over the length of the whole delay towards future or past bins.
	INPUT: list of timebins, index of start, direction of SWAPs ("future" or "past") and the location of
	the orthogonality centre (optional)
	OUTPUT: list of states that changed"""

	Ms = int(np.any(M))
	context_up = ["O,aeJc->eJcOa","zOg,aegJc->ezJcOa","Og,aegJc->eJcOa"]
	context_down = ["bfMd,P->bPfMd","bfhMd,hPz->bPfzMd","bfhMd,hP->bPfMd"]
	Sstr_up = ["eJcx->xeJc","ezJcx->xezJc","eJcx->xeJc"]
	
	for i in range(L-1,0,-1):
		#upper branch
		I = (M+i)%(2*L)
		spec = int(np.any(i))*Ms
		combined_up      = np.einsum(context_up[Ms+spec],F[I],S[0])
		merged,dimS,dimF = merge(combined_up,3+Ms-spec,2) #indices->AB (A=eJc or ezJc, B=Oa)
		svded  = svd_sig(merged,tol) #indices->Ax,xx,xB
		merged = None
		F[I]   = unmerge(svded[2],np.array([1]),dimF) #indices->xB->xOa
		S[0] = np.einsum(Sstr[Ms+spec],unmerge(np.dot(svded[0],svded[1]),dimS,np.array([1]))) #indices->Ax->eJcx or ezJcx->aeJc or aegJc
		dimF  = None
		dimS  = None
		svded = None
		
		#lower branch
		J = (2*L+M-i)%(2*L)
		combined_down    = np.einsum(context_down[Ms+spec],S[1],F[J])
		merged,dimF,dimS = merge(combined_down,2,3+Ms-spec) #indices->AB (A=bP, B=fMd or fzMd)
		svded  = svd_sig(merged,tol) #indices->Ay,yy,yB
		merged = None
		F[J] = unmerge(svded[0],dimF,np.array([1])) #indices->Ay->bPy
		S[1] = unmerge(np.dot(svded[1],svded[2]),np.array([1]),dimS) #indices->yB->yfMd or jfzMd
		dimF  = None
		dimS  = None
		svded = None

	return F,S

######################################################################################################################################################################################################
#############################
### Swapping to perform U ###
#############################
def SWAP_U(M,L,F,S,tol):
	"""Moving the interacting past bin over the length of the whole delay towards future or past bins.
	INPUT: list of timebins, index of start, direction of SWAPs ("future" or "past") and the location of
	the orthogonality centre (optional)
	OUTPUT: list of states that changed"""

	context_up = ["cIx,xOa->OcIa","zcIx,xOa->zOcIa"]
	context_down = ["bPy,ydL->bdLP","bPy,ydLz->bdLPz"]
	Sstr_up = ["eJcx->xeJc","ezJcx->xezJc","eJcx->xeJc"]
	
	for i in range(L-1):
		#upper branch
		I = (M+i+1)%(2*L)
		spec = int(np.any(i))
		combined_up      = np.einsum(context_up[spec],F[M%(2*L)],F[I])
		merged,dimS,dimF = merge(combined_up,1+spec,3) #indices->AB (A=O or zO, B=cIa)
		svded  = svd_sig(merged,tol) #indices->Ax,xx,xB
		merged = None
		F[I]   = unmerge(svded[0],dimF,np.array([1])) #indices->Ax->Ox or zOx
		F[M%(2*L)] = unmerge(np.dot(svded[1],svded[2]),np.array([1]),dimS) #indices->xB->xcIa
		dimF  = None
		dimS  = None
		svded = None
		
		#lower branch
		J = (L+M+i+1)%(2*L)
		combined_down    = np.einsum(context_down[spec],F[J],F[(M+L)%(2*L)])
		merged,dimF,dimS = merge(combined_down,3,1+spec) #indices->AB (A=bdL, B=P or Pz)
		svded  = svd_sig(merged,tol) #indices->Ay,yy,yB
		merged = None
		F[J] = unmerge(svded[2],np.array([1]),dimF) #indices->yB->yP or yPz
		F[(M+L)%(2*L)] = unmerge(np.dot(svded[0],svded[1]),dimS,np.array([1])) #indices->Ay->bdLy
		dimF  = None
		dimS  = None
		svded = None

	return F
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

