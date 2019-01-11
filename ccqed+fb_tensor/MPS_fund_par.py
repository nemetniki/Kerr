import numpy as np
import scipy as sc
from numpy.linalg import svd
import time
import sys
from decimal import Decimal
from math import factorial
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
import multiprocessing


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
	block = np.nan_to_num(block,copy=False)
	#print(block.shape)
	try:
		svd_init  = svd(block,full_matrices=False)
	except:
		kv = min(block.shape)
	# Performing the SVD
		svd_init  = svds(csc_matrix(block),k=kv-2)
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

#####################################################################################################################################################################################################
###################################
### SVD creating 2 link-indices ###
###################################
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
	svd_final = [np.zeros((svd_init[0].shape[0],dim,dim),complex),np.zeros((dim,dim),complex),np.zeros((dim,dim,svd_init[2].shape[1]),complex)]
	for J in range(0,dim**2):
		svd_final[0][:,J%dim,int(J/dim)] = svd_init[0][:,J]
		svd_final[1][J%dim,int(J/dim)]   = svd_init[1][J]
		svd_final[2][J%dim,int(J/dim),:] = svd_init[2][J,:]
	
	# Calculating block 1 and block 2 after U and the SVD
	block_1 = np.einsum("ijk,jk->ijk",svd_final[0],svd_final[1])
	block_2 = svd_final[2]
#	block_2 = np.einsum("ij,ijk->ijk",np.sqrt(svd_final[1]),svd_final[2])
#	block_1 = np.einsum("ijk,jk->ijk",svd_final[0],np.sqrt(svd_final[1]))
#	block_1 = svd_final[0]
#	block_2 = svd_final[2]
	
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
			idx_o[1] = i1
			idx_o[0] = i2
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

#	print("dL:",dL,", dR:",dR)
	#Number of unmerged dimensions on the left and the right
	NL = len(dL)
	NR = len(dR)
	
	spec = int(np.any(which))
	#Dimensions of the merged array
	DL = list(block.shape[:which])*spec+[block.shape[0]]*(1-spec)
	DR = list(block.shape[(-3+which)*spec:])*spec+[block.shape[1]]*(1-spec)
	
	dL = list(dL)*int(np.any(which-2))+DL*(1-int(np.any(which-2))) #dL unless unmerge on the right
	dR = list(dR)*int(np.any(which-1))+DR*(1-int(np.any(which-1))) #dR unless unmerge on the left
	
#	print("state shape:",block.shape," which:",which)
#	print("DL:",DL,", DR:",DR,", dL:",dL,", dR:",dR)
#	print("NL:",NL,", NR:",NR)

	#Initializing the left-merged and the unmerged tensor
	right_block = np.zeros(DL+dR,dtype=np.complex128)
	unmerged_block = np.zeros(dL+dR,dtype=np.complex128)
#	print("right block shape:",right_block.shape,"unmerged shape:",unmerged_block.shape)
		
	#%%%%%%%#
	# RIGHT #
	#%%%%%%%#

	if NR == 1: #No unmerge needed
		right_block += block

	elif NR == 2:
		for J in np.arange(0,DR[0]):
			idx_o = [slice(None)]*block.ndim
			idx_o[-1] = J
			idx_n = [slice(None)]*(right_block.ndim)
			idx_n[-1] = J%dR[1]
			idx_n[-2] = int(J/dR[1])

			right_block[tuple(idx_n)] = block[tuple(idx_o)]

	elif NR == 3:
		for J in np.arange(0,DR[0]):
			idx_o = [slice(None)]*block.ndim
			idx_o[-1] = J
			idx_n = [slice(None)]*(right_block.ndim)
			idx_n[-1] = J%dR[2]
			idx_n[-2] = int(J/dR[2])%dR[1]
			idx_n[-3] = int(J/(dR[2]*dR[1]))

			right_block[tuple(idx_n)] = block[tuple(idx_o)]
		
	elif NR == 4:
		for J in np.arange(0,DR[0]):
			idx_o = [slice(None)]*block.ndim
			idx_o[-1] = J
			idx_n = [slice(None)]*(right_block.ndim)
			idx_n[-1] = J%dR[3]
			idx_n[-2] = int(J/dR[3])%dR[2]
			idx_n[-3] = int(J/(dR[3]*dR[2]))%dR[1]
			idx_n[-4] = int(J/(dR[3]*dR[2]*dR[1]))

			right_block[tuple(idx_n)] = block[tuple(idx_o)]
		
	elif NR == 5:
		for J in np.arange(0,DR[0]):
			idx_o = [slice(None)]*block.ndim
			idx_o[-1] = J
			idx_n = [slice(None)]*(right_block.ndim)
			idx_n[-1] = J%dR[4]
			idx_n[-2] = int(J/dR[4])%dR[3]
			idx_n[-3] = int(J/(dR[4]*dR[3]))%dR[2]
			idx_n[-4] = int(J/(dR[4]*dR[3]*dR[2]))%dR[1]
			idx_n[-5] = int(J/(dR[4]*dR[3]*dR[2]*dR[1]))

			right_block[tuple(idx_n)] = block[tuple(idx_o)]

	#%%%%%%#
	# LEFT #
	#%%%%%%#

	if NL == 1: #No unmerge needed
		unmerged_block += right_block

	elif NL == 2:
		for I in np.arange(0,DL[0]):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[0] = I
			idx_n = [slice(None)]*(right_block.ndim+1)
			idx_n[1] = I%dL[1]
			idx_n[0] = int(I/dL[1])
			
			unmerged_block[tuple(idx_n)] = right_block[tuple(idx_o)]

	elif NL == 3:
		for I in np.arange(0,DL[0]):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[0] = I
			idx_n = [slice(None)]*(right_block.ndim+2)
			idx_n[2] = I%dL[2]
			idx_n[1] = int(I/dL[2])%dL[1]
			idx_n[0] = int(I/(dL[2]*dL[1]))
			
			unmerged_block[tuple(idx_n)] = right_block[tuple(idx_o)]
		
	elif NL == 4:
		for I in np.arange(0,DL[0]):
			idx_o = [slice(None)]*right_block.ndim
			idx_o[0] = I
			idx_n = [slice(None)]*(right_block.ndim+3)
			idx_n[3] = I%dL[3]
			idx_n[2] = int(I/dL[3])%dL[2]
			idx_n[1] = int(I/(dL[3]*dL[2]))%dL[1]
			idx_n[0] = int(I/(dL[3]*dL[2]*dL[1]))
			
			unmerged_block[tuple(idx_n)] = right_block[tuple(idx_o)]

	elif NL == 5:
		for I in np.arange(0,DL[0]):
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
	context = ["IJKab->IbJaK","abLMN->aLbMN","gIeJKab->IbgJaKe","abhLfMN->aLbMhNf"]
	spec    = int(np.any(M))

	#%%%%%%%%%%%%%%%%#
	# INDEX ORDERING #
	#%%%%%%%%%%%%%%%%#
#	print("\nsplit","\nhow:",context[spec*2+which-1],"what:",block.shape)
#	print("block shape:",block.shape,", spec:", spec,", which:", which,", context:",context[spec*2+which-1])
	block = np.einsum(context[spec*2+which-1],block)
#	print("then:",block.shape)
	
	#%%%%%%%%%%%%%%%%%%%%%%#
	# SPLITTING F FROM B&S #
	#%%%%%%%%%%%%%%%%%%%%%%#
	# Merging the indices in block to F+SB
	merged_FSB,dimF,dimSB = merge(block,2,3+2*spec) 
#	print("merged:",merged_FSB.shape, dimF,dimSB)
	#print(merged_FSB.shape,dimF,dimSB)
	#indices -> AB (A=Ib/aL, B=JaK/bMN or B=gJaKe/bMhNf)		
	# Performing the SVD to split the F from SB
	svd_FSB,link_dim1    = SVD_sig(merged_FSB,tol) #indices->Ac/Ad,cc/dd,cB/dB
#	print("F from SB svd:",svd_FSB[0].shape,svd_FSB[2].shape)
	merged_FSB = None
	# Reordering the indices of F
	Fstr = ["Ibc->cIb","aLd->aLd"]
	F = np.einsum(Fstr[which-1],unmerge(svd_FSB[0],dimF,np.array([link_dim1]))) #indices->Ac/Ad->Ibc/aLd->cIb/aLd
#	print("how:",Fstr[which-1],"F:",F.shape)
	
	#%%%%%%%%%%%%%%%%%%%%%#
	# SEPARATING B FROM S #
	#%%%%%%%%%%%%%%%%%%%%%#
	# Keeping the orthognality centre in S. Unmerging the indices of S from B, but merging the indices of S and B separately
	SB,dimS,dimB = merge(unmerge(np.dot(svd_FSB[1],svd_FSB[2]),np.array([link_dim1]),np.array([np.prod(dimSB[:spec+2]),np.prod(dimSB[spec+2:])])),2,1)
#	print("SB merged:",SB.shape,dimS,dimB)
	#indices->cB/dB->cCK/dCN or cCE/dCE (C=Ja/bM or C=gJa/bMh,E=Ke/Nf)->DK/DN or DE (D=cJa/dbM or D=cgJa/dbMh)
	svd_FSB = None
	# Performing SVD to separate S from B
	svd_SB,link_dim2 = SVD_sig(SB,tol) #indices->Dj/Dk,jj/kk,jK/kN or jE/kE
#	print("svd SB:",svd_SB[0].shape,svd_SB[2].shape)
	# Unmerging the potential merged indices in B
	B    = unmerge(svd_SB[2],np.array([link_dim2]),dimSB[2+spec:]) #indices->jK/kN or jKe/kNf
#	print("B:",B.shape)
	Sstr = ["cJae->ceJa","dbMf->bfMd","cgJae->gceJa","dbMhf->bfMdh"]
	# Keeping the orthoginality centre in S, unmerging and reordering the indices of S
	S = np.einsum(Sstr[which-1+2*spec],unmerge(np.dot(svd_SB[0],svd_SB[1]),
								np.concatenate((np.array([dimS[0]]),dimSB[:2+spec])),np.array([link_dim2])))
#	print("how:",Sstr[which-1+2*spec],"S:",S.shape)
	#De/Df->cJae/dbMf or cgJae/dbMhf->ceJa/bfMd or gceJa/bfMdh
	return F,B,S	
	#cIb/aLd, jK/kN or jKe/kNf, ceJa/bfMd or gceJa/bfMdh

######################################################################################################################################################################################################
#######################################
### Swapping back to original place ###
#######################################
def parallel_back1(M,L,F,S,tol,result_queue):
	Ms = int(np.any(M))
	context_up = ["O,ceJa->ceJOa","zOg,gceJa->zceJOa","Og,gceJa->ceJOa"]
	list_FS = [0]*(L)
	list_FS[0] += S[0]
#	print(list_FS[0]==S[0])
	for i in range(L-1,0,-1):
		#upper branch
		I = (M+i)%(2*L)			
		spec = (1-int(np.any(i-1)))*Ms
		#print("before back",I,F[I].shape,S[0].shape,context_up[Ms+spec],Ms+spec)#+spec)
		combined_up      = np.einsum(context_up[Ms+spec],F[I],list_FS[0])
		merged,dimS,dimF = merge(combined_up,3+Ms-spec,2) #indices->AB (A=ceJ or zceJ, B=Oa)
		svded,link_dim  = SVD_sig(merged,tol) #indices->Ax,xx,xB
		merged = None
		list_FS[i]   = unmerge(svded[2],np.array([link_dim]),dimF) #indices->xB->xOa
#		print("at M=",M,"before",list_FS[0].shape)
		list_FS[0] = unmerge(np.dot(svded[0],svded[1]),dimS,np.array([link_dim])) #indices->Ax->ceJx or zceJx
#		print("at M=",M,"after",list_FS[0].shape)
		#print("after back",I,F[I].shape,S[0].shape,context_up[Ms+spec],Ms+spec)#+spec)
		dimF  = None
		dimS  = None
		svded = None
	result_queue.put((0,list_FS))

def parallel_back2(M,L,F,S,tol,result_queue):
	Ms = int(np.any(M))
	context_down = ["bfMd,P->bPfMd","bfMdh,hPz->bPfMdz","bfMdh,hP->bPfMd"]
	list_FS = [0]*(L)
	list_FS[0] += S[1]
#	print(list_FS[0]==S[1])
	for j in range(L-1,0,-1):
		spec = (1-int(np.any(j-1)))*Ms
		#lower branch
		J = (L+M+j)%(2*L)
#		print("before back",J,F[J].shape,S[1].shape,context_down[Ms+spec],Ms+spec)#+spec)
		combined_down    = np.einsum(context_down[Ms+spec],list_FS[0],F[J])
		merged,dimF,dimS = merge(combined_down,2,3+Ms-spec) #indices->AB (A=bP, B=fMd or fMdz)
		svded,link_dim  = SVD_sig(merged,tol) #indices->Ay,yy,yB
		merged = None
		list_FS[j] = unmerge(svded[0],dimF,np.array([link_dim])) #indices->Ay->bPy
#		print("at M=",M,"before",list_FS[0].shape)
		list_FS[0] = unmerge(np.dot(svded[1],svded[2]),np.array([link_dim]),dimS) #indices->yB->yfMd or yfMdz
#		print("at M=",M,"after",list_FS[0].shape)
#		print("after back",J,F[J].shape,S[1].shape,context_down[Ms+spec],Ms+spec)#+spec)
		dimF  = None
		dimS  = None
		svded = None
	result_queue.put((1,list_FS))
	

def SWAP_back(M,L,F,S,tol):
	"""Moving the system bin back to its original place in the loop.
	INPUT: time, delay, list of fibre bins, list of system bins, tolerance of SVD
	OUTPUT: list of fibre states and system states"""

	result_queue = multiprocessing.Queue()	
	proc1 = multiprocessing.Process(target=parallel_back1, args=(M,L,F,S,tol,result_queue))
	proc2 = multiprocessing.Process(target=parallel_back2, args=(M,L,F,S,tol,result_queue))

	proc1.start()
	proc2.start()

	tmp_results = {}
	while len(tmp_results) < 2:
	# two items were put into the Queue: the result index and the result
		result_index, tmp_result = result_queue.get()
		tmp_results[result_index] = tmp_result
	
	proc1.join()
	proc2.join()

	for ind in range(L-1,0,-1):
		F[(M+ind)%(2*L)]   = tmp_results[0][ind]
		F[(L+M+ind)%(2*L)] = tmp_results[1][ind]
	S[0] = tmp_results[0][0]
	S[1] = tmp_results[1][0]

	return F,S
	#xOa and bPy, ceJx or zceJx and yfMd or yfMdz

######################################################################################################################################################################################################
#############################
### Swapping to perform U ###
#############################
def parallel_U1(M,L,F,tol,result_queue):
	context_up = ["cIx,xOa->OcIa","zcIx,xOa->zOcIa"]
	list_F = [0]*(L)
	list_F[-1] += F[M%(2*L)]
	for iU in range(L-1):
		#upper branch
		I = (M+iU+1)%(2*L)
		spec = int(np.any(iU))
		#print("before",I,spec, list_F[-1].shape,F[I].shape,context_up[spec])
		combined_up      = np.einsum(context_up[spec],list_F[-1],F[I])
		merged,dimF,dimFO = merge(combined_up,1+spec,3) #indices->AB (A=O or zO, B=cIa)
		svded,link_dim  = SVD_sig(merged,tol) #indices->Au,uu,uB
		merged = None
		list_F[iU]   = unmerge(svded[0],dimF,np.array([link_dim])) #indices->Au->Ou or zOu
		list_F[-1] = unmerge(np.dot(svded[1],svded[2]),np.array([link_dim]),dimFO) #indices->uB->ucIa
		#print("after",I, list_F[-1].shape,list_F[iU].shape)
		dimF  = None
		dimS  = None
		svded = None
	result_queue.put((0,list_F))

def parallel_U2(M,L,F,tol,result_queue):
	context_down = ["bPy,yLd->bLdP","bPy,yLdz->bLdPz"]
	list_F = [0]*(L)
	list_F[-1] += F[(M+L)%(2*L)]
	for jU in range(L-1):
		spec = int(np.any(jU))
		#lower branch
		J = (L+M+jU+1)%(2*L)
#		print("F shape",F[J].shape,F[(M+L)%(2*L)].shape,F[(J-1)%(2*L)].shape)
		combined_down    = np.einsum(context_down[spec],F[J],list_F[-1])
		merged,dimFO,dimF = merge(combined_down,3,1+spec) #indices->AB (A=bLd, B=P or Pz)
		svded,link_dim  = SVD_sig(merged,tol) #indices->Av,vv,vB
		merged = None
		list_F[jU] = unmerge(svded[2],np.array([link_dim]),dimF) #indices->vB->vP or vPz
		list_F[-1] = unmerge(np.dot(svded[0],svded[1]),dimFO,np.array([link_dim])) #indices->Av->bLdv
		dimF  = None
		dimS  = None
		svded = None
	result_queue.put((1,list_F))

def SWAP_U(M,L,F,tol):
	"""Moving the interacting fibre and system bins next to each other.
	INPUT: time, delay, list of fibre bins, tolerance of SVD
	OUTPUT: list of fibre states"""

	result_queue = multiprocessing.Queue()	
	proc1 = multiprocessing.Process(target=parallel_U1, args=(M,L,F,tol,result_queue))
	proc2 = multiprocessing.Process(target=parallel_U2, args=(M,L,F,tol,result_queue))

	proc1.start()
	proc2.start()

	tmp_results = {}
	while len(tmp_results) < 2:
	# two items were put into the Queue: the result index and the result
		result_index, tmp_result = result_queue.get()
		tmp_results[result_index] = tmp_result

	proc1.join()
	proc2.join()	

	for ind in range(L-1):
		F[(M+ind+1)%(2*L)] = tmp_results[0][ind]
		F[(M+L+ind+1)%(2*L)] = tmp_results[1][ind]
	F[M%(2*L)] = tmp_results[0][-1]
	F[(L+M)%(2*L)] = tmp_results[1][-1]

	return F
	#Ou/vP or zOu/vPz and ucIa/bLdv
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

