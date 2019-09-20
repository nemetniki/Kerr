import numpy as np
import scipy as sc
from scipy.linalg import svd
import time
import sys
from decimal import Decimal
from math import factorial
from opt_einsum import contract

es = np.einsum

#*************************#
#***-------------------***#
#***| ANALYZING TOOLS |***#
#***-------------------***#
#*************************#

############################
### Calculating the norm ###
############################
def normf(M,L,statesB1,statesB2,statesF,statesS,normB1,normB2):
	"""Performing the contractions for the norm
	INPUT: Time step, delay length, Markovian and non-Markovian+system state lists, 
	finally, the stored values of normB1 and normB2 (tensor with indices for state and dual) at timestep M
	OUTPUT: the norm at the present time, the contraction of the past bins which stay constant
	and the precalculated composite system state for observable expectation values."""
	
	#There are two kinds of environment bins. Markovian, that are the input-output channels from the cavities
	#and non-Markovian, that is the connecting fibre. 
	index = M%(L)
	#Contracting B1s

	if M>0:
		if M>1:
			#print("statesB",statesB1[M-1],statesB2[M-1])
			#print("normB1,normB2",normB1.shape,normB2.shape)
			#print("statesB1,statesB2",statesB1[M-1].shape,statesB2[M-1].shape)
			normB1 = contract("ab,aIc,bId->cd",normB1,statesB1[M-1],np.conjugate(statesB1[M-1]))
			normB2 = contract("ab,cIa,dIb->cd",normB2,statesB2[M-1],np.conjugate(statesB2[M-1]))
			#print("normB1,normB2",normB1.shape,normB2.shape)
		elif M==1:
			#print("statesB",statesB1[M-1],statesB2[M-1])
			#print("statesB1,statesB2",statesB1[M-1].shape,statesB2[M-1].shape)
			normB1 = es("Ia,Ib->ab",statesB1[M-1],np.conjugate(statesB1[M-1]))
			normB2 = es("aI,bI->ab",statesB2[M-1],np.conjugate(statesB2[M-1]))
			#print("normB1,normB2",normB1.shape,normB2.shape)
		
		norm_left = 0
		norm_left += normB1
#		print("norm_left",es("ii",norm_left))
		if index!=0:
			for i in range(index):
#				print("index",i)
				norm_left = contract("ab,aIc,bId->cd",norm_left,statesF[i],np.conjugate(statesF[i]))
#				print("norm_left",es("ii",norm_left))
		
			
		norm_right = 0
		norm_right += normB2
#		print("norm_right",es("ii",norm_right))
		for i in range(2*L-1,index+L-1,-1):
#			print("index",i)
			norm_right = contract("ab,cIa,dIb->cd",norm_right,statesF[i],np.conjugate(statesF[i]))
#			print("norm_right",es("ii",norm_right))
		
#		print("S1",es("aIc,aIc",statesS[0],np.conjugate(statesS[0])))
#		print("S2",es("aIc,aIc",statesS[1],np.conjugate(statesS[1])))
		
		if M%(2*L)<L:
#			print("index",index)
			norm_mid = es("aIc,bId->abcd",statesF[index],np.conjugate(statesF[index]))
#			print("norm_mid",norm_mid.shape,es("aabb",norm_mid))
			for i in range(1,L,1):
#				print("index",i+index)
				norm_mid = contract("abcd,cIe,dIf->abef",norm_mid,statesF[index+i],np.conjugate(statesF[index+i]))
#				print("norm_mid",norm_mid.shape,es("aabb",norm_mid))
			sys_state = contract("ab,aIc,bJd,cdef,eKg,fLh,gh->IJKL",norm_left,statesS[0],np.conjugate(statesS[0]),
								norm_mid,statesS[1],np.conjugate(statesS[1]),norm_right)
		elif M%(2*L)>L-1:
#			print("index",index+L-1)
			norm_mid = es("aIc,bId->abcd",statesF[index+L-1],np.conjugate(statesF[index+L-1]))
#			print("norm_mid",norm_mid.shape,es("aabb",norm_mid))
			for i in range(L-2,-1,-1):
#				print("index",i+index)
				norm_mid = contract("abcd,eIa,fIb->efcd",norm_mid,statesF[index+i],np.conjugate(statesF[index+i]))
#				print("norm_mid",norm_mid.shape,es("aabb",norm_mid))
			sys_state = contract("ab,aIc,bJd,cdef,eKg,fLh,gh->KLIJ",norm_left,statesS[1],np.conjugate(statesS[1]),
								norm_mid,statesS[0],np.conjugate(statesS[0]),norm_right)
		#print("sys_state",sys_state)

	elif M==0:
		normB1 = 1.
		normB2 = 1.
		sys_state = contract("I,J,K,L->IJKL",statesS[0],np.conjugate(statesS[0]),statesS[1],np.conjugate(statesS[1]))
							
	norm = contract("IIJJ",sys_state)
	
#	print("norm",norm)
	return np.real(norm),normB1,normB2,sys_state

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

##############################################
### Expectation value of system observable ###
##############################################
def exp_sys(observable,sys_state,which):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the combined system state from the norm function, which system
	OUTPUT: expectation value of the observable"""
	
	context = ["IJKK,IJ","IIJK,JK"]
	obs = contract(context[which-1],sys_state,observable)
	return np.real(obs)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

#################################################
### Expectation value of reservoir observable ###
#################################################
def env_dens(state,when = 3,which=1):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the combined system state from the norm function, which system
	OUTPUT: expectation value of the observable"""
	
	if when==0:
		dens = contract("I,J->IJ",state,np.conjugate(state))
	elif when==1:
		if which==1:
			dens = contract("Ia,Ja->IJ",state,np.conjugate(state))
		elif which==2:
			dens = contract("aI,aJ->IJ",state,np.conjugate(state))
	else:
		dens = contract("aIb,aJb->IJ",state,np.conjugate(state))
	return dens

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

#################################################
### Expectation value of reservoir observable ###
#################################################
def exp_res(observable,density):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the combined system state from the norm function, which system
	OUTPUT: expectation value of the observable"""
	
	obs = contract("IJ,IJ",density,observable)
	return np.real(obs)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

#######################
### Output spectrum ###
#######################
def spectrum(states,freqs,pmax,N_env,dt):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the state of the system and timestep M
	OUTPUT: expectation value of the observable"""

	bval = np.sqrt(np.arange(0,N_env))
	B = np.diag(bval[:-1],1)
	Bd = np.diag(bval[1:],-1)
	coherence = np.zeros(pmax,complex)

	nB      = contract("aIb,IJ,JK,aKb",np.conjugate(states[-1]),Bd,B,states[-1])
	coherence[0] = 1
	sum_B   = coherence[0]*np.exp(1j*freqs*0.*dt)
	next    = contract("aIb,IJ,aJc->bc",np.conjugate(states[-1]),Bd,states[-1])
	for p in range(1,pmax):
		coherence[p] = contract("bc,bId,IJ,cJd",next,np.conjugate(states[-1-p]),B,states[-1-p])/nB
		sum_B   += coherence[p]*np.exp(1j*freqs*p*dt)
		next    = contract("bc,bId,cIe->de",next,np.conjugate(states[-1-p]),states[-1-p])	
	return 2./dt*sum_B, np.real(coherence)
    
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

######################
### g2 correlation ###
######################
def g2_tau(state,N_env,dt,pmax):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the state of the system and timestep M
	OUTPUT: expectation value of the observable"""
	bval = np.sqrt(np.arange(0,N_env))
	B = np.diag(bval[:-1],1)
	g2tau = np.zeros(pmax,complex)

	temp      = contract("IJ,JK,aKb->aIb",B,B,states[-1])
	next_temp = contract("IJ,aJb->aIb",B,states[-1])
	nB        = contract("aIb,aIb",np.conjugate(next_temp),next_temp)
	g2tau[0]  = contract("aIb,aIb",np.conjugate(temp),temp)/nB**2
	next    = contract("aIb,aIc->bc",np.conjugate(next_temp),next_temp)
	for p in range(1,pmax):
		temp     = contract("IJ,bJd->bId",B,states[-1-p])
		g2tau[p] = contract("bc,bId,cId",next,np.conjugate(temp),temp)/nB**2
		next     = contract("bc,bId,cIe->de",next,np.conjugate(states[-1-p]),states[-1-p])
	return g2tau


