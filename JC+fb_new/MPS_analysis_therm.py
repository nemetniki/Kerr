import numpy as np
import scipy as sc
from scipy.linalg import svd
import time
import sys
from decimal import Decimal
from math import factorial
import opt_einsum as oe

#*************************#
#***-------------------***#
#***| ANALYZING TOOLS |***#
#***-------------------***#
#*************************#

############################
### Calculating the norm ###
############################
@profile
def normf(M,L,state,norm_L):
	"""Performing the contractions for the norm
	INPUT: calculated states in list "state" with a delay index-length L 
	and the stored values of norm_L (tensor with indices for state and dual) at timestep M
	OUTPUT: the norm at the present time and the contraction of the past bins which stay constant"""

	# Indices of the timebins initially: 0->furthest past, L->system, L+1->first future timebin
	if M==0:
		norm = np.einsum("ij,ij",state[L],np.conjugate(state[L]))
		norm_L = 1.
	else:
		# Contracting part of the MPS that won't change anymore with its dual and with previously stored tensors
		if np.isscalar(norm_L) or norm_L.size==1:
			norm_L = np.einsum("kmj,lmj->kl",state[M-1],np.conjugate(state[M-1]))*norm_L
		else:
			norm_L = np.einsum("kmj,lmj->kl",np.einsum("kmi,ij->kmj",state[M-1],norm_L),np.conjugate(state[M-1]))
		
		# Contracting the system part of the MPS
		norm_S = np.einsum("ki,kj->ij",state[L+M],np.conjugate(state[L+M]))
		norm = norm_L
		#print(norm)

	# Performing the rest of the reservoir contractions from right to left.
		for i in range(0,L):
		#	print(len(norm.shape))
			#print("state",state[M+i].shape)
			#if np.isscalar(norm) or norm.size==1:
			#	norm = np.einsum("kmi,lmi->kl",state[M+i],np.conjugate(state[M+i]))*norm
			#else:
			norm = np.einsum("kmj,lmj->kl",np.einsum("kmi,ij->kmj",state[M+i],norm),np.conjugate(state[M+i]))
	# Contracting the environment part with the system part
		#if np.isscalar(norm) or norm.size==1:
		#	norm = np.einsum("ii",norm_S)*norm
		#else:
		norm = np.einsum("ij,ij",norm,norm_S)
		norm_S = None
	return np.real(norm),np.real(norm_L)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

##############################################
### Expectation value of system observable ###
##############################################
@profile
def exp_sys(observable,sys,M):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the state of the system and timestep M
	OUTPUT: expectation value of the observable"""

	# Indices of the timebins initially: 0->furthest past, L->system, L+1->first future timebin
	obs = np.einsum("jk,kj",np.einsum("ij,ik->jk",sys,observable),np.conjugate(sys))
	return np.real(obs)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

#######################
### Output spectrum ###
#######################
def spectrum(states,freqs,pmax,N_env,dt,index,thermal):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the state of the system and timestep M
	OUTPUT: expectation value of the observable"""
	def dB(state):
		new_state = np.zeros(state.shape,complex)
		if thermal == False:
			n = np.arange(1,N_env)
			iend = 1
		elif thermal == True:
			n = np.arange(0,N_env*(N_env-1))/N_env).astype(np.int64)+1
			iend = N_env-1
		new_state[:,0:(-iend),:] = np.einsum("i,jik->jik",np.sqrt(n*dt),state[:,iend:,:])
		return new_state
		
	def dBd(state):
		new_state = np.zeros(state.shape,complex)
		if thermal == False:
			n = np.arange(1,N_env)
			iend = 1
		elif thermal == True:
			n = np.arange(0,N_env*(N_env-1))/N_env).astype(np.int64)+1
			iend = N_env-1
		new_state[:,iend:,:] = np.einsum("k,ikl->ikl",np.sqrt(n*dt),state[:,0:-iend,:])
		return new_state
		
	sum_B      = np.zeros(freqs.shape,complex)
	sum_B      = np.einsum("ilk,ilk",dBd(dB(states[index])),np.conjugate(states[index]))*np.ones(freqs.shape)
	next_step  = np.einsum("ilk,ilm->km",dBd(states[index]),np.conjugate(states[index]))
	for p in range(1,pmax):
		sum_B     += (np.einsum("ij,ij",np.einsum("ilk,jlk->ij",dB(states[index-p]),np.conjugate(states[index-p])),
								next_step)*np.exp(1j*freqs*p*dt))
		next_step  = np.einsum("jmk,jml->kl",np.einsum("ij,imk->jmk",next_step,states[index-p]),
								np.conjugate(states[index-p]))
	return 2./dt*np.real(sum_B)
    
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

######################
### g2 correlation ###
######################
@profile
def g2_t(state,N_env,dt,thermal):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the state of the system and timestep M
	OUTPUT: expectation value of the observable"""
	def dB(state):
		new_state = np.zeros(state.shape,complex)
		if thermal == False:
			n = np.arange(1,N_env)
			iend = 1
		elif thermal == True:
			n = np.arange(0,N_env*(N_env-1))/N_env).astype(np.int64)+1
			iend = N_env-1
		new_state[:,0:(-iend),:] = np.einsum("i,jik->jik",np.sqrt(n),state[:,iend:,:])
		return new_state
#    dBd = sc.eye(N_env,N_env,-1)*np.sqrt(dt*np.arange(1,N_env+1))
	temp = dB(dB(state))
	temp2 = dB(state)
	NB = np.einsum("jil,jil",temp2,np.conjugate(temp2))
	g2_t = np.einsum("jil,jil",temp,np.conjugate(temp))/(NB**2)
	temp = None
	return np.real(g2_t),np.real(NB)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#############################
### g2 output correlation ###
#############################
@profile
def g2_out(states,taumax,N_env,dt,index,thermal):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the state of the system and timestep M
	OUTPUT: expectation value of the observable"""
	def dB(state):
		new_state = np.zeros(state.shape,complex)
		if thermal == False:
			n = np.arange(1,N_env)
			iend = 1
		elif thermal == True:
			n = np.arange(0,N_env*(N_env-1))/N_env).astype(np.int64)+1
			iend = N_env-1
		new_state[:,0:(-iend),:] = np.einsum("i,jik->jik",np.sqrt(n),state[:,iend:,:])
		return new_state
	g2_out = np.zeros(taumax)
	tau = np.zeros(taumax)
	temp      = dB(dB(states[index]))
	temp2     = dB(states[index])
	NB = np.einsum("ijk,ijk",temp2, np.conjugate(temp2))
	g2_out[0] = np.einsum("ijk,ijk",temp, np.conjugate(temp))/(NB**2)
	next_step = np.einsum("ijk,ijl->kl",temp2,np.conjugate(temp2))
	for it in range(1,taumax):
		tau[it] = dt*it
		temp    = dB(states[index-it])
		g2_out[it] = np.einsum("njk,njk",np.einsum("mn,mjk->njk",next_step,temp), np.conjugate(temp))/(NB**2)
		next_step = np.einsum("mjk,mjl->kl",np.einsum("ijk,im->mjk",states[index-it],next_step),np.conjugate(states[index-it]))
	return np.real(tau),np.real(g2_out)

################################
### Number of output photons ###
################################
@profile
def NB_out(state,N_env,NB_past,dt,thermal):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the state of the system and timestep M
	OUTPUT: expectation value of the observable"""
	def dB(state):
		new_state = np.zeros(state.shape,complex)
		if thermal == False:
			n = np.arange(1,N_env)
			iend = 1
		elif thermal == True:
			n = np.arange(0,N_env*(N_env-1))/N_env).astype(np.int64)+1
			iend = N_env-1
		new_state[:,0:(-iend),:] = np.einsum("i,jik->jik",np.sqrt(n),state[:,iend:,:])
		return new_state
	temp = dB(state)
	NB_now = np.einsum("jil,jil",temp,np.conjugate(temp))
	NB = NB_now+NB_past
	temp = None
	NB_now = None
	return np.real(NB)

