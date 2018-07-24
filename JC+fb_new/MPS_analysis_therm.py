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
		if np.isscalar(norm_L) or len(norm_L.shape)==0.:
			norm_L = np.einsum("kmj,lmj->kl",state[M-1],np.conjugate(state[M-1]))*norm_L
		else:
			norm_L = oe.contract("kmi,ij,lmj->kl",state[M-1],norm_L,np.conjugate(state[M-1]))
		
		# Contracting the system part of the MPS
		norm_S = np.einsum("ki,kj->ij",state[L+M],np.conjugate(state[L+M]))
		norm = norm_L
		#print(norm)

	# Performing the rest of the reservoir contractions from right to left.
		for i in range(0,L):
		#	print(len(norm.shape))
			#print("state",state[M+i].shape)
			if np.isscalar(norm) or len(norm.shape)==0:
				norm = np.einsum("kmi,lmi->kl",state[M+i],np.conjugate(state[M+i]))*norm
			else:
				norm = oe.contract("kmi,ij,lmj->kl",state[M+i],norm),np.conjugate(state[M+i]))
	# Contracting the environment part with the system part
		if np.isscalar(norm) or len(norm.shape)==0:
			norm = np.einsum("ii",norm_S)*norm
		else:
			norm = np.einsum("ij,ij",norm,norm_S)
			norm_S = None
	return np.real(norm),np.real(norm_L)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

##############################################
### Expectation value of system observable ###
##############################################
def exp_sys(observable,sys,M):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the state of the system and timestep M
	OUTPUT: expectation value of the observable"""

	# Indices of the timebins initially: 0->furthest past, L->system, L+1->first future timebin
	obs = oe.contract("ij,ki,kj",sys,observable,np.conjugate(sys))
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
		if thermal == True:
			n = ((np.linspace(0,(N_env-1)**2-1,(N_env-1)**2)/(N_env-1)).astype(int)+1)[0:(-N_env+1)]
			iend = N_env-1
		else:
			n = np.arange(1,N_env)
			iend = 1
		new_state[:,0:(-iend),:] = np.einsum("i,jik->jik",np.sqrt(n*dt),state[:,iend:,:])
		return new_state
		
	def dBd(state):
		new_state = np.zeros(state.shape,complex)
		if thermal == True:
			n = ((np.linspace(0,(N_env-1)**2-1,(N_env-1)**2)/(N_env-1)).astype(int)+1)[0:(-N_env+1)]
			iend = N_env-1
		else:
			n = np.arange(1,N_env)
			iend = 1
		new_state[:,iend:,:] = np.einsum("k,ikl->ikl",np.sqrt(n*dt),state[:,0:-iend,:])
		return new_state
		
	sum_B      = np.zeros(freqs.shape,complex)
	sum_B      = np.einsum("ilk,ilk",dBd(dB(states[index])),np.conjugate(states[index]))*np.ones(freqs.shape)
	next_step  = np.einsum("ilk,ilm->km",dBd(states[index]),np.conjugate(states[index]))
	for p in range(1,pmax):
		sum_B     += (oe.contract("ilk,jlk,ij",dB(states[index-p]),np.conjugate(states[index-p]),next_step)*
						np.exp(1j*freqs*p*dt))
		next_step  = oe.contract("ij,imk,jml->kl",next_step,states[index-p],np.conjugate(states[index-p]))
	return 2./dt*np.real(sum_B)
    
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

######################
### g2 correlation ###
######################
def g2_t(state,N_env,dt,thermal):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the state of the system and timestep M
	OUTPUT: expectation value of the observable"""
	def dB(state):
		new_state = np.zeros(state.shape,complex)
		legs = len(state.shape)
		if thermal == True:
			n = ((np.linspace(0,(N_env-1)**2-1,(N_env-1)**2)/(N_env-1)).astype(int)+1)[0:(-N_env+1)]
			iend = N_env-1
		else:
			n = np.arange(1,N_env)
			iend = 1
		if legs==1:
			new_state[0:(-iend)] = np.einsum("i,i->i",np.sqrt(n),state[iend:])
		elif legs==2:
			ind = state.shape.index(np.max(state.shape))
			if ind == 0:
				new_state[0:(-iend),:] = np.einsum("i,ij->ij",np.sqrt(n),state[iend:,:])
			elif ind == 1:
				new_state[:,0:(-iend)] = np.einsum("i,ji->ji",np.sqrt(n),state[:,iend:])
		elif legs==3:
			new_state[:,0:(-iend),:] = np.einsum("i,jik->jik",np.sqrt(n),state[:,iend:,:])
		return new_state
#    dBd = sc.eye(N_env,N_env,-1)*np.sqrt(dt*np.arange(1,N_env+1))
	temp = dB(dB(state))
	temp2 = dB(state)
	if len(state.shape)==1:       
		NB = np.einsum("i,i",temp2,np.conjugate(temp2))
		g2_t = np.einsum("i,i",temp,np.conjugate(temp))/(NB**2)
		temp = None
	elif len(state.shape)==2:       
		indp = state.shape.index(np.max(state.shape))
		NB = np.einsum("il,il",temp2,np.conjugate(temp2))
		g2_t = np.einsum("il,il",temp,np.conjugate(temp))/(NB**2)
		temp = None        
	elif len(state.shape)==3:
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
def g2_out(states,taumax,N_env,dt,index,thermal):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the state of the system and timestep M
	OUTPUT: expectation value of the observable"""
	def dB(state):
		new_state = np.zeros(state.shape,complex)
		legs = len(state.shape)
		if thermal == True:
			n = ((np.linspace(0,(N_env-1)**2-1,(N_env-1)**2)/(N_env-1)).astype(int)+1)[0:(-N_env+1)]
			iend = N_env-1
		else:
			n = np.arange(1,N_env)
			iend = 1
		if legs==1:
			new_state[0:(-iend)] = np.einsum("i,i->i",np.sqrt(n),state[iend:])
		elif legs==2:
			ind = state.shape.index(np.max(state.shape))
			if ind == 0:
				new_state[0:(-iend),:] = np.einsum("i,ij->ij",np.sqrt(n),state[iend:,:])
			elif ind == 1:
				new_state[:,0:(-iend)] = np.einsum("i,ji->ji",np.sqrt(n),state[:,iend:])
		elif legs==3:
			new_state[:,0:(-iend),:] = np.einsum("i,jik->jik",np.sqrt(n),state[:,iend:,:])
		return new_state
	g2_out = np.zeros(taumax)
	tau = np.zeros(taumax)
	temp      = dB(dB(states[index]))
	temp2     = dB(states[index])
	if len(states[index].shape)==1:
		next_step = np.einsum("j,j",temp2,np.conjugate(temp2))
		NB = next_step
		g2_out[0] = np.einsum("j,j",temp,np.conjugate(temp))/(NB**2)
	elif len(states[index].shape)==2:
		ind = states[index].shape.index(np.max(states[index].shape))
		NB = np.einsum("jk,jk",temp2, np.conjugate(temp2))
		if ind==0:
			g2_out[0] = np.einsum("jk,jk",temp, np.conjugate(temp))/(NB**2)
			next_step = np.einsum("jk,jl->kl",temp2,np.conjugate(temp2))
		elif ind==1:
			g2_out[0] = np.einsum("kj,kj",temp, np.conjugate(temp))/(NB**2)
			next_step = np.einsum("kj,kj",temp2,np.conjugate(temp2))
	elif len(states[index].shape)==3:
		NB = np.einsum("ijk,ijk",temp2, np.conjugate(temp2))
		g2_out[0] = np.einsum("ijk,ijk",temp, np.conjugate(temp))/(NB**2)
		next_step = np.einsum("ijk,ijl->kl",temp2,np.conjugate(temp2))
	for it in range(1,taumax):
		tau[it] = dt*it
		temp    = dB(states[index-it])
		if len(states[index-it].shape)==1:
			if np.isscalar(next_step):
				g2_out[it] = next_step * np.einsum("j,j",temp,np.conjugate(temp))/(NB**2)
				next_step  = next_step * np.einsum("j,j",states[index-it],np.conjugate(states[index-it]))
			else:
				g2_out[it] = np.einsum("ii",next_step) * np.einsum("j,j",temp,np.conjugate(temp))/(NB**2)
				next_step  = np.einsum("ii",next_step) * np.einsum("j,j",states[index-it],
                                                                   np.conjugate(states[index-it]))
		if len(states[index-it].shape)==2:
			indp = states[index-it].shape.index(np.max(states[index-it].shape))
			if indp == 0:
				if np.isscalar(next_step):
					g2_out[it] = next_step*np.einsum("jk,jk",temp,np.conjugate(temp))/(NB**2)
					next_step = next_step*np.einsum("jk,jl->kl",states[index-it],np.conjugate(states[index-it]))
				else:
					g2_out[it] = np.einsum("ii",next_step)*np.einsum("jk,jk",temp, np.conjugate(temp))/(NB**2)
					next_step = np.einsum("ii",next_step)*np.einsum("jk,jl->kl",states[index-it],
																	np.conjugate(states[index-it]))
			elif indp == 1:
				if np.isscalar(next_step):
					g2_out[it] = next_step*np.einsum("kj,kj",temp, np.conjugate(temp))/(NB**2)
					next_step = next_step*np.einsum("kj,kj",states[index-it],np.conjugate(states[index-it]))
				else:
					g2_out[it] = np.einsum("ki,ki",next_step,np.einsum("kj,ij->ki",temp, np.conjugate(temp)))/(NB**2)
					next_step = np.einsum("ki,ki",next_step,np.einsum("kj,ij->ki",states[index-it],
																		np.conjugate(states[index-it])))
		elif len(states[index-it].shape)==3:
			if np.isscalar(next_step):
				g2_out[it] = next_step*np.einsum("ijk,ijk",temp, np.conjugate(temp))/(NB**2)
				next_step = next_step * np.einsum("ijk,ijl->kl",states[index-it],np.conjugate(states[index-it]))
			else:
				g2_out[it] = np.einsum("mn,mn",next_step,np.einsum("mjk,njk->mn",temp, np.conjugate(temp)))/(NB**2)
				next_step = np.einsum("mjk,mjl->kl",np.einsum("ijk,im->mjk",states[index-it],next_step),
									np.conjugate(states[index-it]))
	return np.real(tau),np.real(g2_out)

################################
### Number of output photons ###
################################
def NB_out(state,N_env,NB_past,dt,thermal):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the state of the system and timestep M
	OUTPUT: expectation value of the observable"""
	def dB(state):
		new_state = np.zeros(state.shape,complex)
		legs = len(state.shape)
		if thermal == True:
			n = ((np.linspace(0,(N_env-1)**2-1,(N_env-1)**2)/(N_env-1)).astype(int)+1)[0:(-N_env+1)]
			iend = N_env-1
		else:
			n = np.arange(1,N_env)
			iend = 1
		if legs==1:
			new_state[0:(-iend)] = np.einsum("i,i->i",np.sqrt(n),state[iend:])
		elif legs==2:
			ind = state.shape.index(np.max(state.shape))
			if ind == 0:
				new_state[0:(-iend),:] = np.einsum("i,ij->ij",np.sqrt(n),state[iend:,:])
			elif ind == 1:
				new_state[:,0:(-iend)] = np.einsum("i,ji->ji",np.sqrt(n),state[:,iend:])
		elif legs==3:
			new_state[:,0:(-iend),:] = np.einsum("i,jik->jik",np.sqrt(n),state[:,iend:,:])
		return new_state
	temp = dB(state)
	if len(state.shape)==1:       
		NB_now = np.einsum("i,i",temp,np.conjugate(temp))
	elif len(state.shape)==2:       
		indp = state.shape.index(np.max(state.shape))
		NB_now = np.einsum("il,il",temp,np.conjugate(temp))
	elif len(state.shape)==3:
		NB_now = np.einsum("jil,jil",temp,np.conjugate(temp))
	NB = NB_now+NB_past
	temp = None
	NB_now = None
	return np.real(NB)

