import numpy as np
import scipy as sc
from scipy.linalg import svd
import time
import sys
from decimal import Decimal
from math import factorial

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
		norm = np.einsum("i,i",state[L],np.conjugate(state[L]))
		norm_L = 1.
	else:
		#print(state[M-1].shape,norm_L.shape)
	# Contracting part of the MPS that won't change anymore with its dual and with previously stored tensors
		if len(state[M-1].shape)==1:
			norm_L = np.einsum("i,i",state[M-1],np.conjugate(state[M-1]))
		elif len(state[M-1].shape)==2:
			if state[M-1].shape[1]>state[M-1].shape[0]:
				if np.isscalar(norm_L):
					norm_L = np.einsum("ik,jk->ij",state[M-1],np.conjugate(state[M-1]))*norm_L
				else:
					norm_L = np.einsum("ik,jk->ij",state[M-1],np.conjugate(state[M-1]))*np.einsum("ii",norm_L)
			else:
				if np.isscalar(norm_L):
					norm_L = np.einsum("ik,ik",state[M-1],np.conjugate(state[M-1]))*norm_L
				else:
					norm_L = np.einsum("ij,ij",np.einsum("ik,ij->kj",state[M-1],np.conjugate(state[M-1])),norm_L)
		else:
			norm_L = np.einsum("kmj,lmj->kl",np.einsum("kmi,ij->kmj",state[M-1],norm_L),np.conjugate(state[M-1]))
        
		# Contracting the system part of the MPS
		if len(state[L+M].shape)==1:
			norm_S = np.dot(state[L+M],np.conjugate(state[L+M]))
		else:
			norm_S = np.einsum("ki,kj->ij",state[L+M],np.conjugate(state[L+M]))
		norm = norm_L

	# Performing the rest of the reservoir contractions from right to left.
		for i in range(0,L):
			#print("state",state[M+i].shape)
			if len(state[M+i].shape)==1:
				norm_past = np.einsum("i,i",state[M+i],np.conjugate(state[M+i]))
				if np.isscalar(norm):
					norm = norm_past*norm
				else:
					norm = norm_past*np.einsum("ii",norm)
				#print("norm",norm.shape)
			elif len(state[M+i].shape)==2:
				if state[M+i].shape[0]>state[M+i].shape[1]:
					norm_past = np.einsum("ji,jk->ik",state[M+i],np.conjugate(state[M+i]))
					if np.isscalar(norm):
						norm = np.einsum("ii",norm_past)*norm
					else:
						norm = np.einsum("ij,ij",norm_past,norm)
				elif state[M+i].shape[0]<state[M+i].shape[1]:
					norm_past = np.einsum("ij,kj->ik",state[M+i],np.conjugate(state[M+i]))
					if np.isscalar(norm):
						norm = norm_past*norm
					else:
						norm = norm_past*np.einsum("ii",norm)
#				print("norm",norm.shape)
			else:
				#print("norm",norm.shape)
				if np.isscalar(norm):
					norm = np.einsum("kmi,lmi->kl",state[M+i],np.conjugate(state[M+i]))*norm
				else:
					norm = np.einsum("kmj,lmj->kl",np.einsum("kmi,ij->kmj",state[M+i],norm),np.conjugate(state[M+i]))
	# Contracting the environment part with the system part
		if len(state[L+M].shape) ==1:
			if np.isscalar(norm):
				norm = norm*norm_S
			else:
				norm = np.einsum("ii",norm)*norm_S
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
	if len(sys.shape)==1:
		obs = np.einsum("i,i",np.einsum("i,ji",sys,observable),np.conjugate(sys))
	else:
		obs = np.einsum("kj,kj",np.einsum("ij,ki->kj",sys,observable),np.conjugate(sys))
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
		legs = len(state.shape)
		if thermal == True:
			n = ((np.linspace(0,(N_env-1)**2-1,(N_env-1)**2)/(N_env-1)).astype(int)+1)[0:(-N_env+1)]
			iend = N_env-1
		else:
			n = np.arange(1,state.shape[0])
			iend = 1
		if legs==1:
			new_state[0:(-iend)] = np.einsum("i,i->i",np.sqrt(n*dt),state[iend:])
		elif legs==2:
			ind = state.shape.index(np.max(state.shape))
			if ind == 0:
				new_state[0:(-iend),:] = np.einsum("i,ij->ij",np.sqrt(n*dt),state[iend:,:])
			elif ind == 1:
				new_state[:,0:(-iend)] = np.einsum("i,ji->ji",np.sqrt(n*dt),state[:,iend:])
		elif legs==3:
			new_state[:,0:(-iend),:] = np.einsum("i,jik->jik",np.sqrt(n*dt),state[:,iend:,:])
		return new_state
	def dBd(state):
		new_state = np.zeros(state.shape,complex)
		legs = len(state.shape)
		if thermal == True:
			n = ((np.linspace(0,(N_env-1)**2-1,(N_env-1)**2)/(N_env-1)).astype(int)+1)[0:(-N_env+1)]
			iend = N_env-1
		else:
			n = np.arange(1,state.shape[0])
			iend = 1
		if legs==1:
			new_state[iend:] = np.einsum("k,k->k",np.sqrt(n*dt),state[0:-iend])
		elif legs==2:
			ind = state.shape.index(np.max(state.shape))
			if ind == 0:
				new_state[iend:,:] = np.einsum("k,kj->kj",np.sqrt(n*dt),state[0:-iend,:])
			elif ind == 0:
				new_state[:,iend:] = np.einsum("k,jk->jk",np.sqrt(n*dt),state[:,0:-iend])
		elif legs==3:
			new_state[:,iend:,:] = np.einsum("k,ikl->ikl",np.sqrt(n*dt),state[:,0:-iend,:])
		return new_state
	sum_B     = np.zeros(freqs.shape,complex)
	if len(states[index].shape)==1:
		sum_B      = np.einsum("l,l",dBd(dB(states[index])),np.conjugate(states[index]))*np.ones(freqs.shape)
		next_step  = np.einsum("l,l",dBd(states[index]),np.conjugate(states[index]))
	elif len(states[index].shape)==2:
		ind = states[index].shape.index(np.max(states[index].shape))
		sum_B      = np.einsum("lk,lk",dBd(dB(states[index])),np.conjugate(states[index]))*np.ones(freqs.shape)
		if ind==0:
			next_step  = np.einsum("lk,lm->km",dBd(states[index]),np.conjugate(states[index]))
		elif ind==1:
			next_step  = np.einsum("kl,kl",dBd(states[index]),np.conjugate(states[index]))
	elif len(states[index].shape)==3:
		sum_B      = np.einsum("ilk,ilk",dBd(dB(states[index])),np.conjugate(states[index]))*np.ones(freqs.shape)
		next_step  = np.einsum("ilk,ilm->km",dBd(states[index]),np.conjugate(states[index]))
	for p in range(1,pmax):
		if len(states[index-p].shape)==1:
			if np.isscalar(next_step):
				sum_B     += (np.einsum("l,l",dB(states[index-p]),np.conjugate(states[index-p]))*next_step*np.exp(1j*freqs*p*dt))
				next_step  = next_step*np.einsum("j,j",states[index-p],np.conjugate(states[index-p]))
			else:
				sum_B     += (np.einsum("ii",next_step)*
								np.einsum("l,l",dB(states[index-p]),np.conjugate(states[index-p]))*np.exp(1j*freqs*p*dt))
				next_step  = np.einsum("ii",next_step)*np.einsum("m,m",states[index-p],np.conjugate(states[index-p]))
		if len(states[index-p].shape)==2:
			indp = states[index-p].shape.index(np.max(states[index-p].shape))
			if indp == 0:
				if np.isscalar(next_step):
					sum_B     += (next_step*np.einsum("lk,lk",dB(states[index-p]),np.conjugate(states[index-p]))*np.exp(1j*freqs*p*dt))
					next_step  = next_step*np.einsum("mk,ml->kl",states[index-p],np.conjugate(states[index-p]))
				else:
					sum_B     += (np.einsum("ii",next_step)*
									np.einsum("lk,lk",dB(states[index-p]),np.conjugate(states[index-p]))*np.exp(1j*freqs*p*dt))
					next_step  = np.einsum("ii",next_step)*np.einsum("mk,lm->kl",states[index-p],
                                                                 np.conjugate(states[index-p]))
			elif indp == 1:
				if np.isscalar(next_step):
					sum_B     += (np.einsum("ii",np.einsum("il,jl->ij",dB(states[index-p]),np.conjugate(states[index-p])))*
									next_step*np.exp(1j*freqs*p*dt))
					next_step  = next_step*np.einsum("ij,ij",states[index-p],np.conjugate(states[index-p]))
				else:
					sum_B     += (np.einsum("ij,ij",np.einsum("il,jl->ij",dB(states[index-p]),np.conjugate(states[index-p])),next_step)*
									np.exp(1j*freqs*p*dt))
					next_step  = np.einsum("kj,jk",np.einsum("ij,ik->kj",next_step,states[index-p]),
											np.conjugate(states[index-p]))
		elif len(states[index-p].shape)==3:
			if np.isscalar(next_step):
				sum_B     += (np.einsum("ilk,ilk",dB(states[index-p]),np.conjugate(states[index-p]))*next_step*np.exp(1j*freqs*p*dt))
				next_step  = next_step*np.einsum("imk,imk",states[index-p],np.conjugate(states[index-p]))
			else:
				sum_B     += (np.einsum("ij,ij",np.einsum("ilk,jlk->ij",dB(states[index-p]),np.conjugate(states[index-p])),next_step)*
								np.exp(1j*freqs*p*dt))
				next_step  = np.einsum("jmk,jml->kl",np.einsum("ij,imk->jmk",next_step,states[index-p]),
										np.conjugate(states[index-p]))
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
			n = np.arange(1,state.shape[0])
			iend = 1
		if legs==1:
			new_state[0:(-iend)] = np.einsum("i,i->i",np.sqrt(n*dt),state[iend:])
		elif legs==2:
			ind = state.shape.index(np.max(state.shape))
			if ind == 0:
				new_state[0:(-iend),:] = np.einsum("i,ij->ij",np.sqrt(n*dt),state[iend:,:])
			elif ind == 1:
				new_state[:,0:(-iend)] = np.einsum("i,ji->ji",np.sqrt(n*dt),state[:,iend:])
		elif legs==3:
			new_state[:,0:(-iend),:] = np.einsum("i,jik->jik",np.sqrt(n*dt),state[:,iend:,:])
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
	return np.real(g2_t),np.real(NB)/dt

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
			n = np.arange(1,state.shape[0])
			iend = 1
		if legs==1:
			new_state[0:(-iend)] = np.einsum("i,i->i",np.sqrt(n*dt),state[iend:])
		elif legs==2:
			ind = state.shape.index(np.max(state.shape))
			if ind == 0:
				new_state[0:(-iend),:] = np.einsum("i,ij->ij",np.sqrt(n*dt),state[iend:,:])
			elif ind == 1:
				new_state[:,0:(-iend)] = np.einsum("i,ji->ji",np.sqrt(n*dt),state[:,iend:])
		elif legs==3:
			new_state[:,0:(-iend),:] = np.einsum("i,jik->jik",np.sqrt(n*dt),state[:,iend:,:])
		return new_state
	g2_out = np.zeros(taumax)
	tau = np.zeros(taumax)
	temp      = dB(dB(states[index]))
	temp2     = dB(states[index])
	if len(states[index].shape)==1:
		g2_out[0] = np.einsum("j,j",temp,np.conjugate(temp))
		next_step = np.einsum("j,j",temp2,np.conjugate(temp2))
	elif len(states[index].shape)==2:
		ind = states[index].shape.index(np.max(states[index].shape))
		if ind==0:
			g2_out[0] = np.einsum("jk,jk",temp, np.conjugate(temp))
			next_step = np.einsum("jk,jl->kl",temp2,np.conjugate(temp2))
		elif ind==1:
			g2_out[0] = np.einsum("kj,kj",temp, np.conjugate(temp))
			next_step = np.einsum("kj,kj",temp2,np.conjugate(temp2))
	elif len(states[index].shape)==3:
			g2_out[0] = np.einsum("ijk,ijk",temp, np.conjugate(temp))
			next_step = np.einsum("ijk,ijl->kl",temp2,np.conjugate(temp2))
	for it in range(1,taumax):
		tau[it] = dt*it
		temp    = dB(states[index-it])
		if len(states[index-it].shape)==1:
			if np.isscalar(next_step):
				g2_out[it] = next_step * np.einsum("j,j",temp,np.conjugate(temp))
				next_step  = next_step * np.einsum("j,j",states[index-it],np.conjugate(states[index-it]))
			else:
				g2_out[it] = np.einsum("ii",next_step) * np.einsum("j,j",temp,np.conjugate(temp))
				next_step  = np.einsum("ii",next_step) * np.einsum("j,j",states[index-it],
                                                                   np.conjugate(states[index-it]))
		if len(states[index-it].shape)==2:
			indp = states[index-it].shape.index(np.max(states[index-it].shape))
			if indp == 0:
				if np.isscalar(next_step):
					g2_out[it] = next_step*np.einsum("jk,jk",temp,np.conjugate(temp))
					next_step = next_step*np.einsum("jk,jl->kl",states[index-it],np.conjugate(states[index-it]))
				else:
					g2_out[it] = np.einsum("ii",next_step)*np.einsum("jk,jk",temp, np.conjugate(temp))
					next_step = np.einsum("ii",next_step)*np.einsum("jk,jl->kl",states[index-it],
																	np.conjugate(states[index-it]))
			elif indp == 1:
				if np.isscalar(next_step):
					g2_out[it] = next_step*np.einsum("kj,kj",temp, np.conjugate(temp))
					next_step = next_step*np.einsum("kj,kj",states[index-it],np.conjugate(states[index-it]))
				else:
					g2_out[it] = np.einsum("ki,ki",next_step,np.einsum("kj,ij->ki",temp, np.conjugate(temp)))
					next_step = np.einsum("ki,ki",next_step,np.einsum("kj,ij->ki",states[index-it],
																		np.conjugate(states[index-it])))
		elif len(states[index-it].shape)==3:
			if np.isscalar(next_step):
				g2_out[it] = next_step*np.einsum("ijk,ijk",temp, np.conjugate(temp))
				next_step = next_step * np.einsum("ijk,ijl->kl",states[index-it],np.conjugate(states[index-it]))
			else:
				g2_out[it] = np.einsum("mn,mn",next_step,np.einsum("mjk,njk->mn",temp, np.conjugate(temp)))
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
			n = np.arange(1,state.shape[0])
			iend = 1
		if legs==1:
			new_state[0:(-iend)] = np.einsum("i,i->i",np.sqrt(n*dt),state[iend:])
		elif legs==2:
			ind = state.shape.index(np.max(state.shape))
			if ind == 0:
				new_state[0:(-iend),:] = np.einsum("i,ij->ij",np.sqrt(n*dt),state[iend:,:])
			elif ind == 1:
				new_state[:,0:(-iend)] = np.einsum("i,ji->ji",np.sqrt(n*dt),state[:,iend:])
		elif legs==3:
			new_state[:,0:(-iend),:] = np.einsum("i,jik->jik",np.sqrt(n*dt),state[:,iend:,:])
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

