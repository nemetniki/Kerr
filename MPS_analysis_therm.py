import numpy as np
import scipy as sc
from scipy.linalg import svd
import time
import sys
from decimal import Decimal
from math import factorial
from opt_einsum import contract

#*************************#
#***-------------------***#
#***| ANALYZING TOOLS |***#
#***-------------------***#
#*************************#

############################
### Calculating the norm ###
############################
def normf(M,L,statesB1,statesB2,statesFS,normB1,normB2):
	"""Performing the contractions for the norm
	INPUT: calculated states in list "state" with a delay index-length L 
	and the stored values of norm_L (tensor with indices for state and dual) at timestep M
	OUTPUT: the norm at the present time and the contraction of the past bins which stay constant"""
	
	def B(M,states,norm_past):
		if len(states[M-1].shape)==1:
			norm_past = np.einsum("i,i",states[M-1],np.conjugate(states[M-1]))*norm_past
		elif len(states[M-1].shape)==2:
			if states[M-1].shape[1]>states[M-1].shape[0]:
				if np.isscalar(norm_past) or len(norm_past)==1:
					norm_past = np.einsum("ik,jk->ij",states[M-1],np.conjugate(states[M-1]))*norm_past
				else:
					norm_past = np.einsum("ik,jk->ij",states[M-1],np.conjugate(states[M-1]))*np.einsum("ii",norm_past)
			else:
				if np.isscalar(norm_past) or len(norm_past)==1:
					norm_past = np.einsum("ik,ik",states[M-1],np.conjugate(states[M-1]))*norm_past
				else:
					norm_past = contract("ik,ij,kj",states[M-1],np.conjugate(states[M-1]),norm_past)
		else:
			if np.isscalar(norm_past) or len(norm_past)==1:
				norm_past = np.einsum("kmj,lmj->kl",states[M-1],np.conjugate(states[M-1]))*norm_past
			else:
				norm_past = contract("kmi,ij,lmj->kl",states[M-1],norm_past,np.conjugate(states[M-1]))
		return norm_past
	def F(M,L,states,ind):
		normF = B(ind,states,1.):
		for i in range(1,L-1):
			if len(states[ind+i].shape)==1:
				temp = np.einsum("i,i",state[M+i],np.conjugate(state[M+i]))
				if np.isscalar(normF) or len(normF)==1:
					normF = temp*normF
				else:
					normF = temp*np.einsum("ii",normF)
			elif len(states[ind+i].shape)==2:
				if state[ind+i].shape[0]>state[ind+i].shape[1]:
					temp = np.einsum("ji,jk->ik",states[ind+i],np.conjugate(states[ind+i]))
					if np.isscalar(normF) or len(normF)==1:
						normF = np.einsum("ii",temp)*normF
					else:
						normF = np.einsum("ij,ij",temp,normF)
				elif state[ind+i].shape[0]<state[ind+i].shape[1]:
					temp = np.einsum("ij,kj->ik",states[ind+i],np.conjugate(states[ind+i]))
					if np.isscalar(normF) or len(normF)==1:
						normF = temp*normF
					else:
						normF = temp*np.einsum("ii",normF)
			else:
				if np.isscalar(normF) or len(normF)==1:
					normF = np.einsum("kmi,lmi->kl",states[ind+i],np.conjugate(states[ind+i]))*normF
				else:
					normF = contract("kmi,ij,lmj->kl",states[ind+i],normF,np.conjugate(states[ind+i]))
		return normF

	# Indices of the timebins initially: 0->furthest past, L->system, L+1->first future timebin
	if M==0:
		normS = np.einsum("i,i",statesFS[0],np.conjugate(statesFS[0]))*np.einsum("i,i",statesFS[2],np.conjugate(statesFS[2]))
		normB1 = 1.
		normB2 = 1.
		normF1 = 1.
		normF2 = 1.
		norm = 0. + normS
	else:

	# Contracting part of the MPS that won't change anymore with its dual and with previously stored tensors
		normB1 = B(M,statesB1,normB1)		
		normB2 = B(M,statesB2,normB2)
		normF1 = F(M,L,statesFS,4)		
		normF2 = F(M,L,statesFS,L+3)
        
	# Contracting the system part of the MPS (indices: IvJKtLqrMN)
	# capital letters: physical indices, small letters: link indices, order: F1,S1,B1,F2,S2,B2
	sys_state = contract("opI,wvoJ,tswL,pqrsM->IvJtLqrM",statesFS[1],statesFS[0],statesFS[3],statesFS[2])
	normS = contract("IvJtLqrM,IaJbLcdM->vatbqcrd",sys_state,np.conjugate(sys_state))
	
	norm = contract("ij,kl,mn,op,ijklmnop",normB1,normF2,normB2,normF1,normS)
	
	return np.real(norm),np.real(normB1),np.real(normB1)

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
			n = np.arange(1,N_env)
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
			n = np.arange(1,N_env)
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

