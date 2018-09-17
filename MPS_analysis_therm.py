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
	
	def B(M,states,norm_past):
	'''Performs the contractions in one Markovian reservoir up to the present bin by contracting with past contributions.
	INPUT: Time step, Markovian reservoir list, previously saved contributions.
	OUTPUT: Present contribution to the norm.'''
		if len(states[M-1].shape)==1:
			norm_past = es("i,i",states[M-1],np.conjugate(states[M-1]))*norm_past
		elif len(states[M-1].shape)==2:
			if states[M-1].shape[1]>states[M-1].shape[0]:
				if np.isscalar(norm_past) or len(norm_past)==1:
					norm_past = es("ik,jk->ij",states[M-1],np.conjugate(states[M-1]))*norm_past
				else:
					norm_past = es("ik,jk->ij",states[M-1],np.conjugate(states[M-1]))*es("ii",norm_past)
			else:
				if np.isscalar(norm_past) or len(norm_past)==1:
					norm_past = es("ik,ik",states[M-1],np.conjugate(states[M-1]))*norm_past
				else:
					norm_past = contract("ik,ij,kj",states[M-1],np.conjugate(states[M-1]),norm_past)
		else:
			if np.isscalar(norm_past) or len(norm_past)==1:
				norm_past = es("kmj,lmj->kl",states[M-1],np.conjugate(states[M-1]))*norm_past
			else:
				norm_past = contract("kmi,ij,lmj->kl",states[M-1],norm_past,np.conjugate(states[M-1]))
		return norm_past
		
	def F(M,L,states,which):
	'''Performs the contractions in the fibre reservoir that is not affected by the present interaction 
	by contracting with past contributions.
	INPUT: Time step, length of the delay, fibre reservoir list, index that specifies which end
	OUTPUT: Present contribution to the norm.'''
		normF   = 1.
		context = ["imk,iml->kl","kmi,lmi->kl","imk,ij,jml->kl","kmi,ij,lmj->kl"]
		for i in range(0,L-2):
			index = abs(-(which-1)*(2*L-1)+i)
			if len(states[index].shape)==1:
				temp = es("i,i",state[index],np.conjugate(state[index]))
				if np.isscalar(normF) or len(normF)==1:
					normF = temp*normF
				else:
					normF = temp*es("ii",normF)
			elif len(states[index].shape)==2:
				if state[index].shape[which-1]<state[index].shape[which%2]:
					temp = es("ji,jk->ik",states[index],np.conjugate(states[index]))
					if np.isscalar(normF) or len(normF)==1:
						normF = es("ii",temp)*normF
					else:
						normF = es("ij,ij",temp,normF)
				elif state[index].shape[which-1]>state[index].shape[which%2]:
					temp = es("ij,kj->ik",states[ind+i],np.conjugate(states[ind+i]))
					if np.isscalar(normF) or len(normF)==1:
						normF = temp*normF
					else:
						normF = temp*es("ii",normF)
			else:
				if np.isscalar(normF) or len(normF)==1:
					normF = es(context[which-1],states[index],np.conjugate(states[index]))*normF
				else:
					normF = contract(context[which+1],states[index],normF,np.conjugate(states[index]))
		return normF

	#Putting everything together. In the first time step there are no link indices, the states are normalized.
	if M==0:
		sys_state = es("i,j->ij",statesS[0],statesS[1])
		normS = es("ij,ij",sys_state,np.conjugate(sys_state))
		normB1 = 1.
		normB2 = 1.
		normF1 = 1.
		normF2 = 1.
		norm = 0. + normS
	else:
	#In other time steps we can calculate the current contributions by using the functions above
		normB1 = B(M,statesB1,normB1)		
		normB2 = B(M,statesB2,normB2)
		normF1 = F(M,L,statesF,1) #F states on the upper branch
		normF2 = F(M,L,statesF,2) #F states on the lower branch
        
		# Contracting the system part of the MPS 
		# capital letters: physical indices, small letters: link indices
		# order for systemFS: S1,F1,S2,F2 -> final: F1,S1,(B1),F2,S2,(B2)
		# indices: I(F1 phys)v(B1 past link)J(S1 phys)K(B1 phys)t(F lower link)
		#	   L(F2 phys)q(F upper link)r(B2 past link)M(S2 phys)N(B2 phys)
		sys_state = contract("opI,wvoJ,tswL,pqrsM->IvJtLqrM",statesF[L-1],statesS[0],statesF[L],statesS[1])
		normS = contract("IvJtLqrM,IaJbLcdM->vatbqcrd",sys_state,np.conjugate(sys_state))
		
		norm = contract("ij,kl,mn,op,ijklmnop",normB1,normF2,normB2,normF1,normS)
	
	return np.real(norm),np.real(normB1),np.real(normB2),sys_state

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
	
	context = ["IvJtLqrM,JK,IvKtLqrM","IvJtLqrM,MK,IvJtLqrK","JM,JK,KM","JM,MK,JK"]
	#Links between bins
	if len(sys_state.shape)>2:
		obs = contract(context[which-1],sys_state,observable,np.conjugate(sys_state))
	#Independent system bins
	else:
		obs = contract(context[which+1],sys_state,observable,np.conjugate(sys_state))
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
			new_state[0:(-iend)] = es("i,i->i",np.sqrt(n*dt),state[iend:])
		elif legs==2:
			ind = state.shape.index(np.max(state.shape))
			if ind == 0:
				new_state[0:(-iend),:] = es("i,ij->ij",np.sqrt(n*dt),state[iend:,:])
			elif ind == 1:
				new_state[:,0:(-iend)] = es("i,ji->ji",np.sqrt(n*dt),state[:,iend:])
		elif legs==3:
			new_state[:,0:(-iend),:] = es("i,jik->jik",np.sqrt(n*dt),state[:,iend:,:])
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
			new_state[iend:] = es("k,k->k",np.sqrt(n*dt),state[0:-iend])
		elif legs==2:
			ind = state.shape.index(np.max(state.shape))
			if ind == 0:
				new_state[iend:,:] = es("k,kj->kj",np.sqrt(n*dt),state[0:-iend,:])
			elif ind == 0:
				new_state[:,iend:] = es("k,jk->jk",np.sqrt(n*dt),state[:,0:-iend])
		elif legs==3:
			new_state[:,iend:,:] = es("k,ikl->ikl",np.sqrt(n*dt),state[:,0:-iend,:])
		return new_state
	sum_B     = np.zeros(freqs.shape,complex)
	if len(states[index].shape)==1:
		sum_B      = es("l,l",dBd(dB(states[index])),np.conjugate(states[index]))*np.ones(freqs.shape)
		next_step  = es("l,l",dBd(states[index]),np.conjugate(states[index]))
	elif len(states[index].shape)==2:
		ind = states[index].shape.index(np.max(states[index].shape))
		sum_B      = es("lk,lk",dBd(dB(states[index])),np.conjugate(states[index]))*np.ones(freqs.shape)
		if ind==0:
			next_step  = es("lk,lm->km",dBd(states[index]),np.conjugate(states[index]))
		elif ind==1:
			next_step  = es("kl,kl",dBd(states[index]),np.conjugate(states[index]))
	elif len(states[index].shape)==3:
		sum_B      = es("ilk,ilk",dBd(dB(states[index])),np.conjugate(states[index]))*np.ones(freqs.shape)
		next_step  = es("ilk,ilm->km",dBd(states[index]),np.conjugate(states[index]))
	for p in range(1,pmax):
		if len(states[index-p].shape)==1:
			if np.isscalar(next_step):
				sum_B     += (es("l,l",dB(states[index-p]),np.conjugate(states[index-p]))*next_step*np.exp(1j*freqs*p*dt))
				next_step  = next_step*es("j,j",states[index-p],np.conjugate(states[index-p]))
			else:
				sum_B     += (es("ii",next_step)*
								es("l,l",dB(states[index-p]),np.conjugate(states[index-p]))*np.exp(1j*freqs*p*dt))
				next_step  = es("ii",next_step)*es("m,m",states[index-p],np.conjugate(states[index-p]))
		if len(states[index-p].shape)==2:
			indp = states[index-p].shape.index(np.max(states[index-p].shape))
			if indp == 0:
				if np.isscalar(next_step):
					sum_B     += (next_step*es("lk,lk",dB(states[index-p]),np.conjugate(states[index-p]))*np.exp(1j*freqs*p*dt))
					next_step  = next_step*es("mk,ml->kl",states[index-p],np.conjugate(states[index-p]))
				else:
					sum_B     += (es("ii",next_step)*
									es("lk,lk",dB(states[index-p]),np.conjugate(states[index-p]))*np.exp(1j*freqs*p*dt))
					next_step  = es("ii",next_step)*es("mk,lm->kl",states[index-p],
                                                                 np.conjugate(states[index-p]))
			elif indp == 1:
				if np.isscalar(next_step):
					sum_B     += (es("ii",es("il,jl->ij",dB(states[index-p]),np.conjugate(states[index-p])))*
									next_step*np.exp(1j*freqs*p*dt))
					next_step  = next_step*es("ij,ij",states[index-p],np.conjugate(states[index-p]))
				else:
					sum_B     += (es("ij,ij",es("il,jl->ij",dB(states[index-p]),np.conjugate(states[index-p])),next_step)*
									np.exp(1j*freqs*p*dt))
					next_step  = es("kj,jk",es("ij,ik->kj",next_step,states[index-p]),
											np.conjugate(states[index-p]))
		elif len(states[index-p].shape)==3:
			if np.isscalar(next_step):
				sum_B     += (es("ilk,ilk",dB(states[index-p]),np.conjugate(states[index-p]))*next_step*np.exp(1j*freqs*p*dt))
				next_step  = next_step*es("imk,imk",states[index-p],np.conjugate(states[index-p]))
			else:
				sum_B     += (es("ij,ij",es("ilk,jlk->ij",dB(states[index-p]),np.conjugate(states[index-p])),next_step)*
								np.exp(1j*freqs*p*dt))
				next_step  = es("jmk,jml->kl",es("ij,imk->jmk",next_step,states[index-p]),
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
			new_state[0:(-iend)] = es("i,i->i",np.sqrt(n),state[iend:])
		elif legs==2:
			ind = state.shape.index(np.max(state.shape))
			if ind == 0:
				new_state[0:(-iend),:] = es("i,ij->ij",np.sqrt(n),state[iend:,:])
			elif ind == 1:
				new_state[:,0:(-iend)] = es("i,ji->ji",np.sqrt(n),state[:,iend:])
		elif legs==3:
			new_state[:,0:(-iend),:] = es("i,jik->jik",np.sqrt(n),state[:,iend:,:])
		return new_state
#    dBd = sc.eye(N_env,N_env,-1)*np.sqrt(dt*np.arange(1,N_env+1))
	temp = dB(dB(state))
	temp2 = dB(state)
	if len(state.shape)==1:       
		NB = es("i,i",temp2,np.conjugate(temp2))
		g2_t = es("i,i",temp,np.conjugate(temp))/(NB**2)
		temp = None
	elif len(state.shape)==2:       
		indp = state.shape.index(np.max(state.shape))
		NB = es("il,il",temp2,np.conjugate(temp2))
		g2_t = es("il,il",temp,np.conjugate(temp))/(NB**2)
		temp = None        
	elif len(state.shape)==3:
		NB = es("jil,jil",temp2,np.conjugate(temp2))
		g2_t = es("jil,jil",temp,np.conjugate(temp))/(NB**2)
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
			new_state[0:(-iend)] = es("i,i->i",np.sqrt(n),state[iend:])
		elif legs==2:
			ind = state.shape.index(np.max(state.shape))
			if ind == 0:
				new_state[0:(-iend),:] = es("i,ij->ij",np.sqrt(n),state[iend:,:])
			elif ind == 1:
				new_state[:,0:(-iend)] = es("i,ji->ji",np.sqrt(n),state[:,iend:])
		elif legs==3:
			new_state[:,0:(-iend),:] = es("i,jik->jik",np.sqrt(n),state[:,iend:,:])
		return new_state
	g2_out = np.zeros(taumax)
	tau = np.zeros(taumax)
	temp      = dB(dB(states[index]))
	temp2     = dB(states[index])
	if len(states[index].shape)==1:
		next_step = es("j,j",temp2,np.conjugate(temp2))
		NB = next_step
		g2_out[0] = es("j,j",temp,np.conjugate(temp))/(NB**2)
	elif len(states[index].shape)==2:
		ind = states[index].shape.index(np.max(states[index].shape))
		NB = es("jk,jk",temp2, np.conjugate(temp2))
		if ind==0:
			g2_out[0] = es("jk,jk",temp, np.conjugate(temp))/(NB**2)
			next_step = es("jk,jl->kl",temp2,np.conjugate(temp2))
		elif ind==1:
			g2_out[0] = es("kj,kj",temp, np.conjugate(temp))/(NB**2)
			next_step = es("kj,kj",temp2,np.conjugate(temp2))
	elif len(states[index].shape)==3:
		NB = es("ijk,ijk",temp2, np.conjugate(temp2))
		g2_out[0] = es("ijk,ijk",temp, np.conjugate(temp))/(NB**2)
		next_step = es("ijk,ijl->kl",temp2,np.conjugate(temp2))
	for it in range(1,taumax):
		tau[it] = dt*it
		temp    = dB(states[index-it])
		if len(states[index-it].shape)==1:
			if np.isscalar(next_step):
				g2_out[it] = next_step * es("j,j",temp,np.conjugate(temp))/(NB**2)
				next_step  = next_step * es("j,j",states[index-it],np.conjugate(states[index-it]))
			else:
				g2_out[it] = es("ii",next_step) * es("j,j",temp,np.conjugate(temp))/(NB**2)
				next_step  = es("ii",next_step) * es("j,j",states[index-it],
                                                                   np.conjugate(states[index-it]))
		if len(states[index-it].shape)==2:
			indp = states[index-it].shape.index(np.max(states[index-it].shape))
			if indp == 0:
				if np.isscalar(next_step):
					g2_out[it] = next_step*es("jk,jk",temp,np.conjugate(temp))/(NB**2)
					next_step = next_step*es("jk,jl->kl",states[index-it],np.conjugate(states[index-it]))
				else:
					g2_out[it] = es("ii",next_step)*es("jk,jk",temp, np.conjugate(temp))/(NB**2)
					next_step = es("ii",next_step)*es("jk,jl->kl",states[index-it],
																	np.conjugate(states[index-it]))
			elif indp == 1:
				if np.isscalar(next_step):
					g2_out[it] = next_step*es("kj,kj",temp, np.conjugate(temp))/(NB**2)
					next_step = next_step*es("kj,kj",states[index-it],np.conjugate(states[index-it]))
				else:
					g2_out[it] = es("ki,ki",next_step,es("kj,ij->ki",temp, np.conjugate(temp)))/(NB**2)
					next_step = es("ki,ki",next_step,es("kj,ij->ki",states[index-it],
																		np.conjugate(states[index-it])))
		elif len(states[index-it].shape)==3:
			if np.isscalar(next_step):
				g2_out[it] = next_step*es("ijk,ijk",temp, np.conjugate(temp))/(NB**2)
				next_step = next_step * es("ijk,ijl->kl",states[index-it],np.conjugate(states[index-it]))
			else:
				g2_out[it] = es("mn,mn",next_step,es("mjk,njk->mn",temp, np.conjugate(temp)))/(NB**2)
				next_step = es("mjk,mjl->kl",es("ijk,im->mjk",states[index-it],next_step),
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
			new_state[0:(-iend)] = es("i,i->i",np.sqrt(n),state[iend:])
		elif legs==2:
			ind = state.shape.index(np.max(state.shape))
			if ind == 0:
				new_state[0:(-iend),:] = es("i,ij->ij",np.sqrt(n),state[iend:,:])
			elif ind == 1:
				new_state[:,0:(-iend)] = es("i,ji->ji",np.sqrt(n),state[:,iend:])
		elif legs==3:
			new_state[:,0:(-iend),:] = es("i,jik->jik",np.sqrt(n),state[:,iend:,:])
		return new_state
	temp = dB(state)
	if len(state.shape)==1:       
		NB_now = es("i,i",temp,np.conjugate(temp))
	elif len(state.shape)==2:       
		indp = state.shape.index(np.max(state.shape))
		NB_now = es("il,il",temp,np.conjugate(temp))
	elif len(state.shape)==3:
		NB_now = es("jil,jil",temp,np.conjugate(temp))
	NB = NB_now+NB_past
	temp = None
	NB_now = None
	return np.real(NB)

