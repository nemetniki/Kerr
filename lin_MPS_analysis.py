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

##############################################
### Expectation value of system observable ###
##############################################
def exp_env(observable,state,k):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the combined system state from the norm function, which system
	OUTPUT: expectation value of the observable"""
	
	context = ["I,IJ,J","Ia,IJ,Ja","aI,IJ,aJ","aIb,IJ,aJb"]
	ND = state.ndim
	
	if ND==3:
		obs = contract(context[3],state,observable,np.conjugate(state))
	elif ND==2:
		if k==1:
			obs = contract(context[1],state,observable,np.conjugate(state))
		elif k==2:
			obs = contract(context[2],state,observable,np.conjugate(state))
	elif ND==1:
		obs = contract(context[0],state,observable,np.conjugate(state))

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
	g2_out = np.zeros(taumax,complex)
	tau = np.zeros(taumax,complex)
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

