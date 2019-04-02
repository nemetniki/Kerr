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
	norm=0.
#	print(M,norm)
#	if np.isscalar(norm_L):
#		print("value: ",norm_L)
#	else:
#		print("value: ",norm_L, "shape: ",norm_L.shape)
	# Indices of the timebins initially: 0->furthest past, L->system, L+1->first future timebin
	if M==0:
		norm = np.einsum("i,i",state[L],np.conjugate(state[L]))
		norm_L = 1.
	else:
		#print(state[M-1].shape,norm_L.shape)
	# Contracting part of the MPS that won't change anymore with its dual and with previously stored tensors
		if len(state[M-1].shape)==1:
			if np.isscalar(norm_L):
				norm_L = np.einsum("i,i",state[M-1],np.conjugate(state[M-1]))*norm_L
			else:
				norm_L = np.einsum("i,i",state[M-1],np.conjugate(state[M-1]))*np.einsum("ii",norm_L)
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
			if np.isscalar(norm_L):
				norm_L = np.einsum("kmj,lmj->kl",state[M-1],np.conjugate(state[M-1]))*norm_L
			else:
				norm_L = np.einsum("kmj,lmj->kl",np.einsum("kmi,ij->kmj",state[M-1],norm_L),np.conjugate(state[M-1]))
        
		# Contracting the system part of the MPS
		if len(state[L+M].shape)==1:
			norm_S = np.dot(state[L+M],np.conjugate(state[L+M]))
		else:
			norm_S = np.einsum("ki,kj->ij",state[L+M],np.conjugate(state[L+M]))
		norm += norm_L

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
#	print(norm_L)
	return np.real(norm),norm_L

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

##############################################
### Expectation value of system observable ###
##############################################
def exp_sys(observable,sys):
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

###################################################
### Expectation value of environment observable ###
###################################################
def env_dens(env):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the state of the system and timestep M
	OUTPUT: expectation value of the observable"""

	# Indices of the timebins initially: 0->furthest past, L->system, L+1->first future timebin
	if len(env.shape)==1:
		dens = np.einsum("i,j->ij",env,np.conjugate(env))
	elif len(env.shape)==2:
		dens = np.einsum("ki,kj->ij",env,np.conjugate(env))
	else:
		dens = np.einsum("ilk,imk->lm",env,np.conjugate(env))
	return dens

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

#############################
### Cavity density matrix ###
#############################
def cav_dens(sys,N):
	"""Calculating the expectation value of a given system observable
	INPUT: observable of interest, the state of the system and timestep M
	OUTPUT: expectation value of the observable"""

	# Indices of the timebins initially: 0->furthest past, L->system, L+1->first future timebin
	if len(sys.shape)==1:
		dens = np.einsum("i,j->ji",sys,np.conjugate(sys))
	elif len(sys.shape)==2:
		dens = np.einsum("ik,jk->ji",sys,np.conjugate(sys))
	
	cav_dens = np.zeros((N,N),complex)
#	for i in range(N):
#		for j in range(N):
#			cav_dens[i][j] = dens[2*i][2*j]+dens[2*i+1][2*j+1]
#	cav_dens[0][0] = cav_dens[0][0]-1
#	print(cav_dens.shape)
	return cav_dens

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

#######################
### Output spectrum ###
#######################
def spectrum(states,freqs,pmax,N_env,dt,index):#,offset):
#spec = spectrum(states,om,N-L-1,N_env+1,dt,N-L-1)
    """Calculating the expectation value of a given system observable
    INPUT: observable of interest, the state of the system and timestep M
    OUTPUT: expectation value of the observable"""
    corr = np.zeros(pmax,complex)
    dB  = sc.eye(N_env,N_env,1)*np.sqrt(dt*np.arange(0,N_env)) 
    dBd = sc.eye(N_env,N_env,-1)*np.sqrt(dt*np.arange(1,N_env+1)) 
    sum_B     = np.zeros(freqs.shape,complex)
#    first_state = np.einsum("ijk,lj->ilk",states[index],dB)
#    corr[0] = np.einsum("ilk,ilk",np.einsum("ijk,lj->ilk",states[index],np.einsum("ik,kj->ij",dB,dBd)),np.conjugate(states[index]))-offset
#    offset    = np.einsum("ilk,ilk",first_state,np.conjugate(first_state))
#    sum_B = corr[0]*np.ones(freqs.shape)
    next_step  = np.einsum("ilk,ilm->km",np.einsum("ijk,lj->ilk",states[index],dBd),np.conjugate(states[index]))
    for p in range(1,pmax):
        if p==1:
            offset = 0.#(np.einsum("ij,ij",np.einsum("ilk,jlk->ij",np.einsum("imk,lm->ilk",states[index-p],dB),
                       #                                   np.conjugate(states[index-p])),next_step))
        if len(states[index-p].shape)==2:
            corr[p] = 	((np.einsum("ij,ij",np.einsum("il,jl->ij",np.einsum("im,lm->il",states[index-p],dB),
                                                              np.conjugate(states[index-p])),next_step)-offset))
            sum_B     += corr[p]*np.exp(1j*freqs*p*dt)
            next_step  = np.einsum("jk,jk",np.einsum("ij,ik->jk",next_step,states[index-p]),
                                           np.conjugate(states[index-p]))
        elif len(states[index-p].shape)==3:
            corr[p]     += ((np.einsum("ij,ij",np.einsum("ilk,jlk->ij",np.einsum("imk,lm->ilk",states[index-p],dB),
                                                          np.conjugate(states[index-p])),next_step)-offset))
            sum_B     += corr[p]*np.exp(1j*freqs*p*dt)
            next_step  = np.einsum("jmk,jml->kl",np.einsum("ij,imk->jmk",next_step,states[index-p]),
                                       np.conjugate(states[index-p]))
#    return 2./dt*np.real(sum_B)
    return 1./dt*np.abs(sum_B),corr
    
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////#

######################
### g2 correlation ###
######################
def g2_t(state,N_env,dt):
    """Calculating the expectation value of a given system observable
    INPUT: observable of interest, the state of the system and timestep M
    OUTPUT: expectation value of the observable"""
    dB  = sc.eye(N_env,N_env,1)*np.sqrt(dt*np.arange(0,N_env)) 
#    dBd = sc.eye(N_env,N_env,-1)*np.sqrt(dt*np.arange(1,N_env+1))
    if len(state.shape)==1:       
        temp = np.einsum("ik,k",np.einsum("ij,jk",dB,dB),state)
        temp2 = np.einsum("ik,k",dB,state)
        NB = np.einsum("i,i",temp2,np.conjugate(temp2))
        g2_t = np.einsum("i,i",temp,np.conjugate(temp))/(NB**2)
        temp = None
    elif len(state.shape)==2:       
        indp = state.shape.index(np.max(state.shape))
        temp = None        
        if indp == 0:
            temp = np.einsum("ik,kl->il",np.einsum("ij,jk",dB,dB),state)
            temp2 = np.einsum("ik,kl->il",dB,state)
            NB = np.einsum("il,il",temp2,np.conjugate(temp2))
            g2_t = np.einsum("il,il",temp,np.conjugate(temp))/(NB**2)
            temp = None        
        elif indp == 1:
            temp = np.einsum("ik,lk->li",np.einsum("ij,jk",dB,dB),state)
            temp2 = np.einsum("ik,lk->li",dB,state)
            NB = np.einsum("li,li",temp2,np.conjugate(temp2))
            g2_t = np.einsum("li,li",temp,np.conjugate(temp))/(NB**2)
            temp = None        
    elif len(state.shape)==3:
        temp = np.einsum("ik,jkl->jil",np.einsum("ij,jk",dB,dB),state)
        temp2 = np.einsum("ik,jkl->jil",dB,state)
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
def g2_out(states,taumax,N_env,dt,index):
    """Calculating the expectation value of a given system observable
    INPUT: observable of interest, the state of the system and timestep M
    OUTPUT: expectation value of the observable"""
    dB  = sc.eye(N_env,N_env,1)*np.sqrt(dt*np.arange(0,N_env))
    g2_out = np.zeros(taumax)
    tau = np.zeros(taumax)
    temp      = np.einsum("ilk,jl->ijk",states[index],np.einsum("jf,fl->jl",dB,dB))
    denom = np.einsum("ilk,jl->ijk",states[index],dB)
    g2_out[0] = np.einsum("ijk,ijk",temp, np.conjugate(temp))/(np.einsum("ijk,ijk",denom,np.conjugate(denom))**2)
    temp2     = np.einsum("ilk,jl->ijk",states[index],dB)
    next_step = np.einsum("ijk,ijl->kl",temp2,np.conjugate(temp2))/(np.einsum("ijk,ijk",temp2,np.conjugate(temp2))**2)
    for it in range(1,taumax):
        tau[it] = dt*it
        if len(states[index-it].shape)==2:
            temp      = np.einsum("kl,jl->kj",states[index-it],dB)
            g2_out[it] = np.einsum("ki,ki",next_step,np.einsum("kj,ij->ki",temp, np.conjugate(temp)))
            next_step = np.einsum("ki,ki",next_step,np.einsum("kj,ij->ki",states[index-it],
                                                                      np.conjugate(states[index-it])))
        elif len(states[index-it].shape)==3:
            temp      = np.einsum("ilk,jl->ijk",states[index-it],dB)
            g2_out[it] = np.einsum("mn,mn",next_step,np.einsum("mjk,njk->mn",temp, np.conjugate(temp)))
            next_step = np.einsum("mjk,mjl->kl",np.einsum("ijk,im->mjk",states[index-it],next_step),
                                      np.conjugate(states[index-it]))
    return np.real(tau),np.real(g2_out)

################################
### Number of output photons ###
################################
def NB_out(state,N_env,NB_past,dt):
    """Calculating the expectation value of a given system observable
    INPUT: observable of interest, the state of the system and timestep M
    OUTPUT: expectation value of the observable"""
    dB  = sc.eye(N_env,N_env,1)*np.sqrt(dt*np.arange(0,N_env)) 
#    dBd = sc.eye(N_env,N_env,-1)*np.sqrt(dt*np.arange(1,N_env+1))
    if len(state.shape)==1:       
        temp = np.einsum("ik,k",dB,state)
        NB_now = np.einsum("i,i",temp,np.conjugate(temp))
    elif len(state.shape)==2:       
        indp = state.shape.index(np.max(state.shape))
        if indp == 0:
            temp = np.einsum("ik,kl->il",dB,state)
            NB_now = np.einsum("il,il",temp,np.conjugate(temp))
        elif indp == 1:
            temp = np.einsum("ik,lk->li",dB,state)
            NB_now = np.einsum("li,li",temp,np.conjugate(temp))
    elif len(state.shape)==3:
        temp = np.einsum("ik,jkl->jil",dB,state)
        NB_now = np.einsum("jil,jil",temp,np.conjugate(temp))
    NB = NB_now+NB_past
    temp = None
    NB_now = None
    return np.real(NB)

