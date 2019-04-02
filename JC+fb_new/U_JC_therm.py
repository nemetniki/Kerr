import numpy as np
import scipy as sc
from scipy.linalg import svd
import time
import sys
from decimal import Decimal
from math import factorial
import opt_einsum as oe

############################
### Evolution operator U ###
############################
#@profile
def U(tk,tS,tl,N_env,M,gamma_L,gamma_R,dt,phi,Ome,Omc,g,Delc,Dele,thermal): #tk: time bin state at k, tS: state of S
	"""Evolution operator up to dt^2
	INPUT: states at the current time (t_k), the delayed time (t_l) and the system state (t_S) at timestep M
	Remember, at M=0 the state is separable
	OUTPUT: combined block of states at t_k, for the system and at t_l"""
    
    #print(tk.shape,tS.shape,tl.shape,M)
    
    ####--------------------------####
    #### Parameters and operators ####
    ####--------------------------####
    
    #####Dimensions of the physical indices#####
	dim_tk,dim_tS,dim_tl = tk.shape[0],tS.shape[0],tl.shape[1]
#	print("dims",dim_tk,dim_tS,dim_tl)

    #####Frequently used operators#####
	#@profile
	def B(state):
		if thermal == False:
			n = np.arange(1,state.shape[0]+1)
			ind = 1
		elif thermal == True:
			n = (np.arange(N_env,N_env*(N_env+1))/N_env).astype(np.int64)
			ind = N_env
		if gamma_L > 0:
			new_state_tk = np.zeros(state.shape,complex)
			new_state_tk[0:-ind,:,:,:] = np.einsum("i,ijkl->ijkl",np.sqrt(n[0:-ind]*gamma_L*dt),state[ind:,:,:,:])
		if gamma_R > 0:
			new_state_tl = np.zeros(state.shape,complex)
			new_state_tl[:,:,0:-ind,:] = np.einsum("k,ijkl->ijkl",np.sqrt(n[0:-ind]*gamma_R*dt)*np.exp(-1j*phi),state[:,:,ind:,:])
		return new_state_tk+new_state_tl

	#@profile
	def Bd(state):
		if thermal == False:
			n = np.arange(1,state.shape[0]+1)
			ind = 1
		elif thermal == True:
			n = (np.arange(N_env,N_env*(N_env+1))/N_env).astype(np.int64)
			ind = N_env
		if gamma_L > 0:
			new_state_tk = np.zeros(state.shape,complex)
			new_state_tk[ind:,:,:,:] = np.einsum("i,ijkl->ijkl",np.sqrt(n[0:-ind]*gamma_L*dt),state[0:-ind,:,:,:])
		if gamma_R > 0:
			new_state_tl = np.zeros(state.shape,complex)
			new_state_tl[:,:,ind:,:] = np.einsum("k,ijkl->ijkl",np.sqrt(n[0:-ind]*gamma_R*dt)*np.exp(1j*phi),state[:,:,0:-ind,:])
		return new_state_tk+new_state_tl

	#@profile
	def c(state):
		new_state = np.zeros(state.shape,complex)
		n=(np.arange(2,dim_tS)/2).astype(np.int64)
		new_state[:,0:-2,:,:] = np.einsum("j,ijkl->ijkl",np.sqrt(n),state[:,2:,:,:])
		return new_state
		
	#@profile
	def cd(state):
		new_state = np.zeros(state.shape,complex)
		n=(np.arange(2,dim_tS)/2).astype(np.int64)
		new_state[:,2:,:,:] = np.einsum("j,ijkl->ijkl",np.sqrt(n),state[:,0:-2,:,:])
		return new_state
		
	#@profile
	def sm(state):
		new_state = np.zeros(state.shape,complex)
		new_state[:,0:dim_tS-1:2,:,:] = state[:,1:dim_tS:2,:,:]
		return new_state

	#@profile
	def sp(state):
		new_state = np.zeros(state.shape,complex)
		new_state[:,1:dim_tS:2,:,:] = state[:,0:dim_tS-1:2,:,:]
		return new_state

	#@profile
	def JC(state):
		new_tS_g = np.zeros(state.shape,complex)
		new_tS_Ome = np.zeros(state.shape,complex)
		n=(np.arange(2,dim_tS)/2).astype(np.int64)
		if g is not 0:
			new_tS_g[:,1:dim_tS-1:2,:,:]   = np.einsum("j,ijkl->ijkl",np.sqrt(n[0:dim_tS-1:2])*g,state[:,2:dim_tS:2,:,:])
			new_tS_g[:,2:dim_tS:2,:,:]   = np.einsum("j,ijkl->ijkl",np.sqrt(n[0:dim_tS-1:2])*g,state[:,1:dim_tS-1:2,:,:])
		if Ome is not 0:
			new_tS_Ome[:,0:dim_tS-2:2,:,:] = Ome*state[:,1:dim_tS-1:2,:,:]
			new_tS_Ome[:,1:dim_tS-1:2,:,:] = Ome*state[:,0:dim_tS-2:2,:,:]
		return new_tS_g+new_tS_Ome
		
	#@profile
	def nc(state):
		n=(np.arange(0,dim_tS)/2).astype(np.int64)
		new_state = np.zeros(state.shape,complex)
		new_state = np.einsum("j,ijkl->ijkl",n,state)
		return new_state
		
	#@profile
	def C(state):
		if Delc == 0. and Omc == 0.:
			return np.zeros(state.shape,complex)
		elif Delc == 0. and Omc > 0.:
			return Omc*(c(state)+cd(state))
		elif (Delc > 0. or Delc < 0.) and Omc == 0.:
			return Delc*nc(state)
		else:
			return Delc*nc(state)+Omc*(c(state)+cd(state))
	
	#@profile
	def MB(state):
		return Bd(c(state))-B(cd(state))
	
	#@profile
	def MS(state):
		new_tS = np.zeros(state.shape,complex)
		new_tS += state
		if Dele == 0.:
			return -1j*dt*(C(state)+JC(state))
		else:
			new_tS[:,0:dim_tS:2,:,:] = 0
			return -1j*dt*(C(state)+JC(state)+Dele*new_tS)
        
    ####----------------------####
    #### Different terms in U ####
    ####----------------------####
    
    #####Initial state#####
#	initial = np.einsum("i,kl,lmn->ikmn",tk,tS,tl,optimize=True)
	initial = np.einsum("ikl,lmn->ikmn",np.tensordot(tk,tS,axes=0),tl)
#    print("init",initial[0,2,0],initial[1,0,0],initial[0,0,1],initial[0,0,0], initial.shape)
	
	if gamma_L > 0. or gamma_R > 0.:
		#####Environment#####
		MBi = np.zeros(initial.shape, complex)
		MBi += initial
		env = np.zeros(MBi.shape,complex)
		for i in range(1,5):
			MBi = MB(MBi)/i
			env += MBi
		MBi = None
		
		#####System-Environment#####
		sys_env = MS(MB(initial))
		sys_env += (.5*(MB(sys_env)) -
					0.5j*dt*( Bd(Delc*c(initial)+Omc*initial+g*sm(initial)) + 
								B(Delc*cd(initial)+Omc*initial+g*sp(initial))-
								(gamma_L+gamma_R)/3.*dt*(C(initial)+g*(c(sp(initial))+cd(sm(initial)))+
								Delc*nc(initial)) + 2/3.*Delc*Bd(B(initial))))
    
    #####System#####
	sys = MS(initial+MS(initial)/2.)
    
#    print("sysenv",sys_env[0,2,0])

	return initial + sys + env + sys_env##