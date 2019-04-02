import numpy as np
import scipy as sc
from scipy.linalg import svd
import time
import sys
from decimal import Decimal
from math import factorial

############################
### Evolution operator U ###
############################
def U(tk,tS,tl,M,gamma_L,gamma_R,dt,phi,Omc,Delc,chipp): #tk: time bin state at k, tS: state of S
	"""Evolution operator up to dt^2
	INPUT: states at the current time (t_k), the delayed time (t_l) and the system state (t_S) at timestep M
	Remember, at M=0 the state is separable
	OUTPUT: combined block of states at t_k, for the system and at t_l"""
    
    #print(tk.shape,tS.shape,tl.shape,M)
    
    ####--------------------------####
    #### Parameters and operators ####
    ####--------------------------####
    
    #####Dimensions of the physical indices#####
	dim_tk = tk.shape[0]
	if len(tS.shape)==1:
		dim_tS = tS.shape[0]
		dim_tl = tl.shape[0]
	else:
		dim_tl = tl.shape[1]
		dim_tS = tS.shape[0]
#    print("dims",dim_tk,dim_tS,dim_tl)

    #####Frequently used operators#####
	def B(state):
		new_state_tk = np.zeros(state.shape,complex)
		new_state_tl = np.zeros(state.shape,complex)
		n = np.arange(1,state.shape[0])
		if legs==3:
			if gamma_L is not 0:
				new_state_tk[0:-1,:,:] = np.einsum("i,ijk->ijk",np.sqrt(n*gamma_L*dt),state[1:,:,:])
			if gamma_R is not 0:
				new_state_tl[:,:,0:-1] = np.einsum("k,ijk->ijk",np.sqrt(n*gamma_R*dt)*np.exp(-1j*phi),state[:,:,1:])
		elif legs==4:
			if gamma_L is not 0:
				new_state_tk[0:-1,:,:,:] = np.einsum("i,ijkl->ijkl",np.sqrt(n*gamma_L*dt),state[1:,:,:,:])
			if gamma_R is not 0:
				new_state_tl[:,:,0:-1,:] = np.einsum("k,ijkl->ijkl",np.sqrt(n*gamma_R*dt)*np.exp(-1j*phi),state[:,:,1:,:])
		return new_state_tk+new_state_tl
	def Bd(state):
		new_state_tk = np.zeros(state.shape,complex)
		new_state_tl = np.zeros(state.shape,complex)
		n = np.arange(1,state.shape[0])
		if legs==3:
			if gamma_L is not 0:
				new_state_tk[1:,:,:] = np.einsum("i,ijk->ijk",np.sqrt(n*gamma_L*dt),state[0:-1,:,:])
			if gamma_R is not 0:
				new_state_tl[:,:,1:] = np.einsum("k,ijk->ijk",np.sqrt(n*gamma_R*dt)*np.exp(-1j*phi),state[:,:,0:-1])
		elif legs==4:
			if gamma_L is not 0:
				new_state_tk[1:,:,:,:] = np.einsum("i,ijkl->ijkl",np.sqrt(n*gamma_L*dt),state[0:-1,:,:,:])
			if gamma_R is not 0:
				new_state_tl[:,:,1:,:] = np.einsum("k,ijkl->ijkl",np.sqrt(n*gamma_R*dt)*np.exp(-1j*phi),state[:,:,0:-1,:])
		return new_state_tk+new_state_tl
	def c(state):
    #        print("c",state[0,2,0], state.shape, len(state.shape))
		new_state = np.zeros(state.shape,complex)
		n = np.arange(1,state.shape[1])
		if legs==3:
			new_state[:,0:-1,:] = np.einsum("j,ijk->ijk",np.sqrt(n),state[:,1:,:])
		elif legs==4:
			new_state[:,0:-1,:,:] = np.einsum("j,ijkl->ijkl",np.sqrt(n),state[:,1:,:,:])
		return new_state
	def cd(state):
    #        print("c",state[0,2,0], state.shape, len(state.shape))
		new_state = np.zeros(state.shape,complex)
		n = np.arange(1,state.shape[1])
		if legs==3:
			new_state[:,1:,:] = np.einsum("j,ijk->ijk",np.sqrt(n),state[:,0:-1,:])
		elif legs==4:
			new_state[:,1:,:,:] = np.einsum("j,ijkl->ijkl",np.sqrt(n),state[:,0:-1,:,:])
		return new_state
	def nc(state,j):
		n = np.arange(0,state.shape[1])+j
		new_state = np.zeros(state.shape,complex)
		if len(state.shape)==3:
			new_state = np.einsum("j,ijk->ijk",n,state)
		elif len(state.shape)==4:
			new_state = np.einsum("j,ijkl->ijkl",n,state)
		return new_state
	def MB(state):
		return Bd(c(state))-B(cd(state))
	def MS(state):
		new_tS = np.zeros(state.shape,complex)
		new_tS += state
		if Delc is 0 and Omc is 0:
			return -1j*dt*chipp*nc(nc(state,0),-1)
		elif Delc is 0 and Omc is not 0:
			return -1j*dt* ( chipp*nc(nc(state,0),-1) + Omc*(c(state)+cd(state)) )
		else:
			return -1j*dt* ( chipp*nc(nc(state,0),-1) + Omc*(c(state)+cd(state)) + Delc*nc(state,0) )
        
    ####----------------------####
    #### Different terms in U ####
    ####----------------------####
    
    #####Initial state#####
	if len(tS.shape)==1:
		initial = np.tensordot(tk,np.tensordot(tS,tl,0),0)
	elif len(tS.shape)==2:
		if len(tl.shape)==3:
			initial = np.tensordot(tk,np.einsum("ij,jkl->ikl",tS,tl),0)
		elif len(tl.shape)==2:
			initial = np.tensordot(tk,np.einsum("ij,jk->ik",tS,tl),0)
	else:
		print("Unusual shape for tS")
#    print("init",initial[0,2,0],initial[1,0,0],initial[0,0,1],initial[0,0,0], initial.shape)
	legs = len(initial.shape)
	
	if gamma_L is not 0 or gamma_R is not 0:
		#####Environment#####
		MBi = np.zeros(initial.shape, complex)
		MBi += initial
		env = np.zeros(MBi.shape,complex)
	#    print("env",env[0,0,0],env[1,0,0],env[0,0,1])
		for i in range(1,5):
			MBi = MB(MBi)/i
			env += MBi
	#        print("env",env[0,0,0],env[1,0,0],env[0,0,1])
		MBi = None
		
		#####System-Environment#####
		sys_env = MS(MB(initial))
		sys_env += (   .5*MB(sys_env) - 1j*0.5*dt*Bd( c(Delc*initial+2*chipp*nc(initial,-1)) + Omc*initial ) -
						1j*0.5*dt*B( cd(Delc*initial+2*chipp*nc(initial,0))  + Omc*initial ) -
						1j*dt/3*Bd(B(Delc*initial+4*chipp*nc(initial,0))) +
						1j/6*(gamma_L+gamma_R)*dt**2*( 2*Delc*nc(initial,0) + 4*chipp*nc(nc(initial,-1),0) +
										Omc*(c(initial)+cd(initial)) )   )
						
    
    #####System#####
	sys = MS(initial+MS(initial)/2.)
    
#    print("sysenv",sys_env[0,2,0])

	return initial + sys + env + sys_env##

