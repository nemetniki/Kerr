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
def U(M,tF1,tF2,tS1,tB1,tB2,tS2,gamma_B1,gamma_B2,gamma_F1,gamma_F2,dt,phi,Ome1,Ome2,Omc1,Omc2,g1,g2,Delc1,Delc2,Dele): #tk: time bin state at k, tS: state of S1
	"""Evolution operator up to dt^2
	INPUT: states at the current time (t_k), the delayed time (t_l) and the system state (t_S) at timestep M
	Remember, at M=0 the state is separable
	OUTPUT: combined block of states at t_k, for the system and at t_l"""
    
    #print(tk.shape,tS.shape,tl.shape,M)
    
    ####--------------------------####
    #### Parameters and operators ####
    ####--------------------------####
    
    #####Dimensions of the physical indices#####
	dim_tB = tB1.shape[0]
	if M==0:
		dim_tS = tS1.shape[0]
		dim_tF = tF1.shape[0]
	else:
		dim_tS = tS1.shape[1]
		dim_tF = tF1.shape[1]
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
	def a(state,which):
		new_state = np.zeros(state.shape,complex)
		n=np.linspace(1,int((dim_tS)/2),dim_tS-2).astype(np.int64)
		if which == 1:
			if M==0:
				new_state[:,:,0:-2,:,:,:] = np.einsum("j,mijknp->mijknp",np.sqrt(n),state[:,:,2:,:,:,:])
			else:
				new_state[:,:,0:-2,:,:,:,:,:] = np.einsum("j,mijklnop->mijklnop",np.sqrt(n),state[:,:,2:,:,:,:,:,:])
		elif which == 2:
			if M==0:
				new_state[:,:,:,0:-2,:,:] = np.einsum("j,mikjnp->mikjnp",np.sqrt(n),state[:,:,:,2:,:,:])
			else:
				new_state[:,:,:,:,0:-2,:,:,:] = np.einsum("j,mikljnop->mikljnop",np.sqrt(n),state[:,:,:,:,2:,:,:,:])
		return new_state
	def ad(state,which):
		new_state = np.zeros(state.shape,complex)
		n=np.linspace(1,int((dim_tS)/2),dim_tS-2).astype(np.int64)
		if which == 1:
			if M==0:
				new_state[:,:,2:,:,:,:] = np.einsum("j,mijknp->mijknp",np.sqrt(n),state[:,:,0:-2,:,:,:])
			else:
				new_state[:,:,2:,:,:,:,:,:] = np.einsum("j,mijklnop->mijklnop",np.sqrt(n),state[:,:,0:-2,:,:,:,:,:])
		elif which == 2:
			if M==0:
				new_state[:,:,:,2:,:,:] = np.einsum("j,mikjnp->mikjnp",np.sqrt(n),state[:,:,:,0:-2,:,:])
			else:
				new_state[:,:,:,:,2:,:,:,:] = np.einsum("j,mikljnop->mikljnop",np.sqrt(n),state[:,:,:,:,0:-2,:,:,:])
		return new_state
	def sm(state,which):
		new_state = np.zeros(state.shape,complex)
		if which == 1:
			if M == 0:
				new_state[:,:,0:dim_tS-1:2,:,:,:] = state[:,:,1:dim_tS:2,:,:,:]
			else:
				new_state[:,:,0:dim_tS-1:2,:,:,:,:,:] = state[:,:,1:dim_tS:2,:,:,:,:,:]
		elif which == 2:
			if M == 0:
				new_state[:,:,:,0:dim_tS-1:2,:,:] = state[:,:,:,1:dim_tS:2,:,:]
			else:
				new_state[:,:,:,:,0:dim_tS-1:2,:,:,:] = state[:,:,:,:,1:dim_tS:2,:,:,:]
		return new_state
	def sp(state,which):
		new_state = np.zeros(state.shape,complex)
		if which == 1:
			if M == 0:
				new_state[:,:,1:dim_tS:2,:,:,:] = state[:,:,0:dim_tS-1:2,:,:,:]
			else:
				new_state[:,:,1:dim_tS:2,:,:,:,:,:] = state[:,:,0:dim_tS-1:2,:,:,:,:,:]
		elif which == 2:
			if M == 0:
				new_state[:,:,:,1:dim_tS:2,:,:] = state[:,:,:,0:dim_tS-1:2,:,:]
			else:
				new_state[:,:,:,:,1:dim_tS:2,:,:,:] = state[:,:,:,:,0:dim_tS-1:2,:,:,:]
		return new_state
	def JC(state,which):
		new_tS_g = np.zeros(state.shape,complex)
		new_tS_Ome = np.zeros(state.shape,complex)
		n=np.linspace(1,int((dim_tS)/2),dim_tS-2).astype(np.int64)
		if which == 1:
			if M == 0:
				if g1 is not 0:
					new_tS_g[:,:,1:dim_tS-1:2,:,:,:]   = np.einsum("j,iljkmn->iljkmn",np.sqrt(n[0:dim_tS-1:2])*g1,state[:,:,2:dim_tS:2,:,:,:])
					new_tS_g[:,:,2:dim_tS:2,:,:,:]   = np.einsum("j,iljkmn->iljkmn",np.sqrt(n[0:dim_tS-1:2])*g1,state[:,:,1:dim_tS-1:2,:,:,:])
				if Ome1 is not 0:
					new_tS_Ome[:,:,0:dim_tS-2:2,:,:,:] = Ome1*state[:,:,1:dim_tS-1:2,:,:,:]
					new_tS_Ome[:,:,1:dim_tS-1:2,:,:,:] = Ome1*state[:,:,0:dim_tS-2:2,:,:,:]
			else:
				if g1 is not 0:
					new_tS_g[:,:,1:dim_tS-1:2,:,:,:,:,:]   = np.einsum("j,imjklnop->imjklnop",np.sqrt(n[0:dim_tS-1:2])*g1,state[:,:,2:dim_tS:2,:,:,:,:,:])
					new_tS_g[:,:,2:dim_tS:2,:,:,:,:,:]   = np.einsum("j,imjklnop->imjklnop",np.sqrt(n[0:dim_tS-1:2])*g1,state[:,:,1:dim_tS-1:2,:,:,:,:,:])
				if Ome1 is not 0:
					new_tS_Ome[:,:,0:dim_tS-2:2,:,:,:,:,:] = Ome1*state[:,:,1:dim_tS-1:2,:,:,:,:,:]
					new_tS_Ome[:,:,1:dim_tS-1:2,:,:,:,:,:] = Ome1*state[:,:,0:dim_tS-2:2,:,:,:,:,:]
		elif which == 2:
			if M == 0:
				if g2 is not 0:
					new_tS_g[:,:,:,1:dim_tS-1:2,:,:]   = np.einsum("j,ilkjmn->ilkjmn",np.sqrt(n[0:dim_tS-1:2])*g2,state[:,:,:,2:dim_tS:2,:,:])
					new_tS_g[:,:,:,2:dim_tS:2,:,:]   = np.einsum("j,ilkjmn->ilkjmn",np.sqrt(n[0:dim_tS-1:2])*g2,state[:,:,:,1:dim_tS-1:2,:,:])
				if Ome2 is not 0:
					new_tS_Ome[:,:,:,0:dim_tS-2:2,:,:] = Ome2*state[:,:,:,1:dim_tS-1:2,:,:]
					new_tS_Ome[:,:,:,1:dim_tS-1:2,:,:] = Ome2*state[:,:,:,0:dim_tS-2:2,:,:]
			else:
				if g2 is not 0:
					new_tS_g[:,:,:,:,1:dim_tS-1:2,:,:,:]   = np.einsum("j,imkljnop->imkljnop",np.sqrt(n[0:dim_tS-1:2])*g2,state[:,:,:,:,2:dim_tS:2,:,:,:])
					new_tS_g[:,:,:,:,2:dim_tS:2,:,:,:]   = np.einsum("j,imkljnop->imkljnop",np.sqrt(n[0:dim_tS-1:2])*g2,state[:,:,:,:,1:dim_tS-1:2,:,:,:])
				if Ome2 is not 0:
					new_tS_Ome[:,:,:,:,0:dim_tS-2:2,:,:,:] = Ome2*state[:,:,:,:,1:dim_tS-1:2,:,:,:]
					new_tS_Ome[:,:,:,:,1:dim_tS-1:2,:,:,:] = Ome2*state[:,:,:,:,0:dim_tS-2:2,:,:,:]
		return new_tS_g+new_tS_Ome
	def nc(state):
		n=np.linspace(0,int((dim_tS)/2),dim_tS).astype(np.int64)
		new_state = np.zeros(state.shape,complex)
		if len(state.shape)==3:
			new_state = np.einsum("j,ijk->ijk",n,state)
		elif len(state.shape)==4:
			new_state = np.einsum("j,ijkl->ijkl",n,state)
		return new_state
	def C(state):
		if Delc is 0 and Omc is 0:
			return 0.
		elif Delc is 0 and Omc is not 0:
			return Omc*(c(state)+cd(state))
		elif Delc is not 0 and Omc is 0:
			return Delc*nc(state)
		else:
			return Delc*nc(state)+Omc*(c(state)+cd(state))
	def MB(state):
		return Bd(c(state))-B(cd(state))
	def MS(state):
		new_tS = np.zeros(state.shape,complex)
		new_tS += state
		if Dele is 0:
			return -1j*dt*(C(state)+JC(state))
		else:
			if legs==3:
				new_tS[:,0:dim_tS:2,:] = 0
			elif legs==4:
				new_tS[:,0:dim_tS:2,:,:] = 0
			return -1j*dt*(C(state)+JC(state)+Dele*new_tS)
	#don't worry about this one!
	def Nn(tk,tS,tl,NBmin,NBmax):
		B_tk,B_tl = B(tk,tl)
		BdB_tk,BdB_tl = Bd(B_tk,B_tl)
		B_tk = None
		B_tl = None
		nc = np.linspace(0,(dim_tS-1)/2+.1,dim_tS).astype(np.int64)
		state = 0
		if legs==3:
			for i in range(NBmin,NBmax+1):
				state = np.tensordot(BdB_tk+(gamma_L+gamma_R)*dt*tk*i, np.tensordot((nc+1-i)*tS,BdB_tl,0),0)+state
		elif legs==4:
			for i in range(NBmin,NBmax+1):
				state = np.tensordot(BdB_tk+(gamma_L+gamma_R)*dt*tk*i, np.einsum("ij,jkl",(nc+1-i)*tS,BdB_tl))+state
		return state
        
    ####----------------------####
    #### Different terms in U ####
    ####----------------------####
    
    #####Initial state#####
	if M==0:
		initial = np.tensordot(tF1,tB1,tS1,tS2,tB2,tF2)
	else:
		initial = np.einsum("oip,j,okqr,pltu,m,unr->ijkqltmn",tFL,tBL,tS1,tS2,tB2,tF2)
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
		sys_env += (.5*(MB(sys_env)) -
					0.5j*dt*( Bd(Delc*c(initial)+Omc*initial+g*sm(initial)) + 
								B(Delc*cd(initial)+Omc*initial+g*sp(initial))-
								(gamma_L+gamma_R)/3.*dt*(C(initial)+g*(c(sp(initial))+cd(sm(initial)))+
								Delc*nc(initial)) + 2/3.*Delc*Bd(B(initial))))
    
    #####System#####
	sys = MS(initial+MS(initial)/2.)
    
#    print("sysenv",sys_env[0,2,0])

	return initial + sys + env + sys_env##

