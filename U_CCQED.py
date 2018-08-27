import numpy as np
import scipy as sc
from scipy.linalg import svd
import time
import sys
from decimal import Decimal
from math import factorial
from opt_einsum import contract

############################
### Evolution operator U ###
############################
def U(M,tF1,tF2,tS1,tB1,tB2,tS2,gamma_B,gamma_F,dt,phi,Ome,Omc,g,Delc,Dele): #tk: time bin state at k, tS: state of S1
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
		if g[which-1] is not 0:
			if which == 1:
				if M == 0:
					new_tS_g[:,:,1:dim_tS-1:2,:,:,:]   = np.einsum("j,iljkmn->iljkmn",np.sqrt(n[0:dim_tS-1:2])*g1,state[:,:,2:dim_tS:2,:,:,:])
					new_tS_g[:,:,2:dim_tS:2,:,:,:]   = np.einsum("j,iljkmn->iljkmn",np.sqrt(n[0:dim_tS-1:2])*g1,state[:,:,1:dim_tS-1:2,:,:,:])
				else:
					new_tS_g[:,:,1:dim_tS-1:2,:,:,:,:,:]   = np.einsum("j,imjklnop->imjklnop",np.sqrt(n[0:dim_tS-1:2])*g1,state[:,:,2:dim_tS:2,:,:,:,:,:])
					new_tS_g[:,:,2:dim_tS:2,:,:,:,:,:]   = np.einsum("j,imjklnop->imjklnop",np.sqrt(n[0:dim_tS-1:2])*g1,state[:,:,1:dim_tS-1:2,:,:,:,:,:])
			elif which == 2:
				if M == 0:
					new_tS_g[:,:,:,1:dim_tS-1:2,:,:]   = np.einsum("j,ilkjmn->ilkjmn",np.sqrt(n[0:dim_tS-1:2])*g2,state[:,:,:,2:dim_tS:2,:,:])
					new_tS_g[:,:,:,2:dim_tS:2,:,:]   = np.einsum("j,ilkjmn->ilkjmn",np.sqrt(n[0:dim_tS-1:2])*g2,state[:,:,:,1:dim_tS-1:2,:,:])
				else:
					new_tS_g[:,:,:,:,1:dim_tS-1:2,:,:,:]   = np.einsum("j,imkljnop->imkljnop",np.sqrt(n[0:dim_tS-1:2])*g2,state[:,:,:,:,2:dim_tS:2,:,:,:])
					new_tS_g[:,:,:,:,2:dim_tS:2,:,:,:]   = np.einsum("j,imkljnop->imkljnop",np.sqrt(n[0:dim_tS-1:2])*g2,state[:,:,:,:,1:dim_tS-1:2,:,:,:])
		if Ome[which-1] is not 0:
			if which == 1:
				if M == 0:
					new_tS_Ome[:,:,0:dim_tS-2:2,:,:,:] = Ome1*state[:,:,1:dim_tS-1:2,:,:,:]
					new_tS_Ome[:,:,1:dim_tS-1:2,:,:,:] = Ome1*state[:,:,0:dim_tS-2:2,:,:,:]
				else:
					new_tS_Ome[:,:,0:dim_tS-2:2,:,:,:,:,:] = Ome1*state[:,:,1:dim_tS-1:2,:,:,:,:,:]
					new_tS_Ome[:,:,1:dim_tS-1:2,:,:,:,:,:] = Ome1*state[:,:,0:dim_tS-2:2,:,:,:,:,:]
			elif which == 2:
				if M == 0:
					new_tS_Ome[:,:,:,0:dim_tS-2:2,:,:] = Ome2*state[:,:,:,1:dim_tS-1:2,:,:]
					new_tS_Ome[:,:,:,1:dim_tS-1:2,:,:] = Ome2*state[:,:,:,0:dim_tS-2:2,:,:]
				else:
					new_tS_Ome[:,:,:,:,0:dim_tS-2:2,:,:,:] = Ome2*state[:,:,:,:,1:dim_tS-1:2,:,:,:]
					new_tS_Ome[:,:,:,:,1:dim_tS-1:2,:,:,:] = Ome2*state[:,:,:,:,0:dim_tS-2:2,:,:,:]
		return new_tS_g+new_tS_Ome
	def nc(state,which):
		n=np.linspace(0,int((dim_tS)/2),dim_tS).astype(np.int64)
		new_state = np.zeros(state.shape,complex)
		if which == 1:
			if M==0:
				new_state = np.einsum("j,imjkno->imjkno",n,state)
			else:
				new_state = np.einsum("j,imjklnop->imjklnop",n,state)
		elif which == 2:
			if M==0:
				new_state = np.einsum("j,imkjno->imkjno",n,state)
			else:
				new_state = np.einsum("j,imkljnop->imkljnop",n,state)
		return new_state
	def C(state,which):
		if Delc[which-1] is 0 and Omc[which-1] is 0:
			return 0.*state
		elif Delc[which-1] is 0 and Omc[which-1] is not 0:
			return Omc[which-1]*(c(state,which)+cd(state,which))
		elif Delc[which-1] is not 0 and Omc[which-1] is 0:
			return Delc[which-1]*nc(state,which)
		else:
			return Delc[which-1]*nc(state,which)+Omc[which-1]*(c(state,which)+cd(state,which))
	def MS(state,which):
		new_tS = np.zeros(state.shape,complex)
		new_tS += state
		if Dele[which-1] is 0:
			return -1j*dt*(C(state,which)+JC(state,which))
		else:
			if which == 1:
				if M==0:
					new_tS[:,:,0:dim_tS:2,:,:,:] = 0
				else:
					new_tS[:,:,0:dim_tS:2,:,:,:,:,:] = 0
			elif which == 1:
				if M==0:
					new_tS[:,:,:,0:dim_tS:2,:,:] = 0
				else:
					new_tS[:,:,:,:,0:dim_tS:2,:,:,:] = 0
			return -1j*dt*(C(state,which)+JC(state,which)+Dele[which-1]*new_tS)
	def E(state,which,N):
		new_state_tB = np.zeros(state.shape,complex)
		new_state_tF = np.zeros(state.shape,complex)
		n = np.arange(N,state.shape[0])
		nprod = 1.
		i = 0
		while i<N:
			nprod *= n-i
			i+=1
		if gamma_B[which-1] != 0.:
			if which==1:
				if M==0:
					new_state_tB[:,0:-N,:,:,:,:] = np.einsum("i,mijkln->mijkln",np.sqrt(nprod*gamma_B[which-1]*dt),state[:,N:,:,:,:,:])
				else:
					new_state_tB[:,0:-N,:,:,:,:,:,:] = np.einsum("i,mijklnop->mijklnop",np.sqrt(nprod*gamma_B[which-1]*dt),state[:,N:,:,:,:,:,:,:])
			elif which==2:
				if M==0:
					new_state_tB[:,:,:,:,0:-N,:] = np.einsum("i,mjklin->mjklin",np.sqrt(nprod*gamma_B[which-1]*dt),state[:,:,:,:,N:,:])
				else:
					new_state_tB[:,:,:,:,:,:,0:-N,:] = np.einsum("i,mjklnoip->mjklnoip",np.sqrt(nprod*gamma_B[which-1]*dt),state[:,:,:,:,:,:,N:,:])
			if gamma_F[which-1] != 0:
			if which==1:
				if M==0:
					new_state_tF[0:-N,:,:,:,:,:] = np.einsum("i,ijklmn->ijklmn",np.sqrt(nprod*gamma_F[which-1]*dt),state[N:,:,:,:,:,:])*np.exp(-1j*phi)
				else:
					new_state_tF[0:-N,:,:,:,:,:,:,:] = np.einsum("i,ijklmnop->ijklmnop",np.sqrt(nprod*gamma_F[which-1]*dt),state[N:,:,:,:,:,:,:,:])*np.exp(-1j*phi)
			elif which==2:
				if M==0:
					new_state_tF[:,:,:,:,:,0:-N] = np.einsum("i,mjklni->mjklni",np.sqrt(nprod*gamma_F[which-1]*dt),state[:,:,:,:,:,N:])*np.exp(-1j*phi)
				else:
					new_state_tF[:,:,:,:,:,:,:,0:-N] = np.einsum("i,mjklnopi->mjklnopi",np.sqrt(nprod*gamma_F[which-1]*dt),state[:,:,:,:,:,:,:,N:])*np.exp(-1j*phi)
		return new_state_tB+new_state_tF
	def Ed(state,which,N):
		new_state_tB = np.zeros(state.shape,complex)
		new_state_tF = np.zeros(state.shape,complex)
		n = np.arange(N,state.shape[0])
		nprod = 1.
		i = 0
		while i<N:
			nprod *= n+i
			i+1
		if gamma_B[which-1] != 0.:
			if which==1:
				if M==0:
					new_state_tB[:,N:,:,:,:,:] = np.einsum("i,mijkln->mijkln",np.sqrt(nprod*gamma_B[which-1]*dt),state[:,0:-N,:,:,:,:])
				else:
					new_state_tB[:,N:,:,:,:,:,:,:] = np.einsum("i,mijklnop->mijklnop",np.sqrt(nprod*gamma_B[which-1]*dt),state[:,0:-N,:,:,:,:,:,:])
			elif which==2:
				if M==0:
					new_state_tB[:,:,:,:,N:,:] = np.einsum("i,mjklin->mjklin",np.sqrt(nprod*gamma_B[which-1]*dt),state[:,:,:,:,0:-N,:])
				else:
					new_state_tB[:,:,:,:,:,:,N:,:] = np.einsum("i,mjklnoip->mjklnoip",np.sqrt(nprod*gamma_B[which-1]*dt),state[:,:,:,:,:,:,0:-N,:])
			if gamma_F[which-1] != 0:
			if which==1:
				if M==0:
					new_state_tF[N:,:,:,:,:,:] = np.einsum("i,ijklmn->ijklmn",np.sqrt(nprod*gamma_F[which-1]*dt),state[0:-N,:,:,:,:,:])*np.exp(-1j*phi)
				else:
					new_state_tF[N:,:,:,:,:,:,:,:] = np.einsum("i,ijklmnop->ijklmnop",np.sqrt(nprod*gamma_F[which-1]*dt),state[0:-N,:,:,:,:,:,:,:])*np.exp(-1j*phi)
			elif which==2:
				if M==0:
					new_state_tF[:,:,:,:,:,N:] = np.einsum("i,mjklni->mjklni",np.sqrt(nprod*gamma_F[which-1]*dt),state[:,:,:,:,:,0:-N])*np.exp(-1j*phi)
				else:
					new_state_tF[:,:,:,:,:,:,:,N:] = np.einsum("i,mjklnopi->mjklnopi",np.sqrt(nprod*gamma_F[which-1]*dt),state[:,:,:,:,:,:,:,0:-N])*np.exp(-1j*phi)
		return new_state_tB+new_state_tF
	def nE(state,which):
	
	def ME(state,which):
		return ad(E(state,which,1),which)-a(Ed(state,which,1),which)
	
        
    ####----------------------####
    #### Different terms in U ####
    ####----------------------####
    
    #####Initial state#####
	if M==0:
		initial = contract("i,j,k,l,m,n->ijklmn",tF1,tB1,tS1,tS2,tB2,tF2)
	else:
		initial = contract("oip,j,okqr,pltu,m,unr->ijkqltmn",tFL,tBL,tS1,tS2,tB2,tF2)
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

