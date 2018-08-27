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
	def a(state,which,N):
		new_state = np.zeros(state.shape,complex)
		n=np.linspace(N,int((dim_tS)/2),dim_tS-2*N).astype(np.int64)
		nprod = 1.
		i = 0
		while i<N:
			nprod *= n-i
			i+=1
		if which == 1:
			if M==0:
				new_state[:,:,0:-(2*N),:,:,:] = np.einsum("j,mijknp->mijknp",np.sqrt(nprod),state[:,:,(2*N):,:,:,:])
			else:
				new_state[:,:,0:-(2*N),:,:,:,:,:] = np.einsum("j,mijklnop->mijklnop",np.sqrt(nprod),state[:,:,(2*N):,:,:,:,:,:])
		elif which == 2:
			if M==0:
				new_state[:,:,:,0:-(2*N),:,:] = np.einsum("j,mikjnp->mikjnp",np.sqrt(nprod),state[:,:,:,(2*N):,:,:])
			else:
				new_state[:,:,:,:,0:-(2*N),:,:,:] = np.einsum("j,mikljnop->mikljnop",np.sqrt(nprod),state[:,:,:,:,(2*N):,:,:,:])
		return new_state
	def ad(state,which,N):
		new_state = np.zeros(state.shape,complex)
		n=np.linspace(N,int((dim_tS)/2),dim_tS-2*N).astype(np.int64)
		nprod = 1.
		i = 0
		while i<N:
			nprod *= n-i
			i+=1
		if which == 1:
			if M==0:
				new_state[:,:,(2*N):,:,:,:] = np.einsum("j,mijknp->mijknp",np.sqrt(nprod),state[:,:,0:-(2*N),:,:,:])
			else:
				new_state[:,:,(2*N):,:,:,:,:,:] = np.einsum("j,mijklnop->mijklnop",np.sqrt(nprod),state[:,:,0:-(2*N),:,:,:,:,:])
		elif which == 2:
			if M==0:
				new_state[:,:,:,(2*N):,:,:] = np.einsum("j,mikjnp->mikjnp",np.sqrt(nprod),state[:,:,:,0:-(2*N),:,:])
			else:
				new_state[:,:,:,:,(2*N):,:,:,:] = np.einsum("j,mikljnop->mikljnop",np.sqrt(nprod),state[:,:,:,:,0:-(2*N),:,:,:])
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
		if g[which-1] != 0:
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
		if Ome[which-1] != 0:
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
	def nc(state,which,const):
		n=np.linspace(0,int((dim_tS)/2),dim_tS).astype(np.int64)
		new_state = np.zeros(state.shape,complex)
		if which == 1:
			if M==0:
				new_state = np.einsum("j,imjkno->imjkno",n+const,state)
			else:
				new_state = np.einsum("j,imjklnop->imjklnop",n+const,state)
		elif which == 2:
			if M==0:
				new_state = np.einsum("j,imkjno->imkjno",n+const,state)
			else:
				new_state = np.einsum("j,imkljnop->imkljnop",n+const,state)
		return new_state
	def C(state,which):
		if Delc[which-1] == 0 and Omc[which-1] == 0:
			return 0.*state
		elif Delc[which-1] == 0 and Omc[which-1] != 0:
			return Omc[which-1]*(c(state,which)+cd(state,which))
		elif Delc[which-1] != 0 and Omc[which-1] == 0:
			return Delc[which-1]*nc(state,which,0)
		else:
			return Delc[which-1]*nc(state,which,0)+Omc[which-1]*(c(state,which)+cd(state,which))
	def MS(state,which):
		new_tS = np.zeros(state.shape,complex)
		new_tS += state
		if Dele[which-1] == 0:
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
		if gamma_B[which-1] > 0.:
			if which==1:
				if M==0:
					new_state_tB[:,0:-N,:,:,:,:] = np.einsum("i,mijkln->mijkln",np.sqrt(nprod*(gamma_B[which-1]*dt)**N),state[:,N:,:,:,:,:])
				else:
					new_state_tB[:,0:-N,:,:,:,:,:,:] = np.einsum("i,mijklnop->mijklnop",np.sqrt(nprod*(gamma_B[which-1]*dt)**N),state[:,N:,:,:,:,:,:,:])
			elif which==2:
				if M==0:
					new_state_tB[:,:,:,:,0:-N,:] = np.einsum("i,mjklin->mjklin",np.sqrt(nprod*(gamma_B[which-1]*dt)**N),state[:,:,:,:,N:,:])
				else:
					new_state_tB[:,:,:,:,:,:,0:-N,:] = np.einsum("i,mjklnoip->mjklnoip",np.sqrt(nprod*(gamma_B[which-1]*dt)**N),state[:,:,:,:,:,:,N:,:])
		if gamma_F[which-1] > 0:
			if which==1:
				if M==0:
					new_state_tF[0:-N,:,:,:,:,:] = np.einsum("i,ijklmn->ijklmn",np.sqrt(nprod*(gamma_F[which-1]*dt)**N),state[N:,:,:,:,:,:])*np.exp(-1j*N*phi)
				else:
					new_state_tF[0:-N,:,:,:,:,:,:,:] = np.einsum("i,ijklmnop->ijklmnop",np.sqrt(nprod*(gamma_F[which-1]*dt)**N),state[N:,:,:,:,:,:,:,:])*np.exp(-1j*N*phi)
			elif which==2:
				if M==0:
					new_state_tF[:,:,:,:,:,0:-N] = np.einsum("i,mjklni->mjklni",np.sqrt(nprod*(gamma_F[which-1]*dt)**N),state[:,:,:,:,:,N:])*np.exp(-1j*N*phi)
				else:
					new_state_tF[:,:,:,:,:,:,:,0:-N] = np.einsum("i,mjklnopi->mjklnopi",np.sqrt(nprod*(gamma_F[which-1]*dt)**N),state[:,:,:,:,:,:,:,N:])*np.exp(-1j*N*phi)
		return new_state_tB+new_state_tF
	def Ed(state,which,N):
		new_state_tB = np.zeros(state.shape,complex)
		new_state_tF = np.zeros(state.shape,complex)
		n = np.arange(N,state.shape[0])
		nprod = 1.
		i = 0
		while i<N:
			nprod *= n-i
			i+1
		if gamma_B[which-1] > 0.:
			if which==1:
				if M==0:
					new_state_tB[:,N:,:,:,:,:] = np.einsum("i,mijkln->mijkln",np.sqrt(nprod*(gamma_B[which-1]*dt)**N),state[:,0:-N,:,:,:,:])
				else:
					new_state_tB[:,N:,:,:,:,:,:,:] = np.einsum("i,mijklnop->mijklnop",np.sqrt(nprod*(gamma_B[which-1]*dt)**N),state[:,0:-N,:,:,:,:,:,:])
			elif which==2:
				if M==0:
					new_state_tB[:,:,:,:,N:,:] = np.einsum("i,mjklin->mjklin",np.sqrt(nprod*(gamma_B[which-1]*dt)**N),state[:,:,:,:,0:-N,:])
				else:
					new_state_tB[:,:,:,:,:,:,N:,:] = np.einsum("i,mjklnoip->mjklnoip",np.sqrt(nprod*(gamma_B[which-1]*dt)**N),state[:,:,:,:,:,:,0:-N,:])
		if gamma_F[which-1] > 0:
			if which==1:
				if M==0:
					new_state_tF[N:,:,:,:,:,:] = np.einsum("i,ijklmn->ijklmn",np.sqrt(nprod*(gamma_F[which-1]*dt)**N),state[0:-N,:,:,:,:,:])*np.exp(-1j*phi)
				else:
					new_state_tF[N:,:,:,:,:,:,:,:] = np.einsum("i,ijklmnop->ijklmnop",np.sqrt(nprod*(gamma_F[which-1]*dt)**N),state[0:-N,:,:,:,:,:,:,:])*np.exp(-1j*phi)
			elif which==2:
				if M==0:
					new_state_tF[:,:,:,:,:,N:] = np.einsum("i,mjklni->mjklni",np.sqrt(nprod*(gamma_F[which-1]*dt)**N),state[:,:,:,:,:,0:-N])*np.exp(-1j*phi)
				else:
					new_state_tF[:,:,:,:,:,:,:,N:] = np.einsum("i,mjklnopi->mjklnopi",np.sqrt(nprod*(gamma_F[which-1]*dt)**N),state[:,:,:,:,:,:,:,0:-N])*np.exp(-1j*phi)
		return new_state_tB+new_state_tF
	def E2mix(state,which):
		n = np.arange(1,state.shape[0])
		new_state_mix = np.zeros(state.shape)
		if gamma_B[which-1]>0 and gamma_F[which-1]>0:
			if which == 1:
				if M==0:
					new_state_mix[:-1,:-1,:,:,:,:] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(-1j*phi)*
														*contract("i,j,ijklmn->ijklmn",n,n,state[1:,1:,:,:,:,:]))
				else:
					new_state_mix[:-1,:-1,:,:,:,:,:,:] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(-1j*phi)*
															*contract("i,j,ijklmnop->ijklmnop",n,n,state[1:,1:,:,:,:,:,:,:]))
			elif which == 2:
				if M==0:
					new_state_mix[:,:,:,:,:-1,:-1] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(-1j*phi)*
														*contract("i,j,klmnij->klmnij",n,n,state[:,:,:,:,1:,1:]))
				else:
					new_state_mix[:,:,:,:,:,:,:-1,:-1] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(-1j*phi)*
															*contract("i,j,klmnopij->klmnopij",n,n,state[:,:,:,:,:,:,1:,1:]))
		return 2*new_state_mix
	def Ed2mix(state,which):
		n = np.arange(1,state.shape[0])
		new_state_mix = np.zeros(state.shape)
		if gamma_B[which-1]>0 and gamma_F[which-1]>0:
			if which == 1:
				if M==0:
					new_state_mix[1:,1:,:,:,:,:] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(1j*phi)*
														*contract("i,j,ijklmn->ijklmn",n,n,state[:-1,:-1,:,:,:,:]))
				else:
					new_state_mix[1:,1:,:,:,:,:,:,:] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(1j*phi)*
															*contract("i,j,ijklmnop->ijklmnop",n,n,state[:-1,:-1,:,:,:,:,:,:]))
			elif which == 2:
				if M==0:
					new_state_mix[:,:,:,:,1:,1:] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(1j*phi)*
														*contract("i,j,klmnij->klmnij",n,n,state[:,:,:,:,:-1,:-1]))
				else:
					new_state_mix[:,:,:,:,:,:,1:,1:] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(1j*phi)*
															*contract("i,j,klmnopij->klmnopij",n,n,state[:,:,:,:,:,:,:-1,:-1]))
		return 2*new_state_mix
	def E3mix(state,which):
		n = np.arange(1,state.shape[0])
		n2 = n[1:]*(n[1:]-1)
		new_state_mix1 = np.zeros(state.shape)
		new_state_mix2 = np.zeros(state.shape)
		if gamma_B[which-1]>0 and gamma_F[which-1]>0:
			if which == 1:
				if M==0:
					new_state_mix1[:-1,:-2,:,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(-1j*phi)*
														*contract("i,j,ijklmn->ijklmn",n,n2,state[1:,2:,:,:,:,:]))
					new_state_mix2[:-2,:-1,:,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(-1j*2*phi)*
														*contract("i,j,ijklmn->ijklmn",n2,n,state[2:,1:,:,:,:,:]))
				else:
					new_state_mix1[:-1,:-2,:,:,:,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(-1j*phi)*
															*contract("i,j,ijklmn->ijklmn",n,n2,state[1:,2:,:,:,:,:,:,:]))
					new_state_mix2[:-2,:-1,:,:,:,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(-1j*2*phi)*
															*contract("i,j,ijklmn->ijklmn",n2,n,state[2:,1:,:,:,:,:,:,:]))
			elif which == 2:
				if M==0:
					new_state_mix1[:,:,:,:,:-2,:-1] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(-1j*phi)*
														*contract("i,j,ijklmn->ijklmn",n,n2,state[:,:,:,:,2:,1:]))
					new_state_mix2[:,:,:,:,:-1,:-2] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(-1j*2*phi)*
														*contract("i,j,ijklmn->ijklmn",n2,n,state[:,:,:,:,1:,2:]))
				else:
					new_state_mix1[:,:,:,:,:,:,:-2,:-1] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(-1j*phi)*
															*contract("i,j,ijklmn->ijklmn",n,n2,state[:,:,:,:,:,:,2:,1:]))
					new_state_mix2[:,:,:,:,:,:,:-1,:-2] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(-1j*2*phi)*
															*contract("i,j,ijklmn->ijklmn",n2,n,state[:,:,:,:,:,:,1:,2:]))
		return 3*(new_state_mix1+new_state_mix2)
	def Ed3mix(state,which):
		n = np.arange(1,state.shape[0])
		n2 = n[1:]*(n[1:]-1)
		new_state_mix1 = np.zeros(state.shape)
		new_state_mix2 = np.zeros(state.shape)
		if gamma_B[which-1]>0 and gamma_F[which-1]>0:
			if which == 1:
				if M==0:
					new_state_mix1[1:,2:,:,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(1j*phi)*
														*contract("i,j,ijklmn->ijklmn",n,n2,state[:-1,:-2,:,:,:,:]))
					new_state_mix2[2:,1:,:,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(1j*2*phi)*
														*contract("i,j,ijklmn->ijklmn",n2,n,state[:-2,:-1,:,:,:,:]))
				else:
					new_state_mix1[1:,2:,:,:,:,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(1j*phi)*
															*contract("i,j,ijklmn->ijklmn",n,n2,state[:-1,:-2,:,:,:,:,:,:]))
					new_state_mix2[2:,1:,:,:,:,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(1j*2*phi)*
															*contract("i,j,ijklmn->ijklmn",n2,n,state[:-2,:-1,:,:,:,:,:,:]))
			elif which == 2:
				if M==0:
					new_state_mix1[:,:,:,:,2:,1:] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(1j*phi)*
														*contract("i,j,ijklmn->ijklmn",n,n2,state[:,:,:,:,:-2,:-1]))
					new_state_mix2[:,:,:,:,1:,2:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(1j*2*phi)*
														*contract("i,j,ijklmn->ijklmn",n2,n,state[:,:,:,:,:-1,:-2]))
				else:
					new_state_mix1[:,:,:,:,:,:,2:,1:] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(1j*phi)*
															*contract("i,j,ijklmn->ijklmn",n,n2,state[:,:,:,:,:,:,:-2,:-1]))
					new_state_mix2[:,:,:,:,:,:,1:,2:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(1j*2*phi)*
															*contract("i,j,ijklmn->ijklmn",n2,n,state[:,:,:,:,:,:,:-1,:-2]))
		return 3*(new_state_mix1+new_state_mix2)
	def E4mix(state,which):
		n = np.arange(1,state.shape[0])
		n2 = n[1:]*(n[1:]-1)
		n3 = n[2:]*(n[2:]-1)*(n[2:]-2)
		new_state_mix1 = np.zeros(state.shape)
		new_state_mix2 = np.zeros(state.shape)
		new_state_mix3 = np.zeros(state.shape)
		if gamma_B[which-1]>0 and gamma_F[which-1]>0:
			if which == 1:
				if M==0:
					new_state_mix1[:-1,:-3,:,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*phi)*
														*contract("i,j,ijklmn->ijklmn",n,n3,state[1:,3:,:,:,:,:]))
					new_state_mix2[:-3,:-1,:,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*3*phi)*
														*contract("i,j,ijklmn->ijklmn",n3,n,state[3:,1:,:,:,:,:]))
					new_state_mix3[:-2,:-2,:,:,:,:] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(-1j*2*phi)*
														*contract("i,j,ijklmn->ijklmn",n2,n2,state[2:,2:,:,:,:,:]))
				else:
					new_state_mix1[:-1,:-3,:,:,:,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*phi)*
														*contract("i,j,ijklmn->ijklmn",n,n3,state[1:,3:,:,:,:,:,:,:]))
					new_state_mix2[:-3,:-1,:,:,:,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*3*phi)*
														*contract("i,j,ijklmn->ijklmn",n3,n,state[3:,1:,:,:,:,:,:,:]))
					new_state_mix3[:-2,:-2,:,:,:,:,:,:] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(-1j*2*phi)*
														*contract("i,j,ijklmn->ijklmn",n2,n2,state[2:,2:,:,:,:,:,:,:]))
			elif which == 2:
				if M==0:
					new_state_mix1[:,:,:,:,:-3,:-1] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*phi)*
														*contract("i,j,ijklmn->ijklmn",n,n3,state[:,:,:,:,3:,1:]))
					new_state_mix2[:,:,:,:,:-1,:-3] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*3*phi)*
														*contract("i,j,ijklmn->ijklmn",n3,n,state[:,:,:,:,1:,3:]))
					new_state_mix3[:,:,:,:,:-2,:-2] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(-1j*2*phi)*
														*contract("i,j,ijklmn->ijklmn",n2,n2,state[:,:,:,:,2:,2:]))
				else:
					new_state_mix1[:,:,:,:,:,:,:-3,:-1] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*phi)*
														*contract("i,j,ijklmn->ijklmn",n,n3,state[:,:,:,:,:,:,3:,1:]))
					new_state_mix2[:,:,:,:,:,:,:-1,:-3] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*3*phi)*
														*contract("i,j,ijklmn->ijklmn",n3,n,state[:,:,:,:,:,:,1:,3:]))
					new_state_mix3[:,:,:,:,:,:,:-2,:-2] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(-1j*2*phi)*
														*contract("i,j,ijklmn->ijklmn",n2,n2,state[:,:,:,:,:,:,2:,2:]))
		return 4*(new_state_mix1+new_state_mix2)+6*new_state_mix3
	def Ed4mix(state,which):
		n = np.arange(1,state.shape[0])
		n2 = n[1:]*(n[1:]-1)
		n3 = n[2:]*(n[2:]-1)*(n[2:]-2)
		new_state_mix1 = np.zeros(state.shape)
		new_state_mix2 = np.zeros(state.shape)
		new_state_mix3 = np.zeros(state.shape)
		if gamma_B[which-1]>0 and gamma_F[which-1]>0:
			if which == 1:
				if M==0:
					new_state_mix1[1:,3:,:,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*phi)*
														*contract("i,j,ijklmn->ijklmn",n,n3,state[:-1,:-3,:,:,:,:]))
					new_state_mix2[3:,1:,:,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*3*phi)*
														*contract("i,j,ijklmn->ijklmn",n3,n,state[:-3,:-1,:,:,:,:]))
					new_state_mix3[2:,2:,:,:,:,:] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(-1j*2*phi)*
														*contract("i,j,ijklmn->ijklmn",n2,n2,state[:-2,:-2,:,:,:,:]))
				else:
					new_state_mix1[1:,3:,:,:,:,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*phi)*
														*contract("i,j,ijklmn->ijklmn",n,n3,state[:-1,:-3,:,:,:,:,:,:]))
					new_state_mix2[3:,1:,:,:,:,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*3*phi)*
														*contract("i,j,ijklmn->ijklmn",n3,n,state[:-3,:-1,:,:,:,:,:,:]))
					new_state_mix3[2:,2:,:,:,:,:,:,:] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(-1j*2*phi)*
														*contract("i,j,ijklmn->ijklmn",n2,n2,state[:-2,:-2,:,:,:,:,:,:]))
			elif which == 2:
				if M==0:
					new_state_mix1[:,:,:,:,3:,1:] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*phi)*
														*contract("i,j,ijklmn->ijklmn",n,n3,state[:,:,:,:,:-3,:-1]))
					new_state_mix2[:,:,:,:,1:,3:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*3*phi)*
														*contract("i,j,ijklmn->ijklmn",n3,n,state[:,:,:,:,:-1,:-3]))
					new_state_mix3[:,:,:,:,2:,2:] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(-1j*2*phi)*
														*contract("i,j,ijklmn->ijklmn",n2,n2,state[:,:,:,:,:-2,:-2]))
				else:
					new_state_mix1[:,:,:,:,:,:,3:,1:] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*phi)*
														*contract("i,j,ijklmn->ijklmn",n,n3,state[:,:,:,:,:,:,:-3,:-1]))
					new_state_mix2[:,:,:,:,:,:,1:,3:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*3*phi)*
														*contract("i,j,ijklmn->ijklmn",n3,n,state[:,:,:,:,:,:,:-1,:-3]))
					new_state_mix3[:,:,:,:,:,:,2:,2:] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(-1j*2*phi)*
														*contract("i,j,ijklmn->ijklmn",n2,n2,state[:,:,:,:,:,:,:-2,:-2]))
		return 4*(new_state_mix1+new_state_mix2)+6*new_state_mix3
						
	def nE(state,which,const):
		new_state_tF = np.zeros(state.shape,complex)
		new_state_tB = np.zeros(state.shape,complex)
		new_state_tFB1 = np.zeros(state.shape,complex)
		new_state_tFB2 = np.zeros(state.shape,complex)
		n = np.arange(0,state.shape[0])
		if gamma_F[which-1]>0:
			if which == 1:
				if M==0:
					new_state_tF = dt*gamma_F[which-1]*np.einsum("i,ijklmn->ijklmn",n+const,state)
				else:
					new_state_tF = dt*gamma_F[which-1]*np.einsum("i,ijklmnop->ijklmnop",n+const,state)
			elif which == 2:
				if M==0:
					new_state_tF = dt*gamma_F[which-1]*np.einsum("i,jklmni->jklmni",n+const,state)
				else:
					new_state_tF = dt*gamma_F[which-1]*np.einsum("i,jklmnopi->jklmnopi",n+const,state)
		elif gamma_B[which-1]>0:
			if which == 1:
				if M==0:
					new_state_tB = dt*gamma_B[which-1]*np.einsum("i,jiklmn->jiklmn",n+const,state)
				else:
					new_state_tB = dt*gamma_B[which-1]*np.einsum("i,jiklmnop->jiklmnop",n+const,state)
			elif which == 2:
				if M==0:
					new_state_tB = dt*gamma_B[which-1]*np.einsum("i,jklmin->jklmin",n+const,state)
				else:
					new_state_tB = dt*gamma_B[which-1]*np.einsum("i,jklmnoip->jklmnoip",n+const,state)
		elif gamma_B[which-1]>0 and gamma_F[which-1]>0:
			if which == 1:
				if M==0:
					new_state_tFB1[1:,:-1,:,:,:,:] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(1j*phi)*
									contract("i,j,ijklmn->ijklmn",np.sqrt(n[1:]),np.sqrt(n[1:]),state[:-1,1:,:,:,:,:]))
					new_state_tFB2[:-1,1:,:,:,:,:] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(-1j*phi)*
									contract("i,j,ijklmn->ijklmn",np.sqrt(n[1:]),np.sqrt(n[1:]),state[1:,:-1,:,:,:,:]))
				else:
					new_state_tFB1[1:,:-1,:,:,:,:,:,:] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(1j*phi)*
									contract("i,j,ijklmnop->ijklmnop",np.sqrt(n[1:]),np.sqrt(n[1:]),state[:-1,1:,:,:,:,:,:,:]))
					new_state_tFB2[:-1,1:,:,:,:,:,:,:] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(-1j*phi)*
									contract("i,j,ijklmnop->ijklmnop",np.sqrt(n[1:]),np.sqrt(n[1:]),state[1:,:-1,:,:,:,:,:,:]))
			elif which == 2:
				if M==0:
					new_state_tFB1[:,:,:,:,:-1,1:] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(1j*phi)*
									contract("i,j,klmnji->klmnji",np.sqrt(n[1:]),np.sqrt(n[1:]),state[:,:,:,::,1:,-1]))
					new_state_tFB2[:,:,:,:,1:,:-1] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(-1j*phi)*
									contract("i,j,klmnji->klmnji",np.sqrt(n[1:]),np.sqrt(n[1:]),state[:,:,:,:,:-1,1:]))
				else:
					new_state_tFB1[:,:,:,:,:,:,:-1,1:] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(1j*phi)*
									contract("i,j,klmnopji->klmnopji",np.sqrt(n[1:]),np.sqrt(n[1:]),state[:,:,:,:,:,::,1:,-1]))
					new_state_tFB2[:,:,:,:,:,:,1:,:-1] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(-1j*phi)*
									contract("i,j,klmnopji->klmnopji",np.sqrt(n[1:]),np.sqrt(n[1:]),state[:,:,:,:,:,:,:-1,1:]))
		return new_state_tF+new_state_tB+new_state_tFB1+new_state_tFB2
	def nEda(state,which):
		return nE(nc(state,which,1),which,0)+nE(nc(state,which,0),which,1)+nE(nc(state,which,-1),which,2)
	def nEad(state,which):
		return nE(nc(state,which,2),which,-1)+nE(nc(state,which,1),which,0)+nE(nc(state,which,0),which,1)
		
	def ME1(state,which):
		return ad(E(state,which,1),which,1)-a(Ed(state,which,1),which,1)
	def ME2(state,which):
		return (ad(E(state,which,2)+E2mix(state,which),which,2)+a(Ed(state,which,2)+Ed2mix(state,which),which,2)-
				nE(2*nc(state,which,0.5),which,0)-(gamma_F[which-1]+gamma_B[which-1])*dt*nc(state,which,0))
	def ME3(state,which):
		return (ad(E(state,which,3)+E3mix(state,which),which,3) - a(Ed(state,which,3)+Ed3mix(state,which),which,3)-
				E(ad(nEad(state,which),which,1),which,1)+Ed(a(nEda(state,which),which,1),which,1))
	def ME4(state,which):
		return (ad(E(state,which,4)+E4mix(state,which),which,4) + a(Ed(state,which,4)+Ed4mix(state,which),which,4)+
				nEad(nE(nc(state,which,1),which,0)-Ed(a(state,which,2),which,2),which)+
				nEda(nE(nc(state,which,0),which,1)-E(ad(state,which,2),which,2),which))

	def D1(state,which):
		new_state = np.zeros(state.shape)
		if Delc[which-1] !=0:
			new_state += Delc[which-1]*(E(ad(state,which,1),which,1)+Ed(a(state,which,1),which,1)
		if g[which-1] != 0:
			new_state += g[which-1]*(sm(Ed(state,which,1),which)+sm(Ed(state,which,1),which))
		if Ome[which-1] != 0:
			new_state += Ome[which-1]*(Ed(state,which,1)+E(state,which,1))
		return -1j*new_state
	def D2(state,which):
		new_state = np.zeros(state.shape)
		if Delc[which-1] !=0:
			new_state += 2*Delc[which-1]*(nE(state,which,0)-(gamma_B[which-1]+gamma_F[which-1])*dt*nc(state,which,0))
		if g[which-1] !=0:
			new_state += g[which-1]*(gamma_B[which-1]+gamma_F[which-1])*dt*(a(sp(state,which),which,1)-ad(sm(state,which),which,1))
		if Omc[which-1] !=0:
			new_state += Omc[which-1]*(gamma_B[which-1]+gamma_F[which-1])*dt*(a(state,which,1)-ad(state,which,1))
		return -1j*new_state

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
	
    #####System#####
	sys = MS(initial+MS(initial)/2.)
	
	#####Environment#####
	env = ME1(ME1(initial,2),1)+.5*ME2(ME2(initial,2),1)
	for i in range(1,3):
		env += .5*ME2(initial,i)+ME3(initial,i)/6.+ME4(initial,i)/24.+ME1(initial,i)*(initial+.5*ME2(initial,3-i)+ME3(initial,3-i)/6.)
			
    #####System+environment#####
	sys_env = 0.
	for i in range(1,3):
		sys_env += .5*(D1(ME1(initial,3-i)+initial,i)+D2(initial,i)/3.+MS(ME1(ME1(initial,i)+initial,i)+ME1(initial,,i)+
				
#    print("sysenv",sys_env[0,2,0])

	return initial + sys + env + sys_env##

