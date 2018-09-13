import numpy as np
import scipy as sc
from scipy.linalg import svd
import time
import sys
from decimal import Decimal
from math import factorial
from opt_einsum import contract

es=np.einsum
############################
### Evolution operator U ###
############################
def U(M,L,tF1,tF2,tS1,tB1,tB2,tS2,gamma_B,gamma_F,dt,phi,Ome,Omc,g,Delc,Dele):
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
		nprod = np.sqrt(nprod)

		ich  = (-1)**(which+1)*(int(np.any(M))*(2-which)+which)
		iend = state.shape[ich]
		context = ["j,ijklmn->ijklmn","m,ijklmn->ijklmn","J,IvJKtLqrMN->IvJKtLqrMN","M,IvJKtLqrMN->IvJKtLqrMN"]

		idx=[slice(None)]*new_state.ndim
		idx[ich]=np.arange(iend-2*N)
		
		new_state[tuple(idx)] = es(context[which-1+int(np.any(M))*2],nprod,np.take(state,np.arange(2*N,iend),axis=ich))

		return new_state

	def ad(state,which,N):
		new_state = np.zeros(state.shape,complex)
		n=np.linspace(N,int((dim_tS)/2),dim_tS-2*N).astype(np.int64)
		nprod = 1.
		i = 0
		while i<N:
			nprod *= n-i
			i+=1
		nprod = np.sqrt(nprod)

		ich  = (-1)**(which+1)*(int(np.any(M))*(2-which)+which)
		iend = state.shape[ich]
		context = ["j,ijklmn->ijklmn","m,ijklmn->ijklmn","J,IvJKtLqrMN->IvJKtLqrMN","M,IvJKtLqrMN->IvJKtLqrMN"]

		idx=[slice(None)]*new_state.ndim
		idx[ich]=np.arange(2*N,iend)
		
		new_state[tuple(idx)] = es(context[which-1+int(np.any(M))*2],nprod,np.take(state,np.arange(iend-2*N),axis=ich))
		return new_state

	def sm(state,which):
		new_state = np.zeros(state.shape,complex)
		ich  = (-1)**(which+1)*(int(np.any(M))*(2-which)+which)
		iend = state.shape[ich]

		idx=[slice(None)]*new_state.ndim
		idx[ich]=np.arange(0,dim_tS-1,2)
		
		new_state[tuple(idx)] = np.take(state,np.arange(1,dim_tS,2),axis=ich)
		return new_state

	def sp(state,which):
		new_state = np.zeros(state.shape,complex)
		ich  = (-1)**(which+1)*(int(np.any(M))*(2-which)+which)
		iend = state.shape[ich]

		idx=[slice(None)]*new_state.ndim
		idx[ich]=np.arange(1,dim_tS,2)
		
		new_state[tuple(idx)] = np.take(state,np.arange(0,dim_tS-1,2),axis=ich)
		return new_state

	def JC(state,which):
		new_tS_g = np.zeros(state.shape,complex)
		new_tS_Ome = np.zeros(state.shape,complex)
		n = np.linspace(1,int((dim_tS)/2),dim_tS-2).astype(np.int64)
		n = np.sqrt(n[0:dim_tS-1:2])

		ich  = (-1)**(which+1)*(int(np.any(M))*(2-which)+which)
		iend = state.shape[ich]

		if g[which-1] != 0:
			context = ["j,ijklmn->ijklmn","m,ijklmn->ijklmn","J,IvJKtLqrMN->IvJKtLqrMN","M,IvJKtLqrMN->IvJKtLqrMN"]

			idx1=[slice(None)]*new_state.ndim
			idx2=[slice(None)]*new_state.ndim
			idx1[ich]=np.arange(1,dim_tS-1,2)
			idx2[ich]=np.arange(2,dim_tS,2)
			
			new_tS_g[tuple(idx1)] = es(context[which-1+int(np.any(M))*2],n*g[which-1],np.take(state,np.arange(2,dim_tS,2),axis=ich))
			new_tS_g[tuple(idx2)] = es(context[which-1+int(np.any(M))*2],n*g[which-1],np.take(state,np.arange(1,dim_tS-1,2),axis=ich))

		if Ome[which-1] != 0:
			idx1=[slice(None)]*new_state.ndim
			idx2=[slice(None)]*new_state.ndim
			idx1[ich]=np.arange(0,dim_tS-2,2)
			idx2[ich]=np.arange(1,dim_tS-1,2)

			new_tS_Ome[tuple(idx1)] = np.take(state,np.arange(1,dim_tS-1,2),axis=ich)
			new_tS_Ome[tuple(idx2)] = np.take(state,np.arange(0,dim_tS-2,2),axis=ich)
		return new_tS_g+new_tS_Ome

	def nc(state,which,const):
		n=np.linspace(0,int((dim_tS)/2),dim_tS).astype(np.int64)
		new_state = np.zeros(state.shape,complex)
		context = ["j,ijklmn->ijklmn","m,ijklmn->ijklmn","J,IvJKtLqrMN->IvJKtLqrMN","M,IvJKtLqrMN->IvJKtLqrMN"]

		new_state = es(context[which-1+int(np.any(M))*2],n+const,state)
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
			ich  = (-1)**(which+1)*(int(np.any(M))*(2-which)+which)
			iend = state.shape[ich]

			idx=[slice(None)]*new_state.ndim
			idx[ich]=np.arange(0,dim_tS,2)
			
			new_state[tuple(idx)] = 0
			return -1j*dt*(C(state,which)+JC(state,which)+Dele[which-1]*new_tS)

	def MStot(state):
		return MS(state,1)+MS(state,2)

	def E(state,which,N):
		new_state_tB = np.zeros(state.shape,complex)
		new_state_tF = np.zeros(state.shape,complex)
		n = np.arange(N,state.shape[0])
		nprod = 1.
		i = 0
		if M<L:
			phi=0.
		while i<N:
			nprod *= n-i
			i+=1
		nprod = sqrt(nprod)

		if gamma_B[which-1] > 0.:
			ich  = (-1)**(which+1)*(int(np.any(M))*(2-which)+(3-which))
			iend = state.shape[ich]
			context = ["k,ijklmn->ijklmn","n,ijklmn->ijklmn","K,IvJKtLqrMN->IvJKtLqrMN","N,IvJKtLqrMN->IvJKtLqrMN"]

			idx=[slice(None)]*new_state.ndim
			idx[ich]=np.arange(0,iend-N)
			
			new_state_tB[tuple(idx)] = es(context[which-1+int(np.any(M))*2],nprod*np.sqrt(gamma_B[which-1]*dt)**N,np.take(state,np.arange(N,iend),axis=ich))

		if gamma_F[which-1] > 0:
			ich  = (2*int(np.any(M))+3)*(1-which)
			iend = state.shape[ich]
			context = ["i,ijklmn->ijklmn","l,ijklmn->ijklmn","I,IvJKtLqrMN->IvJKtLqrMN","L,IvJKtLqrMN->IvJKtLqrMN"]

			idx=[slice(None)]*new_state.ndim
			idx[ich]=np.arange(0,iend-N)
			
			new_state_tF[tuple(idx)] = es(context[which-1+int(np.any(M))*2],nprod*np.sqrt(gamma_F[which-1]*dt)**N),np.take(state,np.arange(N,iend),axis=ich))*np.exp(-1j*N*phi)

		return new_state_tB+new_state_tF

	def Ed(state,which,N):
		new_state_tB = np.zeros(state.shape,complex)
		new_state_tF = np.zeros(state.shape,complex)
		n = np.arange(N,state.shape[0])
		nprod = 1.
		i = 0
		if M<L:
			phi=0.
		while i<N:
			nprod *= n-i
			i+1
		nprod = sqrt(nprod)

		if gamma_B[which-1] > 0.:
			ich  = (-1)**(which+1)*(int(np.any(M))*(2-which)+(3-which))
			iend = state.shape[ich]
			context = ["k,ijklmn->ijklmn","n,ijklmn->ijklmn","K,IvJKtLqrMN->IvJKtLqrMN","N,IvJKtLqrMN->IvJKtLqrMN"]

			idx=[slice(None)]*new_state.ndim
			idx[ich]=np.arange(N,iend)
			
			new_state_tB[tuple(idx)] = es(context[which-1+int(np.any(M))*2],nprod*np.sqrt(gamma_B[which-1]*dt)**N,np.take(state,np.arange(0,iend-N),axis=ich))

		if gamma_F[which-1] > 0:
			ich  = (2*int(np.any(M))+3)*(1-which)
			iend = state.shape[ich]
			context = ["i,ijklmn->ijklmn","l,ijklmn->ijklmn","I,IvJKtLqrMN->IvJKtLqrMN","L,IvJKtLqrMN->IvJKtLqrMN"]

			idx=[slice(None)]*new_state.ndim
			idx[ich]=np.arange(N,iend)
			
			new_state_tF[tuple(idx)] = es(context[which-1+int(np.any(M))*2],nprod*np.sqrt(gamma_F[which-1]*dt)**N),np.take(state,np.arange(0,iend-N),axis=ich))*np.exp(1j*N*phi)

		return new_state_tB+new_state_tF

	def E2mix(state,which):
		if M<L:
			phi=0.
		n = np.arange(1,state.shape[0])
		new_state_mix = np.zeros(state.shape)
		if gamma_B[which-1]>0 and gamma_F[which-1]>0:
			if which == 1:
				if M==0:
					new_state_mix[:-1,:,:-1,:,:,:] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(-1j*phi)*
														*contract("i,k,ijklmn->ijklmn",n,n,state[1:,:,1:,:,:,:]))
				else:
					new_state_mix[:-1,:,:,:-1,:,:,:,:,:,:] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(-1j*phi)*
															*contract("I,K,IvJKtLqrMN->IvJKtLqrMN",n,n,state[1:,:,:,1:,:,:,:,:,:,:]))
			elif which == 2:
				if M==0:
					new_state_mix[:,:,:,:-1,:,:-1] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(-1j*phi)*
														*contract("l,n,ijklmn->ijklmn",n,n,state[:,:,:,1:,:,1:]))
				else:
					new_state_mix[:,:,:,:,:,:-1,:,:,:,:-1] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(-1j*phi)*
															*contract("L,N,IvJKtLqrMN->IvJKtLqrMN",n,n,state[:,:,:,:,:,1:,:,:,:,1:]))
		return 2*new_state_mix
	def Ed2mix(state,which):
		if M<L:
			phi=0.
		n = np.arange(1,state.shape[0])
		new_state_mix = np.zeros(state.shape)
		if gamma_B[which-1]>0 and gamma_F[which-1]>0:
			if which == 1:
				if M==0:
					new_state_mix[1:,:,1:,:,:,:] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(1j*phi)*
														*contract("i,k,ijklmn->ijklmn",n,n,state[:-1,:,:-1,:,:,:]))
				else:
					new_state_mix[1:,:,:,1:,:,:,:,:,:,:] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(1j*phi)*
															*contract("I,K,IvJKtLqrMN->IvJKtLqrMN",n,n,state[:-1,:,:,:-1,:,:,:,:,:,:,]))
			elif which == 2:
				if M==0:
					new_state_mix[:,:,:,1:,:,1:] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(1j*phi)*
														*contract("l,n,ijklmn->ijklmn",n,n,state[:,:,:,:-1,:,:-1]))
				else:
					new_state_mix[:,:,:,:,:,1:,:,:,:,1:] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt*np.exp(1j*phi)*
															*contract("L,N,IvJKtLqrMN->IvJKtLqrMN",n,n,state[:,:,:,:,:,:-1,:,:,:,:-1]))
		return 2*new_state_mix
	def E3mix(state,which):
		if M<L:
			phi=0.
		n = np.arange(1,state.shape[0])
		n2 = n[1:]*(n[1:]-1)
		new_state_mix1 = np.zeros(state.shape)
		new_state_mix2 = np.zeros(state.shape)
		if gamma_B[which-1]>0 and gamma_F[which-1]>0:
			if which == 1:
				if M==0:
					new_state_mix1[:-1,:,:-2,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(-1j*phi)*
														*contract("i,k,ijklmn->ijklmn",n,n2,state[1:,:,2:,:,:,:]))
					new_state_mix2[:-2,:,:-1,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(-1j*2*phi)*
														*contract("k,i,ijklmn->ijklmn",n2,n,state[2:,:,1:,:,:,:]))
				else:
					new_state_mix1[:-1,:,:,:-2,:,:,:,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(-1j*phi)*
															*contract("I,K,IvJKtLqrMN->IvJKtLqrMN",n,n2,state[1:,:,:,2:,:,:,:,:,:,:]))
					new_state_mix2[:-2,:,:,:-1,:,:,:,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(-1j*2*phi)*
															*contract("K,I,IvJKtLqrMN->IvJKtLqrMN",n2,n,state[2:,:,:,1:,:,:,:,:,:,:]))
			elif which == 2:
				if M==0:
					new_state_mix1[:,:,:,:-2,:,:-1] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(-1j*phi)*
														*contract("l,n,ijklmn->ijklmn",n2,n,state[:,:,:,2:,:,1:]))
					new_state_mix2[:,:,:,:-1,:,:-2] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(-1j*2*phi)*
														*contract("n,l,ijklmn->ijklmn",n2,n,state[:,:,:,1:,:,2:]))
				else:
					new_state_mix1[:,:,:,:,:,:-2,:,:,:,:-1] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(-1j*phi)*
															*contract("L,N,IvJKtLqrMN->IvJKtLqrMN",n2,n,state[:,:,:,:,:,2:,:,:,:,1:]))
					new_state_mix2[:,:,:,:,:,:-1,:,:,:,:-2] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(-1j*2*phi)*
															*contract("N,L,IvJKtLqrMN->IvJKtLqrMN",n2,n,state[:,:,:,:,:,1:,:,:,:,2:]))
		return 3*(new_state_mix1+new_state_mix2)
	def Ed3mix(state,which):
		if M<L:
			phi=0.
		n = np.arange(1,state.shape[0])
		n2 = n[1:]*(n[1:]-1)
		new_state_mix1 = np.zeros(state.shape)
		new_state_mix2 = np.zeros(state.shape)
		if gamma_B[which-1]>0 and gamma_F[which-1]>0:
			if which == 1:
				if M==0:
					new_state_mix1[1:,:,2:,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(1j*phi)*
														*contract("i,k,ijklmn->ijklmn",n2,n,state[:-1,:,:-2,:,:,:]))
					new_state_mix2[2:,:,1:,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(1j*2*phi)*
														*contract("k,i,ijklmn->ijklmn",n2,n,state[:-2,:,:-1,:,:,:]))
				else:
					new_state_mix1[1:,:,:,2:,:,:,:,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(1j*phi)*
															*contract("I,K,IvJKtLqrMN->IvJKtLqrMN",n,n2,state[:-1,:,:,:-2,:,:,:,:,:,:]))
					new_state_mix2[2:,:,:,1:,:,:,:,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(1j*2*phi)*
															*contract("K,I,IvJKtLqrMN->IvJKtLqrMN",n2,n,state[:-2,:,:,:-1,:,:,:,:,:,:]))
			elif which == 2:
				if M==0:
					new_state_mix1[:,:,:,2:,:,1:] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(1j*phi)*
														*contract("l,n,ijklmn->ijklmn",n,n2,state[:,:,:,:-2,:,:-1]))
					new_state_mix2[:,:,:,1:,:,2:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(1j*2*phi)*
														*contract("n,l,ijklmn->ijklmn",n2,n,state[:,:,:,:-1,:,:-2]))
				else:
					new_state_mix1[:,:,:,:,:,2:,:,:,:,1:] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(1j*phi)*
															*contract("L,N,IvJKtLqrMN->IvJKtLqrMN",n,n2,state[:,:,:,:,:,:-2,:,:,:,:-1]))
					new_state_mix2[:,:,:,:,:,1:,:,:,:,2:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(1j*2*phi)*
															*contract("N,L,IvJKtLqrMN->IvJKtLqrMN",n2,n,state[:,:,:,:,:,:-1,:,:,:,:-2]))
		return 3*(new_state_mix1+new_state_mix2)
	def E4mix(state,which):
		if M<L:
			phi=0.
		n = np.arange(1,state.shape[0])
		n2 = n[1:]*(n[1:]-1)
		n3 = n[2:]*(n[2:]-1)*(n[2:]-2)
		new_state_mix1 = np.zeros(state.shape)
		new_state_mix2 = np.zeros(state.shape)
		new_state_mix3 = np.zeros(state.shape)
		if gamma_B[which-1]>0 and gamma_F[which-1]>0:
			if which == 1:
				if M==0:
					new_state_mix1[:-1,:,:-3,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*phi)*
														*contract("i,k,ijklmn->ijklmn",n,n3,state[1:,:,3:,:,:,:]))
					new_state_mix2[:-3,:,:-1,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*3*phi)*
														*contract("k,i,ijklmn->ijklmn",n3,n,state[3:,:,1:,:,:,:]))
					new_state_mix3[:-2,:,:-2,:,:,:] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(-1j*2*phi)*
														*contract("i,k,ijklmn->ijklmn",n2,n2,state[2:,:,2:,:,:,:]))
				else:
					new_state_mix1[:-1,:,:,:-3,:,:,:,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*phi)*
														*contract("I,K,IvJKtLqrMN->IvJKtLqrMN",n,n3,state[1:,:,:,3:,:,:,:,:,:,:]))
					new_state_mix2[:-3,:,:,:-1,:,:,:,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*3*phi)*
														*contract("K,I,IvJKtLqrMN->IvJKtLqrMN",n3,n,state[3:,:,:,1:,:,:,:,:,:,:]))
					new_state_mix3[:-2,:,:,:-2,:,:,:,:,:,:] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(-1j*2*phi)*
														*contract("I,K,IvJKtLqrMN->IvJKtLqrMN",n2,n2,state[2:,:,:,2:,:,:,:,:,:,:]))
			elif which == 2:
				if M==0:
					new_state_mix1[:,:,:,:-3,:,:-1] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*phi)*
														*contract("l,n,ijklmn->ijklmn",n,n3,state[:,:,:,3:,:,1:]))
					new_state_mix2[:,:,:,:-1,:,:-3] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*3*phi)*
														*contract("n,l,ijklmn->ijklmn",n3,n,state[:,:,:,1:,:,3:]))
					new_state_mix3[:,:,:,:-2,:,:-2] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(-1j*2*phi)*
														*contract("l,n,ijklmn->ijklmn",n2,n2,state[:,:,:,2:,:,2:]))
				else:
					new_state_mix1[:,:,:,:,:,:-3,:,:,:,:-1] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*phi)*
														*contract("L,N,IvJKtLqrMN->IvJKtLqrMN",n,n3,state[:,:,:,:,:,3:,:,:,:,1:]))
					new_state_mix2[:,:,:,:,:,:-1,:,:,:,:-3] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*3*phi)*
														*contract("N,L,IvJKtLqrMN->IvJKtLqrMN",n3,n,state[:,:,:,:,:,1:,:,:,:,3:]))
					new_state_mix3[:,:,:,:,:,:-2,:,:,:,:-2] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(-1j*2*phi)*
														*contract("L,N,IvJKtLqrMN->IvJKtLqrMN",n2,n2,state[:,:,:,:,:,2:,:,:,:,2:]))
		return 4*(new_state_mix1+new_state_mix2)+6*new_state_mix3
	def Ed4mix(state,which):
		if M<L:
			phi=0.
		n = np.arange(1,state.shape[0])
		n2 = n[1:]*(n[1:]-1)
		n3 = n[2:]*(n[2:]-1)*(n[2:]-2)
		new_state_mix1 = np.zeros(state.shape)
		new_state_mix2 = np.zeros(state.shape)
		new_state_mix3 = np.zeros(state.shape)
		if gamma_B[which-1]>0 and gamma_F[which-1]>0:
			if which == 1:
				if M==0:
					new_state_mix1[1:,:,3:,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(1j*phi)*
														*contract("i,k,ijklmn->ijklmn",n,n3,state[:-1,:,:-3,:,:,:]))
					new_state_mix2[3:,:,1:,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(1j*3*phi)*
														*contract("k,i,ijklmn->ijklmn",n3,n,state[:-3,:,:-1,:,:,:]))
					new_state_mix3[2:,:,2:,:,:,:] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(1j*2*phi)*
														*contract("i,k,ijklmn->ijklmn",n2,n2,state[:-2,:,:-2,:,:,:]))
				else:
					new_state_mix1[1:,:,:,3:,:,:,:,:,:,:] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(1j*phi)*
														*contract("I,K,IvJKtLqrMN->IvJKtLqrMN",n,n3,state[:-1,:,:,:-3,:,:,:,:,:,:]))
					new_state_mix2[3:,:,:,1:,:,:,:,:,:,:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(1j*3*phi)*
														*contract("K,I,IvJKtLqrMN->IvJKtLqrMN",n3,n,state[:-3,:,:,:-1,:,:,:,:,:,:]))
					new_state_mix3[2:,:,:,2:,:,:,:,:,:,:] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(1j*2*phi)*
														*contract("I,K,IvJKtLqrMN->IvJKtLqrMN",n2,n2,state[:-2,:,:,:-2,:,:,:,:,:,:]))
			elif which == 2:
				if M==0:
					new_state_mix1[:,:,:,3:,:,1:] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(1j*phi)*
														*contract("l,n,ijklmn->ijklmn",n,n3,state[:,:,:,:-3,:,:-1]))
					new_state_mix2[:,:,:,1:,:,3:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(1j*3*phi)*
														*contract("n,l,ijklmn->ijklmn",n3,n,state[:,:,:,:-1,:,:-3]))
					new_state_mix3[:,:,:,2:,:,2:] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(1j*2*phi)*
														*contract("l,n,ijklmn->ijklmn",n2,n2,state[:,:,:,:-2,:,:-2]))
				else:
					new_state_mix1[:,:,:,:,:,3:,:,:,:,1:] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(1j*phi)*
														*contract("L,N,IvJKtLqrMN->IvJKtLqrMN",n,n3,state[:,:,:,:,:,:-3,:,:,:,:-1]))
					new_state_mix2[:,:,:,:,:,1:,:,:,:,3:] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(1j*3*phi)*
														*contract("N,L,IvJKtLqrMN->IvJKtLqrMN",n3,n,state[:,:,:,:,:,:-1,:,:,:,:-3]))
					new_state_mix3[:,:,:,:,:,2:,:,:,:,2:] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(1j*2*phi)*
														*contract("L,N,IvJKtLqrMN->IvJKtLqrMN",n2,n2,state[:,:,:,:,:,:-2,:,:,:,:-2]))
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
					new_state_tF = dt*gamma_F[which-1]*es("i,ijklmn->ijklmn",n+const,state)
				else:
					new_state_tF = dt*gamma_F[which-1]*es("I,IvJKtLqrMN->IvJKtLqrMN",n+const,state)
			elif which == 2:
				if M==0:
					new_state_tF = dt*gamma_F[which-1]*es("l,ijklmn->ijklmn",n+const,state)
				else:
					new_state_tF = dt*gamma_F[which-1]*es("L,IvJKtLqrMN->IvJKtLqrMN",n+const,state)
		elif gamma_B[which-1]>0:
			if which == 1:
				if M==0:
					new_state_tB = dt*gamma_B[which-1]*es("k,ijklmn->ijklmn",n+const,state)
				else:
					new_state_tB = dt*gamma_B[which-1]*es("K,IvJKtLqrMN->IvJKtLqrMN",n+const,state)
			elif which == 2:
				if M==0:
					new_state_tB = dt*gamma_B[which-1]*es("n,ijklmn->ijklmn",n+const,state)
				else:
					new_state_tB = dt*gamma_B[which-1]*es("N,IvJKtLqrMN->IvJKtLqrMN",n+const,state)
		elif gamma_B[which-1]>0 and gamma_F[which-1]>0:
			if M<L:
				phi=0.
			if which == 1:
				if M==0:
					new_state_tFB1[1:,:,:-1,:,:,:] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(1j*phi)*
									contract("i,k,ijklmn->ijklmn",np.sqrt(n[1:]),np.sqrt(n[1:]),state[:-1,:,1:,:,:,:]))
					new_state_tFB2[:-1,:,1:,:,:,:] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(-1j*phi)*
									contract("i,k,ijklmn->ijklmn",np.sqrt(n[1:]),np.sqrt(n[1:]),state[1:,:,:-1,:,:,:]))
				else:
					new_state_tFB1[1:,:,:,:-1,:,:,:,:,:,:] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(1j*phi)*
									contract("I,K,IvJKtLqrMN->IvJKtLqrMN",np.sqrt(n[1:]),np.sqrt(n[1:]),state[:-1,:,:,1:,:,:,:,:,:,:]))
					new_state_tFB2[:-1,:,:,1:,:,:,:,:,:,:] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(-1j*phi)*
									contract("I,K,IvJKtLqrMN->IvJKtLqrMN",np.sqrt(n[1:]),np.sqrt(n[1:]),state[1:,:,:,:-1,:,:,:,:,:,:]))
			elif which == 2:
				if M==0:
					new_state_tFB1[:,:,:,:-1,:,1:] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(1j*phi)*
									contract("l,n,ijklmn->ijklmn",np.sqrt(n[1:]),np.sqrt(n[1:]),state[:,:,:,1:,:,:-1]))
					new_state_tFB2[:,:,:,1:,:,:-1] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(-1j*phi)*
									contract("l,n,ijklmn->ijklmn",np.sqrt(n[1:]),np.sqrt(n[1:]),state[:,:,:,:-1,:,1:]))
				else:
					new_state_tFB1[:,:,:,:,:,:-1,:,:,:,1:] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(1j*phi)*
									contract("L,N,IvJKtLqrMN->IvJKtLqrMN",np.sqrt(n[1:]),np.sqrt(n[1:]),state[:,:,:,:,:,:,1:,:,:,:-1]))
					new_state_tFB2[:,:,:,:,:,1:,:,:,:,:-1] = dt*(np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(-1j*phi)*
									contract("N,L,IvJKtLqrMN->IvJKtLqrMN",np.sqrt(n[1:]),np.sqrt(n[1:]),state[:,:,:,:,:,:-1,:,:,:,1:]))
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
		return (ad(E(state,which,4)+E4mix(state,which),which,4) + a(Ed(state,which,4)+Ed4mix(state,which),which,4)-
				nE(nc(ad(E(state,which,2)+E2mix(state,which,2),which,2),which,-2),which,3)-
				nE(nc(a(Ed(state,which,2)+Ed2mix(state,which,2),which,2),which,3),which,-2)-
				nEad(nE(nc(state,which,1),which,0)-a(Ed(state,which,2)+Ed2mix(state,which,2),which)+
				nEda(nE(nc(state,which,0),which,1)-ad(E(state,which,2)+E2mix(state,which,2),which))

	def D1(state):
		new_state = np.zeros(state.shape)
		for which in range(0,3):
			if Delc[which-1] !=0:
				new_state += Delc[which-1]*(E(ad(state,which,1),which,1)+Ed(a(state,which,1),which,1)
			if g[which-1] != 0:
				new_state += g[which-1]*(sm(Ed(state,which,1),which)+sm(Ed(state,which,1),which))
			if Ome[which-1] != 0:
				new_state += Ome[which-1]*(Ed(state,which,1)+E(state,which,1))
		return -1j*new_state
	def D2(state):
		new_state = np.zeros(state.shape)
		for which in range(0,3):
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
		initial = contract("i,j,k,l,m,n->ijklmn",tF1,tS1,tB1,tF2,tS2,tB2)
	else:
		initial = contract("oip,uvok,j,stun,pqrsl,m->IvJKtLqrMN",tF1,tS1,tB1,tF2,tS2,tB2)
#    print("init",initial[0,2,0],initial[1,0,0],initial[0,0,1],initial[0,0,0], initial.shape)
	#legs = len(initial.shape)
	
	#####System#####
	s1 = MStot(initial)
	sys = s1+MStot(s1)*.5
	
	#####Environment#####
	env = ME1(ME1(initial,2),1)+.25*ME2(ME2(initial,2),1)
	for i in range(1,3):
		env += .5*ME2(initial,i)+ME3(initial,i)/6.+ME4(initial,i)/24.+ME1(initial+.5*ME2(initial,3-i)+ME3(initial,3-i)/6.,i)
			
	#####System+environment#####
	sys_env = .5*D1(initial)+D2(initial)/6.+ME1(ME1(s1,2),1)
	for i in range(1,3):
		sys_env += ME1(s1+.5*D1(initial),i)+.5*ME2(s1,i)
				
#    print("sysenv",sys_env[0,2,0])

	return initial + sys + env + sys_env

