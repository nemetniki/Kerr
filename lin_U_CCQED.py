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

def U(M,L,tF,tS,tB,gamma_B,gamma_F,dt,phi,Ome,Omc,g,Delc,Dele):
	"""Evolution operator up to dt^2
	INPUT: states at the current time (t_k), the delayed time (t_l) and the system state (t_S) at timestep M
	Remember, at M=0 the state is separable
	OUTPUT: combined block of states at t_k, for the system and at t_l"""
    
    #print(tk.shape,tS.shape,tl.shape,M)
    
    ####--------------------------####
    #### Parameters and operators ####
    ####--------------------------####
    
    #####Dimensions of the physical indices#####
	dim_tB = tB[0].shape[0]
	if M==0:
		dim_tS = tS[0].shape[0]
		dim_tF = tF[0].shape[0]
	else:
		dim_tS = tS[0].shape[1]
		dim_tF = tF[0].shape[1]

	#print("dims",dim_tF,dim_tS,dim_tB)

    #####Frequently used operators#####
	#@profile
	#tB[0],tF[0],tS[0],tF[1],tS[1],tB[1]	
	def a(state,which,N):
		new_state = np.zeros(state.shape,complex)
		if dim_tS>2*N:
			n=np.linspace(N,int((dim_tS)/2),dim_tS-2*N).astype(np.int64)
			nprod = 1.
			i = 0
			while i<N:
				nprod *= np.sqrt(n-i)
				i+=1

			ich  = (-1)**(which+1)*(2+int(np.any(M)))#which=1->2/3, which=2->-2/-3
			iend = state.shape[ich]
			context = ["k,ijklmn->ijklmn","m,ijklmn->ijklmn","K,aIJKLMNe->aIJKLMNe","M,aIJKLMNe->aIJKLMNe"]

			idx=[slice(None)]*state.ndim
			idx[ich]=slice(None,iend-2*N)
			
			new_state[tuple(idx)] = es(context[which-1+int(np.any(M))*2],nprod,np.take(state,np.arange(2*N,iend),axis=ich))

		return new_state
	
	#@profile
	def ad(state,which,N):
		new_state = np.zeros(state.shape,complex)
		if dim_tS>2*N:
			n=np.linspace(N,int((dim_tS)/2),dim_tS-2*N).astype(np.int64)
			nprod = 1.
			i = 0
			while i<N:
				nprod *= np.sqrt(n-i)
				i+=1

			ich  = (-1)**(which+1)*(2+int(np.any(M)))#which=1->2/3, which=2->-2/-3
			iend = state.shape[ich]
			context = ["k,ijklmn->ijklmn","m,ijklmn->ijklmn","K,aIJKLMNe->aIJKLMNe","M,aIJKLMNe->aIJKLMNe"]

			idx=[slice(None)]*state.ndim
			idx[ich]=slice(2*N,iend)
			
			new_state[tuple(idx)] = es(context[which-1+int(np.any(M))*2],nprod,np.take(state,np.arange(iend-2*N),axis=ich))
		return new_state

	#@profile
	def sm(state,which):
		new_state = np.zeros(state.shape,complex)
		ich  = (-1)**(which+1)*(2+int(np.any(M)))#which=1->2/3, which=2->-2/-3
		iend = state.shape[ich]

		idx=[slice(None)]*state.ndim
		idx[ich]=slice(0,dim_tS-1,2)
		
		new_state[tuple(idx)] = np.take(state,np.arange(1,dim_tS,2),axis=ich)
		return new_state

	#@profile
	def sp(state,which):
		new_state = np.zeros(state.shape,complex)
		ich  = (-1)**(which+1)*(2+int(np.any(M)))#which=1->2/3, which=2->-2/-3
		iend = state.shape[ich]

		idx=[slice(None)]*state.ndim
		idx[ich]=slice(1,dim_tS,2)
		
		new_state[tuple(idx)] = np.take(state,np.arange(0,dim_tS-1,2),axis=ich)
		return new_state

	#@profile
	def JC(state,which):
		new_tS_g = np.zeros(state.shape,complex)
		new_tS_Ome = np.zeros(state.shape,complex)
		n = np.linspace(1,int((dim_tS)/2),dim_tS-2).astype(np.int64)
		n = np.sqrt(n[0:dim_tS-1:2])

		ich  = (-1)**(which+1)*(2+int(np.any(M)))#which=1->2/3, which=2->-2/-3
		iend = state.shape[ich]

		if g[which-1] != 0:
			context = ["k,ijklmn->ijklmn","m,ijklmn->ijklmn","K,aIJKLMNe->aIJKLMNe","M,aIJKLMNe->aIJKLMNe"]

			idx1=[slice(None)]*state.ndim
			idx2=[slice(None)]*state.ndim
			idx1[ich]=slice(1,dim_tS-1,2)
			idx2[ich]=slice(2,dim_tS,2)
			
			new_tS_g[tuple(idx1)] = es(context[which-1+int(np.any(M))*2],n*g[which-1],np.take(state,np.arange(2,dim_tS,2),axis=ich))
			new_tS_g[tuple(idx2)] = es(context[which-1+int(np.any(M))*2],n*g[which-1],np.take(state,np.arange(1,dim_tS-1,2),axis=ich))

		if Ome[which-1] != 0:
			idx1=[slice(None)]*state.ndim
			idx2=[slice(None)]*state.ndim
			idx1[ich]=slice(0,dim_tS-2,2)
			idx2[ich]=slice(1,dim_tS-1,2)

			new_tS_Ome[tuple(idx1)] = np.take(state,np.arange(1,dim_tS-1,2),axis=ich)
			new_tS_Ome[tuple(idx2)] = np.take(state,np.arange(0,dim_tS-2,2),axis=ich)
		return new_tS_g+new_tS_Ome

	#@profile
	def nc(state,which,const):
		n=np.linspace(0,int((dim_tS)/2),dim_tS).astype(np.int64)
		new_state = np.zeros(state.shape,complex)
		context = ["k,ijklmn->ijklmn","m,ijklmn->ijklmn","K,aIJKLMNe->aIJKLMNe","M,aIJKLMNe->aIJKLMNe"]

		new_state = es(context[which-1+int(np.any(M))*2],n+const,state)
		return new_state

	#@profile
	def C(state,which):
		if Delc[which-1] == 0 and Omc[which-1] == 0:
			return 0.*state
		elif Delc[which-1] == 0 and Omc[which-1] != 0:
			return Omc[which-1]*(c(state,which)+cd(state,which))
		elif Delc[which-1] != 0 and Omc[which-1] == 0:
			return Delc[which-1]*nc(state,which,0)
		else:
			return Delc[which-1]*nc(state,which,0)+Omc[which-1]*(c(state,which)+cd(state,which))

	#@profile
	def MS(state,which):
		new_tS = np.zeros(state.shape,complex)
		new_tS += state
		if Dele[which-1] == 0:
			return -1j*dt*(C(state,which)+JC(state,which))
		else:
			ich  = (-1)**(which+1)*(2+int(np.any(M)))#which=1->2/3, which=2->-2/-3
			iend = state.shape[ich]

			idx=[slice(None)]*new_state.ndim
			idx[ich]=slice(0,dim_tS,2)
			
			new_tS[tuple(idx)] = 0
			return -1j*dt*(C(state,which)+JC(state,which)+Dele[which-1]*new_tS)

	#@profile
	def MStot(state):
		return MS(state,1)+MS(state,2)

	#@profile
	def E(state,which,N):
		#print(state.shape, N)
		new_state_tB = np.zeros(state.shape,complex)
		new_state_tF = np.zeros(state.shape,complex)
		if N<dim_tB:
			n = np.arange(N,dim_tB)
			nprod = 1.
			i = 0
			if M<L:
				phip=0.
			else:
				phip=phi#[which-1]
			#print(phip)
			while i<N:
				nprod *= np.sqrt(n-i)
				i+=1

			if gamma_B[which-1] > 0.:
				ich = (-1)**(which+1)*(which-1+int(np.any(M)))#which=1->0/1, which=2->-1/-2
				iend = state.shape[ich]
				context = ["i,ijklmn->ijklmn","n,ijklmn->ijklmn","I,aIJKLMNe->aIJKLMNe","N,aIJKLMNe->aIJKLMNe"]

				idx=[slice(None)]*state.ndim
				idx[ich]=np.arange(0,iend-N)
				
	#			print(new_state_tB[tuple(idx)].shape,np.take(state,np.arange(N,iend),axis=ich).shape,nprod.shape,N)
				new_state_tB[tuple(idx)] = es(context[which-1+int(np.any(M))*2],nprod*np.sqrt(gamma_B[which-1]*dt)**N,np.take(state,np.arange(N,iend),axis=ich))

			if gamma_F[which-1] > 0:
				ich = (-1)**(which+1)*(2*which-1+int(np.any(M)))#which=1->1/2, which=2->-3/-4
				iend = state.shape[ich]
				context = ["j,ijklmn->ijklmn","l,ijklmn->ijklmn","J,aIJKLMNe->aIJKLMNe","L,aIJKLMNe->aIJKLMNe"]

				idx=[slice(None)]*state.ndim
				idx[ich]=np.arange(0,iend-N)
				
	#			print(new_state_tF[tuple(idx)].shape,np.take(state,np.arange(N,iend),axis=ich).shape,nprod.shape,N)
				new_state_tF[tuple(idx)] = es(context[which-1+int(np.any(M))*2],nprod*np.sqrt(gamma_F[which-1]*dt)**N,np.take(state,np.arange(N,iend),axis=ich))*np.exp(-1j*N*phip)

		return new_state_tB+new_state_tF

	#@profile
	def Ed(state,which,N):
		new_state_tB = np.zeros(state.shape,complex)
		new_state_tF = np.zeros(state.shape,complex)
		if N<dim_tB:
			n = np.arange(N,dim_tB)
			nprod = 1.
			i = 0
			if M<L:
				phip=0.
			else:
				phip=phi#[which-1]
			while i<N:
				nprod *= np.sqrt(n-i)
				i+=1

			if gamma_B[which-1] > 0.:
				ich = (-1)**(which+1)*(which-1+int(np.any(M)))#which=1->0/1, which=2->-1/-2
				iend = state.shape[ich]
				context = ["i,ijklmn->ijklmn","n,ijklmn->ijklmn","I,aIJKLMNe->aIJKLMNe","N,aIJKLMNe->aIJKLMNe"]

				idx=[slice(None)]*state.ndim
				idx[ich]=slice(N,iend)
	#			print("context:", context[which-1+int(np.any(M))*2],", which:", which,", M:", M)
				new_state_tB[tuple(idx)] = es(context[which-1+int(np.any(M))*2],nprod*np.sqrt(gamma_B[which-1]*dt)**N,np.take(state,np.arange(0,iend-N),axis=ich))

			if gamma_F[which-1] > 0:
				ich = (-1)**(which+1)*(2*which-1+int(np.any(M)))#which=1->1/2, which=2->-3/-4
				iend = state.shape[ich]
				context = ["j,ijklmn->ijklmn","l,ijklmn->ijklmn","J,aIJKLMNe->aIJKLMNe","L,aIJKLMNe->aIJKLMNe"]

				idx=[slice(None)]*state.ndim
				idx[ich]=slice(N,iend)
				
				new_state_tF[tuple(idx)] = es(context[which-1+int(np.any(M))*2],nprod*np.sqrt(gamma_F[which-1]*dt)**N,np.take(state,np.arange(0,iend-N),axis=ich))*np.exp(1j*N*phip)

		return new_state_tB+new_state_tF

	#@profile
	def E2mix(state,which):
		new_state_mix = np.zeros(state.shape,complex)
		if 2<dim_tB:
			if M<L:
				phip=0.
			else:
				phip=phi#[which-1]
			n = np.sqrt(np.arange(1,dim_tB))
			if gamma_B[which-1]>0 and gamma_F[which-1]>0:
				ich1 = (-1)**(which+1)*(which-1+int(np.any(M)))#which=1->0/1, which=2->-1/-2
				ich2 = (-1)**(which+1)*(2*which-1+int(np.any(M)))#which=1->1/2, which=2->-3/-4
				iend1 = state.shape[ich1]
				iend2 = state.shape[ich2]
				context = ["j,i,ijklmn->ijklmn","l,n,ijklmn->ijklmn","J,I,aIJKLMNe->aIJKLMNe","L,N,aIJKLMNe->aIJKLMNe"]

				idx_n=[slice(None)]*state.ndim
				idx_n[ich1]=slice(None,iend1-1)
				idx_n[ich2]=slice(None,iend2-1)
				idx_o=[slice(None)]*state.ndim
				idx_o[ich1]=slice(1,iend1)
				idx_o[ich2]=slice(1,iend2)
				
				new_state_mix[tuple(idx_n)] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(-1j*phip)*dt*
								contract(context[which-1+int(np.any(M))*2],n,n,state[tuple(idx_o)]))
		return 2*new_state_mix

	#@profile
	def Ed2mix(state,which):
		new_state_mix = np.zeros(state.shape,complex)
		if 2<dim_tB:
			if M<L:
				phip=0.
			else:
				phip=phi#[which-1]
			n = np.sqrt(np.arange(1,dim_tB))
			if gamma_B[which-1]>0 and gamma_F[which-1]>0:

				ich1 = (-1)**(which+1)*(which-1+int(np.any(M)))#which=1->0/1, which=2->-1/-2
				ich2 = (-1)**(which+1)*(2*which-1+int(np.any(M)))#which=1->1/2, which=2->-3/-4
				iend1 = state.shape[ich1]
				iend2 = state.shape[ich2]
				context = ["j,i,ijklmn->ijklmn","l,n,ijklmn->ijklmn","J,I,aIJKLMNe->aIJKLMNe","L,N,aIJKLMNe->aIJKLMNe"]

				idx_o=[slice(None)]*state.ndim
				idx_o[ich1]=slice(None,iend1-1)
				idx_o[ich2]=slice(None,iend2-1)
				idx_n=[slice(None)]*state.ndim
				idx_n[ich1]=slice(1,iend1)
				idx_n[ich2]=slice(1,iend2)
				
				new_state_mix[tuple(idx_n)] = (np.sqrt(gamma_B[which-1]*gamma_F[which-1])*np.exp(1j*phip)*dt*
								contract(context[which-1+int(np.any(M))*2],n,n,state[tuple(idx_o)]))
		return 2*new_state_mix

	#@profile
	def E3mix(state,which):

		new_state_mix1 = np.zeros(state.shape,complex)
		new_state_mix2 = np.zeros(state.shape,complex)
		if 2<dim_tB:
			if M<L:
				phip=0.
			else:
				phip=phi#[which-1]
			n = np.arange(1,dim_tB)
			n2 = np.sqrt(n[1:]*(n[1:]-1))
			n = np.sqrt(n)
			if gamma_B[which-1]>0 and gamma_F[which-1]>0:
				ich1 = (-1)**(which+1)*(which-1+int(np.any(M)))#which=1->0/1, which=2->-1/-2
				ich2 = (-1)**(which+1)*(2*which-1+int(np.any(M)))#which=1->1/2, which=2->-3/-4
				iend1 = state.shape[ich1]
				iend2 = state.shape[ich2]
				context = ["j,i,ijklmn->ijklmn","l,n,ijklmn->ijklmn","J,I,aIJKLMNe->aIJKLMNe","L,N,aIJKLMNe->aIJKLMNe"]

				idx_n1=[slice(None)]*state.ndim
				idx_n1[ich1]=slice(None,iend1-1)
				idx_n1[ich2]=slice(None,iend2-2)
				idx_n2=[slice(None)]*state.ndim
				idx_n2[ich1]=slice(None,iend1-2)
				idx_n2[ich2]=slice(None,iend2-1)
				idx_o1=[slice(None)]*state.ndim
				idx_o1[ich1]=slice(1,iend1)
				idx_o1[ich2]=slice(2,iend2)
				idx_o2=[slice(None)]*state.ndim
				idx_o2[ich1]=slice(2,iend1)
				idx_o2[ich2]=slice(1,iend2)
				
				new_state_mix1[tuple(idx_n1)] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(-2j*phip)*
								contract(context[which-1+int(np.any(M))*2],n2,n,state[tuple(idx_o1)]))
				new_state_mix2[tuple(idx_n2)] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(-1j*phip)*
								contract(context[which-1+int(np.any(M))*2],n,n2,state[tuple(idx_o2)]))
		return 3*(new_state_mix1+new_state_mix2)

	#@profile
	def Ed3mix(state,which):

		new_state_mix1 = np.zeros(state.shape,complex)
		new_state_mix2 = np.zeros(state.shape,complex)
		if 2<dim_tB:
			if M<L:
				phip=0.
			else:
				phip=phi#[which-1]
			n = np.arange(1,dim_tB)
			n2 = np.sqrt(n[1:]*(n[1:]-1))
			n = np.sqrt(n)
			if gamma_B[which-1]>0 and gamma_F[which-1]>0:
				ich1 = (-1)**(which+1)*(which-1+int(np.any(M)))#which=1->0/1, which=2->-1/-2
				ich2 = (-1)**(which+1)*(2*which-1+int(np.any(M)))#which=1->1/2, which=2->-3/-4
				iend1 = state.shape[ich1]
				iend2 = state.shape[ich2]
				context = ["j,i,ijklmn->ijklmn","l,n,ijklmn->ijklmn","J,I,aIJKLMNe->aIJKLMNe","L,N,aIJKLMNe->aIJKLMNe"]

				idx_o1=[slice(None)]*state.ndim
				idx_o1[ich1]=slice(None,iend1-1)
				idx_o1[ich2]=slice(None,iend2-2)
				idx_o2=[slice(None)]*state.ndim
				idx_o2[ich1]=slice(None,iend1-2)
				idx_o2[ich2]=slice(None,iend2-1)
				idx_n1=[slice(None)]*state.ndim
				idx_n1[ich1]=slice(1,iend1)
				idx_n1[ich2]=slice(2,iend2)
				idx_n2=[slice(None)]*state.ndim
				idx_n2[ich1]=slice(2,iend1)
				idx_n2[ich2]=slice(1,iend2)
				
				new_state_mix1[tuple(idx_n1)] = (gamma_F[which-1]*np.sqrt(gamma_B[which-1])*dt**1.5*np.exp(2j*phip)*
								contract(context[which-1+int(np.any(M))*2],n2,n,state[tuple(idx_o1)]))
				new_state_mix2[tuple(idx_n2)] = (gamma_B[which-1]*np.sqrt(gamma_F[which-1])*dt**1.5*np.exp(1j*phip)*
								contract(context[which-1+int(np.any(M))*2],n,n2,state[tuple(idx_o2)]))

		return 3*(new_state_mix1+new_state_mix2)

	#@profile
	def E4mix(state,which):

		new_state_mix1 = np.zeros(state.shape,complex)
		new_state_mix2 = np.zeros(state.shape,complex)
		new_state_mix3 = np.zeros(state.shape,complex)

		if 2<dim_tB:
			if M<L:
				phip=0.
			else:
				phip=phi#[which-1]
			n = np.arange(1,dim_tB)
			n2 = np.sqrt(n[1:]*(n[1:]-1))
			if 3<dim_tB:
				n3 = np.sqrt(n[2:]*(n[2:]-1)*(n[2:]-2))
			n = np.sqrt(n)
			if gamma_B[which-1]>0 and gamma_F[which-1]>0:
				ich1 = (-1)**(which+1)*(which-1+int(np.any(M)))#which=1->0/1, which=2->-1/-2
				ich2 = (-1)**(which+1)*(2*which-1+int(np.any(M)))#which=1->1/2, which=2->-3/-4
				iend1 = state.shape[ich1]
				iend2 = state.shape[ich2]
				context = ["j,i,ijklmn->ijklmn","l,n,ijklmn->ijklmn","J,I,aIJKLMNe->aIJKLMNe","L,N,aIJKLMNe->aIJKLMNe"]

				if 3<dim_tB:
					idx_n1=[slice(None)]*state.ndim
					idx_n1[ich1]=slice(None,iend1-3)
					idx_n1[ich2]=slice(None,iend2-1)
					idx_n2=[slice(None)]*state.ndim
					idx_n2[ich1]=slice(None,iend1-1)
					idx_n2[ich2]=slice(None,iend2-3)
					idx_o1=[slice(None)]*state.ndim
					idx_o1[ich1]=slice(3,iend1)
					idx_o1[ich2]=slice(1,iend2)
					idx_o2=[slice(None)]*state.ndim
					idx_o2[ich1]=slice(1,iend1)
					idx_o2[ich2]=slice(3,iend2)
					new_state_mix1[tuple(idx_n1)] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(-1j*phip)*
									contract(context[which-1+int(np.any(M))*2],n,n3,state[tuple(idx_o1)]))
					new_state_mix2[tuple(idx_n2)] = (gamma_F[which-1]*np.sqrt(gamma_F[which-1]*gamma_B[which-1])*dt**2*np.exp(-3j*phip)*
									contract(context[which-1+int(np.any(M))*2],n3,n,state[tuple(idx_o2)]))
				idx_n3=[slice(None)]*state.ndim
				idx_n3[ich1]=slice(None,iend1-2)
				idx_n3[ich2]=slice(None,iend2-2)
				idx_o3=[slice(None)]*state.ndim
				idx_o3[ich1]=slice(2,iend1)
				idx_o3[ich2]=slice(2,iend2)
				
				new_state_mix3[tuple(idx_n3)] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(-2j*phip)*
								contract(context[which-1+int(np.any(M))*2],n2,n2,state[tuple(idx_o3)]))

		return 4*(new_state_mix1+new_state_mix2)+6*new_state_mix3

	#@profile
	def Ed4mix(state,which):
		new_state_mix1 = np.zeros(state.shape,complex)
		new_state_mix2 = np.zeros(state.shape,complex)
		new_state_mix3 = np.zeros(state.shape,complex)
		if 2<dim_tB:
			if M<L:
				phip=0.
			else:
				phip=phi#[which-1]
			n = np.arange(1,dim_tB)
			n2 = np.sqrt(n[1:]*(n[1:]-1))
			if 3<dim_tB:
				n3 = np.sqrt(n[2:]*(n[2:]-1)*(n[2:]-2))
			n = np.sqrt(n)
			if gamma_B[which-1]>0 and gamma_F[which-1]>0:
				ich1 = (-1)**(which+1)*(which-1+int(np.any(M)))#which=1->0/1, which=2->-1/-2
				ich2 = (-1)**(which+1)*(2*which-1+int(np.any(M)))#which=1->1/2, which=2->-3/-4
				iend1 = state.shape[ich1]
				iend2 = state.shape[ich2]
				context = ["j,i,ijklmn->ijklmn","l,n,ijklmn->ijklmn","J,I,aIJKLMNe->aIJKLMNe","L,N,aIJKLMNe->aIJKLMNe"]

				if 3<dim_tB:
					idx_o1=[slice(None)]*state.ndim
					idx_o1[ich1]=slice(None,iend1-3)
					idx_o1[ich2]=slice(None,iend2-1)
					idx_o2=[slice(None)]*state.ndim
					idx_o2[ich1]=slice(None,iend1-1)
					idx_o2[ich2]=slice(None,iend2-3)
					idx_n1=[slice(None)]*state.ndim
					idx_n1[ich1]=slice(3,iend1)
					idx_n1[ich2]=slice(1,iend2)
					idx_n2=[slice(None)]*state.ndim
					idx_n2[ich1]=slice(1,iend1)
					idx_n2[ich2]=slice(3,iend2)
					new_state_mix1[tuple(idx_n1)] = (gamma_B[which-1]*np.sqrt(gamma_B[which-1]*gamma_F[which-1])*dt**2*np.exp(1j*phip)*
									contract(context[which-1+int(np.any(M))*2],n,n3,state[tuple(idx_o1)]))
					new_state_mix2[tuple(idx_n2)] = (gamma_F[which-1]*np.sqrt(gamma_F[which-1]*gamma_B[which-1])*dt**2*np.exp(3j*phip)*
									contract(context[which-1+int(np.any(M))*2],n3,n,state[tuple(idx_o2)]))
				idx_o3=[slice(None)]*state.ndim
				idx_o3[ich1]=slice(None,iend1-2)
				idx_o3[ich2]=slice(None,iend2-2)
				idx_n3=[slice(None)]*state.ndim
				idx_n3[ich1]=slice(2,iend1)
				idx_n3[ich2]=slice(2,iend2)
				
				new_state_mix3[tuple(idx_n3)] = (gamma_B[which-1]*gamma_F[which-1]*dt**2*np.exp(2j*phip)*
								contract(context[which-1+int(np.any(M))*2],n2,n2,state[tuple(idx_o3)]))

		return 4*(new_state_mix1+new_state_mix2)+6*new_state_mix3
						
	#@profile
	def nE(state,which,const):
		new_state_tF = np.zeros(state.shape,complex)
		new_state_tB = np.zeros(state.shape,complex)
		new_state_tFB1 = np.zeros(state.shape,complex)
		new_state_tFB2 = np.zeros(state.shape,complex)
		n = np.arange(0,dim_tB)
		if gamma_F[which-1]>0:
			context = ["j,ijklmn->ijklmn","l,ijklmn->ijklmn","J,aIJKLMNe->aIJKLMNe","L,aIJKLMNe->aIJKLMNe"]

			new_state_tF = dt*gamma_F[which-1]*es(context[which-1+2*int(np.any(M))],n+const,state)

		if gamma_B[which-1]>0:
			context = ["i,ijklmn->ijklmn","n,ijklmn->ijklmn","I,aIJKLMNe->aIJKLMNe","N,aIJKLMNe->aIJKLMNe"]

			new_state_tB = dt*gamma_B[which-1]*es(context[which-1+2*int(np.any(M))],n+const,state)

		if gamma_B[which-1]>0 and gamma_F[which-1]>0:

			if M<L:
				phip=0.
			else:
				phip=phi#[which-1]

			context = ["j,i,ijklmn->ijklmn","l,n,ijklmn->ijklmn","J,I,aIJKLMNe->aIJKLMNe","L,N,aIJKLMNe->aIJKLMNe"]
			ich1 = (-1)**(which+1)*(which-1+int(np.any(M)))#which=1->0/1, which=2->-1/-2
			ich2 = (-1)**(which+1)*(2*which-1+int(np.any(M)))#which=1->1/2, which=2->-3/-4
			iend1 = state.shape[ich1]
			iend2 = state.shape[ich2]
			n = np.sqrt(n[1:])

			idx_n1=[slice(None)]*state.ndim
			idx_n1[ich1]=slice(None,iend1-1)
			idx_n1[ich2]=slice(1,iend2)
			idx_n2=[slice(None)]*state.ndim
			idx_n2[ich1]=slice(1,iend1)
			idx_n2[ich2]=slice(None,iend2-1)
			idx_o1=[slice(None)]*state.ndim
			idx_o1[ich1]=slice(1,iend1)
			idx_o1[ich2]=slice(None,iend2-1)
			idx_o2=[slice(None)]*state.ndim
			idx_o2[ich1]=slice(None,iend1-1)
			idx_o2[ich2]=slice(1,iend2)
			
			new_state_tFB1[tuple(idx_n1)] = (np.sqrt(gamma_F[which-1]*gamma_B[which-1])*dt*np.exp(1j*phip)*
							contract(context[which-1+int(np.any(M))*2],n,n,state[tuple(idx_o1)]))
			new_state_tFB2[tuple(idx_n2)] = (np.sqrt(gamma_F[which-1]*gamma_B[which-1])*dt*np.exp(-1j*phip)*
							contract(context[which-1+int(np.any(M))*2],n,n,state[tuple(idx_o2)]))

		return new_state_tF+new_state_tB+new_state_tFB1+new_state_tFB2

	#@profile
	def nEda(state,which):
		return nE(nc(state,which,1),which,0)+nE(nc(state,which,0),which,1)+nE(nc(state,which,-1),which,2)

	#@profile
	def nEad(state,which):
		return nE(nc(state,which,2),which,-1)+nE(nc(state,which,1),which,0)+nE(nc(state,which,0),which,1)
		
	#@profile
	def ME1(state,which):
		return ad(E(state,which,1),which,1)-a(Ed(state,which,1),which,1)

	#@profile
	def ME2(state,which):
		return (ad(E(state,which,2)+E2mix(state,which),which,2)+a(Ed(state,which,2)+Ed2mix(state,which),which,2)-
				nE(2*nc(state,which,0.5),which,0)-(gamma_F[which-1]+gamma_B[which-1])*dt*nc(state,which,0))

	#@profile
	def ME3(state,which):
		return (ad(E(state,which,3)+E3mix(state,which),which,3) - a(Ed(state,which,3)+Ed3mix(state,which),which,3)-
				E(ad(nEad(state,which),which,1),which,1)+Ed(a(nEda(state,which),which,1),which,1))

	#@profile
	def ME4(state,which):
		return (ad(E(state,which,4)+E4mix(state,which),which,4) + a(Ed(state,which,4)+Ed4mix(state,which),which,4)-
				nE(nc(ad(E(state,which,2)+E2mix(state,which),which,2),which,-2),which,3)-
				nE(nc(a(Ed(state,which,2)+Ed2mix(state,which),which,2),which,3),which,-2)-
				nEad(nE(nc(state,which,1),which,0)-a(Ed(state,which,2)+Ed2mix(state,which),which,2),which)+
				nEda(nE(nc(state,which,0),which,1)-ad(E(state,which,2)+E2mix(state,which),which,2),which))

	#@profile
	def D1(state):
		new_state = np.zeros(state.shape,complex)
		for which in range(1,3):
			if Delc[which-1] !=0:
				new_state += Delc[which-1]*(E(ad(state,which,1),which,1)+Ed(a(state,which,1),which,1))
			if g[which-1] != 0:
				new_state += g[which-1]*(sm(Ed(state,which,1),which)+sp(E(state,which,1),which))
			if Ome[which-1] != 0:
				new_state += Ome[which-1]*(Ed(state,which,1)+E(state,which,1))
		return -1j*new_state*dt

	#@profile
	def D2(state):
		new_state = np.zeros(state.shape,complex)
		for which in range(1,3):
			if Delc[which-1] !=0:
				new_state += 2*Delc[which-1]*(nE(state,which,0)-(gamma_B[which-1]+gamma_F[which-1])*dt*nc(state,which,0))
			if g[which-1] !=0:
				new_state += g[which-1]*(gamma_B[which-1]+gamma_F[which-1])*dt*(a(sp(state,which),which,1)-ad(sm(state,which),which,1))
			if Omc[which-1] !=0:
				new_state += Omc[which-1]*(gamma_B[which-1]+gamma_F[which-1])*dt*(a(state,which,1)-ad(state,which,1))
		return -1j*new_state*dt

    ####----------------------####
    #### Different terms in U ####
    ####----------------------####
    
    #####Initial state#####
	
	##[0] refers to timebins correponding to system 1 for forward direction (t=i+(2n+1)M) 
	##and system 2 for reverse direction (t=i+2nM)
	
	#print("shapes", tB[0].shape,tF[0].shape,tS[0].shape,tF[1].shape,tS[1].shape,tB[1].shape)
	if M==0:
		initial = contract("i,j,k,l,m,n->ijklmn",tB[0],tF[0],tS[0],tF[1],tS[1],tB[1])
	else:
		initial = contract("I,bJc,aKb,dLe,cMd,N->aIJKLMNe",tB[0],tF[0],tS[0],tF[1],tS[1],tB[1])

		
	#print("initial",initial.shape)
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

