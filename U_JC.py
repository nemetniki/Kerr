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
def U(tk,tS,tl,M,gamma_L,gamma_R,dt,phi,Ome,Omc,g,Delc,Dele): #tk: time bin state at k, tS: state of S
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
        if len(state.shape)==3:
            for i in range(dim_tk-1):
                new_state_tk[i,:,:] = np.sqrt((i+1)*gamma_L*dt)*state[i+1,:,:]
                new_state_tl[:,:,i] = np.sqrt((i+1)*gamma_R*dt)*np.exp(-1j*phi)*state[:,:,i+1]
        elif len(state.shape)==4:
            for i in range(dim_tk-1):
                new_state_tk[i,:,:,:] = np.sqrt((i+1)*gamma_L*dt)*state[i+1,:,:,:]
                new_state_tl[:,:,i,:] = np.sqrt((i+1)*gamma_R*dt)*np.exp(-1j*phi)*state[:,:,i+1,:]
        else:
            print("Unusual shape for tS")
        return new_state_tk+new_state_tl
    def Bd(state):
        new_state_tk = np.zeros(state.shape,complex)
        new_state_tl = np.zeros(state.shape,complex)
        if len(state.shape)==3:
            for i in range(dim_tk-1):
                new_state_tk[i+1,:,:] = np.sqrt((i+1)*gamma_L*dt)*state[i,:,:]
                new_state_tl[:,:,i+1] = np.sqrt((i+1)*gamma_R*dt)*np.exp(1j*phi)*state[:,:,i]
        elif len(state.shape)==4:
            for i in range(dim_tk-1):
                new_state_tk[i+1,:,:,:] = np.sqrt((i+1)*gamma_L*dt)*state[i,:,:,:]
                new_state_tl[:,:,i+1,:] = np.sqrt((i+1)*gamma_R*dt)*np.exp(1j*phi)*state[:,:,i,:]
        else:
            print("Unusual shape for tS")
        return new_state_tk+new_state_tl
    def c(state):
#        print("c",state[0,2,0], state.shape, len(state.shape))
        new_state = np.zeros(state.shape,complex)
        if len(state.shape)==3:
            for i in range(dim_tS-2):
                new_state[:,i,:] = np.sqrt(int((i+2)/2))*state[:,i+2,:]
        elif len(state.shape)==4:
            for i in range(dim_tS-2):
                new_state[:,i,:,:] = np.sqrt(int((i+2)/2))*state[:,i+2,:,:]
        else:
            print("Unusual shape for tS")
#        print("c",new_state[0,0,0])
        return new_state
    def cd(state):
#        print("cd",state[0,2,0])
        new_state = np.zeros(state.shape,complex)
        if len(state.shape)==3:
            for i in range(dim_tS-2):
                new_state[:,i+2,:] = np.sqrt(int((i+2)/2))*state[:,i,:]
        elif len(state.shape)==4:
            for i in range(dim_tS-2):
                new_state[:,i+2,:,:] = np.sqrt(int((i+2)/2))*state[:,i,:,:]
        else:
            print("Unusual shape for tS")
#        print("cd",new_state[0,4,0])
        return new_state
    def sm(state):
        new_state = np.zeros(state.shape,complex)
        if len(state.shape)==3:
            for i in range(dim_tS-1):
                if i%2==0:
                    new_state[:,i,:] = state[:,i+1,:]
        elif len(state.shape)==4:
            for i in range(dim_tS-1):
                if i%2==0:
                    new_state[:,i,:,:] = state[:,i+1,:,:]
        else:
            print("Unusual shape for tS")
        return new_state
    def sp(state):
        new_state = np.zeros(state.shape,complex)
        if len(state.shape)==3:
            for i in range(dim_tS-1):
                if i%2==0:
                    new_state[:,i+1,:] = state[:,i,:]
        elif len(state.shape)==4:
            for i in range(dim_tS-1):
                if i%2==0:
                    new_state[:,i+1,:,:] = state[:,i,:,:]
        else:
            print("Unusual shape for tS")
        return new_state
    def JC(state):
        new_tS_g = np.zeros(state.shape,complex)
        new_tS_Ome = np.zeros(state.shape,complex)
        if len(state.shape)==3:
            for i in range(dim_tS-2):
                if i%2==0:
                    new_tS_g[:,i+1,:]   = np.sqrt(np.int((i+2)/2))*g*state[:,i+2,:]
                    new_tS_Ome[:,i,:] = Ome*state[:,i+1,:]
    #                elif i%2==1:
                    new_tS_g[:,i+2,:]   = np.sqrt(np.int((i+2)/2))*g*state[:,i+1,:]
                    new_tS_Ome[:,i+1,:] = Ome*state[:,i,:]
        elif len(state.shape)==4:
            for i in range(dim_tS-2):
                if i%2==0:
                    new_tS_g[:,i+1,:,:]   = np.sqrt(np.int((i+2)/2))*g*state[:,i+2,:,:]
                    new_tS_Ome[:,i,:,:] = Ome*state[:,i+1,:,:]
    #                elif i%2==0:
                    new_tS_g[:,i+2,:,:]   = np.sqrt(np.int((i+2)/2))*g*state[:,i+1,:,:]
                    new_tS_Ome[:,i+1,:,:] = Ome*state[:,i,:,:]
        else:
            print("Unusual shape for tS")
        return new_tS_g+new_tS_Ome
    def nc(state):
        new_state = np.zeros(state.shape,complex)
        if len(state.shape)==3:
            for i in range(dim_tS):
                new_state[:,i,:] = np.int(i/2)*state[:,i,:]
        elif len(state.shape)==4:
            for i in range(dim_tS):
                new_state[:,i,:,:] = np.int(i/2)*state[:,i,:,:]
        else:
            print("Unusual shape for tS")
        return new_state
    def C(state):
        return Delc*nc(state)+Omc*(c(state)+cd(state))
    def MB(state):
        return Bd(c(state))-B(cd(state))
    def MS(state):
        new_tS = np.zeros(state.shape,complex)
        new_tS += state
        if len(state.shape)==3:
            for i in range(dim_tS):
                if i%2==0:
                    new_tS[:,i,:] = np.zeros((dim_tk,dim_tl),complex)
        elif len(state.shape)==4:
            for i in range(dim_tS):
                if i%2==0:
                    new_tS[:,i,:,:] = np.zeros((dim_tk,dim_tl,tl.shape[-1]),complex)
        else:
            print("Unusual shape for tS")
        return -1j*dt*(C(state)+JC(state)+Dele*new_tS)
    def Nn(tk,tS,tl,NBmin,NBmax):
        B_tk,B_tl = B(tk,tl)
        BdB_tk,BdB_tl = Bd(B_tk,B_tl)
        B_tk = None
        B_tl = None
        nc = np.linspace(0,(dim_tS-1)/2+.1,dim_tS).astype(np.int64)
        state = 0
        if len(state.shape)==3:
            for i in range(NBmin,NBmax+1):
                state = np.tensordot(BdB_tk+(gamma_L+gamma_R)*dt*tk*i, np.tensordot((nc+1-i)*tS,BdB_tl,0),0)+state
        elif len(state.shape)==4:
            for i in range(NBmin,NBmax+1):
                state = np.tensordot(BdB_tk+(gamma_L+gamma_R)*dt*tk*i, np.einsum("ij,jkl",(nc+1-i)*tS,BdB_tl))+state
        else:
            print("Unusual shape for tS")
        return state
        
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
    
    #####System#####
    sys = MS(initial+MS(initial)/2.)
    
    #####System-Environment#####
    sys_env = MS(MB(initial))
    sys_env += (.5*(MB(sys_env)) -
                0.5j*dt*( Bd(Delc*c(initial)+Omc*initial+g*sm(initial)) + 
                          B(Delc*cd(initial)+Omc*initial+g*sp(initial))-
                          (gamma_L+gamma_R)/3.*dt*(C(initial)+g*(c(sp(initial))+cd(sm(initial)))+
                                                   Delc*nc(initial)) + 2/3.*Delc*Bd(B(initial))))
#    print("sysenv",sys_env[0,2,0])

    return initial + sys + env + sys_env##

