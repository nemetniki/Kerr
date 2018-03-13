
# coding: utf-8

# # MPS code

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.linalg import svd
import matplotlib as mpl
import time
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rc('font',family='FreeSerif')
mpl.rc('xtick',labelsize=30)
mpl.rc('ytick',labelsize=30)

### Linestyles ###
colors={'red':(241/255.,88/255.,84/255.),        'orange':(250/255,164/255.,58/255.),        'pink':(241/255,124/255.,176/255.),        'brown':(178/255,145/255.,47/255.),        'purple':(178/255,118/255.,178/255.),        'green':(96/255,189/255.,104/255.),        'blue':(93/255,165/255.,218/255.),        'yellow':(222/255., 207/255., 63/255),        'black':(0.,0.,0.)}
collab = ['brown','green','blue','pink','black']
linewidth = [2,2,3,3,4]
linestyle = ['solid','dashed','dashdot','dotted','solid']


# ## 1. Hamiltonian in the rotating frame
# 
# The system Hamiltonian:
# 
# \begin{align}
# \hat{H}_S=\hbar\Delta_c\hat{c}^\dagger\hat{c} + \hbar\Delta_e\hat{\sigma}_+\hat{\sigma}_-
#             +\hbar g\left(\hat{c}^\dagger\hat{\sigma}_-+\hat{\sigma}_+\hat{c}\right)
#             +\frac{\hbar\Omega_c}{2}\left(\hat{c}+\hat{c}^\dagger\right)
#             +\frac{\hbar\Omega_e}{2}\left(\hat{\sigma}_-+\hat{\sigma}_+\right)
# \end{align}
# 
# The interaction with the environment with feedback:
# 
# \begin{align}
# \hat{H}_{fb}(t)=-i\hbar\left\{\left[\sqrt{\gamma_R}\hat{b}(t-\tau)e^{-i\phi}+\sqrt{\gamma_L}\hat{b}(t)\right]\hat{c}^\dagger-\left[\sqrt{\gamma_R}\hat{b}^\dagger(t-\tau)e^{i\phi}+\sqrt{\gamma_L}\hat{b}^\dagger(t)\right]\hat{c}\right\}
# \end{align}
# 
# where the feedback phase is
# \begin{align}
# \phi=\pi-\omega_L\tau
# \end{align}
# 
# The bath is originally Markovain with
# \begin{align}
# \left[\hat{b}(t),\hat{b}^\dagger(t^\prime)\right]=\delta(t-t^\prime)
# \end{align}
# 

# ## 2. Time-evolution
# 
# \begin{align}
# |\Psi(t_{k+1})\rangle=U(t_{k+1},t_k)|\Psi(t_k)\rangle
# \end{align}
# 
# with $\Psi$ written as a Matrix Product State. The time-evolution operator can be expanded as
# \begin{align}
# U(t_{k+1},t_k) &= \exp{\left[-\frac{i}{\hbar}\left(H_S\Delta t+\int_{t_k}^{t_{k+1}}H_{fb}(t)dt\right)\right]}\\
# {\bf U}&=\exp{\left({\bf M}_S+{\bf M}_B\right)} = \sum_{n=0}^\infty\frac{1}{n!}\left({\bf M}_S+{\bf M}_B\right)^n
# \end{align}
# where $t_k=k\Delta t$ and $\Delta B(t_k) = \int_{t_k}^{t_{k+1}}b(t)dt$. This means that
# \begin{align}
# \left[\Delta B(t_k),\Delta B^\dagger(t_j)\right] = \Delta t \delta_{k,j}
# \end{align}
# 
# Therefore the different orders of the expansion above are:
# \begin{align}
# {\bf U} &=\mathbb{1}+\color{red}{{\bf M}_B}+\color{orange}{{\bf M}_S+\frac{1}{2}{\bf M}_B^2}+
#             \color{green}{\frac{1}{2}\left({\bf M}_S{\bf M}_B+{\bf M}_B{\bf M}_S\right)+\frac{1}{6}{\bf M}_B^3}+
#             \color{blue}{\frac{1}{2}{\bf M}_S^2+\frac{1}{6}\left({\bf M}_S{\bf M}_B^2+{\bf M}_B{\bf M}_S{\bf M}_B+{\bf M}_B^2{\bf M}_S\right)+\frac{1}{24}{\bf M}_B^4}+\mathcal{O}(\Delta t^{5/2})
# \end{align}
# This means that to first order we have:
# <img src='U_mat_JCM.png'>

# ## 3. Test case #1: no feedback
# Let us consider a TLS on a waveguide. We need up to 4 photons in the environment and a system vector with size 2. $\gamma_R=0$
# <img src='U_mat_TLS.png'>

# In[6]:


#index in the init list = timebin index: 0=16, ..., 15=1, 16=0, 17=S, 18=-1, 19=-2, 20=-3
start = time.time()

gamma_L = 1.
Om_TLS  = 1.5*np.pi#*gamma_L
Delta_T = 1.0
dt      = .0001
endt=int(12/dt)+3
init = [0]*endt
l=3
initTLS = np.array([0,1]) #starting at |e>
initenv = np.zeros(2)
initenv[0] = 1
# Initial state of the system
init[endt-4] = initTLS

### This function basically implements the formula above without feedback ###
def U(tk,tS): #tk: time bin state at k, tS: state of S
    U_m      = sc.eye(tk.size,tk.size,1)*np.sqrt(np.arange(0,tk.size)) # decreasing the number of photons in the environment bin with 1
    #print(np.dot(U_m,tk))
    U_p      = sc.eye(tk.size,tk.size,-1)*np.sqrt(np.arange(1,tk.size+1)) # increasing the number of photons in the environment bin with 1
    #print(np.dot(U_p,tk))
    sm       = sc.eye(2,2,1) # sigma_-
    sp       = sc.eye(2,2,-1) # sigma_+
    
    ### Different terms in U ###
    
    #####Identity#####
    T_0      = np.tensordot(tk,tS,0) #identity operation
    
    #####Terms with \delta_{i_Tg}\Delta_{j_Te}#####
    U_tk_ig_je = - 1j*dt*Om_TLS*sc.eye(tk.size,tk.size) +                 ( -1 + dt*gamma_L/6*np.arange(0,tk.size) + .5j*Delta_T*dt )*np.sqrt(dt*gamma_L)*U_m 
    T_ig_je = np.tensordot(np.dot(U_tk_ig_je,tk),np.dot(sp,tS),0)    
    
    #####Terms with \delta_{i_Te}\Delta_{j_Tg}#####
    U_tk_ie_jg = - 1j*dt*Om_TLS*sc.eye(tk.size,tk.size) +                 ( 1- dt*gamma_L/6*np.arange(1,tk.size+1) - .5*Delta_T*dt )*np.sqrt(dt*gamma_L)*U_p 
    T_ie_jg = np.tensordot(np.dot(U_tk_ie_jg,tk),np.dot(sm,tS),0)    

    #####Terms with \delta_{i_Te}\Delta_{j_Te}#####
    U_tk_ie_je = -(.5*gamma_L*np.arange(1,tk.size+1) + 1j*Delta_T)*dt*sc.eye(tk.size,tk.size) -                1j*Om_TLS/2.*np.sqrt(gamma_L)*dt**1.5*(U_p - U_m)
    T_ie_je = np.tensordot(np.dot(U_tk_ie_je,tk), np.dot(np.array([[0,0],[0,1]]),tS),0)

    #####Terms with \delta_{i_Tg}\Delta_{j_Tg}#####
    U_tk_ig_jg = -(.5*gamma_L*np.arange(0,tk.size))*dt*sc.eye(tk.size,tk.size) +                1j*Om_TLS/2.*np.sqrt(gamma_L)*dt**1.5*(U_m - U_p)
    T_ig_jg = np.tensordot(np.dot(U_tk_ig_jg,tk), np.dot(np.array([[1,0],[0,0]]),tS),0)
    
    #nextstep = T_0 + T_sqdt_1 + T_sqdt_2 + T_dt_S_1 + T_dt_S_2 + T_dt_S_3 + T_dt_B_1 + T_dt_B_2    
    nextstep = T_0 + T_ig_je + T_ie_jg + T_ie_je + T_ig_jg
    return(nextstep)

print("initial excitation:",np.einsum("k,k",np.einsum("i,ik",initTLS,np.array([[0,0],[0,1]])),np.conjugate(initTLS)))
print("initial ground probability:",np.einsum("k,k",np.einsum("i,ik",initTLS,np.array([[1,0],[0,0]])),np.conjugate(initTLS)))
m=0
exc2 = np.zeros(endt-3,complex)
norm = np.zeros(endt-3,complex)
norm2 = np.zeros(endt-2,complex)
norm2[endt-3] = np.dot(init[endt-4],np.conjugate(init[endt-4]))
norm3 = np.zeros(endt-2,complex)
norm3[endt-3] = np.dot(init[endt-4],np.conjugate(init[endt-4]))
exc = np.zeros(endt-3,complex)

#IMPORTANT: One needs to be careful with the final indices that come out of the summation of einsum, better to define them.

for m in np.arange(endt-4,-1,-1):
#    print("before\n",init[m].shape)
#    print(init[m])
#    if m<endt-4:
#        print(init[m+1].shape)
#    print(m)

    ### Calculating the norm ###
    if m==endt-4:
        #print(init[m])
        nor11 = np.dot(init[m],np.conjugate(init[m]))
        #print("nor1",nor1)
    elif m==endt-5:
        nor21 = np.einsum("ik,jk->ij",init[m+1],np.conjugate(init[m+1]))
        nor11 = np.einsum("ij,ij",np.einsum("li,lj->ij",init[m],np.conjugate(init[m])),nor21)
    else:
        nor31 = np.einsum("mli,nlj->minj",init[m+1],np.conjugate(init[m+1]))
        nor21 = np.einsum("minj,ij->mn",nor31,nor21)
        #print("nor2=",nor2)
        #print(nor2.shape)
        nor11 = np.einsum("mn,mn",np.einsum("lm,ln->mn",init[m],np.conjugate(init[m])),nor21)
    norm[m] = nor11

    ### Determining the excitation level of the qubit ###
    if m==endt-4:
        exc[m]=np.einsum("k,k",np.einsum("i,ik",init[m],np.array([[1,0],[0,0]])),np.conjugate(init[m]))
        exc2[m]=np.einsum("k,k",np.einsum("i,ik",init[m],np.array([[0,0],[0,1]])),np.conjugate(init[m]))
    else:
#        exc2[m]=np.einsum("jk,kj",np.einsum("ij,ik->jk",init[m],np.array([[0,0],[0,1]])),np.conjugate(init[m]))
        exc2_pre=np.einsum("ik,kj->ij",np.einsum("li,lk->ik",init[m],np.array([[0,0],[0,1]])),np.conjugate(init[m]))
        exc2[m] = np.einsum("ij,ij",exc2_pre,nor21)
        exc_pre=np.einsum("ik,kj->ij",np.einsum("li,lk->ik",init[m],np.array([[1,0],[0,0]])),np.conjugate(init[m]))
        exc[m] = np.einsum("ij,ij",exc_pre,nor21)
#        exc[m]=np.einsum("jk,kj",np.einsum("ij,ik->jk",init[m],np.array([[1,0],[0,0]])),np.conjugate(init[m]))
    
    ### A time-step ###
    # We consider up to 4 excitations in the environmental timebin (first array)
    together = U(initenv,init[m]) #normalized
#    print("together shape\n",together.shape)
    if m==endt-4:
        tog_swap = np.einsum("ij->ji",together)
        tog_svd  = svd(tog_swap,full_matrices=False)
    else:
        tog_swap = np.einsum("ijk->jik",together)
        di = tog_swap[0,:,0].size
        dj = tog_swap[:,0,0].size
        dk = tog_swap[0,0,:].size
#        print("di=",di,", dk=",dk)
        tog_swap_conc = np.zeros((dj,di*dk),dtype=np.complex128)
        for i in range(0,tog_swap[0,:,0].size):
            for k in range(0,tog_swap[0,0,:].size):
                tog_swap_conc[:,i+di*k]=tog_swap[:,i,k]
        tog_svd_conc = svd(tog_swap_conc,full_matrices=False)
        tog_svd = [0.+0*1j]*3
        tog_svd[2] = np.zeros((dj,di,dk),dtype=np.complex128)
        for I in range(0,tog_swap_conc[:,0].size):
            tog_svd[2][:,I%di,int((I-(I%di))/di)]  = tog_svd_conc[0][:,I]
        tog_svd[1] = tog_svd_conc[1]
        tog_svd[0] = tog_svd_conc[0]

    ### Storing the different Matrices of the timebins and system (I'm not convinced that this is necessary) ###
    init[m] = tog_svd[2]
    init[m-1] = np.dot(tog_svd[0],np.diag(tog_svd[1]))

#    print("after\n",init[m].shape)
#    print(init[m-1].shape)
#print(init[-1])

### Plotting to be happy ###
t=np.arange(0,endt-3)*dt*gamma_L
t2=np.arange(0,endt-2)*dt*gamma_L
end = time.time()
timeh = int((end-start)/3600)
timem = int((end-start)/60-timeh*60)
times = int((end-start)-timeh*3600-60*timem)
print("elapsed time: %02d:%02d:%02d" % (timeh,timem,times))


plt.figure(figsize = (12,7))
plt.plot(t,np.flip(np.real(exc),0),lw=2,label="ground",color=colors["green"])
plt.plot(t,np.flip(np.real(exc2),0),lw=2,label="excited",color=colors["blue"])
#plt.plot(t,np.exp(-gamma_L*t),ls="--",lw=4,color=colors["orange"],label="$\exp{(-\gamma_Lt)}$")
#plt.xlim(0,8)
#plt.ylim(-0.01,1.01)
plt.xlabel("$\gamma_Lt$",fontsize=40)
plt.ylabel("$\left<\sigma_+\sigma_-\\right>$",fontsize=40)
plt.legend(fontsize=25,loc="upper right")
plt.grid(True)
plt.figure(2,figsize = (12,7))
plt.plot(t,np.flip(np.real(norm-1),0),lw=2,label="norm",color=colors["purple"])
plt.grid(True)
plt.ylabel("$\left<\psi(t)|\psi(t)\\right>-1$",fontsize=40)
plt.xlabel("$\gamma_Lt$",fontsize=40)
plt.show()


# In[9]:


plt.figure(figsize = (12,7))
plt.plot(t,np.flip(np.real(exc),0),lw=2,label="ground",color=colors["green"])
plt.plot(t,np.flip(np.real(exc2),0),lw=2,label="excited",color=colors["blue"])
plt.plot(t,.5+0*t,"--",color="grey")
#plt.plot(t,np.exp(-gamma_L*t),ls="--",lw=4,color=colors["orange"],label="$\exp{(-\gamma_Lt)}$")
plt.xlim(0.0,2)
plt.ylim(0.45,0.55)
plt.xlabel("$\gamma_Lt$",fontsize=40)
plt.ylabel("$\left<\sigma_+\sigma_-\\right>$",fontsize=40)
plt.legend(fontsize=25,loc="upper right")
plt.grid(True)


# In[123]:


print(spin)
plt.semilogx(delt,spin)
plt.ylim(0.5,0.6)
plt.grid(True)


# ## 3. Test case #1: no feedback
# Let us consider a TLS on a waveguide. We need up to 4 photons in the environment and a system vector with size 2. $\gamma_R=0$
# <img src='2nd_order.png'>

# In[2]:


# Useful blocks:

def U(tk,tS,tl,M): #tk: time bin state at k, tS: state of S
    dim_tk = tk.shape[0]
    dim_tS = tS.shape[0]
    if M==0:
        dim_tl = tl.shape[0]
    else:
        dim_tl = tl.shape[1]

    U_m_tk = sc.eye(dim_tk,dim_tk,1)*np.sqrt(dt*gamma_L*np.arange(0,dim_tk)) # decreasing the number of photons in the environment bin with 1
    U_p_tk = sc.eye(dim_tk,dim_tk,-1)*np.sqrt(dt*gamma_L*np.arange(1,dim_tk+1)) # increasing the number of photons in the environment bin with 1
    U_m_tl = np.exp(-1j*phi)*sc.eye(dim_tl,dim_tl,1)*np.sqrt(dt*gamma_R*np.arange(0,dim_tl)) # decreasing the number of photons in the environment bin with 1
    U_p_tl = np.exp(1j*phi)*sc.eye(dim_tl,dim_tl,-1)*np.sqrt(dt*gamma_R*np.arange(1,dim_tl+1)) # increasing the number of photons in the environment bin with 1

    U_2m_tk = np.dot(U_m_tk,U_m_tk)
    U_2m_tl = np.dot(U_m_tl,U_m_tl)
    U_2p_tk = np.dot(U_p_tk,U_p_tk)
    U_2p_tl = np.dot(U_p_tl,U_p_tl)

    U_2np1_tk = (2*np.arange(0,dim_tk)+1)*gamma_L*dt
    U_2np1_tl = (2*np.arange(0,dim_tl)+1)*gamma_R*dt
    U_np_tk = np.arange(0,dim_tk)*gamma_L*dt
    U_np_tl = np.arange(0,dim_tl)*gamma_R*dt

    sm     = sc.eye(2,2,1) # sigma_-
    sp     = sc.eye(2,2,-1) # sigma_+
    
    ### Different terms in U ###
    
    #####Identity#####
    if M == 0:
        T_0      = np.tensordot(np.tensordot(tk,tS,0),tl,0) #identity operation
    else:
        T_0      = np.tensordot(tk,np.dot(tS,tl),0) #identity operation
    
    #####Terms with \delta_{i_Tg}\Delta_{j_Te}#####
    U_ig_je_0 = - 1j*dt*Om_TLS*( 1. -1j*.5*dt*Delta_T)
    U_ig_je_k = ( -1 + .5j*Delta_T*dt )*U_m_tk +                np.dot(np.diag(U_np_tk)/6.,U_m_tk)-                1j*dt*Om_TLS/6.*U_2m_tk+                1j*Om_TLS*dt**2/6.*np.diag(U_2np1_tk)
    U_ig_je_l = ( -1 + .5j*Delta_T*dt)*U_m_tl +                 np.dot(np.diag(U_np_tl)/6.,U_m_tl)-                1j*dt*Om_TLS/6.*U_2m_tl+                1j*Om_TLS*dt**2/6.*np.diag(U_2np1_tl)
    S_ig_je = np.dot(sp,tS)
    if M==0:
        T_ig_je = U_ig_je_0*np.tensordot(np.tensordot(tk,S_ig_je,0),tl,0) +                np.tensordot(np.tensordot(np.dot(U_ig_je_k,tk),S_ig_je,0),tl,0) +                np.tensordot(np.tensordot(tk,S_ig_je,0),np.dot(U_ig_je_l,tl),0) +                np.tensordot(np.tensordot(np.dot(U_p_tk,tk),S_ig_je,0),np.dot(U_2m_tl,tl),0)/6. +                np.tensordot(np.tensordot(np.dot(U_2m_tk,tk),S_ig_je,0),np.dot(U_p_tl,tl),0)/6.+                np.tensordot(np.tensordot(np.dot(U_m_tk,tk),S_ig_je,0),U_2np1_tl*tl,0)/6.+                np.tensordot(np.tensordot(U_2np1_tk*tk,S_ig_je,0),np.dot(U_m_tl,tl),0)/6.+                1j*dt*Om_TLS/3.*(-np.tensordot(np.tensordot(np.dot(U_m_tk,tk),S_ig_je,0),np.dot(U_m_tl,tl))+                                 np.tensordot(np.tensordot(np.dot(U_m_tk,tk),S_ig_je,0),np.dot(U_p_tl,tl))+                                 np.tensordot(np.tensordot(np.dot(U_p_tk,tk),S_ig_je,0),np.dot(U_m_tl,tl)))
    else:
        T_ig_je = U_ig_je_0*np.tensordot(tk,np.dot(S_ig_je,tl),0) +                np.tensordot(np.dot(U_ig_je_k,tk),np.dot(S_ig_je,tl),0) +                np.tensordot(tk,np.dot(S_ig_je,np.einsum("ij,kil",U_ig_je_l,tl)),0) +                np.tensordot(np.dot(U_p_tk,tk),np.dot(S_ig_je,np.einsum("ij,kil",U_2m_tl,tl)),0)/6. +                np.tensordot(np.dot(U_2m_tk,tk),np.dot(S_ig_je,np.einsum("ij,kil",U_p_tl,tl)),0)/6.+                np.tensordot(np.dot(U_m_tk,tk),np.dot(S_ig_je,np.einsum("ij,kil",np.diag(U_2np1_tl),tl)),0)/6.+                np.tensordot(U_2np1_tk*tk,np.dot(S_ig_je,np.einsum("ij,kil",U_m_tl,tl)),0)/6.+                1j*dt*Om_TLS/3.*(-np.tensordot(np.dot(U_m_tk,tk),np.dot(S_ig_je,np.einsum("ij,kil",U_m_tl,tl),0))+                                 np.tensordot(np.dot(U_m_tk,tk),np.dot(S_ig_je,np.einsum("ij,kil",U_p_tl,tl),0))+                                 np.tensordot(np.dot(U_p_tk,tk),np.dot(S_ig_je,np.einsum("ij,kil",U_m_tl,tl),0)))
    U_ig_je_0 = None
    U_ig_je_k = None
    U_ig_je_l = None
    S_ig_je = None
    
    #####Terms with \delta_{i_Te}\Delta_{j_Tg}#####
    U_ie_jg_0 = - 1j*dt*Om_TLS*( 1. -1j*.5*dt*Delta_T)
    U_ie_jg_k = ( 1. - .5j*Delta_T*dt ) * U_p_tk -                 np.dot(np.diag(U_np_tk+1)/6.,U_p_tk)-                 1j*dt*Om_TLS/6.*U_2p_tk+                1j*dt**2*Om_TLS*gamma_L/6.*(2*np.arange(0,dim_tk)+1)*sc.eye(dim_tk)
    U_ie_jg_l = ( 1. - .5j*Delta_T*dt )*U_p_tl -                 np.dot(np.diag(U_np_tl+1)/6.,U_p_tl)-                 1j*dt*Om_TLS/6.*U_2p_tl+                1j*dt**2*Om_TLS*gamma_R/6.*(2*np.arange(0,dim_tl)+1)*sc.eye(dim_tl)
    S_ie_jg = np.dot(sm,tS)
    if M==0:
        T_ie_jg = U_ie_jg_0*np.tensordot(np.tensordot(tk,S_ie_jg,0),tl,0) +                np.tensordot(np.tensordot(np.dot(U_ie_jg_k,tk),S_ie_jg,0),tl,0) +                np.tensordot(np.tensordot(tk,S_ie_jg,0),np.dot(U_ie_jg_l,tl),0) +                np.tensordot(np.tensordot(np.dot(U_m_tk,tk),S_ie_jg,0),np.dot(U_2p_tl,tl),0)/6. +                np.tensordot(np.tensordot(np.dot(U_2p_tk,tk),S_ie_jg,0),np.dot(U_m_tl,tl),0)/6.+                np.tensordot(np.tensordot(np.dot(U_p_tk,tk),S_ig_je,0),U_2np1_tl*tl,0)/6.+                np.tensordot(np.tensordot(U_2np1_tk*tk,S_ig_je,0),np.dot(U_p_tl,tl),0)/6.+                1j*dt*Om_TLS/3.*(-np.tensordot(np.tensordot(np.dot(U_p_tk,tk),S_ie_jg,0),np.dot(U_p_tl,tl))+                                 np.tensordot(np.tensordot(np.dot(U_m_tk,tk),S_ie_jg,0),np.dot(U_p_tl,tl))+                                 np.tensordot(np.tensordot(np.dot(U_p_tk,tk),S_ie_jg,0),np.dot(U_m_tl,tl)))
    else:
        T_ie_jg = U_ie_jg_0*np.tensordot(tk,np.dot(S_ie_jg,tl),0) +                np.tensordot(np.dot(U_ie_jg_k,tk),np.dot(S_ie_jg,tl),0) +                np.tensordot(tk,np.dot(S_ie_jg,np.einsum("ij,kil",U_ie_jg_l,tl)),0) +                np.tensordot(np.dot(U_m_tk,tk),np.dot(S_ie_jg,np.einsum("ij,kil",U_2p_tl,tl)),0)/6. +                np.tensordot(np.dot(U_2p_tk,tk),np.dot(S_ie_jg,np.einsum("ij,kil",U_m_tl,tl)),0)/6.+                np.tensordot(np.dot(U_p_tk,tk),np.dot(S_ig_je,np.einsum("ij,kil",np.diag(U_2np1_tl),tl)),0)/6.+                np.tensordot(U_2np1_tk*tk,np.dot(S_ig_je,np.einsum("ij,kil",U_p_tl,tl)),0)/6.+                1j*dt*Om_TLS/3.*(-np.tensordot(np.dot(U_p_tk,tk),np.dot(S_ie_jg,np.einsum("ij,kil",U_p_tl,tl),0))+                                 np.tensordot(np.dot(U_m_tk,tk),np.dot(S_ie_jg,np.einsum("ij,kil",U_p_tl,tl),0))+                                 np.tensordot(np.dot(U_p_tk,tk),np.dot(S_ie_jg,np.einsum("ij,kil",U_m_tl,tl),0)))
    U_ie_jg_0 = None
    U_ie_jg_k = None
    U_ie_jg_l = None
    S_ie_jg = None
    
    #####Terms with \delta_{i_Te}\Delta_{j_Te}#####
    U_ie_je_0 = -1j*dt*Delta_T-.5*dt**2*(Delta_T**2+Om_TLS**2)
    U_ie_je_k = np.diag(( -.5 + 1j/3.*Delta_T*dt + 1/12.*(U_np_tk+1) )*(U_np_tk+1))+                .5j*dt*Om_TLS*(U_m_tk-U_p_tk)
    U_ie_je_l = np.diag(( -.5 + 1j/3.*Delta_T*dt + 1/12.*(U_np_tl+1) )*(U_np_tl+1))+                .5j*dt*Om_TLS*(U_m_tl-U_p_tl)
    S_ie_je = np.dot(np.dot(sp,sm),tS)
    if M==0:
        T_ie_je = U_ie_je_0*np.tensordot(np.tensordot(tk,S_ie_je,0),tl,0) +                np.tensordot(np.tensordot(np.dot(U_ie_je_k,tk),S_ie_je,0),tl,0) +                np.tensordot(np.tensordot(tk,S_ie_je,0),np.dot(U_ie_je_l,tl),0) +                (-.5+1j*dt/3.*Delta_T)*np.tensordot(np.tensordot(np.dot(U_p_tk,tk),S_ie_je,0),np.dot(U_m_tl,tl),0) +                (-.5+1j*dt/3.*Delta_T)*np.tensordot(np.tensordot(np.dot(U_m_tk,tk),S_ie_je,0),np.dot(U_p_tl,tl),0) +                np.tensordot(np.tensordot(np.dot(np.dot(np.diag(U_2np1_tk),U_m_tk),tk),S_ie_je,0),np.dot(U_p_tl,tl),0)/12.+                np.tensordot(np.tensordot(np.dot(np.dot(np.diag(U_2np1_tk+1),U_p_tk),tk),S_ie_je,0),np.dot(U_m_tl,tl),0)/12.+                np.tensordot(np.tensordot(U_2np1_tk*tk,S_ie_je,0),(U_np_tl+1)*tl,0)/12.+                np.tensordot(np.tensordot((U_np_tk+1)*tk,S_ie_je,0),U_2np1_tl*tl,0)/12.+                np.tensordot(np.tensordot(np.dot(U_2m_tk,tk),S_ie_je),np.dot(U_2p_tl,tl),0)/12.+                np.tensordot(np.tensordot(np.dot(U_2p_tk,tk),S_ie_je),np.dot(U_2m_tl,tl),0)/12.+                np.tensordot(np.tensordot(np.dot(U_m_tk,tk),S_ie_je),np.dot(np.dot(np.diag(U_2np1_tl+1),U_p_tl),tl),0)/12.+                np.tensordot(np.tensordot(np.dot(U_p_tk,tk),S_ie_je),np.dot(np.dot(np.diag(U_2np1_tl),U_m_tl),tl),0)/12.
    else:
        T_ie_je = U_ie_je_0*np.tensordot(tk,np.dot(S_ie_je,tl),0) +                np.tensordot(np.dot(U_ie_je_k,tk),np.dot(S_ie_je,tl),0) +                np.tensordot(tk,np.dot(S_ie_je,np.einsum("ij,kil",U_ie_je_l,tl)),0) +                (-.5+1j*dt/3.*Delta_T)*np.tensordot(np.dot(U_p_tk,tk),np.dot(S_ie_je,np.einsum("ij,kil",U_m_tl,tl)),0) +                (-.5+1j*dt/3.*Delta_T)*np.tensordot(np.dot(U_m_tk,tk),np.dot(S_ie_je,np.einsum("ij,kil",U_p_tl,tl)),0) +                np.tensordot(np.dot(np.dot(np.diag(U_2np1_tk),U_m_tk),tk),np.dot(S_ie_je,np.einsum("ij,kil",U_p_tl,tl)),0)/12.+                np.tensordot(np.dot(np.dot(np.diag(U_2np1_tk+1),U_p_tk),tk),np.dot(S_ie_je,np.einsum("ij,kil",U_m_tl,tl)),0)/12.+                np.tensordot(U_2np1_tk*tk,np.dot(S_ie_je,np.einsum("ij,kil",np.diag(U_np_tl),tl)),0)/12.+                np.tensordot(U_np_tk*tk,np.dot(S_ie_je,np.einsum("ij,kil",np.diag(U_2np1_tl),tl)),0)/12.+                np.tensordot(np.dot(U_2m_tk,tk),np.dot(S_ie_je,np.einsum("ij,kil",U_2p_tl,tl)),0)/12.+                np.tensordot(np.dot(U_2p_tk,tk),np.dot(S_ie_je,np.einsum("ij,kil",U_2m_tl,tl)),0)/12.+                np.tensordot(np.dot(U_m_tk,tk),np.dot(S_ie_je,np.einsum("ij,kil",np.dot(np.diag(U_2np1_tl+1),U_p_tl),tl)),0)/12.+                np.tensordot(np.dot(U_p_tk,tk),np.dot(S_ie_je,np.einsum("ij,kil",np.dot(np.diag(U_2np1_tl),U_m_tl),tl)),0)/12.

    U_ie_je_0 = None
    U_ie_je_k = None
    U_ie_je_l = None
    S_ie_je = None

    #####Terms with \delta_{i_Tg}\Delta_{j_Tg}#####
    U_ig_jg_0 = -.5*dt**2*Om_TLS**2
    U_ig_jg_k = np.diag(( -.5 + 1j/6.*Delta_T*dt + 1/12.*U_np_tk )*U_np_tk)+                .5j*dt*Om_TLS*(U_m_tk-U_p_tk)
    U_ig_jg_l = np.diag(( -.5 + 1j/6.*Delta_T*dt + 1/12.*U_np_tl )*U_np_tl)+                .5j*dt*Om_TLS*(U_m_tl-U_p_tl)
    S_ig_jg = np.dot(np.dot(sm,sp),tS)
    if M==0:
        T_ig_jg = U_ig_jg_0*np.tensordot(np.tensordot(tk,S_ig_jg,0),tl,0) +                np.tensordot(np.tensordot(np.dot(U_ig_jg_k,tk),S_ig_jg,0),tl,0) +                np.tensordot(np.tensordot(tk,S_ig_jg,0),np.dot(U_ig_jg_l,tl),0) +                (-.5+1j*dt/6.*Delta_T)*np.tensordot(np.tensordot(np.dot(U_p_tk,tk),S_ig_jg,0),np.dot(U_m_tl,tl),0) +                (-.5+1j*dt/6.*Delta_T)*np.tensordot(np.tensordot(np.dot(U_m_tk,tk),S_ig_jg,0),np.dot(U_p_tl,tl),0) +                np.tensordot(np.tensordot(np.dot(np.dot(np.diag(U_2np1_tk),U_p_tk),tk),S_ig_jg,0),np.dot(U_m_tl,tl),0)/12.+                np.tensordot(np.tensordot(np.dot(np.dot(np.diag(U_2np1_tk-1),U_m_tk),tk),S_ie_je,0),np.dot(U_p_tl,tl),0)/12.+                np.tensordot(np.tensordot(U_2np1_tk*tk,S_ie_je,0),U_np_tl*tl,0)/12.+                np.tensordot(np.tensordot(U_np_tk*tk,S_ie_je,0),U_2np1_tl*tl,0)/12.+                np.tensordot(np.tensordot(np.dot(U_2m_tk,tk),S_ie_je),np.dot(U_2p_tl,tl),0)/12.+                np.tensordot(np.tensordot(np.dot(U_2p_tk,tk),S_ie_je),np.dot(U_2m_tl,tl),0)/12.+                np.tensordot(np.tensordot(np.dot(U_m_tk,tk),S_ie_je),np.dot(np.dot(np.diag(U_2np1_tl),U_p_tl),tl),0)/12.+                np.tensordot(np.tensordot(np.dot(U_p_tk,tk),S_ie_je),np.dot(np.dot(np.diag(U_2np1_tl-1),U_m_tl),tl),0)/12.
    else:
        T_ig_jg = U_ig_jg_0*np.tensordot(tk,np.dot(S_ig_jg,tl),0) +                np.tensordot(np.dot(U_ig_jg_k,tk),np.dot(S_ig_jg,tl),0) +                np.tensordot(tk,np.dot(S_ig_jg,np.einsum("ij,kil",U_ig_jg_l,tl)),0) +                (-.5+1j*dt/6.*Delta_T)*np.tensordot(np.dot(U_p_tk,tk),np.dot(S_ig_jg,np.einsum("ij,kil",U_m_tl,tl)),0) +                (-.5+1j*dt/6.*Delta_T)*np.tensordot(np.dot(U_m_tk,tk),np.dot(S_ig_jg,np.einsum("ij,kil",U_p_tl,tl)),0) +                np.tensordot(np.dot(np.dot(np.diag(U_2np1_tk),U_p_tk),tk),np.dot(S_ig_jg,np.einsum("ij,kil",U_m_tl,tl)),0)/12.+                np.tensordot(np.dot(np.dot(np.diag(U_2np1_tk-1),U_m_tk),tk),np.dot(S_ie_je,np.einsum("ij,kil",U_p_tl,tl)),0)/12.+                np.tensordot(U_2np1_tk*tk,np.dot(S_ig_jg,np.einsum("ij,kil",np.diag(U_np_tl),tl)),0)/12.+                np.tensordot(U_np_tk*tk,np.dot(S_ig_jg,np.einsum("ij,kil",np.diag(U_2np1_tl),tl)),0)/12.+                np.tensordot(np.dot(U_2m_tk,tk),np.dot(S_ig_jg,np.einsum("ij,kil",U_2p_tl,tl)),0)/12.+                np.tensordot(np.dot(U_2p_tk,tk),np.dot(S_ig_jg,np.einsum("ij,kil",U_2m_tl,tl)),0)/12.+                np.tensordot(np.dot(U_m_tk,tk),np.dot(S_ig_jg,np.einsum("ij,kil",np.dot(np.diag(U_2np1_tl),U_p_tl),tl)),0)/12.+                np.tensordot(np.dot(U_p_tk,tk),np.dot(S_ig_jg,np.einsum("ij,kil",np.dot(np.diag(U_2np1_tl-1),U_m_tl),tl)),0)/12.

    U_ig_jg_0 = None
    U_ig_jg_k = None
    U_ig_jg_l = None
    S_ig_jg = None
    
    #nextstep = T_0 + T_sqdt_1 + T_sqdt_2 + T_dt_S_1 + T_dt_S_2 + T_dt_S_3 + T_dt_B_1 + T_dt_B_2    
    nextstep = T_0 + T_ig_je + T_ie_jg + T_ie_je + T_ig_jg
    return(nextstep)



print("initial excitation:",np.einsum("k,k",np.einsum("i,ik",initTLS,np.array([[0,0],[0,1]])),np.conjugate(initTLS)))
print("initial ground probability:",np.einsum("k,k",np.einsum("i,ik",initTLS,np.array([[1,0],[0,0]])),np.conjugate(initTLS)))
m=0
exc2 = np.zeros(endt-3,complex)
norm = np.zeros(endt-3,complex)
norm2 = np.zeros(endt-2,complex)
norm2[endt-3] = np.dot(init[endt-4],np.conjugate(init[endt-4]))
norm3 = np.zeros(endt-2,complex)
norm3[endt-3] = np.dot(init[endt-4],np.conjugate(init[endt-4]))
exc = np.zeros(endt-3,complex)

#IMPORTANT: One needs to be careful with the final indices that come out of the summation of einsum, better to define them.

for m in np.arange(endt-4,-1,-1):
#    print("before\n",init[m].shape)
#    print(init[m])
#    if m<endt-4:
#        print(init[m+1].shape)
#    print(m)

    ### Calculating the norm ###
    if m==endt-4:
        #print(init[m])
        nor11 = np.dot(init[m],np.conjugate(init[m]))
        #print("nor1",nor1)
    elif m==endt-5:
        nor21 = np.einsum("ik,jk->ij",init[m+1],np.conjugate(init[m+1]))
        nor11 = np.einsum("ij,ij",np.einsum("li,lj->ij",init[m],np.conjugate(init[m])),nor21)
    else:
        nor31 = np.einsum("mli,nlj->minj",init[m+1],np.conjugate(init[m+1]))
        nor21 = np.einsum("minj,ij->mn",nor31,nor21)
        #print("nor2=",nor2)
        #print(nor2.shape)
        nor11 = np.einsum("mn,mn",np.einsum("lm,ln->mn",init[m],np.conjugate(init[m])),nor21)
    norm[m] = nor11

    ### Determining the excitation level of the qubit ###
    if m==endt-4:
        exc[m]=np.einsum("k,k",np.einsum("i,ik",init[m],np.array([[1,0],[0,0]])),np.conjugate(init[m]))
        exc2[m]=np.einsum("k,k",np.einsum("i,ik",init[m],np.array([[0,0],[0,1]])),np.conjugate(init[m]))
    else:
#        exc2[m]=np.einsum("jk,kj",np.einsum("ij,ik->jk",init[m],np.array([[0,0],[0,1]])),np.conjugate(init[m]))
        exc2_pre=np.einsum("ik,kj->ij",np.einsum("li,lk->ik",init[m],np.array([[0,0],[0,1]])),np.conjugate(init[m]))
        exc2[m] = np.einsum("ij,ij",exc2_pre,nor21)
        exc_pre=np.einsum("ik,kj->ij",np.einsum("li,lk->ik",init[m],np.array([[1,0],[0,0]])),np.conjugate(init[m]))
        exc[m] = np.einsum("ij,ij",exc_pre,nor21)
#        exc[m]=np.einsum("jk,kj",np.einsum("ij,ik->jk",init[m],np.array([[1,0],[0,0]])),np.conjugate(init[m]))
    
    ### A time-step ###
    # We consider up to 4 excitations in the environmental timebin (first array)
    together = U(initenv,init[m]) #normalized
#    print("together shape\n",together.shape)
    if m==endt-4:
        tog_swap = np.einsum("ij->ji",together)
        tog_svd  = svd(tog_swap,full_matrices=False)
    else:
        tog_swap = np.einsum("ijk->jik",together)
        di = tog_swap[0,:,0].size
        dj = tog_swap[:,0,0].size
        dk = tog_swap[0,0,:].size
#        print("di=",di,", dk=",dk)
        tog_swap_conc = np.zeros((dj,di*dk),dtype=np.complex128)
        for i in range(0,tog_swap[0,:,0].size):
            for k in range(0,tog_swap[0,0,:].size):
                tog_swap_conc[:,i+di*k]=tog_swap[:,i,k]
        tog_svd_conc = svd(tog_swap_conc,full_matrices=False)
        tog_svd = [0.+0*1j]*3
        tog_svd[2] = np.zeros((dj,di,dk),dtype=np.complex128)
        for I in range(0,tog_swap_conc[:,0].size):
            tog_svd[2][:,I%di,int((I-(I%di))/di)]  = tog_svd_conc[0][:,I]
        tog_svd[1] = tog_svd_conc[1]
        tog_svd[0] = tog_svd_conc[0]

    ### Storing the different Matrices of the timebins and system (I'm not convinced that this is necessary) ###
    init[m] = tog_svd[2]
    init[m-1] = np.dot(tog_svd[0],np.diag(tog_svd[1]))

#    print("after\n",init[m].shape)
#    print(init[m-1].shape)
#print(init[-1])

### Plotting to be happy ###
t=np.arange(0,endt-3)*dt*gamma_L
t2=np.arange(0,endt-2)*dt*gamma_L
end = time.time()
timeh = int((end-start)/3600)
timem = int((end-start)/60-timeh*60)
times = int((end-start)-timeh*3600-60*timem)
print("elapsed time: %02d:%02d:%02d" % (timeh,timem,times))


plt.figure(figsize = (12,7))
plt.plot(t,np.flip(np.real(exc),0),lw=2,label="ground",color=colors["green"])
plt.plot(t,np.flip(np.real(exc2),0),lw=2,label="excited",color=colors["blue"])
#plt.plot(t,np.exp(-gamma_L*t),ls="--",lw=4,color=colors["orange"],label="$\exp{(-\gamma_Lt)}$")
#plt.xlim(0,8)
#plt.ylim(-0.01,1.01)
plt.xlabel("$\gamma_Lt$",fontsize=40)
plt.ylabel("$\left<\sigma_+\sigma_-\\right>$",fontsize=40)
plt.legend(fontsize=25,loc="upper right")
plt.grid(True)
plt.figure(2,figsize = (12,7))
plt.plot(t,np.flip(np.real(norm-1),0),lw=2,label="norm",color=colors["purple"])
plt.grid(True)
plt.ylabel("$\left<\psi(t)|\psi(t)\\right>-1$",fontsize=40)
plt.xlabel("$\gamma_Lt$",fontsize=40)
plt.show()


# In[47]:


a1 = np.array([0,1,2,3,4])
a2 = np.array([0,1,2,3])
a3 = np.array([0,1,2])
a4 = np.array([0,1])

b = np.tensordot(np.tensordot(a1,a2,0),a3,0)
c = np.tensordot(a3,a4,0)
d = np.dot(b,c)
d.shape


# In[11]:


a = np.arange(0,4)
b = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])

print(b*a)
print(np.dot(a*sc.eye(4),b))
print(np.dot(np.diag(a),b))


# In[21]:


Um=sc.eye(4,4,1)*np.sqrt(np.arange(0,4))
Up=sc.eye(4,4,-1)*np.sqrt(np.arange(1,5))
v=np.arange(0,4)-1
np.dot(np.dot(np.dot(Um,Up),Um),np.array([0,0,1,0]))

