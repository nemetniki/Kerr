#!/usr/bin/python3.5
# coding: utf-8

# # MPS code

# In[7]:


import numpy as np
import scipy as sc
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.linalg import svd
import time
import sys
import argparse
from decimal import Decimal
from math import factorial
from MPS_fund_therm import *
from U_JC_therm import *
from MPS_analysis_therm import *

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

#**************#
#***--------***#
#***| CODE |***#
#***--------***#
#**************#

#############
### Timer ###
#############
start = time.time()

###################
### Code inputs ###
###################
def str2bool(v):
	if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser(prog='MPS Jaynes-Cummings+feedback',
				description = '''Calculating the evolution of the occupation of the TLS levels
						the photon number inside the cavity and the norm 
						for different driving and feedback conditions''',
				formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("ID",type=int,help='ID number for the process')
parser.add_argument("gamma_L",type=float,help='gamma_L')
parser.add_argument("gamma_R",type=float,help='gamma_R')
parser.add_argument("g",type=float,help='g: cavity atom coupling')
parser.add_argument("phi",type=float,help='phi/pi (note that constructive interference is pi, destructive is 0)')
parser.add_argument("-init_ind",type=int,default = 0,help='initial index of system vector')
parser.add_argument("-Om_e",type=float,default = 0,help='Om_e: direct driving strength of the TLS')
parser.add_argument("-Om_c",type=float,default = 0,help='Om_c: driving strength for the cavity')
parser.add_argument("-tol",type=float,default = -3,help='tolerance')
parser.add_argument("-Nphot",type=int,default = 50,help='maximal boson number')
parser.add_argument("-L",type=int,default = 10,help='delay as number of dt')
parser.add_argument("-endt",type=float, default = 10, help ='end of time evolution')
parser.add_argument("-dt",type=float,default = 0.005, help ='dt')
parser.add_argument("-cohC",type=float, default = 0.,help='coherent initial state for the cavity')
parser.add_argument("-cohE",type=float, default = 0.,help='coherent initial state for the environment')
parser.add_argument("-nT",type=float, default = 0.,help='thermal photon number')

args = parser.parse_args()


##################
### Parameters ###
##################
tol     = 10**(args.tol)
endt  = args.endt
dt    = args.dt
L     = args.L#50
N     = int(endt/dt)+L+1
N_env = args.Nphot

g       = args.g#2*np.pi/(L*dt)
gamma_L = args.gamma_L#2*g#7.5
gamma_R = args.gamma_R#2*g#7.5
Ome     = args.Om_e#0.#1.5*np.pi
Omc     = args.Om_c#0.#1.5*np.pi
Dele    = 0.#1.0
Delc    = 0.#1.0
phi     = args.phi*np.pi#pi
thermal = False
################################
### MPS state initialization ###
################################
initJC     = np.zeros(2*N_env+1,complex)
if args.cohC>0.:
	preinitJC = coherent(args.cohC,0,np.zeros(N_env+1,complex))
	if args.init_ind==0:
		initJC[0::2] = preinitJC
	elif args.init_ind==1:
		initJC[1::2] = preinitJC[:-1]
	preinitJC = None
else:
	initJC[args.init_ind]  = 1. #starting at |e>
initenv    = np.zeros(N_env+1,complex)
if args.cohE>0.:
	initenv = coherent(args.cohE,0,initenv)
elif args.nT>0.:
	thermal = True
	phots = np.linspace(0,N_env-1,N_env)
	rhotherm = np.diag(1./np.sqrt(1+args.nT)*(args.nT/(args.nT+1))**(phots/2.))
	initenv = rhotherm.reshape(N_env**2)
else:
	initenv[0] = 1.
states     = [initenv]*L+(N-L)*[0.]
ind_sys       = L
states[ind_sys] = initJC/np.sqrt(np.sum(initJC**2))

g2_ta,NB = g2_t(states[ind_sys-1],N_env+1,dt,thermal)
NB_outa = 0.
normL = 1.

z = np.zeros(2*N_env+1,complex)
z[np.arange(0,2*N_env+1,2)]=np.ones(N_env+1)
sgg = np.diag(z)
see = np.identity(2*N_env+1)-sgg
ncdiag = np.linspace(0,N_env+.1,2*N_env+1).astype(np.int64)
nc = np.diag(ncdiag)
gcdiag = np.zeros(N_env+1,complex)
for i in range(1,N_env+1):
	gcdiag[i] = gcdiag[i-1]+2*(i-1)
g2c = np.diag(np.sort(np.concatenate((gcdiag[:-1],gcdiag))))


if args.cohC>0:
	filename = "../Data/JC+fb/Old/evol%04d_coherent_cavity.txt" % (args.ID)
	outname = "../Data/JC+fb/Old/OUT%04d_coherent_cavity.txt" % (args.ID)
	specname = "../Data/JC+fb/Old/spec%04d_coherent_cavity.txt" % (args.ID)
	g2tau = "../Data/JC+fb/Old/g2tau%04d_coherent_cavity.txt" % (args.ID)
elif args.cohE>0:
	filename = "../Data/JC+fb/Old/evol%04d_coherent_environment.txt" % (args.ID)
	outname = "../Data/JC+fb/Old/OUT%04d_coherent_environment.txt" % (args.ID)
	specname = "../Data/JC+fb/Old/spec%04d_coherent_environment.txt" % (args.ID)
	g2tau = "../Data/JC+fb/Old/g2tau%04d_coherent_environment.txt" % (args.ID)
elif Ome>0:
	filename = "../Data/JC+fb/Old/evol%04d_atom_drive.txt" % (args.ID)
	outname = "../Data/JC+fb/Old/OUT%04d_atom_drive.txt" % (args.ID)
	specname = "../Data/JC+fb/Old/spec%04d_atom_drive.txt" % (args.ID)
	g2tau = "../Data/JC+fb/Old/g2tau%04d_atom_drive.txt" % (args.ID)
elif Omc>0:
	filename = "../Data/JC+fb/Old/evol%04d_cavity_drive.txt" % (args.ID)
	outname = "../Data/JC+fb/Old/OUT%04d_cavity_drive.txt" % (args.ID)
	specname = "../Data/JC+fb/Old/spec%04d_cavity_drive.txt" % (args.ID)
	g2tau = "../Data/JC+fb/Old/g2tau%04d_cavity_drive.txt" % (args.ID)
else:
	filename = "../Data/JC+fb/Old/evol%04d.txt" % (args.ID)
	outname = "../Data/JC+fb/Old/OUT%04d.txt" % (args.ID)
	specname = "../Data/JC+fb/Old/spec%04d.txt" % (args.ID)
	g2tau = "../Data/JC+fb/Old/g2tau%04d.txt" % (args.ID)
	
file_out = open(outname,"a")
file_out.close()
file_out = open(outname,"r+")
file_out.truncate()
file_out.close()
file_out = open(outname,"a")

file_out.write("""\ngamma_L = %f, gamma_R = %f, Om_e = %f, Om_c = %f, phi = %fpi,
Delta_e = %f, Delta_c = %f, g = %f, Nphot_max = %d, initind = %d,
tolerance = %.0E, delay_L = %d, endt = %.0f, dt = %f\n
coherent initial state amplitude for cavity: %f and the environment %f\n
thermal photon number: %f\n
Data file: M*dt,norm,exc_pop,gr_pop,nc_exp,g2_ta,NB,NB_outa\n""" % (gamma_L, gamma_R, Ome, Omc, args.phi, Dele, Delc, g, N_env, args.init_ind, Decimal(tol), L, endt, dt,args.cohC,args.cohE,args.nT))
file_out.close()

file_evol = open(filename,"a")
file_evol.close()
file_evol = open(filename,"r+")
file_evol.truncate()
file_evol.close()
file_evol = open(filename,"a")

######################
### Time evolution ###
######################
for M in range(0,N-L-1):
#    print(M*dt)
    percent10 = (N-L-1)/10.
    count = 0
    if M%(int(percent10))==0:
#        count=count+5
        print("M =",M, " out of ",N-L-1)
        sys.stdout.flush()
    
#        print("%d percent" % count)
 
    # After the first time step, bring the interacting past bin next to the system bin
    if M>0:
        # Relocating the orthogonality centre to the next interacting past bin if applicable
        states[M],states[M-1] = OC_reloc(states[M],states[M-1],"left",tol)
        states[M:M+L] = SWAP(states,M,"future",L,tol)
                
    # Relocating the orthogonality centre from the past bin to the system bin before the evolution
    # operator's action if applicable
    states[ind_sys],states[ind_sys-1] = OC_reloc(states[ind_sys],states[ind_sys-1],"left",tol)
        
    norm,normL = normf(M,L,states,normL)
    nc_exp = exp_sys(nc,states[ind_sys],M)
    exc_pop = exp_sys(see,states[ind_sys],M)
    gr_pop  = exp_sys(sgg,states[ind_sys],M)
    g2_tac  = exp_sys(g2c,states[ind_sys],M)/nc_exp**2
    file_evol.write("%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\n" %(M*dt,norm,exc_pop,gr_pop,nc_exp,g2_tac,g2_ta,NB,NB_outa))
    file_evol.flush()
    file_out.close()


#    # Erase the remnants of the states that will not influence the dynamics anymore:
#    if M>0:
#        states[M-1] = None
#   It is needed for the spectral calculations

    # The time evolution operator acting on the interacting state bins
#    print("initial shapes",initenv.shape, states[ind_sys].shape, states[ind_sys-1].shape)
    U_block = U(initenv,states[ind_sys],states[ind_sys-1],N_env,M,gamma_L,gamma_R,dt,phi,Ome,Omc,g,Delc,Dele,thermal)
#    print("U block",U_block.shape)

    # Merging of the link index on the right into a new tensor if applicable    
    U_right_merge=False
    if len(U_block.shape)>3:
        U_right_merge=True
        U_block,U_right_dims = merge(U_block,"right")
    # Exchanging the position of the system and the present bin in the MPS state list
    U_block  = np.einsum("ijk->jik",U_block)
#    print("U block merge right",U_block.shape)
    
    # Separating the system state from the environment bins
    states[ind_sys+1],U_small_block = cut(U_block,tol,"right")
#    print("tS and rest",states[ind_sys+1].shape,U_small_block.shape)
    U_block = None
    # Separating the present time bin from the interacting past bin
    states[ind_sys],states[ind_sys-1] = cut(U_small_block,tol,"left")
#    print("tk and tl",states[ind_sys].shape,states[ind_sys-1].shape)
    U_small_block=None
    # Unmerging of the previously merged link index on the right if applicable
    if U_right_merge:
        if len(states[ind_sys-1].shape)==1:
            U_dims = U_right_dims
        else:
            U_dims = np.concatenate((np.array([states[ind_sys-1].shape[0]]),U_right_dims),axis = 0)
        states[ind_sys-1] = unmerge(states[ind_sys-1],U_dims,"right")
        U_dims = None
#    print("tl final",states[ind_sys-1].shape)
#    print("U done, states done", states[ind_sys+1].shape, states[ind_sys].shape, states[ind_sys-1].shape)
        
    # Moving the interacting past bin's state back to its original position in the MPS
    states[(ind_sys-L):(ind_sys)] = SWAP(states,(ind_sys-2),"past",L,tol)
    g2_ta,NB = g2_t(states[M],N_env+1,dt,thermal)
    NB_outa = NB_out(states[M],N_env+1,NB_outa,dt,thermal)
    # Preparing for the next step with the system index
    ind_sys =1+ind_sys

# restoring the normalization after the time step with moving the past bin next to the system state
# and relocating the orthogonality centre
if len(states[M].shape)>1:
    states[M+1],states[M] = OC_reloc(states[M+1],states[M],"left",tol)

#Calculating the output spectra over a range of frequencies
om = np.linspace(-20,20,5000)
index=int(20/dt)
spec = spectrum(states,om,index,N_env+1,dt,index,thermal)
tau,g2_outa = g2_out(states,N-L-1,N_env+1,dt,N-L-1,thermal)
time_out = np.transpose(np.vstack((om,spec)))
f = open(specname, 'a')
f.close()
f = open(specname, 'r+')
f.truncate()
np.savetxt(specname,time_out)
time_out = np.transpose(np.vstack((tau,g2_outa)))
f = open(g2tau, 'a')
f.close()
f = open(g2tau, 'r+')
f.truncate()
np.savetxt(g2tau,time_out)
time_out=None

states[M+1:M+L+1] = SWAP(states,M+1,"future",L,tol)
if len(states[ind_sys-1].shape)>1:
    states[ind_sys],states[ind_sys-1] = OC_reloc(states[ind_sys],states[ind_sys-1],"left",tol)

# Calculating the last value of the norm and the excited and ground population in time
norm,normL = normf(N-L-1,L,states,normL)
nc_exp = exp_sys(nc,states[N-1],N-L-1)
exc_pop = exp_sys(see,states[N-1],N-L-1)
gr_pop  = exp_sys(sgg,states[N-1],N-L-1)
#	file_evol = open("./Data/TLS+feedback_gL=%dp10_gR=%dp10_Om=%dp10_phi=%dp10pi.txt" % \
#			(gamma_L*10, gamma_R*10, Om_TLS*10, args.phi*10),"a")
g2_tac  = exp_sys(g2c,states[N-1],N-L-1)/nc_exp**2
file_evol.write("%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\n" %(M*dt,norm,exc_pop,gr_pop,nc_exp,g2_tac,g2_ta,NB,NB_outa))

end = time.time()-start
h = int(end/3600)
m = int((end-3600*h)/60)
s = int(end-3600*h-60*m)
#print(#"final excited state population:",exc_pop[-1],
#      "\nfinal ground state population:",gr_pop[-1],
file_out = open(outname,"a")
file_out.write("\nfinished in: %02d:%02d:%02d" %(h,m,s))
file_out.close()
