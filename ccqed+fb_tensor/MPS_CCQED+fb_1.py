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
from U_CCQED_1 import *
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

########################################################################################################################################################################################################
#############
### Timer ###
#############

start = time.time()

########################################################################################################################################################################################################
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
parser.add_argument("findex",type=int,help='Index number for the file')
parser.add_argument("gamB1",type=float,help='gamma_B1')
parser.add_argument("gamB2",type=float,help='gamma_B2')
parser.add_argument("gamF1",type=float,help='gamma_F1')
parser.add_argument("gamF2",type=float,help='gamma_F2')
parser.add_argument("g1",type=float,help='g1: cavity-atom coupling on the left')
parser.add_argument("g2",type=float,help='g2: cavity-atom coupling on the right')
parser.add_argument("phi",type=float,help='phi/pi phase of the propagation from one cavity to another')
parser.add_argument("-initind1",type=int,default = 0,help='initial index of system 1 vector')
parser.add_argument("-initind2",type=int,default = 0,help='initial index of system 2 vector')
parser.add_argument("-Ome1",type=float,default = 0,help='Omega_e1: direct driving strength of the TLS on the left')
parser.add_argument("-Ome2",type=float,default = 0,help='Omega_e2: direct driving strength of the TLS on the right')
parser.add_argument("-Omc1",type=float,default = 0,help='Omega_c1: driving strength for the cavity on the left')
parser.add_argument("-Omc2",type=float,default = 0,help='Omega_c2: driving strength for the cavity on the right')
parser.add_argument("-tol",type=float,default = -3,help='tolerance')
parser.add_argument("-Nphot",type=int,default = 4,help='maximal boson number')
parser.add_argument("-L",type=int,default = 10,help='delay as number of dt')
parser.add_argument("-endt",type=float, default = 10., help ='end of time evolution')
parser.add_argument("-dt",type=float,default = 0.5, help ='dt')
parser.add_argument("-cohC1",type=float, default = 0.,help='coherent initial state for cavity 1')
parser.add_argument("-cohC2",type=float, default = 0.,help='coherent initial state for cavity 2')
parser.add_argument("-cohB1",type=float, default = 0.,help='coherent initial state for the environment on the left')
parser.add_argument("-cohB2",type=float, default = 0.,help='coherent initial state for the environment on the right')
parser.add_argument("-cohF",type=float, default = 0.,help='coherent initial state for the connecting fibre')
parser.add_argument("-nT",type=float, default = 0.,help='thermal photon number')

args = parser.parse_args()

########################################################################################################################################################################################################
##################
### Parameters ###
##################

tol     = 10**(args.tol)
endt    = args.endt
dt      = args.dt
L       = args.L#50
N       = int(endt/dt)
len_env = args.Nphot+1
len_sys = 2*args.Nphot+1

g       = np.array([args.g1,args.g2])
gamma_B = np.array([args.gamB1,args.gamB2])
gamma_F = np.array([args.gamF1,args.gamF2])
Ome     = np.array([args.Ome1,args.Ome2])
Omc     = np.array([args.Omc1,args.Omc2])
Dele    = np.array([0.,0.])
Delc    = np.array([0.,0.])
phi     = args.phi*np.pi
thermal = False

########################################################################################################################################################################################################
################################
### MPS state initialization ###
################################

#%%%%%%%%#
# SYSTEM #
#%%%%%%%%#
initJC1 = np.zeros(len_sys,complex)
initJC2 = np.zeros(len_sys,complex)
#Coherent driving for cavities
if args.cohC1>0. or args.cohC2>0.:
	preJC   = coherent(args.cohC,0,np.zeros(N_env+1,complex))
	prenorm = np.sqrt(np.sum(preJC**2))
	if args.cohC1>0. and args.cohC2==0.:
		if args.initind1==0:
			initJC1[0::2] = preJC/prenorm
		elif args.initind1==1:
			initJC1[1::2] = preJC[:-1]/prenorm
		initJC2[args.initind2]  = 1.
	elif args.cohC1>0. and args.cohC2==0.:
		initJC1[args.initind1]  = 1.
		if args.initind2==0:
			initJC2[0::2] = preJC/prenorm
		elif args.initind2==1:
			initJC2[1::2] = preJC[:-1]/prenorm
	else:
		if args.initind1==0:
			initJC1[0::2] = preJC/prenorm
		elif args.initind1==1:
			initJC1[1::2] = preJC[:-1]/prenorm
		if args.initind2==0:
			initJC2[0::2] = preJC/prenorm
		elif args.initind2==1:
			initJC2[1::2] = preJC[:-1]/prenorm
	preJC = None
#Fock initial state
else:
	initJC1[args.initind1] = 1.
	initJC2[args.initind2] = 1.

#%%%%%%%%%%%%%#
# ENVIRONMENT #
#%%%%%%%%%%%%%#
initB1 = np.zeros(len_env,complex)
initB2 = np.zeros(len_env,complex)
initF  = np.zeros(len_env,complex)

#Coherent driving for environments
if args.cohB1>0. or args.cohB2>0. or args.cohF>0.:
	if args.cohB1>0.:
		initB1 = coherent(args.cohB1,0,initB1)
	if args.cohB2>0.:
		initB2 = coherent(args.cohB2,0,initB2)
	if args.cohF>0.:
		initF  = coherent(args.cohF,0,initF)
#Thermal state for environments
elif args.nT>0.:
	thermal = True
	phots = np.linspace(0,N_env-1,N_env)
	rhotherm = np.diag(1./np.sqrt(1+args.nT)*(args.nT/(args.nT+1))**(phots/2.))
	initB1  = rhotherm.reshape(N_env**2)
	initB2 += initB1
	initF  += initB1
#Vacuum initial state of environments
else:
	initB1[0] = 1.
	initB2[0] = 1.
	initF[0]  = 1.

#Markovian environment lists
statesB1 = [initB1]*N
statesB2 = [initB2]*N
#Non-Markovian environment+system list
statesF  = 2*L*[initF]
statesS  = [initJC1]+[initJC2]
#print("statesS",statesS)

#g2_ta1,NB1,NB2 = g2_t(states[ind_sys-1],N_env+1,dt,thermal)
#NB_outa = 0.

#Initial contribution for the norm
normB1 = 1.
normB2 = 1.

# sigma_z construction
z      = np.zeros(len_sys,complex)
z[np.arange(0,len_sys,2)]=np.ones(len_env)
sgg    = np.diag(z)
see    = np.identity(len_sys)-sgg
sz     = see-sgg
#print("sigmaz",sz)

# n_c and g2_c construction
ncdiag = (np.linspace(0,len_sys-1,len_sys)/2.).astype(np.int64)
nc     = np.diag(ncdiag)
#print("nc",nc)
gcdiag = np.zeros(len_env,complex)
for i in range(1,len_env):
	gcdiag[i] = gcdiag[i-1]+2*(i-1)
g2c    = np.diag(np.sort(np.concatenate((gcdiag[:-1],gcdiag))))


########################################################################################################################################################################################################
###########################
### File initialization ###
###########################

filename = "./Data/CCQED+fb_%d.txt" % (args.findex)
outname = "./Data/OUT_CCQED+fb_%d.txt" % (args.findex)
#	specname = "./Data/spec_JC+fb_gL=%dp1000_gR=%dp1000_g=%dp10_phi=%dp10pi_initind=%d_ome=%dp10_omc=%dp10_L=%d.txt" % (gamma_L*1000, gamma_R*1000, g*10, args.phi*10,args.init_ind,Ome*10,Omc*10,L)
#	g2tau = "./Data/g2tau_JC+fb_gL=%dp1000_gR=%dp1000_g=%dp10_phi=%dp10pi_initind=%d_ome=%dp10_omc=%dp10_L=%d.txt" % (gamma_L*1000, gamma_R*1000, g*10, args.phi*10,args.init_ind,Ome*10,Omc*10,L)
	
#%%%%%%%%%%%#
# Info file #
#%%%%%%%%%%%#
file_out = open(outname,"a")
file_out.close()
file_out = open(outname,"r+")
file_out.truncate()
file_out.close()
file_out = open(outname,"a")

file_out.write("""Data file index: %d
\nSystem1 parameters: g1 = %f, Delta_e1 = %f, Delta_c1 = %f, initial index: %d, Omega_e1 = %f, Omega_c1 = %f
\nSystem 2 parameters: g2 = %f, Delta_e2 = %f, Delta_c2 = %f, initial index: %d, Omega_e2 = %f, Omega_c2 = %f
\nEnvironment parameters: gamma_B1 = %f, gamma_B2 = %f
\nConnecting fibre parameters: gamma_F1 = %f, gamma_F2 = %f,phi = %fpi, delay_L = %d
\nNumerical parameters: Nphot_max = %d, tolerance = %.0E, endt = %.0f, dt = %f
\nCoherent initial state amplitude for cavity1: %f, cavity2: %f, the environment on the left %f, on the right: %f and in the fibre: %f
\nthermal photon number: %f
\nData file: M*dt,norm,pop1,pop2,nc1_exp,nc2_exp,g2_1_exp,g2_2_exp\n""" % (args.findex,g[0],Dele[0],Delc[0],args.initind1,Ome[0],Omc[0],
								g[1],Dele[1],Delc[1],args.initind2,Ome[1],Omc[1],
								gamma_B[0],gamma_B[1],gamma_F[0],gamma_F[1],phi,L,
								args.Nphot,tol,endt,dt,args.cohC1,args.cohC2,
								args.cohB1,args.cohB2,args.cohF,args.nT))
file_out.close()

#%%%%%%%%%%%#
# Data file #
#%%%%%%%%%%%#
file_evol = open(filename,"a")
file_evol.close()
file_evol = open(filename,"r+")
file_evol.truncate()
file_evol.close()
file_evol = open(filename,"a")

########################################################################################################################################################################################################
######################
### Time evolution ###
######################

for M in range(0,N):
	#    print(M*dt)
	percent10 = (N)/10.
	count = 0
	if M%(int(percent10))==0:
	#        count=count+5
		print("M =",M, " out of ",N)
	sys.stdout.flush()
	
	#%%%%%%%%%%%%%%%#
	# SWAP BEFORE U #
	#%%%%%%%%%%%%%%%#

	# After the first time step, bring the interacting past bin next to the system bin
	#SWAP_U(M,L,F,S,tol)
	if M>0:
		statesF = SWAP_U(M,L,statesF,tol)

	#%%%%%%#
	# NORM #
	#%%%%%%#
	##normf(M,L,statesB1,statesB2,statesF,statesS,normB1,normB2)
	##return np.real(norm),np.real(normB1),np.real(normB2),sys_state
	norm,normB1,normB2,sys_state = normf(M,L,statesB1,statesB2,statesF,statesS,normB1,normB2)

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%#
	# SYSTEM EXPECTATION VALUES #
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%#
	#exp_sys(observable,sys_state,which)
	nc_exp1  = exp_sys(nc,sys_state,1) #photon number in cavity 1
	nc_exp2  = exp_sys(nc,sys_state,2) #photon number in cavity 2
	pop_exp1 = exp_sys(sz,sys_state,1) #atomic population inversion in cavity 1
	pop_exp2 = exp_sys(sz,sys_state,2) #atomic population inversion in cavity 2
	g2_exp1  = exp_sys(g2c,sys_state,1)/nc_exp1**2 #correlation function from the field in cavity 1 
	g2_exp2  = exp_sys(g2c,sys_state,2)/nc_exp2**2 #correlation function from the field in cavity 2 
	file_evol.write("%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\n" %(M*dt,norm,pop_exp1,pop_exp2,nc_exp1,nc_exp2,g2_exp1,g2_exp2))
	file_evol.flush()
	file_out.close()


	#    # Erase the remnants of the states that will not influence the dynamics anymore:
	#    if M>0:
	#        statesB[M-1] = None
	#   It is needed for the spectral calculations

	#%%%%%%%%%%%%%%%%%%%%#
	# TIME EVOLUTION MAP #
	#%%%%%%%%%%%%%%%%%%%%#
	
	F_ind = M # current index for F, which interacts with system 1
	
	# Time evolution map
	#--------------------
	##U(M,L,tF1,tF2,tS,tB1,tB2,gamma_B,gamma_F,dt,phi,Ome,Omc,g,Delc,Dele)
#	print(len(statesB1),len(statesB2),M,len(statesF),F_ind,F_ind+L)
	U_block = U(M,L,statesF[F_ind%(2*L)],statesF[(F_ind+L)%(2*L)],statesS,statesB1[M],statesB2[M],gamma_B,gamma_F,dt,phi,Ome,Omc,g,Delc,Dele)
	#    print("U block",U_block.shape)

	# Split into original blocks
	#----------------------------
	Sdim = 3 + int(np.any(M))*2 # Number of indices for each side (S+F+B)
	# Merging indices per side
	U_block,U1_dim,U2_dim = merge(U_block,Sdim,Sdim)
	# Separating the two sides with 2 link indices
	U1_block,U2_block = SVD_split(U_block,tol)
	# Unmerging the indices of block 1 on the left
	##unmerge(block,dL,dR,which=0)
#	print("u1_block dim:",U1_dim,", u1 block:", U1_block.shape)
	U1_block = unmerge(U1_block,U1_dim,np.array([1]),1)
#	print("u1 block:", U1_block.shape)
	# Saving the new states from block 1 in their corresponding lists
	##split(M,block,which,tol)
	statesF[(F_ind)%(2*L)],statesB1[M],statesS[0] = split(M,U1_block,1,tol)
#	print("M=",M,", F1:",statesF[(F_ind)%(2*L)].shape,", B1:",statesB1[M].shape,", S1:",statesS[0].shape)
	# Unmerging the indices of block 2 on the right
	U2_block = unmerge(U2_block,np.array([1]),U2_dim,2)
	# Saving the new states from block 2 in their corresponding lists
	statesF[(F_ind+L)%(2*L)],statesB2[M],statesS[1] = split(M,U2_block,2,tol)
#	print("M=",M,", F2:",statesF[(F_ind+L)%(2*L)].shape,", B2:",statesB2[M].shape,", S2:",statesS[1].shape)

	# SWAP
	#------
	# Moving the interacting past bin's state back to its original position in the MPS
	#SWAP_back(M,L,F,S,tol)
	statesF,statesS = SWAP_back(M,L,statesF,statesS,tol)
#	print("system",statesS[0].shape,statesS[1].shape)
#	print("fibre",statesF[(M)%(2*L)].shape,statesF[(M+1)%(2*L)].shape,statesF[(M+L)%(2*L)].shape,statesF[(M+L+1)%(2*L)].shape)
#	g2_ta,NB = g2_t(states[M],N_env+1,dt,thermal)
#	NB_outa = NB_out(states[M],N_env+1,NB_outa,dt,thermal)


#Calculating the output spectra over a range of frequencies
#om = np.linspace(-20,20,5000)
#spec = spectrum(states,om,N-L-1,N_env+1,dt,N-L-1,thermal)
#tau,g2_outa = g2_out(states,N-L-1,N_env+1,dt,N-L-1,thermal)
#time_out = np.transpose(np.vstack((om,spec)))
#f = open(specname, 'a')
#f.close()
#f = open(specname, 'r+')
#f.truncate()
#np.savetxt(specname,time_out)
#time_out = np.transpose(np.vstack((tau,g2_outa)))
#f = open(g2tau, 'a')
#f.close()
#f = open(g2tau, 'r+')
#f.truncate()
#np.savetxt(g2tau,time_out)
#time_out=None

# SWAP back to calculate the norm and the expectation values
statesF = SWAP_U(M+1,L,statesF,tol)

#%%%%%%#
# NORM #
#%%%%%%#
##normf(M,L,statesB1,statesB2,statesF,statesS,normB1,normB2)
##return np.real(norm),np.real(normB1),np.real(normB2),sys_state
norm,normB1,normB2,sys_state = normf(M+1,L,statesB1,statesB2,statesF,statesS,normB1,normB2)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# SYSTEM EXPECTATION VALUES #
#%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#exp_sys(observable,sys_state,which)
nc_exp1  = exp_sys(nc,sys_state,1) #photon number in cavity 1
nc_exp2  = exp_sys(nc,sys_state,2) #photon number in cavity 2
pop_exp1 = exp_sys(sz,sys_state,1) #atomic population inversion in cavity 1
pop_exp2 = exp_sys(sz,sys_state,2) #atomic population inversion in cavity 2
g2_exp1  = exp_sys(g2c,sys_state,1)/nc_exp1**2 #correlation function from the field in cavity 1 
g2_exp2  = exp_sys(g2c,sys_state,2)/nc_exp2**2 #correlation function from the field in cavity 2 
file_evol.write("%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\n" %((M+1)*dt,norm,pop_exp1,pop_exp2,nc_exp1,nc_exp2,g2_exp1,g2_exp2))
file_evol.flush()
file_out.close()

end = time.time()-start
h = int(end/3600)
m = int((end-3600*h)/60)
s = int(end-3600*h-60*m)
#print(#"final excited state population:",exc_pop[-1],
#      "\nfinal ground state population:",gr_pop[-1],
file_out = open(outname,"a")
file_out.write("\nfinished in: %02d:%02d:%02d" %(h,m,s))
file_out.close()
