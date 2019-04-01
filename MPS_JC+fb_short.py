#!/usr/bin/python3.5
# coding: utf-8

# # MPS code

# In[7]:


import numpy as np
import scipy as sc
import os
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
from U_JC import *
from MPS_analysis import *

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
parser.add_argument("-Wigner",type=bool,default = False,help='Wigner function')
parser.add_argument("-spectrum",type=bool,default = False,help='spectrum and g2(tau)')
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
parser.add_argument("-Dele",type=float, default = 0.,help='Detuning of the atom')
parser.add_argument("-Delc",type=float, default = 0.,help='Detuning of the cavity')
parser.add_argument("-spec_cutoff",type=int, default = 80,help='length of time in terms of tau over which the spectrum is calculated')

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
Dele    = args.Dele#1.0
Delc    = args.Delc#1.0
phi     = args.phi*np.pi#pi

################################
### MPS state initialization ###
################################
initJC     = np.zeros(2*N_env+1,complex)
if args.cohC>0.:
	preinitJC = coherentS(np.sqrt(args.cohC),np.zeros(N_env+1,complex))
	initJC[0::2] = preinitJC
	preinitJC = None
else:
	initJC[args.init_ind]  = 1. #starting at |e>

initenv    = np.zeros(N_env+1,complex)
if args.cohE>0.:
	initenv = coherentE(args.cohE/np.sqrt(gamma_L),initenv)
else:
	initenv[0] = 1.
states     = [initenv]*L+(N-L)*[0.]
ind_sys       = L
states[ind_sys] = initJC/np.sqrt(np.sum(initJC**2))

NB_outa = 0.
normL = 1.

z = np.zeros(2*N_env+1,complex)
z[np.arange(0,2*N_env+1,2)]=np.ones(N_env+1)
sgg = np.diag(z)
see = np.identity(2*N_env+1)-sgg
ncdiag = np.linspace(0,N_env+.1,2*N_env+1).astype(np.int64)
nc = np.diag(ncdiag)
nc2 = np.diag(ncdiag-1)
g2cop = nc*nc2

nBdiag = np.linspace(0,N_env,N_env+1).astype(np.int64)
nB = np.diag(nBdiag)
nB2 = np.diag(nBdiag-1)
g2Bop = nB*nB2

Bdens = env_dens(states[0])
Cdens = cav_dens(states[ind_sys],N_env)
NB    = np.einsum("ij,ij",Bdens,nB)
g2_t = np.einsum("ij,ij",Bdens,g2Bop)/NB**2

##################################
### Output file initialization ###
##################################

if args.cohC>0:
	filename = "./Data/JC+fb/New/evol%04d_coherent_cavity.txt" % (args.ID)
	outname = "./Data/JC+fb/New/OUT%04d_coherent_cavity.txt" % (args.ID)
	specname = "./Data/JC+fb/New/spec%04d_coherent_cavity.txt" % (args.ID)
	g2tau = "./Data/JC+fb/New/g2tau%04d_coherent_cavity.txt" % (args.ID)
	corrf = "./Data/JC+fb/New/corr%04d_coherent_cavity.txt" % (args.ID)
	Bdenspath = "./Data/JC+fb/New/Bdens%04d_coherent_cavity" % (args.ID)
	Cdenspath = "./Data/JC+fb/New/Cdens%04d_coherent_cavity" % (args.ID)
elif args.cohE>0:
	filename = "./Data/JC+fb/New/evol%04d_coherent_environment.txt" % (args.ID)
	outname = "./Data/JC+fb/New/OUT%04d_coherent_environment.txt" % (args.ID)
	specname = "./Data/JC+fb/New/spec%04d_coherent_environment.txt" % (args.ID)
	g2tau = "./Data/JC+fb/New/g2tau%04d_coherent_environment.txt" % (args.ID)
	corrf = "./Data/JC+fb/New/corr%04d_coherent_environment.txt" % (args.ID)
	Bdenspath = "./Data/JC+fb/New/Bdens%04d_coherent_environment" % (args.ID)
	Cdenspath = "./Data/JC+fb/New/Cdens%04d_coherent_environment" % (args.ID)
elif Ome>0:
	filename = "./Data/JC+fb/New/evol%04d_atom_drive.txt" % (args.ID)
	outname = "./Data/JC+fb/New/OUT%04d_atom_drive.txt" % (args.ID)
	specname = "./Data/JC+fb/New/spec%04d_atom_drive.txt" % (args.ID)
	g2tau = "./Data/JC+fb/New/g2tau%04d_atom_drive.txt" % (args.ID)
	corrf = "./Data/JC+fb/New/corr%04d_atom_drive.txt" % (args.ID)
	Bdenspath = "./Data/JC+fb/New/Bdens%04d_atom_drive" % (args.ID)
	Cdenspath = "./Data/JC+fb/New/Cdens%04d_atom_drive" % (args.ID)
elif Omc>0:
	filename = "./Data/JC+fb/New/evol%04d_cavity_drive.txt" % (args.ID)
	outname = "./Data/JC+fb/New/OUT%04d_cavity_drive.txt" % (args.ID)
	specname = "./Data/JC+fb/New/spec%04d_cavity_drive.txt" % (args.ID)
	g2tau = "./Data/JC+fb/New/g2tau%04d_cavity_drive.txt" % (args.ID)
	corrf = "./Data/JC+fb/New/corr%04d_cavity_drive.txt" % (args.ID)
	Bdenspath = "./Data/JC+fb/New/Bdens%04d_cavity_drive" % (args.ID)
	Cdenspath = "./Data/JC+fb/New/Cdens%04d_cavity_drive" % (args.ID)
else:
	filename = "./Data/JC+fb/New/evol%04d.txt" % (args.ID)
	outname = "./Data/JC+fb/New/OUT%04d.txt" % (args.ID)
	specname = "./Data/JC+fb/New/spec%04d.txt" % (args.ID)
	g2tau = "./Data/JC+fb/New/g2tau%04d.txt" % (args.ID)
	corrf = "./Data/JC+fb/New/corr%04d.txt" % (args.ID)
	Bdenspath = "./Data/JC+fb/New/Bdens%04d" % (args.ID)
	Cdenspath = "./Data/JC+fb/New/Cdens%04d" % (args.ID)
if not os.path.exists(Bdenspath):
	os.makedirs(Bdenspath)
if not os.path.exists(Cdenspath):
	os.makedirs(Cdenspath)
	
file_out = open(outname,"a")
file_out.close()
file_out = open(outname,"r+")
file_out.truncate()
file_out.close()
file_out = open(outname,"a")

file_out.write("""ID = %04d\ngamma_L = %f, gamma_R = %f, Om_e = %f, Om_c = %f, phi = %fpi,
Delta_e = %f, Delta_c = %f, g = %f, Nphot_max = %d, initind = %d,
tolerance = %.0E, delay_L = %d, endt = %.0f, dt = %f\n
coherent initial state amplitude for cavity: %f and the environment %f
Data file: M*dt,norm,exc_pop,gr_pop,nc_exp,g2_ta,g2_t,NB\n""" % (args.ID,gamma_L, gamma_R, Ome, Omc, args.phi, Dele, Delc, g, N_env, args.init_ind, Decimal(tol), L, endt, dt,args.cohC,args.cohE))
file_out.close()

file_evol = open(filename,"a")
file_evol.close()
file_evol = open(filename,"r+")
file_evol.truncate()
file_evol.close()
file_evol = open(filename,"a")

if args.Wigner:
	file_Bdens = open(Bdenspath+"/000.txt","a")
	file_Bdens.close()
	file_Bdens = open(Bdenspath+"/000.txt","r+")
	file_Bdens.truncate()
	file_Bdens.close()
	file_Bdens = open(Bdenspath+"/000.txt","a")

	np.savetxt(Bdenspath+"/000.txt",Bdens.view(float))
	file_Bdens.write("\n")
	file_Bdens.flush()
	Bdens_count=0
	file_Bdens.close()

	file_Cdens = open(Cdenspath+"/000.txt","a")
	file_Cdens.close()
	file_Cdens = open(Cdenspath+"/000.txt","r+")
	file_Cdens.truncate()
	file_Cdens.close()
	file_Cdens = open(Cdenspath+"/000.txt","a")

	np.savetxt(Cdenspath+"/000.txt",Cdens.view(float))
	file_Cdens.write("\n")
	file_Cdens.flush()
	file_Cdens.close()

######################
### Time evolution ###
######################
for M in range(0,N-L-1):
	percent10 = (N-L-1)/10.
	count = 0
	if M%(int(percent10))==0:
		print("M =",M, " out of ",N-L-1)
		sys.stdout.flush()
	
	#%%%%%%#
	# SWAP #
	#%%%%%%#
	if M>0:
		#% Relocating the orthogonality centre to the next interacting past bin if applicable %#
#		print("M and M-1 shape",states[M].shape,states[M-1].shape)
		states[M],states[M-1] = OC_reloc(states[M],states[M-1],"left",tol)
		#% After the first time step, bring the interacting past bin next to the system bin %#
#		print("ind_sys",ind_sys,states[ind_sys].shape)
#		print("M and M+L-1 shape",M,M+L-1,states[M].shape,states[M+L-1].shape)
		states[M:M+L] = SWAP(states,M,"future",M,L,tol)
#		print("M and M+L-1 shape",states[M].shape,states[M+L-1].shape)
#		print("ind_sys",ind_sys,states[ind_sys].shape)
		#states[M:M+L] = SWAP(states,M,"future",L,tol)
                #% Relocating the orthogonality centre from the past bin to the system bin before the evolution %#
#		print("ind_sys-1",states[ind_sys-1].shape)
		states[ind_sys],states[ind_sys-1] = OC_reloc(states[ind_sys],states[ind_sys-1],"left",tol)
        
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%#
	# System expectation values #
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%#
	norm,normL = normf(M,L,states,normL)
	nc_exp = exp_sys(nc,states[ind_sys])
	exc_pop = exp_sys(see,states[ind_sys])
	gr_pop  = exp_sys(sgg,states[ind_sys])
	g2_ta =  exp_sys(g2cop,states[ind_sys])/nc_exp**2
	file_evol.write("%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\n" %(M*dt,norm,exc_pop,gr_pop,nc_exp,g2_ta,g2_t,NB))
	file_evol.flush()
	file_out.close()

	Cdens = cav_dens(states[ind_sys],N_env)

	if args.Wigner:
		if M*dt<15.1 and (M*dt)%1==0:
			file_Cdens = open(Cdenspath+"/%03d.txt"%((M+1)*dt),"a")
			file_Cdens.close()
			file_Cdens = open(Cdenspath+"/%03d.txt"%((M+1)*dt),"r+")
			file_Cdens.truncate()
			file_Cdens.close()
			file_Cdens = open(Cdenspath+"/%03d.txt"%((M+1)*dt),"a")

			np.savetxt(Cdenspath+"/%03d.txt"%((M+1)*dt),Cdens.view(float))
			file_Cdens.write("\n")
			file_Cdens.flush()
			file_Cdens.close()

		elif M*dt>15 and (M*dt)%10==0:
	 
			file_Cdens = open(Cdenspath+"/%03d.txt"%((M+1)*dt),"a")
			file_Cdens.close()
			file_Cdens = open(Cdenspath+"/%03d.txt"%((M+1)*dt),"r+")
			file_Cdens.truncate()
			file_Cdens.close()
			file_Cdens = open(Cdenspath+"/%03d.txt"%((M+1)*dt),"a")

			np.savetxt(Cdenspath+"/%03d.txt"%((M+1)*dt),Cdens.view(float))
			file_Cdens.write("\n")
			file_Cdens.flush()
			file_Cdens.close()

	#%%%%%%%%%%%%%%#
	# Unitary step #
	#%%%%%%%%%%%%%%#
	U_block = U(initenv,states[ind_sys],states[ind_sys-1],M,gamma_L,gamma_R,dt,phi,Ome,Omc,g,Delc,Dele)
#	print("U block",U_block.shape)
	#   -------
	#  |       |--
	#   -------
	#    | | |

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%#
	# Separating the system bin #
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%#
	#% Merging of the link index on the right into a new tensor if applicable %#   
	U_right_merge=False
	if len(U_block.shape)>3:
		U_right_merge=True
		U_block,U_right_dims = MERGE(U_block,[2,3])
	#   -------
	#  |       | 
	#   -------
	#    | | |

	#% Exchanging the position of the system and the present bin in the MPS state list %#
	U_block  = np.einsum("ijk->jik",U_block)
#	print("U block merge right",U_block.shape)
    
	#% Separating the system state from the environment bins %#
	sys_env_merged,merge_dims = MERGE(U_block,[1,2])
	#   -------
	#  |       | 
	#   -------
	#    |   |
	sys_env_merged_svd,link_dim = SVD_sig(sys_env_merged,tol)
	sys_env_merged=None
	states[ind_sys+1] = sys_env_merged_svd[0]
	#   ---    -----
	#  (   |--|     | 
	#   ---    -----
	#    |       |
	env_dims = np.array([link_dim,merge_dims[0],merge_dims[1]])
	env_merged = UNMERGE(np.einsum("ij,jk->ik",sys_env_merged_svd[1],sys_env_merged_svd[2]),
			     env_dims,merge_dims,[1,2])
	#   ---    -----
	#  (   |--|     | 
	#   ---    -----
	#    |      | |
	sys_env_merged_svd=None
		

#	states[ind_sys+1],U_small_block = cut(U_block,tol,"right")
#	print("tS and rest",states[ind_sys+1].shape,env_merged.shape)
	U_block = None
	env_dims=None

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
	# Separating the present bin #
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
	#    -----
	# --|     | 
	#    -----
	#     | |
	env_merged,merge_dims = MERGE(env_merged,[0,1])
	#  -----
	# |     | 
	#  -----
	#   | |
	env_merged_svd,link_dim = SVD_sig(env_merged,tol)
	env_merged=None
	present_dims = np.array([merge_dims[0],merge_dims[1],link_dim])
	states[ind_sys] = UNMERGE(env_merged_svd[0],present_dims,merge_dims,[0,1])
	#   ---    ---
	#--(   |--|   | 
	#   ---    ---
	#    |      |
	states[ind_sys-1] = np.dot(env_merged_svd[1],env_merged_svd[2])
	sys_env_merged_svd=None
#	states[ind_sys],states[ind_sys-1] = cut(U_small_block,tol,"left")
#	print("tk and tl",states[ind_sys].shape,states[ind_sys-1].shape)
	U_small_block=None
	present_dims=None

	#% Unmerging of the previously merged link index on the right if applicable %#
	if U_right_merge:
		U_dims = np.concatenate((np.array([states[ind_sys-1].shape[0]]),U_right_dims),axis = 0)
		states[ind_sys-1] = UNMERGE(states[ind_sys-1],U_dims,U_right_dims,[1,2])
		U_dims = None
		U_right_dims=None
	#   ---    ---
	#--(   |--|   |-- 
	#   ---    ---
	#    |      |
#	print("tl final",states[ind_sys-1].shape)
#	print("U done, states done", states[ind_sys+1].shape, states[ind_sys].shape, states[ind_sys-1].shape)
        
	#%%%%%%%%%%%#
	# SWAP back #
	#%%%%%%%%%%%#
	states[(ind_sys-L):(ind_sys)] = SWAP(states,ind_sys-1,"past",M,L,tol)

#	states[(ind_sys-L):(ind_sys)] = SWAP(states,(ind_sys-2),"past",L,tol)
#	g2_ta,NB = g2_t(states[M],N_env+1,dt)
	Bdens = env_dens(states[M])
	if args.Wigner:
		if M*dt<15.1 and ((M+1)*dt)%1==0:
			file_Bdens = open(Bdenspath+"/%03d.txt"%((M+1)*dt),"a")
			file_Bdens.close()
			file_Bdens = open(Bdenspath+"/%03d.txt"%((M+1)*dt),"r+")
			file_Bdens.truncate()
			file_Bdens.close()
			file_Bdens = open(Bdenspath+"/%03d.txt"%((M+1)*dt),"a")

			np.savetxt(Bdenspath+"/%03d.txt"%((M+1)*dt),Bdens.view(float))
			file_Bdens.write("\n")
			file_Bdens.flush()
			file_Bdens.close()

		elif ((M+1)*dt)>15 and ((M+1)*dt)%10==0:
	 
			file_Bdens = open(Bdenspath+"/%03d.txt"%((M+1)*dt),"a")
			file_Bdens.close()
			file_Bdens = open(Bdenspath+"/%03d.txt"%((M+1)*dt),"r+")
			file_Bdens.truncate()
			file_Bdens.close()
			file_Bdens = open(Bdenspath+"/%03d.txt"%((M+1)*dt),"a")

			np.savetxt(Bdenspath+"/%03d.txt"%((M+1)*dt),Bdens.view(float))
			file_Bdens.write("\n")
			file_Bdens.flush()
			file_Bdens.close()

	NB    = np.einsum("ij,ij",Bdens,nB)
	g2_t  = np.einsum("ij,ij",Bdens,g2Bop)/NB**2
	#NB_outa = NB_out(states[M],N_env+1,NB_outa,dt)

	#% Preparing for the next step with the system index %#
	ind_sys =1+ind_sys
#	print("ind_sys",ind_sys,states[ind_sys].shape)

#############################
### Output spectra and g2 ###
#############################

if args.spectrum:
	om = np.linspace(-20,20,5001)
#	om = np.fftfreq(N-L-1,dt)
	spec,corr = spectrum(states,om,args.spec_cutoff*L,N_env+1,dt,N-L-2)#,NB*dt)
	tau,g2_outa = g2_out(states,N-L-2,N_env+1,dt,N-L-2)
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
	corr_out = np.transpose(np.vstack((np.arange(len(corr))*dt,corr)))
	f = open(corrf, 'a')
	f.close()
	f = open(corrf, 'r+')
	f.truncate()
	np.savetxt(corrf,corr_out)
	corr_out=None

###################################
### Relocating OC to system bin ###
###################################

for i in range(M,ind_sys):
    if len(states[i].shape)>1:
        states[i+1],states[i] = OC_reloc(states[i+1],states[i],"left",tol)

#################################
### System expectation values ###
#################################
norm,normL = normf(N-L-1,L,states,normL)
nc_exp = exp_sys(nc,states[N-1])
exc_pop = exp_sys(see,states[N-1])
gr_pop  = exp_sys(sgg,states[N-1])
g2_ta  = exp_sys(g2cop,states[N-1])
#	file_evol = open("./Data/TLS+feedback_gL=%dp10_gR=%dp10_Om=%dp10_phi=%dp10pi.txt" % \
#			(gamma_L*10, gamma_R*10, Om_TLS*10, args.phi*10),"a")
file_evol.write("%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\t%.20f\n" %(M*dt,norm,exc_pop,gr_pop,nc_exp,g2_ta,g2_t,NB))

##################
### Timer ends ###
##################

end = time.time()-start
h = int(end/3600)
m = int((end-3600*h)/60)
s = int(end-3600*h-60*m)
#print(#"final excited state population:",exc_pop[-1],
#      "\nfinal ground state population:",gr_pop[-1],
file_out = open(outname,"a")
file_out.write("\nfinished in: %02d:%02d:%02d" %(h,m,s))
file_out.close()
