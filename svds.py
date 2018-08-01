import numpy as np
from scipy.linalg import svd
import scipy.sparse as ss
from scipy.sparse.linalg import svds
from pypropack import svdp
from sparsesvd import sparsesvd
#from sklearn.decomposition import TruncatedSVD

def conv(x):
    return x.replace('+-', '-').encode()

a = np.genfromtxt((conv(x) for x in open("../JC+fb_new/mat.txt")),dtype=complex)
#a = np.random.rand(600,600)
print(np.max(a.imag))
print(a[1,0])
res1 = svd(a,full_matrices=False)
b = ss.csc_matrix(a)
lim = 14
start = np.array([1,0,0])
res = svds(b,k=lim,which="LM")
#resp = svdp(a,k=lim)
#ress = sparsesvd(b, lim)
#svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)

print(ress[0].shape,ress[1].shape,ress[2].shape)

print(np.einsum("ij,jk->ik",np.einsum("ji,j->ij",ress[0],ress[1]),ress[2])[1,0])
print(np.einsum("ij,jk->ik",np.einsum("ij,j->ij",res1[0],res1[1]),res1[2])[1,0])
print(res1[1])
print(res1[0][1,0])
#print(ress[1]-res1[1][:lim])
#print(ress[1])

