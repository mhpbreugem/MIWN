import sys, numpy as np, time, mpmath as mp
mp.mp.dps=40
sys.path.insert(0,"/home/user/MIWN/standards/methods/solver")
sys.path.insert(0,"/home/user/MIWN/numerics/ree_K3")
from code.contour_K3_halo import init_no_learning_K3
from code.halo import replace_inner
from dd_phi import dd_phi
from dd_numba import dd_sub
a=np.load("solutions/pool/ree_K3/v0011/data/solution.npz",allow_pickle=True)
tau=float(a["tau"]); gamma=float(a["gamma"]); u_full=a["u_full"]; ui=a["u_grid_inner"]
mpstr=a["P_inner_mp_str"]; G=mpstr.shape[0]; pad=int(a["pad"])
tv,gv,Wv=np.full(3,tau),np.full(3,gamma),np.full(3,1.0)
du=ui[1]-ui[0]; h=max(0.005,0.05*du); lo,hi=pad,pad+G
halo=init_no_learning_K3(u_full,tv,gv,Wv)
Phi=halo.copy(); Plo=np.zeros_like(halo)
for i in range(G):
  for j in range(G):
    for l in range(G):
      v=mp.mpf(mpstr[i,j,l]); hh=float(v); Phi[lo+i,lo+j,lo+l]=hh; Plo[lo+i,lo+j,lo+l]=float(v-mp.mpf(hh))
t=time.time(); Nh,Nl=dd_phi(Phi,Plo,u_full,lo,hi,tv,gv,Wv,h)
# DD residual over inner
m=0.0
for i in range(lo,hi):
  for j in range(lo,hi):
    for l in range(lo,hi):
      dh,dl=dd_sub(Nh[i,j,l],Nl[i,j,l],Phi[i,j,l],Plo[i,j,l]); a_=abs(dh+dl)
      if a_>m: m=a_
print(f"numba-DD residual ||phi(P)-P||inf on v0011 mp solution = {m:.3e}")
print(f"v0011 meta F_inf (mpmath)                              = {float(a['F_inf']):.3e}")
print(f"dd_phi eval time = {time.time()-t:.1f}s  (mpmath phi_K3_smooth_mp ~ minutes)")
