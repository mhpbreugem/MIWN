import sys, numpy as np
sys.path.insert(0,"/home/user/MIWN/standards/methods/solver")
sys.path.insert(0,"/home/user/MIWN/numerics/ree_K3")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from code.halo import extract_inner, replace_inner
from ode_sweep import anderson_solve
from coarea import phi_K3_coarea
tau,gamma,W=0.5,1.0,1.0; tv,gv,Wv=np.full(3,tau),np.full(3,gamma),np.full(3,W)
G,pad,um=18,4,3.0; du=2*um/(G-1); Gf=G+2*pad
u=np.array([-um+(q-pad)*du for q in range(Gf)]); lo,hi=pad,pad+G; ui=u[lo:hi]
halo=init_no_learning_K3(u,tv,gv,Wv)
def solve(mapfn,extra=(),tol=1e-10,mi=4000):
    def phi(P):
        Pf=replace_inner(halo,P,lo,hi)
        return extract_inner(mapfn(Pf,u,lo,hi,tv,gv,Wv,*extra),lo,hi)
    Ps,res=anderson_solve(phi,extract_inner(halo,lo,hi),tol=tol,max_iter=mi,m=8)
    return Ps,res
Ph,_=solve(phi_K3_halo_smooth,(0.0333,))          # finite bandwidth (noise)
P0,r0=solve(phi_K3_coarea,())                       # bandwidth-free h=0
print(f"coarea(h=0) residual {r0:.1e}",flush=True)
# weights, T*, decomposition
U1,U2,U3=np.meshgrid(ui,ui,ui,indexing='ij')
T=tau*(U1+U2+U3)
f1=np.sqrt(tau/2/np.pi)*np.exp(-tau/2*(ui-0.5)**2); f0=np.sqrt(tau/2/np.pi)*np.exp(-tau/2*(ui+0.5)**2)
F1=f1[:,None,None]*f1[None,:,None]*f1[None,None,:]; F0=f0[:,None,None]*f0[None,:,None]*f0[None,None,:]
Wt=0.5*(F1+F0)
def decomp(P):
    eps=1e-4; m=(P>eps)&(P<1-eps)
    L=np.log(P/(1-P)); t=T[m]; l=L[m]; w=Wt[m]; w=w/w.sum()
    def wr2(deg):
        c=np.polyfit(t,l,deg,w=np.sqrt(w)); pred=np.polyval(c,t)
        lm=np.average(l,weights=w); vt=np.average((l-lm)**2,weights=w); vr=np.average((l-pred)**2,weights=w)
        return vr/vt
    lin=wr2(1); curve=wr2(5)
    return lin,curve
for name,P in [("h=0.0333 (finite bandwidth/noise)",Ph),("h=0  (bandwidth-free)",P0)]:
    lin,curve=decomp(P)
    print(f"{name}:  1-R2_linear={lin:.5f}   scatter(1-R2 vs poly5 in T*)={curve:.5f}   Jensen/curvature gap={lin-curve:.5f}")
np.savez("/tmp/jensen_data.npz",P0=P0,Ph=Ph,ui=ui,T=T,Wt=Wt)
