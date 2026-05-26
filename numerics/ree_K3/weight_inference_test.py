import sys, numpy as np, time
sys.path.insert(0,"/home/user/MIWN/standards/methods/solver"); sys.path.insert(0,"/home/user/MIWN/numerics/ree_K3")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_cubic
from code.halo import extract_inner, replace_inner
from coarea import phi_K3_coarea
tau,gamma,W=2.0,1.0,1.0; tv,gv,Wv=np.full(3,tau),np.full(3,gamma),np.full(3,W)
G,pad,um=16,4,5.0; du=2*um/(G-1); Gf=G+2*pad
u=np.array([-um+(q-pad)*du for q in range(Gf)]); lo,hi=pad,pad+G; ui=u[lo:hi]
def deficit(P):
    U1,U2,U3=np.meshgrid(ui,ui,ui,indexing='ij'); T=tau*(U1+U2+U3)
    f1=np.exp(-tau/2*(ui-0.5)**2);f0=np.exp(-tau/2*(ui+0.5)**2)
    F1=f1[:,None,None]*f1[None,:,None]*f1[None,None,:];F0=f0[:,None,None]*f0[None,:,None]*f0[None,None,:]
    Wt=0.5*(F1+F0);m=(P>1e-4)&(P<1-1e-4);t=T[m];L=np.log(P[m]/(1-P[m]));w=Wt[m];w=w/w.sum();lm=np.average(L,weights=w)
    c1=np.polyfit(t,L,1,w=np.sqrt(w));c5=np.polyfit(t,L,5,w=np.sqrt(w))
    return (np.average((L-np.polyval(c1,t))**2,weights=w)/np.average((L-lm)**2,weights=w),
            np.average((L-np.polyval(c5,t))**2,weights=w)/np.average((L-lm)**2,weights=w))
def run(weighted,n=400,alpha=0.3,ces=200):
    halo=init_no_learning_K3(u,tv,gv,Wv); P=extract_inner(halo,lo,hi).copy()
    acc=None;cnt=0;best=(P.copy(),1e9)
    for it in range(n):
        Pf=replace_inner(halo,P,lo,hi)
        Q=extract_inner(phi_K3_coarea(Pf,u,lo,hi,tv,gv,Wv) if weighted else phi_K3_halo_cubic(Pf,u,lo,hi,tv,gv,Wv),lo,hi)
        F=float(np.max(np.abs(Q-P)))
        if F<best[1]:best=(P.copy(),F)
        P=(1-alpha)*P+alpha*Q;P=np.clip(P,1e-9,1-1e-9)
        if it>=n-ces: acc=Q.copy() if acc is None else acc+Q; cnt+=1
    Pc=acc/cnt; lin,scat=deficit(Pc); return lin,scat,best[1]
nl=deficit(extract_inner(init_no_learning_K3(u,tv,gv,Wv),lo,hi))
print(f"no-learning seed deficit (gamma=1,tau=2): 1-R2_lin={nl[0]:.5f} scatter={nl[1]:.5f}",flush=True)
t=time.time(); lw,sw,rw=run(True); print(f"WEIGHTED (1/|gradP|) REE: 1-R2={lw:.5f} scatter={sw:.5f} res={rw:.1e} ({time.time()-t:.0f}s)",flush=True)
t=time.time(); lc,sc,rc=run(False); print(f"WEIGHTLESS (arc/unit)  REE: 1-R2={lc:.5f} scatter={sc:.5f} res={rc:.1e} ({time.time()-t:.0f}s)",flush=True)
