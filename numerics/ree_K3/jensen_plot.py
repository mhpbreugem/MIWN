import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=np.load("/tmp/jensen_data.npz"); P0=d["P0"]; Ph=d["Ph"]; ui=d["ui"]; T=d["T"]; Wt=d["Wt"]
tau=0.5; G=len(ui)
fig=plt.figure(figsize=(16,5.2))
# Panel A: h=0 price contours, slice u3=0
ax=fig.add_subplot(1,3,1); mid=int(np.argmin(np.abs(ui)))
CS=ax.contour(ui,ui,P0[:,:,mid].T,levels=np.linspace(0.1,0.9,9),cmap="viridis")
ax.clabel(CS,inline=True,fontsize=7,fmt="%.2f")
for c in (-4,-2,0,2,4):
    ax.plot(ui,c-ui,"r--",lw=0.8,alpha=0.6)
ax.set_xlim(ui[0],ui[-1]); ax.set_ylim(ui[0],ui[-1])
ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$")
ax.set_title("h=0 price contours, slice $u_3$=0\n(red dashed: $u_1+u_2$=const)\nlevel sets ∥ sum ⇒ P depends on $\\Sigma u$",fontsize=10)
# Panels B,C: logit P vs T*
def scatterpanel(ax,P,title):
    eps=1e-4; m=(P>eps)&(P<1-eps)
    t=T[m]; L=np.log(P[m]/(1-P[m])); w=Wt[m]; ww=w/w.max()
    o=np.argsort(t)
    ax.scatter(t,L,s=6+40*ww,c=ww,cmap="plasma",alpha=0.5,edgecolors="none")
    cl=np.polyfit(t,L,1,w=np.sqrt(w/w.sum())); cp=np.polyfit(t,L,5,w=np.sqrt(w/w.sum()))
    tt=np.linspace(t.min(),t.max(),200)
    ax.plot(tt,np.polyval(cl,tt),"k--",lw=1.4,label="linear fit")
    ax.plot(tt,np.polyval(cp,tt),"g-",lw=2,label="curve g(T*) (poly5)")
    ax.set_xlabel("$T^*=\\tau\\,\\Sigma u$"); ax.set_ylabel("logit $P$"); ax.set_title(title,fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.2)
scatterpanel(fig.add_subplot(1,3,2),P0,"h=0 (no noise): points hug a near-straight curve\nscatter≈0.001, Jensen≈0.001 ⇒ full revelation")
scatterpanel(fig.add_subplot(1,3,3),Ph,"h=0.033 (noise): scatter (width) + curvature (bend)\nscatter=0.026 (true deficit) + Jensen gap=0.017")
plt.tight_layout(); plt.savefig("/home/user/MIWN/numerics/ree_K3/jensen_contours.png",dpi=150); print("saved")
