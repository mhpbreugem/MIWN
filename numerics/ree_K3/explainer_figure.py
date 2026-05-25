import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
d=np.load("/tmp/jensen_data.npz"); P0=d["P0"]; Ph=d["Ph"]; ui=d["ui"]; T=d["T"]; Wt=d["Wt"]; tau=0.5
fig=plt.figure(figsize=(17,10))

# ---- A: setup & question ----
ax=fig.add_subplot(2,3,1); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
ax.set_title("(A)  The setup", fontsize=13, fontweight="bold", loc="left")
for k,(x,lab) in enumerate([(1.5,"$u_1$"),(1.5,"$u_2$"),(1.5,"$u_3$")]):
    y=8-2.3*k; ax.add_patch(Circle((x,y),0.6,fc="#cde",ec="navy"))
    ax.text(x,y,f"trader\n{k+1}",ha="center",va="center",fontsize=8)
    ax.text(x+0.95,y,f"signal {lab}",fontsize=10,va="center")
    ax.add_patch(FancyArrowPatch((x+2.5,y),(5.4,5),arrowstyle="-|>",mutation_scale=12,color="gray"))
ax.add_patch(FancyBboxPatch((5.4,4.2),2.2,1.6,boxstyle="round,pad=0.1",fc="#fe9",ec="k"))
ax.text(6.5,5,"market\nclears",ha="center",va="center",fontsize=9)
ax.add_patch(FancyArrowPatch((7.7,5),(9,5),arrowstyle="-|>",mutation_scale=14,color="k"))
ax.text(9.2,5,"price\n$p$",fontsize=11,va="center")
ax.text(0.2,1.8,"No noise traders.  Question: does $p$ reveal the\naggregate signal $T^*=\\tau\\Sigma u_k$ (efficient) or not?",fontsize=10)

# ---- B: inference on the price contour ----
ax=fig.add_subplot(2,3,2)
mid=int(np.argmin(np.abs(ui)))
CS=ax.contour(ui,ui,Ph[:,:,mid].T,levels=np.linspace(0.15,0.85,8),colors="0.7",linewidths=0.8)
pc=ax.contour(ui,ui,Ph[:,:,mid].T,levels=[0.5],colors="crimson",linewidths=2.5)
ax.clabel(pc,fmt={0.5:"$P=p$"},fontsize=10)
ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$")
ax.set_title("(B)  Inference lives on the contour",fontsize=13,fontweight="bold",loc="left")
ax.text(0.02,1.02,"agent integrates signal density ALONG $P=p$:\n"
        r"$A_v=\int f_v\,/\,|\nabla P|\,ds$",transform=ax.transAxes,fontsize=9.5,va="bottom")

# ---- C: bandwidth h = noise ----
ax=fig.add_subplot(2,3,3); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
ax.set_title("(C)  Kernel bandwidth $h$ = price noise",fontsize=13,fontweight="bold",loc="left")
xx=np.linspace(1,9,100); yc=5+2.2*np.sin((xx-1)/2.4)
for off,al in [(0,1)]:
    ax.plot(xx,yc,color="crimson",lw=2.4,zorder=3)
ax.fill_between(xx,yc-1.1,yc+1.1,color="orange",alpha=0.3,zorder=1)
ax.annotate("",xy=(5,yc[50]+1.1),xytext=(5,yc[50]-1.1),arrowprops=dict(arrowstyle="<->",color="k"))
ax.text(5.2,yc[50],"$\\sim h$",fontsize=12)
ax.text(0.3,1.4,"With $h>0$ the agent can't resolve the price finer than $h$:\n"
        "infers from a BAND of width $h$ around the contour\n"
        r"$\equiv$ observing the price corrupted by noise of size $h$.",fontsize=10)
ax.text(0.3,9.2,"$h\\to0$  $\\Rightarrow$  exact price  $\\Rightarrow$  no noise",fontsize=11,color="darkgreen",fontweight="bold")

# ---- D: deficit vanishes as h->0 ----
ax=fig.add_subplot(2,3,4)
H=np.array([0.0333,0.025,0.020]); D=np.array([0.0432,0.0285,0.0197])
ax.plot(H,D,"o",ms=11,color="navy",label="grid-converged 1$-$R$^2$")
p,la=np.polyfit(np.log(H),np.log(D),1); hh=np.linspace(0,0.036,100)
ax.plot(hh,np.exp(la)*hh**p,"-",color="crimson",lw=2,label=f"$\\sim h^{{{p:.1f}}}\\to0$")
ax.scatter([0.0333],[0.05234],marker="*",s=240,color="darkorange",ec="k",zorder=6,label="published 0.052 (G=10)")
ax.scatter([0],[0.002],marker="s",s=80,color="green",zorder=6,label="$h{=}0$ direct $\\approx$0.002")
ax.set_xlabel("bandwidth $h$  (= noise)"); ax.set_ylabel("revelation deficit 1$-$R$^2$")
ax.set_title("(D)  Deficit $\\to$ 0 as noise $\\to$ 0",fontsize=13,fontweight="bold",loc="left")
ax.set_xlim(-0.001,0.037); ax.set_ylim(-0.002,0.058); ax.legend(fontsize=8.5); ax.grid(alpha=0.25)

# ---- E: Jensen gap ----
ax=fig.add_subplot(2,3,5)
for P,nm,col in [(Ph,"h=0.033",None)]:
    m=(P>1e-4)&(P<1-1e-4); t=T[m]; L=np.log(P[m]/(1-P[m])); w=Wt[m]; ww=w/w.max()
    ax.scatter(t,L,s=5+30*ww,c=ww,cmap="plasma",alpha=0.5,edgecolors="none")
    cl=np.polyfit(t,L,1,w=np.sqrt(w/w.sum())); cp=np.polyfit(t,L,5,w=np.sqrt(w/w.sum()))
    tt=np.linspace(t.min(),t.max(),200)
    ax.plot(tt,np.polyval(cl,tt),"k--",lw=1.5,label="linear fit (what 1$-$R$^2$ uses)")
    ax.plot(tt,np.polyval(cp,tt),"g-",lw=2.2,label="true curve $g(T^*)$")
ax.set_xlabel("$T^*=\\tau\\,\\Sigma u$"); ax.set_ylabel("logit $P$")
ax.set_title("(E)  The metric over-counts: Jensen gap",fontsize=13,fontweight="bold",loc="left")
ax.text(0.03,0.80,"deficit 0.043 =\n  scatter 0.026 (real)\n + curvature 0.017 (Jensen,\n   NOT inefficiency)",
        transform=ax.transAxes,fontsize=9.5,color="darkgreen")
ax.legend(fontsize=8.5,loc="lower right"); ax.grid(alpha=0.2)

# ---- F: h=0 full revelation ----
ax=fig.add_subplot(2,3,6)
CS=ax.contour(ui,ui,P0[:,:,mid].T,levels=np.linspace(0.1,0.9,9),cmap="viridis")
ax.clabel(CS,inline=True,fontsize=7,fmt="%.2f")
for c in (-4,-2,0,2,4): ax.plot(ui,c-ui,"r--",lw=0.9,alpha=0.6)
ax.set_xlim(ui[0],ui[-1]); ax.set_ylim(ui[0],ui[-1])
ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$")
ax.set_title("(F)  $h{=}0$: contours $\\parallel$  $\\Sigma u$",fontsize=13,fontweight="bold",loc="left")
ax.text(0.03,1.02,"level sets follow $u_1{+}u_2$=const (red): price is a\nfunction of $\\Sigma u$  $\\Rightarrow$  FULL revelation",
        transform=ax.transAxes,fontsize=9.5,va="bottom",color="darkgreen")

fig.suptitle("Why the ~0.05 'inefficiency without noise' is a bandwidth (noise) + metric artifact   —   K=3, $\\gamma$=1, $\\tau$=0.5",
             fontsize=14,fontweight="bold",y=1.00)
plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig("/home/user/MIWN/numerics/ree_K3/explainer.png",dpi=140,bbox_inches="tight")
print("saved")
