"""dps=40 proof that FULL REVELATION survives asymmetric (gamma,tau).
With heterogeneous gammas and taus, P=sigma(sum_k tau_k u_k) is an exact REE:
each agent's contour posterior equals the price and excess demand is zero
(no trade), so 1-R2=0 against the CORRECT precision-weighted statistic
T*=sum tau_k u_k.  (Using the wrong T*, e.g. unweighted sum, manufactures a
spurious deficit -- a metric artifact, not inefficiency.)"""
import mpmath as mp
mp.mp.dps = 40
def fv(u, m, tau): return mp.e ** (-tau / 2 * (u - m) ** 2)
def sigma(x): return 1 / (1 + mp.e ** (-x))
G = [mp.mpf('0.3'), mp.mpf('1.0'), mp.mpf('4.0')]
T = [mp.mpf('0.5'), mp.mpf('2.0'), mp.mpf('5.0')]
def Av_line(c, tk, tj, v):
    m = mp.mpf('0.5') if v == 1 else mp.mpf('-0.5')
    return mp.quad(lambda x: fv(x, m, tk) * fv((c - tk * x) / tj, m, tj), [-mp.inf, mp.inf]) / tj
def posterior(k, uvec):
    Ts = sum(T[i] * uvec[i] for i in range(3)); o = [i for i in range(3) if i != k]
    c = Ts - T[k] * uvec[k]
    a1 = fv(uvec[k], mp.mpf('0.5'), T[k]) * Av_line(c, T[o[0]], T[o[1]], 1)
    a0 = fv(uvec[k], mp.mpf('-0.5'), T[k]) * Av_line(c, T[o[0]], T[o[1]], 0)
    return a1 / (a0 + a1)
def x_demand(mu, p, gamma, W=1):
    z = (mp.log(mu / (1 - mu)) - mp.log(p / (1 - p))) / gamma; e = mp.e ** z
    return W * (e - 1) / ((1 - p) + p * e)
if __name__ == "__main__":
    wm = mp.mpf(0); wc = mp.mpf(0)
    for uvec in [[mp.mpf('0.4'), mp.mpf('-1.1'), mp.mpf('0.7')],
                 [mp.mpf('-2.0'), mp.mpf('0.3'), mp.mpf('1.5')],
                 [mp.mpf('1.0'), mp.mpf('1.0'), mp.mpf('-0.5')]]:
        p = sigma(sum(T[i] * uvec[i] for i in range(3)))
        mus = [posterior(k, uvec) for k in range(3)]
        wm = max(wm, max(abs(m - p) for m in mus))
        wc = max(wc, abs(sum(x_demand(mus[k], p, G[k]) for k in range(3))))
    print("asymmetric gamma=[0.3,1,4] tau=[0.5,2,5], dps=40")
    print("worst |posterior - price| =", mp.nstr(wm, 5))
    print("worst |excess demand|     =", mp.nstr(wc, 5), "(0 => no trade, full revelation)")
