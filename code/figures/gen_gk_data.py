import numpy as np, pandas as pd, duckdb
from pathlib import Path
from scipy.interpolate import RBFInterpolator

ROOT = Path(r"E:\Research\iv_surface")
D4, D5 = ROOT/"04_iv", ROOT/"05_splits"
K_PAD_FRAC, K_GRID_N, FD_H = 0.2, 61, 1e-3

date, expiry = "2023-09-06", "2023-09-28"

panel = duckdb.connect().execute(f"SELECT date, expiry, strike, k, tau, iv FROM read_parquet('{D4/'panel.parquet'}')").df()
held = duckdb.connect().execute(f"SELECT * FROM read_parquet('{D5/'held_out_strikes.parquet'}')").df()
panel = panel.merge(held, on=["date","expiry","strike"], how="left")
panel["w"] = panel["iv"]**2 * panel["tau"]

vis_range = duckdb.connect().execute(f"SELECT * FROM read_parquet('{D5/'visible_k_range.parquet'}')").df()

gday = panel[panel["date"]==date]
vis_day = gday[~gday["is_heldout"]]
print("visible points that day (all expiries):", len(vis_day))

rbf = RBFInterpolator(vis_day[["k","tau"]].to_numpy(), vis_day["iv"].to_numpy(), kernel="thin_plate_spline", smoothing=1e-6)

vr = vis_range[(vis_range["date"]==date)&(vis_range["expiry"]==expiry)]
kmin, kmax = float(vr["k_vis_min"].iloc[0]), float(vr["k_vis_max"].iloc[0])
pad = (kmax-kmin)*K_PAD_FRAC
kgrid = np.linspace(kmin-pad, kmax+pad, K_GRID_N)
tau = float(vis_day[vis_day["expiry"]==expiry]["tau"].iloc[0])

def w_at(kk):
    pts = np.stack([kk, np.full_like(kk, tau)], axis=1)
    iv = rbf(pts)
    return iv**2*tau, iv

w, iv_grid = w_at(kgrid)
w_plus,_ = w_at(kgrid+FD_H)
w_minus,_ = w_at(kgrid-FD_H)
wk = (w_plus-w_minus)/(2*FD_H)
wkk = (w_plus-2*w+w_minus)/(FD_H**2)

def durrleman_g(k,w,wk,wkk):
    w_safe = np.maximum(w,1e-8)
    return (1-k*wk/(2*w_safe))**2 - (wk**2/4.0)*(1.0/w_safe+0.25) + wkk/2.0

g = durrleman_g(kgrid, w, wk, wkk)
in_observed = (kgrid>=kmin)&(kgrid<=kmax)
print("n_viol:", int(np.sum(g<0)), "of", K_GRID_N)

out = pd.DataFrame(dict(k=kgrid, iv=iv_grid, w=w, g=g, in_observed=in_observed))
out["date"]=date; out["expiry"]=expiry; out["tau"]=tau; out["kmin"]=kmin; out["kmax"]=kmax
out.to_csv(ROOT/"tier1_writing_gk_data.csv", index=False)

# also grab the actual observed quotes for this slice for context
obs = gday[gday["expiry"]==expiry][["k","iv"]].sort_values("k")
obs.to_csv(ROOT/"tier1_writing_gk_obs.csv", index=False)
print("saved")
