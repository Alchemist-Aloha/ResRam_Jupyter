import resram_ng.resram_core as resram_core
from resram_ng.resram_core import load_input, resram_data, param_init, raman_residual, run_save
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np
import lmfit
import matplotlib.pyplot as plt

# Ensure Python-only implementation by disabling Rust
resram_core.HAS_RUST = False
print("Running with Python implementation (Rust disabled).")

# Initialize the 'fit_obj' object of class 'load_input'.
# Load from input files in root dir or specified dir.
fit_obj = load_input() 

fit_switch = np.ones(len(fit_obj.delta)+7)
fit_switch[len(fit_obj.delta)] = 1  # fit gamma?
fit_switch[len(fit_obj.delta)+1] = 1  # fit M?
fit_switch[len(fit_obj.delta)+2] = 1  # fit theta?
fit_switch[len(fit_obj.delta)+3] = 0  # fit kappa?
fit_switch[len(fit_obj.delta)+5] = 0  # fit E0?

params_lmfit = param_init(fit_switch, fit_obj)   # Initialize parameters for fitting

for name in params_lmfit.keys():
    print(params_lmfit[name])

# Initial call to populate fit_obj attributes
raman_residual(params_lmfit, fit_obj)
print(f"Initial correlation: {fit_obj.correlation}")

# fit with powell method
# fit_kws = dict(tol=1e-10)    # Set fitting tolerance
# max_nfev = 1000000    # Set maximum number of function evaluations

# # Initialize tqdm progress bar for fitting process
# with tqdm(total=max_nfev, desc="Fitting progress") as pbar:
#     def update_progress(params, iteration, resid, *args, **kwargs):
#         pbar.update(1)
#         try:
#             pbar.set_description(
#                 f"Iteration {iteration}, Loss: {resid[0]:.6f}, Abs_corr: {fit_obj.correlation:.6f}, Fl_corr: {fit_obj.fl_correlation:.6f}")
#         except Exception:
#             pass

#     # Perform the fitting using lmfit.minimize function
#     result = lmfit.minimize(raman_residual, params_lmfit, args=(fit_obj,), method='powell', **fit_kws,
#                             max_nfev=max_nfev, iter_cb=update_progress)

# fit with nelder (Nelder-Mead simplex) method — works with scalar residuals
max_nfev=1000000
with tqdm(total=max_nfev, desc="Fitting progress") as pbar:
    def update_progress(params, iteration, resid, *args, **kwargs):
        pbar.update(1)
        try:
            pbar.set_description(
                f"Iteration {iteration}, Loss: {resid[0]:.6f}, Abs_corr: {fit_obj.correlation:.6f}, Fl_corr: {fit_obj.fl_correlation:.6f}")
        except Exception:
            pass

    # Perform the fitting using lmfit.minimize function
    result = lmfit.minimize(raman_residual, params_lmfit, args=(fit_obj,), method='nelder', max_nfev=max_nfev, iter_cb=update_progress,tol=1e-10)

# fit with emcee (MCMC) method, but it is commented out for now. You can uncomment the following lines to use emcee for fitting.
# fit_kws = dict(nwalkers=150, steps=2000, progress=True)    # emcee has its own progress bar
# result = lmfit.minimize(raman_residual, params_lmfit, args=(fit_obj,), method='emcee', **fit_kws)

print(lmfit.fit_report(result))

current_time_str = datetime.now().strftime("%Y%m%d_%H-%M-%S") 
output = run_save(fit_obj, current_time_str)  # save all parameters and results to a new folder

# Save history lists
np.savetxt(current_time_str+"_data/corr_list.dat", fit_obj.correlation_list)
np.savetxt(current_time_str+"_data/sigma_list.dat", fit_obj.sigma_list)
np.savetxt(current_time_str+"_data/loss_list.dat", fit_obj.loss_list)

print(f"Fitting completed. Results saved in {current_time_str}_data/")
