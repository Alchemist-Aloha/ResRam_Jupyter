"""
Resonance Raman (ResRAM) Core Module
====================================
This module implements the theoretical framework for calculating absorption,
fluorescence, and resonance Raman excitation profiles (REPs) using the
Independent Mode Displaced Harmonic Oscillator (IMDHO) formalism.

It utilizes the time-dependent wavepacket approach and the Brownian oscillator
model to account for solvent-induced line broadening and dynamics.

Theory Reference:
- Absorption/Fluorescence: Equation S6/S9 in Supplementary Information.
- Resonance Raman: Equation S5 (First-order approximation).
- Brownian Oscillator: Multimode lineshape theory (Mukamel, 1995).

Key Parameters:
- E0: Vertical/0-0 transition energy.
- gamma (Γ): Homogeneous broadening (solvent dynamics).
- theta (θ): Static inhomogeneous broadening (Gaussian distribution).
- kappa (κ): Brownian oscillator parameter (ratio of damping to frequency).
- delta (Δ): Dimensionless displacements of vibrational modes.
"""

import sys
from datetime import datetime
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from matplotlib.colors import ListedColormap
import lmfit

DEFAULT_LOSS_COEFFICIENTS = np.array([1.0, 30.0, 30.0], dtype=float)

if os.environ.get("RESRAM_DISABLE_RUST", "").lower() in {"1", "true", "yes", "on"}:
    resram_rust = None
    HAS_RUST = False
else:
    try:
        import resram_rust
        HAS_RUST = all(
            hasattr(resram_rust, name)
            for name in ("cross_sections_rust", "raman_residual_rust")
        )
        if not HAS_RUST:
            resram_rust = None
    except (ImportError, OSError):
        resram_rust = None
        HAS_RUST = False


def get_default_example_dir() -> str:
    """Return the package-relative example data directory."""
    return str((Path(__file__).resolve().parent / "example"))


def ensure_loss_coefficients(obj):
    """Return the three loss coefficients, adding defaults to older objects."""
    coeffs = np.asarray(
        getattr(obj, "loss_coefficients", DEFAULT_LOSS_COEFFICIENTS),
        dtype=float,
    )
    if coeffs.shape != (3,):
        raise ValueError(
            "loss_coefficients must contain three values: "
            "[total_sigma, 1 - abs_corr, 1 - fl_corr]"
        )
    obj.loss_coefficients = coeffs
    return coeffs



class load_input:
    """
    Class to load input files and calculate physical parameters for the simulation.

    Loads vibrational frequencies, displacements, and global simulation parameters
    from text files and initializes the system constants (hbar, kbT, etc.).

    Attributes:
        dir (str): Directory path containing input files.
        wg (np.ndarray): Ground state normal mode frequencies (cm^-1).
        we (np.ndarray): Excited state normal mode frequencies (cm^-1).
        delta (np.ndarray): Dimensionless displacements for each mode.
        S (np.ndarray): Huang-Rhys factors (S = delta^2 / 2).
        T (float): Temperature in Kelvin.
        gamma (float): Homogeneous broadening parameter (cm^-1).
        theta (float): Static inhomogeneous broadening parameter (cm^-1).
        E0 (float): Vertical transition energy (cm^-1).
        k (float): Kappa (κ) parameter for the Brownian oscillator.
        M (float): Transition dipole moment length (Angstroms).
        n (float): Refractive index of the medium.
    """

    def __init__(self, dir=None):
        if dir is None:
            self.dir = get_default_example_dir()
        else:
            self.dir = dir

        self.dir = os.path.abspath(self.dir)
        if not self.dir.endswith(os.sep):
            self.dir += os.sep

        def _safe_loadtxt(filename):
            path = os.path.join(self.dir, filename)
            if not os.path.exists(path):
                return None
            return np.atleast_1d(np.asarray(np.loadtxt(path), dtype=float))

        # Ground state normal mode frequencies cm^-1 (freqs.dat)
        freqs = _safe_loadtxt("freqs.dat")
        if freqs is None:
            print(f"Missing required file: {os.path.join(self.dir, 'freqs.dat')}. Exiting.")
            sys.exit(1)
        else:
            self.wg = freqs
        # Excited state normal mode frequencies cm^-1
        self.we = np.copy(self.wg)
        # Dimensionless displacements (deltas.dat)
        deltas = _safe_loadtxt("deltas.dat")
        if deltas is None:
            print(f"Missing required file: {os.path.join(self.dir, 'deltas.dat')}. Exiting.")
            sys.exit(1)
        else:
            self.delta = deltas

        if len(self.delta) != len(self.wg):
            n = min(len(self.delta), len(self.wg))
            if n == 0:
                self.wg = np.array([], dtype=float)
                self.we = np.array([], dtype=float)
                self.delta = np.array([], dtype=float)
            else:
                print("freqs.dat and deltas.dat length mismatch. Truncating to shortest length.")
                self.wg = self.wg[:n]
                self.we = self.we[:n]
                self.delta = self.delta[:n]

        # UI/Plotting helpers
        self.colors = plt.cm.hsv(np.linspace(0, 1, len(self.wg)))
        self.cmap = ListedColormap(self.colors)

        # Huang-Rhys factor calculation
        self.S = (self.delta**2) / 2

        # Load parameters from inp.txt
        self.inp_txt()
        self.raman_maxcalc = float(self.inp[10])
        self.EL_reach = float(self.inp[6])
        self.update_params()

        # Load experimental spectra if available
        try:
            self.abs_exp_raw = np.loadtxt(self.dir + "abs_exp.dat")
            if self.abs_exp_raw[0, 0] > self.abs_exp_raw[-1, 0]:
                print("Experimental absorption spectrum appears to be inverted.")
                self.abs_exp_raw[:, 0] = self.abs_exp_raw[::-1, 0]
                self.abs_exp_raw[:, 1] = self.abs_exp_raw[::-1, 1]
            self.update_abs_exp_interpolant()
        except Exception:
            print("No experimental absorption spectrum found in directory/")

        try:
            self.fl_exp_raw = np.loadtxt(self.dir + "fl_exp.dat")
            if self.fl_exp_raw[0, 0] > self.fl_exp_raw[-1, 0]:
                print("Experimental fluorescence spectrum appears to be inverted.")
                self.fl_exp_raw[:, 0] = self.fl_exp_raw[::-1, 0]
                self.fl_exp_raw[:, 1] = self.fl_exp_raw[::-1, 1]
            self.update_fl_exp_interpolant()
        except Exception:
            print("No experimental fluorescence spectrum found in directory/")

        try:
            self.profs_exp = np.loadtxt(self.dir + "profs_exp.dat")
        except Exception:
            print("No experimental Raman cross section found in directory/")

        # Initialize result containers
        (
            self.abs_cross,
            self.fl_cross,
            self.raman_cross,
            self.boltz_state,
            self.boltz_coef,
        ) = None, None, None, None, None
        self.sigma = np.zeros_like(self.delta)  # Raman cross sections per mode
        self.correlation = None  # Correlation between calc and expt absorption
        self.fl_correlation = None  # Correlation between calc and expt fluorescence
        self.loss_coefficients = DEFAULT_LOSS_COEFFICIENTS.copy()
        self.total_sigma = None  # Total Raman cross section
        self.loss = None
        self.correlation_list = []
        self.fl_correlation_list = []
        self.sigma_list = []
        self.loss_list = []

    def update_abs_exp_interpolant(self):
        """Refresh absorption data interpolated onto the current calculation grid."""
        abs_spec_interp = np.interp(
            self.convEL, self.abs_exp_raw[:, 0], self.abs_exp_raw[:, 1]
        )
        self.abs_exp = np.stack((self.convEL, abs_spec_interp), axis=0).T

    def update_fl_exp_interpolant(self):
        """Refresh fluorescence data interpolated onto the current calculation grid."""
        fl_exp_interp = np.interp(self.convEL, self.fl_exp_raw[:, 0], self.fl_exp_raw[:, 1])
        self.fl_exp = np.stack((self.convEL, fl_exp_interp), axis=0).T

    def update_experimental_interpolants(self):
        """Refresh experimental interpolants after the calculation grid changes."""
        if hasattr(self, "abs_exp_raw"):
            self.update_abs_exp_interpolant()
        if hasattr(self, "fl_exp_raw"):
            self.update_fl_exp_interpolant()

    def inp_txt(self):
        """
        Reads simulation parameters from 'inp.txt' and calculates derived physical constants.

        Derived constants include:
        - beta (β): 1/kbT
        - eta (η): Average thermal occupation numbers (Bose-Einstein statistics)
        - Brownian oscillator parameters: D, L (Lambda)
        - Reorganization energies: s_reorg (solvent), w_reorg (vibrational)
        - Time and energy grids for integration.
        """
        try:
            with open(self.dir + "inp.txt", "r") as i:
                self.inp = [line.partition("#")[0].rstrip() for line in i.readlines()]
        except Exception:
            with open(self.dir + "inp_new.txt", "r") as i:
                self.inp = [line.partition("#")[0].rstrip() for line in i.readlines()]

        # Physical constants
        self.hbar = 5.3088  # Planck's constant (cm^-1 * ps)
        self.T = float(self.inp[13])  # Temperature (K)
        self.kbT = 0.695 * self.T  # Thermal energy in wavenumbers (cm^-1)
        self.cutoff = self.kbT * 0.1  # Cutoff for Boltzmann distribution

        # Thermal occupation factors (η)
        if self.T > 0.1:
            self.beta = 1 / self.kbT  # beta in cm^-1
            self.eta = 1 / (np.exp(self.wg / self.kbT) - 1)
        else:
            self.beta = 1.0e10  # Effectively infinite for T=0
            self.eta = np.zeros(len(self.wg))

        # Broadening parameters
        self.gamma = float(self.inp[0])  # Homogeneous broadening (cm^-1)
        self.theta = float(self.inp[1])  # Inhomogeneous broadening (cm^-1)
        self.E0 = float(self.inp[2])  # 0-0 Transition energy (cm^-1)

        ## Brownian Oscillator parameters ##
        self.k = float(self.inp[3])  # kappa parameter (κ)
        # Solvent fluctuation parameters derived from κ and Γ
        self.D = (
            self.gamma
            * (1 + 0.85 * self.k + 0.88 * self.k**2)
            / (2.355 + 1.76 * self.k)
        )
        self.L = self.k * self.D  # LAMBDA parameter (Λ)

        # Reorganization energies
        self.s_reorg = (
            self.beta * (self.L / self.k) ** 2 / 2
        )  # Solvent reorganization energy (cm^-1)
        self.w_reorg = 0.5 * np.sum(
            (self.delta) ** 2 * self.wg
        )  # Vibrational reorganization energy
        self.reorg = self.w_reorg + self.s_reorg  # Total reorganization energy

        ## Time and energy range definition ##
        self.ts = float(self.inp[4])  # Time step (ps)
        self.ntime = float(self.inp[5])  # Number of time steps
        self.UB_time = self.ntime * self.ts  # Upper bound in time range
        self.t = np.linspace(0, self.UB_time, int(self.ntime))  # time range array in ps

        self.EL_reach = float(self.inp[6])  # How far plus and minus E0 you want
        self.EL = np.linspace(self.E0 - self.EL_reach, self.E0 + self.EL_reach, 1000)
        # Static inhomogeneous convolution range
        self.E0_range = np.linspace(-self.EL_reach * 0.5, self.EL_reach * 0.5, 501)

        self.th = np.array(self.t / self.hbar)  # Scaled time (t/hbar)

        # Higher-order calculation parameters (rotational coordinates)
        self.ntime_rot = self.ntime / np.sqrt(2)
        self.ts_rot = self.ts / np.sqrt(2)
        self.UB_time_rot = self.ntime_rot * self.ts_rot
        self.tp = np.linspace(0, self.UB_time_rot, int(self.ntime_rot))
        self.tm = None
        self.tm = np.append(-np.flip(self.tp[1:], axis=0), self.tp)

        # Grid after convolution with inhomogeneous distribution
        self.convEL = np.linspace(
            self.E0 - self.EL_reach * 0.5,
            self.E0 + self.EL_reach * 0.5,
            (
                max(len(self.E0_range), len(self.EL))
                - min(len(self.E0_range), len(self.EL))
                + 1
            ),
        )

        self.M = float(self.inp[7])  # Transition dipole length (Angstroms)
        self.n = float(self.inp[8])  # Refractive index

        # Raman pump wavelengths
        try:
            self.rpumps = np.asarray(np.loadtxt(self.dir + "rpumps.dat"))
            self.rp = np.zeros_like(self.rpumps)
            diffs = np.abs(self.convEL[:, np.newaxis] - self.rpumps)
            self.rp = np.argmin(diffs, axis=0)
            self.rp = self.rp.astype(int)
        except Exception:
            print("No rpumps.dat file found in directory/. Skipping Raman calculation.")

        # Raman shift axis (output Raman spectra range)
        self.rshift = np.arange(
            float(self.inp[9]), float(self.inp[10]), float(self.inp[11])
        )
        self.res = float(self.inp[12])  # Peak width (resolution) for Raman spectra

        # Determine if thermal averaging (Boltzmann) is used
        self.convergence = float(self.inp[14])
        self.boltz_toggle = int(self.inp[15])

        if self.boltz_toggle == 1:
            self.boltz_state, self.boltz_coef, self.dos_energy = self.boltz_states()
            if self.T == 0.0:
                self.state = 0
            else:
                self.state = min(
                    range(len(self.boltz_coef)),
                    key=lambda j: abs(self.boltz_coef[j] - self.convergence),
                )

            if self.state == 0:
                self.order = 1
            else:
                self.order = max(max(self.boltz_state[: self.state])) + 1
        if self.boltz_toggle == 0:
            self.boltz_state, self.boltz_coef, self.dos_energy = [0, 0, 0]
            self.order = 1

        self.a = np.arange(self.order)
        self.b = self.a
        self.Q = np.identity(len(self.wg), dtype=int)

        # wq = None
        # wq = np.append(wg,wg)

        ## Prefactors for absorption and Raman cross-sections ##
        if self.order == 1:
            # (0.3/pi) puts it in differential cross section
            self.preR = 2.08e-20 * (self.ts**2)
        elif self.order > 1:
            self.preR = 2.08e-20 * (self.ts_rot**2)

        self.preA = ((5.744e-3) / self.n) * self.ts
        self.preF = self.preA * self.n**2

    def update_params(self):
        """
        Refreshes derived parameters from the current set of base parameters.
        This includes thermal factors, Brownian oscillator parameters, reorganization 
        energies, and integration time/energy grids.
        """
        # Physical constants
        self.kbT = 0.695 * self.T  # Thermal energy in wavenumbers (cm^-1)
        self.cutoff = self.kbT * 0.1  # Cutoff for Boltzmann distribution

        # Thermal occupation factors (η)
        if self.T > 0.1:
            self.beta = 1 / self.kbT  # beta in cm^-1
            self.eta = 1 / (np.exp(self.wg / self.kbT) - 1)
        else:
            self.beta = 1.0e10  # Effectively infinite for T=0
            self.eta = np.zeros(len(self.wg))

        ## Brownian Oscillator parameters ##
        # Solvent fluctuation parameters derived from κ and Γ
        self.D = (
            self.gamma
            * (1 + 0.85 * self.k + 0.88 * self.k**2)
            / (2.355 + 1.76 * self.k)
        )
        self.L = self.k * self.D  # LAMBDA parameter (Λ)

        # Reorganization energies
        self.s_reorg = (
            self.beta * (self.L / self.k) ** 2 / 2
        )  # Solvent reorganization energy (cm^-1)
        self.w_reorg = 0.5 * np.sum(
            (self.delta) ** 2 * self.wg
        )  # Vibrational reorganization energy
        self.reorg = self.w_reorg + self.s_reorg  # Total reorganization energy

        ## Time and energy range definition ##
        self.UB_time = self.ntime * self.ts  # Upper bound in time range
        self.t = np.linspace(0, self.UB_time, int(self.ntime))  # time range array in ps
        self.th = np.array(self.t / self.hbar)  # Scaled time (t/hbar)

        self.EL = np.linspace(self.E0 - self.EL_reach, self.E0 + self.EL_reach, 1000)
        # static inhomogeneous convolution range
        self.E0_range = np.linspace(-self.EL_reach * 0.5, self.EL_reach * 0.5, 501)

        # Higher-order calculation parameters (rotational coordinates)
        self.ntime_rot = self.ntime / np.sqrt(2)
        self.ts_rot = self.ts / np.sqrt(2)
        self.UB_time_rot = self.ntime_rot * self.ts_rot
        self.tp = np.linspace(0, self.UB_time_rot, int(self.ntime_rot))
        self.tm = np.append(-np.flip(self.tp[1:], axis=0), self.tp)

        # Grid after convolution with inhomogeneous distribution
        self.convEL = np.linspace(
            self.E0 - self.EL_reach * 0.5,
            self.E0 + self.EL_reach * 0.5,
            (
                max(len(self.E0_range), len(self.EL))
                - min(len(self.E0_range), len(self.EL))
                + 1
            ),
        )

        # Refresh Huang-Rhys factors
        self.S = (self.delta**2) / 2

        # Update Raman shift axis (output Raman spectra range)
        # Using end rshift from inp[10] if raman_maxcalc is not set
        r_end = getattr(self, "raman_maxcalc", float(self.inp[10]))
        self.rshift = np.arange(
            float(self.inp[9]), r_end, float(self.inp[11])
        )
        self.res = float(self.inp[12])

        # Update Raman pump indices
        try:
            diffs = np.abs(self.convEL[:, np.newaxis] - self.rpumps)
            self.rp = np.argmin(diffs, axis=0)
            self.rp = self.rp.astype(int)
        except Exception:
            pass

        # Update prefactors
        if self.order == 1:
            self.preR = 2.08e-20 * (self.ts**2)
        elif self.order > 1:
            self.preR = 2.08e-20 * (self.ts_rot**2)

        self.preA = ((5.744e-3) / self.n) * self.ts
        self.preF = self.preA * self.n**2

    def boltz_states(self):
        """
        Calculates possible initial vibrational states and their Boltzmann coefficients.

        Returns:
            list: Combinations of vibrational states.
            list: Corresponding Boltzmann coefficients (normalized).
            list: Energy of each state.
        """
        wg = self.wg.astype(int)
        cutoff = range(int(self.cutoff))
        dos = range(len(self.cutoff))
        states = []
        dos_energy = []

        def count_combs(left, i, comb, add):
            if add:
                comb.append(add)
            if left == 0 or (i + 1) == len(wg):
                if (i + 1) == len(wg) and left > 0:
                    if left % wg[i]:  # can't get the exact score with this kind of wg
                        return 0  # so give up on this recursive branch
                    comb.append((left / wg[i], wg[i]))  # fix the amount here
                    i += 1
                while i < len(wg):
                    comb.append((0, wg[i]))
                    i += 1
                states.append([x[0] for x in comb])
                return 1
            cur = wg[i]
            return sum(
                count_combs(left - x * cur, i + 1, comb[:], (x, cur))
                for x in range(0, int(left / cur) + 1)
            )

        boltz_dist = []  # np.zeros(len(dos))
        for i in range(len(cutoff)):
            dos[i] = count_combs(self.cutoff[i], 0, [], None)
            if dos[i] > 0.0:
                boltz_dist.append([np.exp(-cutoff[i] * self.beta)])
                dos_energy.append(cutoff[i])

        norm = np.sum(boltz_dist)
        np.reshape(states, -1, len(cutoff))

        return states, boltz_dist / norm, dos_energy


def g(t, obj):
    """
    Brownian oscillator lineshape function g(t).

    Implements Equation S9 in the reference. This function accounts for
    homogeneous broadening and the dynamics of the environment.

    Args:
        t (float or np.ndarray): Time variable.
        obj: Object containing Brownian parameters (D, L, beta).

    Returns:
        complex: The lineshape function g(t).
    """
    g = ((obj.D / obj.L) ** 2) * (obj.L * t - 1 + np.exp(-obj.L * t)) + 1j * (
        (obj.beta * obj.D**2) / (2 * obj.L)
    ) * (1 - np.exp(-obj.L * t))
    return g


def A(t, obj):
    """
    Time-correlator A(t) for absorption and fluorescence.
    
    Implements Equation S6. This function calculates the electronic 
    transition dipole correlation function weighted by the vibrational 
    overlaps (IMDHO model).
    # old A function
    def A(t,obj):
    # K=np.zeros((len(p.wg),len(t)),dtype=complex)
    # Initialize K matrix based on the type of t provided
    if type(t) == np.ndarray:
        K = np.zeros((len(obj.wg), len(obj.th)), dtype=complex)
    else:
        K = np.zeros((len(obj.wg), 1), dtype=complex)
    # Calculate the K matrix
    for l in np.arange(len(obj.wg)):
        K[l, :] = (1+obj.eta[l])*obj.S[l]*(1-np.exp(-1j*obj.wg[l]*t)) + \
            obj.eta[l]*obj.S[l]*(1-np.exp(1j*obj.wg[l]*t))
    # Calculate the function A based on the K matrix
    A = obj.M**2*np.exp(-np.sum(K, axis=0))
    return A
    Args:
        t (float or np.ndarray): Time variable.
        obj: Object containing mode parameters (wg, S, eta, M).

    Returns:
        complex: The correlation function A(t).
    """
    # Vectorized calculation of the vibrational factor K(t)
    if isinstance(t, np.ndarray):
        K = (1 + obj.eta[:, np.newaxis]) * obj.S[:, np.newaxis] * (
            1 - np.exp(-1j * obj.wg[:, np.newaxis] * t)
        ) + obj.eta[:, np.newaxis] * obj.S[:, np.newaxis] * (
            1 - np.exp(1j * obj.wg[:, np.newaxis] * t)
        )
    else:
        K = (1 + obj.eta) * obj.S * (1 - np.exp(-1j * obj.wg * t)) + obj.eta * obj.S * (
            1 - np.exp(1j * obj.wg * t)
        )

    # A(t) = M^2 * exp(-sum_k K_k(t))
    A = obj.M**2 * np.exp(-np.sum(K, axis=0))
    return A


def R(t1, t2, obj):
    """
    Higher-order Raman correlation function.

    Calculates the multi-dimensional correlation function required for
    higher-order Raman cross sections or multi-mode coupling effects.
    # old R function
    def R(t1, t2,obj):
    # Initialize Ra and R arrays for calculations
    Ra = np.zeros((len(obj.a), len(obj.wg), len(obj.wg), len(obj.EL)), dtype=complex)
    R = np.zeros((len(obj.wg), len(obj.wg), len(obj.EL)), dtype=complex)
    # for l in np.arange(len(p.wg)):
    # 	for q in p.Q:
    for idxq, q in enumerate(obj.Q, start=0):
        for idxl, l in enumerate(q, start=0):

            wg = obj.wg[idxl]
            S = obj.S[idxl]
            eta = obj.eta[idxl]
            if l == 0:
                for idxa, a in enumerate(obj.a, start=0):
                    Ra[idxa, idxq, idxl, :] = ((1./factorial(a))**2)*((eta*(1+eta))**a)*S**(2*a)*(
                        ((1-np.exp(-1j*wg*t1))*np.conj((1-np.exp(-1j*wg*t1))))*((1-np.exp(-1j*wg*t1))*np.conj((1-np.exp(-1j*wg*t1)))))**a
                R[idxq, idxl, :] = np.sum(Ra[:, idxq, idxl, :], axis=0)
            elif l > 0:
                for idxa, a in enumerate(obj.a[l:], start=0):
                    Ra[idxa, idxq, idxl, :] = ((1./(factorial(a)*factorial(a-l))))*(((1+eta)*S*(1-np.exp(-1j*wg*t1))*(
                        1-np.exp(1j*wg*t2)))**a)*(eta*S*(1-np.exp(1j*wg*t1))*(1-np.exp(-1j*wg*t2)))**(a-l)
                R[idxq, idxl, :] = np.sum(Ra[:, idxq, idxl, :], axis=0)
            elif l < 0:
                for idxa, a in enumerate(obj.b[-l:], start=0):
                    Ra[idxa, idxq, idxl, :] = ((1./(factorial(a)*factorial(a+l))))*(((1+eta)*S*(1-np.exp(-1j*wg*t1))*(
                        1-np.exp(1j*wg*t2)))**(a+l))*(eta*S*(1-np.exp(1j*wg*t1))*(1-np.exp(-1j*wg*t2)))**(a)
                R[idxq, idxl, :] = np.sum(Ra[:, idxq, idxl, :], axis=0)
    return np.prod(R, axis=1)
    Args:
        t1, t2: Time variables.
        obj: Simulation object.

    Returns:
        complex: The Raman correlation function R(t1, t2).
    """
    # Initialize Ra and R arrays for calculations
    Ra = np.zeros((len(obj.a), len(obj.Q), len(obj.wg), len(obj.EL)), dtype=complex)
    R = np.zeros((len(obj.Q), len(obj.wg), len(obj.EL)), dtype=complex)

    # Create meshgrid of indices for efficient indexing
    idx_a, idx_q, idx_l = np.meshgrid(
        np.arange(len(obj.a)),
        np.arange(len(obj.Q)),
        np.arange(len(obj.wg)),
        indexing="ij",
    )

    wg = obj.wg[:, np.newaxis, np.newaxis]
    S = obj.S[:, np.newaxis, np.newaxis]
    eta = obj.eta[:, np.newaxis, np.newaxis]

    # Partition calculation based on vibrational quantum number change Q
    Ra = np.where(
        obj.Q[:, np.newaxis, np.newaxis] == 0,
        ((1.0 / factorial(obj.a)) ** 2)
        * ((eta * (1 + eta)) ** obj.a)
        * S ** (2 * obj.a)
        * (
            ((1 - np.exp(-1j * wg * t1)) * np.conj((1 - np.exp(-1j * wg * t1))))
            * ((1 - np.exp(-1j * wg * t1)) * np.conj((1 - np.exp(-1j * wg * t1))))
        )
        ** obj.a,
        np.where(
            obj.Q[:, np.newaxis, np.newaxis] > 0,
            (
                (
                    1.0
                    / (
                        factorial(obj.a)
                        * factorial(obj.a - obj.Q[:, np.newaxis, np.newaxis])
                    )
                )
                * (
                    (
                        (1 + eta)
                        * S
                        * (1 - np.exp(-1j * wg * t1))
                        * (1 - np.exp(1j * wg * t2))
                    )
                    ** obj.a
                )
                * (eta * S * (1 - np.exp(1j * wg * t1)) * (1 - np.exp(-1j * wg * t2)))
                ** (obj.a - obj.Q[:, np.newaxis, np.newaxis])
            ),
            (
                (
                    1.0
                    / (
                        factorial(obj.a)
                        * factorial(obj.a + obj.Q[:, np.newaxis, np.newaxis])
                    )
                )
                * (
                    (
                        (1 + eta)
                        * S
                        * (1 - np.exp(-1j * wg * t1))
                        * (1 - np.exp(1j * wg * t2))
                    )
                    ** (obj.a + obj.Q[:, np.newaxis, np.newaxis])
                )
            )
            * (eta * S * (1 - np.exp(1j * wg * t1)) * (1 - np.exp(-1j * wg * t2)))
            ** obj.a,
        ),
    )

    # Calculate R using np.sum along axis
    R = np.sum(Ra, axis=0)

    return np.prod(R, axis=1)


def cross_sections(obj):
    """
    Main calculation engine for Absorption, Fluorescence, and Raman cross sections.

    This function performs the time integrations of the correlation functions
    to obtain the frequency-domain spectra.

    1. Calculates Absorption and Fluorescence using A(t) and g(t).
    2. Convolves results with a Gaussian (inhomogeneous broadening θ).
    3. Calculates Raman Excitation Profiles (REPs) using Equation S5.

    Args:
        obj: Simulation object containing all parameters.

    Returns:
        tuple: (abs_cross, fl_cross, raman_cross, boltz_state, boltz_coef)
    """
    obj.update_params()
    sqrt2 = np.sqrt(2)

    if HAS_RUST and obj.order == 1:
        abs_c, fl_c, ram_c = resram_rust.cross_sections_rust(
            obj.wg,
            obj.S,
            obj.eta,
            obj.delta,
            obj.th,
            obj.EL,
            obj.convEL,
            obj.E0_range,
            obj.D,
            obj.L,
            obj.beta,
            obj.E0,
            obj.M,
            obj.preA,
            obj.preF,
            obj.preR,
            obj.theta,
            obj.ts,
        )
        obj.abs_cross, obj.fl_cross, obj.raman_cross = abs_c, fl_c, ram_c
        return obj.abs_cross, obj.fl_cross, obj.raman_cross, obj.boltz_state, obj.boltz_coef

    # Work arrays
    q_r = np.ones((len(obj.wg), len(obj.wg), len(obj.th)), dtype=complex)
    K_r = np.zeros((len(obj.wg), len(obj.EL), len(obj.th)), dtype=complex)
    # elif p.order > 1:
    # 	K_r = np.zeros((len(p.tm),len(p.tp),len(p.wg),len(p.EL)),dtype=complex)
    integ_r1 = np.zeros((len(obj.tm), len(obj.EL)), dtype=complex)
    integ_r = np.zeros((len(obj.wg), len(obj.EL)), dtype=complex)
    obj.raman_cross = np.zeros((len(obj.wg), len(obj.convEL)), dtype=complex)

    # Inhomogeneous broadening distribution (H)
    if obj.theta == 0.0:
        H = 1.0  # np.ones(len(p.E0_range))
    else:
        H = (1 / (obj.theta * np.sqrt(2 * np.pi))) * np.exp(
            -((obj.E0_range) ** 2) / (2 * obj.theta**2)
        )

    thth, ELEL = np.meshgrid(obj.th, obj.EL, sparse=True)

    # Absorption and Fluorescence kernels
    K_a = np.exp(1j * (ELEL - (obj.E0)) * thth - g(thth, obj)) * A(thth, obj)
    K_f = np.exp(1j * (ELEL - (obj.E0)) * thth - np.conj(g(thth, obj))) * np.conj(
        A(thth, obj)
    )

    ## First-order Raman Approximation (Equation S5) ##
    if obj.order == 1:
        for idxq, q in enumerate(obj.Q, start=0):
            for idxl, l in enumerate(q, start=0):
                if q[idxl] > 0:
                    # Positive change in vibrational quantum number. This corresponds to Stokes scattering where the vibrational mode is excited.
                    q_r[idxq, idxl, :] = (
                        np.sqrt((1.0 / factorial(q[idxl])))
                        * (((1 + obj.eta[idxl]) ** (0.5) * obj.delta[idxl]) / sqrt2)
                        ** (q[idxl])
                        * (1 - np.exp(-1j * obj.wg[idxl] * thth)) ** (q[idxl])
                    )
                elif q[idxl] < 0:
                    # Negative change in vibrational quantum number. This corresponds to anti-Stokes scattering where the vibrational mode is de-excited.
                    q_r[idxq, idxl, :] = (
                        np.sqrt(1.0 / factorial(np.abs(q[idxl])))
                        * (((obj.eta[idxl]) ** (0.5) * obj.delta[idxl]) / sqrt2) ** (-q[idxl])
                        * (1 - np.exp(1j * obj.wg[idxl] * thth)) ** (-q[idxl])
                    )
            # Combine mode factor with absorption correlator to get Raman kernel
            K_r[idxq, :, :] = K_a * (np.prod(q_r, axis=1)[idxq])

    # If the order is greater than 1, carry out the sums R and compute the full double integral
    ##### Higher order is still broken, need to fix #####
    elif obj.order > 1:
        tpp, tmm, ELEL = np.meshgrid(obj.tp, obj.tm, obj.EL, sparse=True)
        # *A((tpp+tmm)/(np.sqrt(2)))*np.conj(A((tpp-tmm)/(np.sqrt(2))))#*R((tpp+tmm)/(np.sqrt(2)),(tpp-tmm)/(np.sqrt(2)))
        K_r = np.exp(
            1j * (ELEL - obj.E0) * sqrt2 * tmm
            - g(tpp + tmm, obj) / (sqrt2)
            - np.conj(g((tpp - tmm) / (sqrt2), obj))
        )

        for idxtm, tm in enumerate(obj.tm, start=0):
            integ_r1[idxtm, :] = np.trapezoid(
                K_r[(np.abs(len(obj.tm) / 2 - idxtm)) :, idxtm, :], axis=0
            )

        integ = np.trapezoid(integ_r1, axis=0)
    ######################################################

    # Time integration and Convolution with inhomogeneous broadening
    integ_a = np.trapezoid(K_a, axis=1)
    obj.abs_cross = (
        obj.preA * obj.convEL * np.convolve(integ_a, np.real(H), "valid") / (np.sum(H))
    )

    integ_f = np.trapezoid(K_f, axis=1)
    obj.fl_cross = (
        obj.preF * obj.convEL * np.convolve(integ_f, np.real(H), "valid") / (np.sum(H))
    )

    for idx, wg_value in enumerate(obj.wg):
        if obj.order == 1:
            integ_r = np.trapezoid(K_r[idx, :, :], axis=1)
            # Raman cross section scaling by (omega_L - omega_v)^3
            obj.raman_cross[idx, :] = (
                obj.preR
                * obj.convEL
                * (obj.convEL - wg_value) ** 3
                * np.convolve(integ_r * np.conj(integ_r), np.real(H), "valid")
                / np.sum(H)
            )
        elif obj.order > 1:
            integ_r = np.trapezoid(K_r[idx, :, :], axis=1)
            obj.raman_cross[idx, :] = (
                obj.preR
                * obj.convEL
                * (obj.convEL - wg_value) ** 3
                * np.convolve(integ_r, np.real(H), "valid")
                / np.sum(H)
            )

    return obj.abs_cross, obj.fl_cross, obj.raman_cross, obj.boltz_state, obj.boltz_coef


def run_save(obj, current_time_str=None):
    """
    Executes the simulation and saves all resulting data to a time-stamped directory.

    Generates:
    - Absorption and Fluorescence spectra files.
    - Raman excitation profiles (profs.dat).
    - Raman spectra at specific pumps (raman_spec.dat).
    - Summary output file (output.txt).
    """
    if current_time_str is None:
        current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    abs_cross, fl_cross, raman_cross, boltz_states, boltz_coef = cross_sections(obj)
    try:
        raman_spec = np.zeros((len(obj.rshift), len(obj.rpumps)))

        # Convert REPs to Raman spectra by placing Lorentzians at mode frequencies
        for i, rp in enumerate(obj.rp):
            for j, wg in enumerate(obj.wg):
                raman_spec[:, i] += (
                    np.real((raman_cross[j, rp]))
                    * (1 / np.pi)
                    * (0.5 * obj.res)
                    / ((obj.rshift - wg) ** 2 + (0.5 * obj.res) ** 2)
                )
    except Exception:
        raman_spec = None
        print("Raman calculation skipped, no rpumps.dat file found in directory/")

    # Create data folder
    try:
        os.mkdir("./" + current_time_str + "_data")
    except FileExistsError:
        pass

    # Finalize reorganization energies for saving
    obj.s_reorg = obj.beta * (obj.L / obj.k) ** 2 / 2
    obj.w_reorg = 0.5 * np.sum((obj.delta) ** 2 * obj.wg)
    obj.reorg = obj.w_reorg + obj.s_reorg

    # Save results to disk
    np.set_printoptions(threshold=sys.maxsize)
    np.savetxt(
        current_time_str + "_data/profs.dat",
        np.real(np.transpose(raman_cross)),
        delimiter="\t",
    )
    try:
        np.savetxt(
            current_time_str + "_data/raman_spec.dat", raman_spec, delimiter="\t"
        )
        np.savetxt(current_time_str + "_data/rpumps.dat", obj.rpumps)
    except Exception:
        pass
    np.savetxt(current_time_str + "_data/EL.dat", obj.convEL)
    np.savetxt(current_time_str + "_data/deltas.dat", obj.delta)
    np.savetxt(current_time_str + "_data/Abs.dat", np.real(abs_cross))
    np.savetxt(current_time_str + "_data/Fl.dat", np.real(fl_cross))
    np.savetxt(current_time_str + "_data/rshift.dat", obj.rshift)

    # Save current parameters for reproducibility
    inp_list = [float(x) for x in obj.inp]
    inp_list[7] = obj.M
    inp_list[0] = obj.gamma
    inp_list[1] = obj.theta
    inp_list[2] = obj.E0
    inp_list[3] = obj.k
    inp_list[8] = obj.n

    np.savetxt(current_time_str + "_data/inp.dat", inp_list)
    np.savetxt(current_time_str + "_data/freqs.dat", obj.wg)

    try:
        np.savetxt(current_time_str + "_data/abs_exp.dat", obj.abs_exp, delimiter="\t")
    except Exception:
        print("No experimental absorption spectrum found in directory/")
    try:
        np.savetxt(current_time_str + "_data/fl_exp.dat", obj.fl_exp, delimiter="\t")
    except Exception:
        print("No experimental absorption spectrum found in directory/")
    try:
        np.savetxt(
            current_time_str + "_data/profs_exp.dat", obj.profs_exp, delimiter="\t"
        )
    except Exception:
        print("No experimental Raman cross section found in directory/")

    # Write summary report
    with open(current_time_str + "_data/output.txt", "w") as o:
        o.write("E00 = "), o.write(str(obj.E0)), o.write(" cm-1 \n")
        o.write("gamma = "), o.write(str(obj.gamma)), o.write(" cm-1 \n")
        o.write("theta = "), o.write(str(obj.theta)), o.write(" cm-1 \n")
        o.write("M = "), o.write(str(obj.M)), o.write(" Angstroms \n")
        o.write("n = "), o.write(str(obj.n)), o.write("\n")
        o.write("T = "), o.write(str(obj.T)), o.write(" Kelvin \n")
        (
            o.write("solvent reorganization energy = "),
            o.write(str(obj.s_reorg)),
            o.write(" cm-1 \n"),
        )
        (
            o.write("internal reorganization energy = "),
            o.write(str(obj.w_reorg)),
            o.write(" cm-1 \n"),
        )
        (
            o.write("reorganization energy = "),
            o.write(str(obj.reorg)),
            o.write(" cm-1 \n\n"),
        )
        o.write("Boltzmann averaged states and their corresponding weights \n")
        o.write(str(obj.boltz_coef)), o.write("\n")
        o.write(str(obj.boltz_state)), o.write("\n")
    o.close()

    # Create new input file based on current optimization
    with open(current_time_str + "_data/inp_new.txt", "w") as file:
        file.write(f"{obj.gamma} # gamma linewidth parameter (cm^-1)\n")
        file.write(
            f"{obj.theta} # theta static inhomogeneous linewidth parameter (cm^-1)\n"
        )
        file.write(f"{obj.E0} # E0 (cm^-1)\n")
        file.write(f"{obj.k} # kappa solvent parameter\n")
        file.write(f"{obj.ts} # time step (ps)\n")
        file.write(f"{obj.ntime} # number of time steps\n")
        file.write(
            f"{obj.EL_reach} # range plus and minus E0 to calculate lineshapes\n"
        )
        file.write(f"{obj.M} # transition length M (Angstroms)\n")
        file.write(f"{obj.n} # refractive index n\n")
        file.write(f"{obj.inp[9]} # start raman shift axis (cm^-1)\n")
        file.write(f"{obj.inp[10]} # end raman shift axis (cm^-1)\n")
        file.write(f"{obj.inp[11]} # rshift axis step size (cm^-1)\n")
        file.write(f"{obj.inp[12]} # raman spectrum resolution (cm^-1)\n")
        file.write(f"{obj.T} # Temperature (K)\n")
        file.write(
            f"{obj.inp[14]} # convergence for sums # no effect since order > 1 broken\n"
        )
        file.write(f"{obj.inp[15]} # Boltz Toggle\n")

    return resram_data(current_time_str + "_data/")


class resram_data:
    """
    Class for managing, loading, and plotting ResRAM simulation results.

    Can be initialized with a directory path containing results from run_save.
    """

    def __init__(self, input=None):
        if input is None:
            # Default initialization (runs a new simulation)
            self.obj = load_input()
            abs_cross, fl_cross, raman_cross, boltz_state, boltz_coef = cross_sections(
                self.obj
            )
            self.raman_spec = np.zeros((len(self.obj.rshift), len(self.obj.rpumps)))
            for i, rp in enumerate(self.obj.rp):
                for j, wg in enumerate(self.obj.wg):
                    self.raman_spec[:, i] += (
                        np.real((raman_cross[j, rp]))
                        * (1 / np.pi)
                        * (0.5 * self.obj.res)
                        / ((self.obj.rshift - wg) ** 2 + (0.5 * self.obj.res) ** 2)
                    )
            self.fl = np.real(fl_cross)
            self.abs = np.real(abs_cross)

            self.EL = self.obj.convEL
            self.filename = None
            self.wg = self.obj.wg
            self.rpumps = self.obj.rpumps
            self.delta = self.obj.delta
            self.rshift = self.obj.rshift
            self.profs = np.real(np.transpose(raman_cross))
            self.M = self.obj.M
            self.gamma = self.obj.gamma
            self.theta = self.obj.theta
            self.E0 = self.obj.E0
            self.kappa = self.obj.k
            self.n = self.obj.n
            try:
                self.abs_exp = self.obj.abs_exp
            except Exception:
                print("No experimental absorption spectrum found in directory/")
            try:
                self.fl_exp = self.obj.fl_exp
            except Exception:
                print("No experimental fluorescence spectrum found in directory/")
            try:
                self.profs_exp = self.obj.profs_exp
            except Exception:
                print("No experimental Raman cross section found in directory/")

        else:
            # Initialization from saved files
            self.filename = input
            self.wg = np.loadtxt(input + "/freqs.dat")
            self.delta = np.loadtxt(input + "/deltas.dat")
            self.abs = np.loadtxt(input + "/Abs.dat")
            self.EL = np.loadtxt(input + "/EL.dat")
            try:
                self.fl = np.loadtxt(input + "/Fl.dat")
            except Exception:
                print("No fluorescence spectrum found in directory " + input)
            self.raman_spec = np.loadtxt(input + "/raman_spec.dat")
            self.rshift = np.loadtxt(input + "/rshift.dat")
            self.rpumps = None
            try:
                self.profs = np.loadtxt(input + "/profs.dat")
                self.rpumps = np.loadtxt(input + "/rpumps.dat")
            except Exception:
                print("No rpumps.dat or profs.dat file found in directory " + input)
            self.inp = np.loadtxt(input + "/inp.dat")
            self.M = self.inp[7]
            self.gamma = self.inp[0]
            self.theta = self.inp[1]
            self.E0 = self.inp[2]
            self.kappa = self.inp[3]
            self.n = self.inp[8]
            try:
                self.abs_exp = np.loadtxt(input + "/abs_exp.dat")
            except Exception:
                print("No experimental absorption spectrum found in directory " + input)
            try:
                self.profs_exp = np.loadtxt(input + "/profs_exp.dat")
            except Exception:
                print("No experimental Raman cross section found in directory " + input)
            try:
                self.fl_exp = np.loadtxt(input + "/fl_exp.dat")
            except Exception:
                print(
                    "No experimental fluorescence spectrum found in directory " + input
                )

    def plot(self):
        """
        Generates plots for Raman spectra, Raman excitation profiles, and Abs/Fl spectra.
        """
        # divide color map to number of freqs
        colors = plt.cm.hsv(np.linspace(0, 1, len(self.wg)))
        cmap = ListedColormap(colors)

        # Plot Raman spectra at all excitation wavelengths
        if self.rpumps is not None:
            self.fig_raman, self.ax_raman = plt.subplots(figsize=(8, 6))
            for i in np.arange(len(self.rpumps)):
                self.ax_raman.plot(
                    self.rshift,
                    self.raman_spec[:, i],
                    label=str(self.rpumps[i]) + " cm-1",
                )
            self.ax_raman.set_title("Raman spectra")
            self.ax_raman.set_xlabel("Raman Shift (cm-1)")
            self.ax_raman.set_ylabel("Raman Cross Section (1e-14 A**2/Molecule)")
            self.ax_raman.legend()
            self.fig_raman.show()

            ax_list = []
            fig_list = []
            for j in range(len(self.wg)):
                fig, ax = plt.subplots(figsize=(8, 6))
                fig_list.append(fig)
                ax_list.append(ax)
                ax.set_title(
                    "Raman Excitation Profile for " + str(self.wg[j]) + " cm-1"
                )
                ax.set_xlabel("Excitation Wavenumber (cm$^{{-1}}$)")
                ax.set_ylabel("Raman Cross Section (10$^{{-14}}$ Å$^{{2}}$/Molecule)")
                ax.legend(fontsize=8)
                ax.set_xlim(self.EL[0], self.EL[-1])
        else:
            print("no rpumps.dat file found in directory/, skipping Raman spectra plot")

        # Plot Excitation Profiles (REPs) vs experimental points
        if self.rpumps is not None:
            self.fig_profs, self.ax_profs = plt.subplots(figsize=(8, 6))
            for i in np.arange(len(self.rpumps)):  # iterate over pump wn
                min_diff = float("inf")
                rp = None

                # find index of closest pump wavelength
                for rps in range(len(self.EL)):
                    diff = np.absolute(self.EL[rps] - self.rpumps[i])
                    if diff < min_diff:
                        min_diff = diff
                        rp = rps

                for j in range(len(self.wg)):  # iterate over all raman freqs
                    color = cmap(j)
                    self.ax_profs.plot(
                        self.EL,
                        self.profs[:, j],
                        color=color,
                        label=str(self.wg[j]) + " cm-1",
                    ) if i == 0 else self.ax_profs.plot(
                        self.EL, self.profs[:, j], color=color
                    )
                    ax_list[j].plot(
                        self.EL,
                        self.profs[:, j],
                        color=color,
                        label=str(self.wg[j]) + " cm-1",
                    )
                    try:
                        self.ax_profs.plot(
                            self.EL[rp], self.profs_exp[j, i], "o", color=color
                        )
                        ax_list[j].plot(
                            self.EL[rp], self.profs_exp[j, i], "o", color=color
                        )

                    except Exception:
                        print("no experimental Raman cross section data")
                        continue

            self.ax_profs.set_title("Raman Excitation Profiles")
            self.ax_profs.set_xlabel("Excitation Wavenumber (cm$^{{-1}}$)")
            self.ax_profs.set_ylabel(
                "Raman Cross Section (10$^{{-14}}$ Å$^{{2}}$/Molecule)"
            )
            self.ax_profs.legend(ncol=2, fontsize=8)
            self.fig_profs.show()
        else:
            print(
                "no rpumps.dat file found in directory/, skipping Raman excitation profile plot"
            )

        # Plot Absorption and Fluorescence spectra
        self.fig_absfl, self.ax_fl = plt.subplots(figsize=(8, 6))
        if self.fl is not None:
            # Fluorescence correction by (omega_L/omega_0)^2 factor
            self.fl_w3 = self.fl * self.EL**2 / self.E0**2
        # fig,ax = plt.subplots(figsize=(8, 6))
        # ax.plot(self.EL, self.fl_w3/self.fl_w3.max(), label="fl w3 correction")
        # ax.plot(self.EL, self.fl/self.fl.max(), label="fl")
        # ax.legend()
        self.ax_fl.plot(self.EL, self.fl_w3, label="Calc. Fl.", color="red")
        self.ax_abs = self.ax_fl.twinx()
        self.ax_abs.plot(self.EL, self.abs, label="Calc. Abs.", color="blue")

        try:
            self.ax_abs.plot(
                self.EL,
                self.abs_exp[:, 1],
                label="Expt. Abs.",
                color="blue",
                linestyle="dashed",
            )
        except Exception:
            print("no experimental absorption data")
        try:
            self.ax_fl.plot(
                self.EL,
                self.fl_w3.max() * self.fl_exp[:, 1] / self.fl_exp[:, 1].max(),
                label="Expt. Fl.",
                color="red",
                linestyle="dashed",
            )
        except Exception:
            print("no experimental fluorescence data")

        self.ax_fl.set_title("Absorption and Fluorescence Spectra")
        self.ax_fl.set_xlabel("Wavenumber (cm$^{{-1}}$)", fontsize=16)
        self.ax_fl.set_ylabel("Fluorescence intensity (a.u.)", fontsize=16)
        self.ax_fl.legend(fontsize=18, loc="upper left")
        self.ax_fl.tick_params(axis="both", which="major", labelsize=14)
        self.ax_abs.legend(fontsize=18, loc="upper right")
        self.ax_abs.set_ylabel("Cross Section (Å$^{{2}}$/Molecule)", fontsize=16)
        self.ax_abs.tick_params(axis="both", which="major", labelsize=14)
        self.fig_absfl.show()


def raman_residual(param, fit_obj=None):
    """
    Objective function for optimization using lmfit.

    Calculates the 'loss' which is a combination of:
    1. Residual sum of squares for Raman excitation profiles.
    2. Correlation between calculated and experimental absorption spectra.
    3. Correlation between calculated and experimental fluorescence spectra.

    Args:
        param: lmfit.Parameters object.
        fit_obj: load_input object.

    Returns:
        tuple: (total_loss, total_raman_sigma, spectral_mismatch)
    """
    if fit_obj is None:
        fit_obj = load_input()

    # Update object parameters from optimizer
    fit_obj.delta = np.array(
        [param.valuesdict()["delta" + str(i)] for i in np.arange(len(fit_obj.delta))]
    )
    fit_obj.gamma = param.valuesdict()["gamma"]
    fit_obj.M = param.valuesdict()["transition_length"]
    fit_obj.k = param.valuesdict()["kappa"]
    fit_obj.theta = param.valuesdict()["theta"]
    fit_obj.E0 = param.valuesdict()["E0"]

    fit_obj.update_params()
    loss_coefficients = ensure_loss_coefficients(fit_obj)

    if HAS_RUST and fit_obj.order == 1:
        if fit_obj.profs_exp.ndim == 1:
            fit_obj.profs_exp = np.reshape(fit_obj.profs_exp, (-1, 1))
        try:
            loss, sigma, mismatch, abs_corr, fl_corr = resram_rust.raman_residual_rust(
                fit_obj.wg,
                fit_obj.eta,
                fit_obj.delta,
                fit_obj.th,
                fit_obj.EL,
                fit_obj.convEL,
                fit_obj.E0_range,
                fit_obj.abs_exp[:, 1],
                fit_obj.fl_exp[:, 1],
                fit_obj.profs_exp,
                fit_obj.rp.astype(np.uintp),
                loss_coefficients,
                fit_obj.gamma,
                fit_obj.M,
                fit_obj.k,
                fit_obj.theta,
                fit_obj.E0,
                fit_obj.beta,
                fit_obj.preA,
                fit_obj.preF,
                fit_obj.preR,
                fit_obj.ts,
            )
            fit_obj.loss = loss
            fit_obj.total_sigma = sigma
            fit_obj.correlation = abs_corr
            fit_obj.fl_correlation = fl_corr

            # Track history
            if not hasattr(fit_obj, 'loss_list'): fit_obj.loss_list = []
            if not hasattr(fit_obj, 'correlation_list'): fit_obj.correlation_list = []
            if not hasattr(fit_obj, 'fl_correlation_list'): fit_obj.fl_correlation_list = []
            if not hasattr(fit_obj, 'sigma_list'): fit_obj.sigma_list = []
            
            fit_obj.loss_list.append(fit_obj.loss)
            fit_obj.correlation_list.append(fit_obj.correlation)
            fit_obj.fl_correlation_list.append(fit_obj.fl_correlation)
            fit_obj.sigma_list.append(fit_obj.total_sigma)
            # print("Rust optimization step: Loss =", fit_obj.loss, "Sigma =", fit_obj.total_sigma, "Correlation =", fit_obj.correlation)
            return fit_obj.loss, fit_obj.total_sigma, mismatch
        except TypeError:
            pass

    # Run calculation with new parameters
    cross_sections(fit_obj)

    # Calculate absorption correlation
    fit_obj.correlation = np.corrcoef(
        np.real(fit_obj.abs_cross), fit_obj.abs_exp[:, 1]
    )[0, 1]
    fit_obj.fl_correlation = np.corrcoef(
        np.real(fit_obj.fl_cross), fit_obj.fl_exp[:, 1]
    )[0, 1]

    # Calculate Raman residual
    if fit_obj.profs_exp.ndim == 1:
        fit_obj.profs_exp = np.reshape(fit_obj.profs_exp, (-1, 1))

    fit_obj.sigma = np.zeros_like(fit_obj.delta)
    intermediate = (
        1e7 * (np.real(fit_obj.raman_cross[:, fit_obj.rp]) - fit_obj.profs_exp) ** 2
    )
    fit_obj.sigma += intermediate.sum(axis=1)
    fit_obj.total_sigma = np.sum(fit_obj.sigma)

    abs_mismatch = 1 - fit_obj.correlation
    fl_mismatch = 1 - fit_obj.fl_correlation
    fit_obj.loss = (
        loss_coefficients[0] * fit_obj.total_sigma
        + loss_coefficients[1] * abs_mismatch
        + loss_coefficients[2] * fl_mismatch
    )

    # Track history
    if fit_obj.loss_list == []:
        fit_obj.loss_list = [fit_obj.loss]
    else:
        fit_obj.loss_list.append(fit_obj.loss)
    if fit_obj.correlation_list == []:
        fit_obj.correlation_list = [fit_obj.correlation]
    else:
        fit_obj.correlation_list.append(fit_obj.correlation)
    if not hasattr(fit_obj, "fl_correlation_list") or fit_obj.fl_correlation_list == []:
        fit_obj.fl_correlation_list = [fit_obj.fl_correlation]
    else:
        fit_obj.fl_correlation_list.append(fit_obj.fl_correlation)
    if fit_obj.sigma_list == []:
        fit_obj.sigma_list = [fit_obj.total_sigma]
    else:
        fit_obj.sigma_list.append(fit_obj.total_sigma)
    # print("Optimization step: Loss =", fit_obj.loss, "Sigma =", fit_obj.total_sigma, "Correlation =", fit_obj.correlation)
    return fit_obj.loss, fit_obj.total_sigma, 100 * (abs_mismatch + fl_mismatch)


def param_init(fit_switch, obj=None):
    """
    Initializes lmfit Parameters based on a switching array.

    Args:
        fit_switch (list): Binary array indicating which parameters to vary.
        obj: Object containing initial values.

    Returns:
        lmfit.Parameters: Initialized parameters for optimization.
    """
    if obj is None:
        obj = load_input()
    params_lmfit = lmfit.Parameters()

    # Mode displacements
    for i in range(len(obj.delta)):
        if fit_switch[i] == 1:
            params_lmfit.add("delta" + str(i), value=obj.delta[i], min=0.0, max=1.0)
        else:
            params_lmfit.add("delta" + str(i), value=obj.delta[i], vary=False)

    # Global linewidths and energies
    if fit_switch[len(obj.delta)] == 1:
        params_lmfit.add("gamma", value=obj.gamma, min=10, max=1000)
    else:
        params_lmfit.add("gamma", value=obj.gamma, vary=False)

    if fit_switch[len(obj.delta) + 1] == 1:
        params_lmfit.add(
            "transition_length", value=obj.M, min=0.8 * obj.M, max=1.2 * obj.M
        )
    else:
        params_lmfit.add("transition_length", value=obj.M, vary=False)

    if fit_switch[len(obj.delta) + 2] == 1:
        params_lmfit.add("theta", value=obj.theta, min=0, max=10)
    else:
        params_lmfit.add("theta", value=obj.theta, vary=False)

    if fit_switch[len(obj.delta) + 3] == 1:
        params_lmfit.add("kappa", value=obj.k, min=0, max=1)
    else:
        params_lmfit.add("kappa", value=obj.k, vary=False)

    if fit_switch[len(obj.delta) + 5] == 1:
        params_lmfit.add("E0", value=obj.E0, min=0.95 * obj.E0, max=1.05 * obj.E0)
    else:
        params_lmfit.add("E0", value=obj.E0, vary=False)

    return params_lmfit


def orca_freq(inp):
    """
    Utility to parse vibrational frequencies from an ORCA output file and save to freqs.dat.

    Args:
        inp (str): Path to ORCA output/frequency file.

    Returns:
        list: Frequencies in cm^-1.
    """
    import re

    freq = []
    with open(inp, "r") as file:
        for i in file:
            # Match numbers like 1234.56
            num = float(re.findall(r"\d+\.\d+(?=\s|$)", i)[0])
            freq.append(num)
    print(freq)
    np.savetxt("freqs.dat", freq)
    file.close()
    return freq
