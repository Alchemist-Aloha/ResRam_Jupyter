# ResRAM_NG: Resonance Raman Excitation Profile Analysis 🧪

Welcome! This program helps scientists calculate and "fit" (match) theoretical models to experimental data from **Resonance Raman Spectroscopy**. 

It implements the Independent Mode Displaced Harmonic Oscillator (IMDHO) formalism and Brownian oscillator theory to benchmark DFT functionals against experimental Femtosecond Stimulated Raman Spectroscopy (FSRS) data.

---

## 1. Quick Start (GUI) 🚀

The easiest way to use ResRAM is through the interactive graphical interface.

### Running from source
If you have `uv` installed, simply run:
```bash
uv run resram-gui
```

---

## 2. GUI Features ✨

The ResRAM GUI is optimized for a smooth, responsive experience:
*   **Real-time Visualization:** See Absorption, Fluorescence, and Raman Excitation Profiles update as you change parameters.
*   **Background Calculations:** Heavy spectroscopic simulations run in a separate thread, keeping the interface snappy.
*   **Debounced Updates:** Table edits are debounced (300ms) to prevent lag while typing.
*   **Interactive Control:** 
    *   Toggle individual vibrational modes to include/exclude them from plots.
    *   Select which parameters (deltas, gamma, E0, etc.) should be varied during the automated fitting process.
*   **Integrated Fitting:** Run the `lmfit` optimizer directly from the GUI and see the "best fit" results in real-time.

---

## 3. Installation & Development 🛠️

If you want to use the core library in your own scripts or notebooks:

### Using `uv` (Recommended)
```bash
uv pip install -e .
```

### Using `pip`
```bash
pip install -e .
```

### Required Python Libraries
If not using `uv`, manual installation:
```bash
pip install numpy matplotlib scipy lmfit PyQt6 pyqtgraph
```

---

## 4. Optional: Rust Acceleration ⚡

This project includes an optional **Rust backend** (`src/resram_rust`) that provides up to **~4x performance speedup**.

### How to Compile the Rust Backend
1. **Install Rust:** [rustup.rs](https://rustup.rs/)
2. **Install Maturin**
   ```bash
   uv pip install maturin
   ```
3. **Build wheel:**
   ```bash
   uv run resram-rust-build
   ```
3. **Install wheel:**
   ```bash
   uv run resram-rust-install
   ```

Alternative one-step local development install:
```bash
uv run resram-rust-develop
   ```

If `rustc` is not installed, Rust compile commands are skipped and the Python implementation is used.

---

## 5. Input Data Format 📂

The program (GUI and Notebook) expects specific `.dat` files in the working directory:

*   **`inp.txt`**: Global settings (Temperature, Refractive Index, etc.).
*   **`freqs.dat`**: Vibrational frequencies (cm⁻¹).
*   **`deltas.dat`**: Initial dimensionless displacements.
*   **`abs_exp.dat`**: Experimental absorption spectrum.
*   **`fl_exp.dat`**: Experimental fluorescence spectrum.
*   **`profs_exp.dat`**: Experimental Raman excitation profiles.
*   **`rpumps.dat`**: Experimental laser excitation wavenumbers.

---

## 6. Python Scripts Usage (`.py`) 📓



---

## Acknowledgments 🙌
Developed by **Likun Cai**, based on theoretical frameworks and research from:
*   **Dr. Zachary Piontkowski, Dr. Juan S. Sandoval & Prof. David W. McCamant** (University of Rochester) 
*   **Mukamel et al.** - Brownian oscillator models for solvation.

Happy scientific computing!
