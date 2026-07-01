use ndarray::prelude::*;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Implements a "valid" convolution similar to np.convolve(a, v, 'valid')
fn convolve_valid(input: &Array1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
    let (a, v) = if input.len() >= kernel.len() {
        (input, kernel)
    } else {
        (kernel, input)
    };
    let n = a.len();
    let m = v.len();
    let out_len = n - m + 1;
    let mut output = Array1::zeros(out_len);
    for i in 0..out_len {
        let mut sum = 0.0;
        for j in 0..m {
            // v[m-1-j] effectively reverses the kernel, matching np.convolve
            sum += a[i + j] * v[m - 1 - j];
        }
        output[i] = sum;
    }
    output
}

fn calculate_cross_sections(
    wg: &ArrayView1<f64>,
    s_factors: &ArrayView1<f64>,
    eta: &ArrayView1<f64>,
    delta: &ArrayView1<f64>,
    th: &ArrayView1<f64>,
    el: &ArrayView1<f64>,
    conv_el: &ArrayView1<f64>,
    e0_range: &ArrayView1<f64>,
    d: f64,
    l: f64,
    beta: f64,
    e0: f64,
    m: f64,
    pre_a: f64,
    pre_f: f64,
    pre_r: f64,
    theta: f64,
    _dt_unused: f64, // Spacing is calculated from 'th' to ensure consistency with np.trapz
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let th_len = th.len();
    let el_full_len = el.len();
    let wg_len = wg.len();

    // Spacing for trapezoidal integration
    // Python's np.trapezoid defaults to dx=1.0 when x is not provided
    let d_th = 1.0;

    // 1. Pre-calculate time-dependent Kernels (Trapezoidal weighted)
    // kernel = A(t) * exp(-g(t)) * weight
    let mut kernels = Vec::with_capacity(th_len);
    let mut kernels_f = Vec::with_capacity(th_len);

    for j in 0..th_len {
        let t = th[j];

        // g(t)
        let g_real = (d / l).powi(2) * (l * t - 1.0 + (-l * t).exp());
        let g_imag = (beta * d.powi(2)) / (2.0 * l) * (1.0 - (-l * t).exp());
        let g_t = Complex64::new(g_real, g_imag);

        // A(t)
        let mut sum_k = Complex64::new(0.0, 0.0);
        for i in 0..wg_len {
            let k_k =
                (1.0 + eta[i]) * s_factors[i] * (1.0 - Complex64::from_polar(1.0, -wg[i] * t))
                    + eta[i] * s_factors[i] * (1.0 - Complex64::from_polar(1.0, wg[i] * t));
            sum_k += k_k;
        }
        let a_t = Complex64::new(m.powi(2), 0.0) * (-sum_k).exp();

        let weight = if j == 0 || j == th_len - 1 { 0.5 } else { 1.0 };
        kernels.push(a_t * (-g_t).exp() * weight);
        kernels_f.push(a_t.conj() * (-g_t).conj().exp() * weight);
    }

    // 2. Absorption and Fluorescence Integrals (over full EL grid)
    let (integ_a_vec, integ_f_vec): (Vec<f64>, Vec<f64>) = (0..el_full_len)
        .into_par_iter()
        .map(|i| {
            let energy = el[i];
            let mut sum_a = Complex64::new(0.0, 0.0);
            let mut sum_f = Complex64::new(0.0, 0.0);

            for j in 0..th_len {
                let phase = (energy - e0) * th[j];
                let exp_phase = Complex64::from_polar(1.0, phase);
                sum_a += exp_phase * kernels[j];
                sum_f += exp_phase * kernels_f[j];
            }

            ((sum_a * d_th).re, (sum_f * d_th).re)
        })
        .unzip();

    // 3. Inhomogeneous Broadening Kernel H
    let h: Array1<f64> = if theta < 1e-10 {
        let mut h_delta = Array1::from_elem(e0_range.len(), 0.0);
        h_delta[e0_range.len() / 2] = 1.0;
        h_delta
    } else {
        e0_range.mapv(|val| {
            (1.0 / (theta * (2.0 * PI).sqrt())) * (-val.powi(2) / (2.0 * theta.powi(2))).exp()
        })
    };
    let h_sum: f64 = h.sum();

    // 4. Convolution and Final Scaling for Abs/Fl
    let abs_conv = convolve_valid(&Array1::from_vec(integ_a_vec), &h) / h_sum;
    let fl_conv = convolve_valid(&Array1::from_vec(integ_f_vec), &h) / h_sum;

    let mut abs_cross = Array1::<f64>::zeros(conv_el.len());
    let mut fl_cross = Array1::<f64>::zeros(conv_el.len());
    for i in 0..conv_el.len() {
        abs_cross[i] = abs_conv[i] * pre_a * conv_el[i];
        fl_cross[i] = fl_conv[i] * pre_f * conv_el[i];
    }

    // 5. Raman Cross Sections (Parallelized over modes)
    let raman_cross_vec: Vec<Array1<f64>> = (0..wg_len)
        .into_par_iter()
        .map(|idx| {
            let mut integ_r_mag_sq = Array1::<f64>::zeros(el_full_len);
            let factor = ((1.0 + eta[idx]).sqrt() * delta[idx]) / 2.0f64.sqrt();

            for i in 0..el_full_len {
                let energy = el[i];
                let mut sum_r = Complex64::new(0.0, 0.0);
                for j in 0..th_len {
                    let phase = (energy - e0) * th[j];
                    let exp_phase = Complex64::from_polar(1.0, phase);
                    let q_r = factor * (1.0 - Complex64::from_polar(1.0, -wg[idx] * th[j]));
                    sum_r += exp_phase * kernels[j] * q_r;
                }
                // Magnitude squared BEFORE convolution (matching Python)
                integ_r_mag_sq[i] = (sum_r * d_th).norm_sqr();
            }

            let conv_r = convolve_valid(&integ_r_mag_sq, &h) / h_sum;
            let mut final_r = Array1::<f64>::zeros(conv_el.len());
            for k in 0..conv_el.len() {
                // obj.preR * convEL * (convEL - wg)**3 * conv_result
                final_r[k] = pre_r * conv_el[k] * (conv_el[k] - wg[idx]).powi(3) * conv_r[k];
            }
            final_r
        })
        .collect();

    let mut raman_cross = Array2::<f64>::zeros((wg_len, conv_el.len()));
    for i in 0..wg_len {
        raman_cross.row_mut(i).assign(&raman_cross_vec[i]);
    }

    (abs_cross, fl_cross, raman_cross)
}

fn corrcoef(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.sum() / n;
    let mean_y = y.sum() / n;
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    num / (den_x.sqrt() * den_y.sqrt())
}

#[pyfunction]
fn raman_residual_rust(
    wg: PyReadonlyArray1<f64>,
    eta: PyReadonlyArray1<f64>,
    delta: PyReadonlyArray1<f64>,
    th: PyReadonlyArray1<f64>,
    el: PyReadonlyArray1<f64>,
    conv_el: PyReadonlyArray1<f64>,
    e0_range: PyReadonlyArray1<f64>,
    abs_exp: PyReadonlyArray1<f64>,
    fl_exp: PyReadonlyArray1<f64>,
    profs_exp: PyReadonlyArray2<f64>,
    rp: PyReadonlyArray1<usize>,
    loss_coefficients: PyReadonlyArray1<f64>,
    gamma: f64,
    m: f64,
    kappa: f64,
    theta: f64,
    e0: f64,
    beta: f64,
    pre_a: f64,
    pre_f: f64,
    pre_r: f64,
    dt: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let wg = wg.as_array();
    let eta = eta.as_array();
    let delta = delta.as_array();
    let th = th.as_array();
    let el = el.as_array();
    let conv_el = conv_el.as_array();
    let e0_range = e0_range.as_array();
    let abs_exp = abs_exp.as_array();
    let fl_exp = fl_exp.as_array();
    let profs_exp = profs_exp.as_array();
    let rp = rp.as_array();
    let loss_coefficients = loss_coefficients.as_array();

    if loss_coefficients.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "loss_coefficients must contain three values: [total_sigma, 1 - abs_corr, 1 - fl_corr]",
        ));
    }

    let s_factors = delta.mapv(|d| d.powi(2) / 2.0);
    let d_param = gamma * (1.0 + 0.85 * kappa + 0.88 * kappa.powi(2)) / (2.355 + 1.76 * kappa);
    let l_param = kappa * d_param;

    let (abs_cross, fl_cross, raman_cross) = calculate_cross_sections(
        &wg,
        &s_factors.view(),
        &eta,
        &delta,
        &th,
        &el,
        &conv_el,
        &e0_range,
        d_param,
        l_param,
        beta,
        e0,
        m,
        pre_a,
        pre_f,
        pre_r,
        theta,
        dt,
    );

    let abs_correlation = corrcoef(&abs_cross.view(), &abs_exp);
    let fl_correlation = corrcoef(&fl_cross.view(), &fl_exp);

    let mut total_sigma = 0.0;
    for (idx, &mode_rp) in rp.iter().enumerate() {
        for mode_idx in 0..wg.len() {
            let diff = raman_cross[[mode_idx, mode_rp]] - profs_exp[[mode_idx, idx]];
            total_sigma += 1e7 * diff.powi(2);
        }
    }

    let abs_mismatch = 1.0 - abs_correlation;
    let fl_mismatch = 1.0 - fl_correlation;
    let loss = loss_coefficients[0] * total_sigma
        + loss_coefficients[1] * abs_mismatch
        + loss_coefficients[2] * fl_mismatch;
    let spectral_mismatch = 100.0 * (abs_mismatch + fl_mismatch);

    Ok((
        loss,
        total_sigma,
        spectral_mismatch,
        abs_correlation,
        fl_correlation,
    ))
}

#[pyfunction]
fn cross_sections_rust(
    py: Python<'_>,
    wg: PyReadonlyArray1<f64>,
    s_factors: PyReadonlyArray1<f64>,
    eta: PyReadonlyArray1<f64>,
    delta: PyReadonlyArray1<f64>,
    th: PyReadonlyArray1<f64>,
    el: PyReadonlyArray1<f64>,
    conv_el: PyReadonlyArray1<f64>,
    e0_range: PyReadonlyArray1<f64>,
    d: f64,
    l: f64,
    beta: f64,
    e0: f64,
    m: f64,
    pre_a: f64,
    pre_f: f64,
    pre_r: f64,
    theta: f64,
    dt: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    let (abs_cross, fl_cross, raman_cross) = calculate_cross_sections(
        &wg.as_array(),
        &s_factors.as_array(),
        &eta.as_array(),
        &delta.as_array(),
        &th.as_array(),
        &el.as_array(),
        &conv_el.as_array(),
        &e0_range.as_array(),
        d,
        l,
        beta,
        e0,
        m,
        pre_a,
        pre_f,
        pre_r,
        theta,
        dt,
    );

    Ok((
        abs_cross.into_pyarray(py).unbind(),
        fl_cross.into_pyarray(py).unbind(),
        raman_cross.into_pyarray(py).unbind(),
    ))
}

#[pymodule]
fn resram_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cross_sections_rust, m)?)?;
    m.add_function(wrap_pyfunction!(raman_residual_rust, m)?)?;
    Ok(())
}
