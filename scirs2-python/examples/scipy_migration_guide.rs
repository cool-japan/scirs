//! SciPy → scirs2 Migration Guide
//!
//! Run with:
//! ```bash
//! cargo run --example scipy_migration_guide -p scirs2-python
//! ```
//!
//! Prints a formatted table mapping common SciPy functions to their scirs2
//! equivalents.  The table is grouped by sub-module and includes notes on
//! interface differences.

fn main() {
    print_migration_guide();
}

/// One row in the migration table.
struct MigrationEntry {
    scipy: &'static str,
    scirs2: &'static str,
    notes: &'static str,
}

impl MigrationEntry {
    const fn new(scipy: &'static str, scirs2: &'static str, notes: &'static str) -> Self {
        Self {
            scipy,
            scirs2,
            notes,
        }
    }
}

/// Print a formatted migration guide to stdout.
fn print_migration_guide() {
    // ── column widths ─────────────────────────────────────────────────────────
    let w_scipy = 40_usize;
    let w_scirs2 = 40_usize;
    let w_notes = 45_usize;

    let border = format!(
        "+{:-<w_scipy$}+{:-<w_scirs2$}+{:-<w_notes$}+",
        "",
        "",
        "",
        w_scipy = w_scipy + 2,
        w_scirs2 = w_scirs2 + 2,
        w_notes = w_notes + 2,
    );

    let header = |section: &str| {
        println!("{border}");
        println!(
            "| {:<w_scipy$} | {:<w_scirs2$} | {:<w_notes$} |",
            format!("=== {} ===", section),
            "",
            "",
            w_scipy = w_scipy,
            w_scirs2 = w_scirs2,
            w_notes = w_notes,
        );
        println!("{border}");
        println!(
            "| {:<w_scipy$} | {:<w_scirs2$} | {:<w_notes$} |",
            "SciPy",
            "scirs2",
            "Notes",
            w_scipy = w_scipy,
            w_scirs2 = w_scirs2,
            w_notes = w_notes,
        );
        println!("{border}");
    };

    let row = |e: &MigrationEntry| {
        println!(
            "| {:<w_scipy$} | {:<w_scirs2$} | {:<w_notes$} |",
            e.scipy,
            e.scirs2,
            e.notes,
            w_scipy = w_scipy,
            w_scirs2 = w_scirs2,
            w_notes = w_notes,
        );
    };

    // ── Linear Algebra ────────────────────────────────────────────────────────
    let linalg: &[MigrationEntry] = &[
        MigrationEntry::new("scipy.linalg.det", "scirs2.linalg.det_py", "Same interface"),
        MigrationEntry::new("scipy.linalg.inv", "scirs2.linalg.inv_py", "Same interface"),
        MigrationEntry::new(
            "scipy.linalg.solve",
            "scirs2.linalg.solve_py",
            "Same interface",
        ),
        MigrationEntry::new(
            "scipy.linalg.lstsq",
            "scirs2.linalg.lstsq_py",
            "Same interface",
        ),
        MigrationEntry::new(
            "scipy.linalg.lu",
            "scirs2.linalg.lu_py",
            "Returns (P, L, U)",
        ),
        MigrationEntry::new("scipy.linalg.qr", "scirs2.linalg.qr_py", "Same interface"),
        MigrationEntry::new("scipy.linalg.svd", "scirs2.linalg.svd_py", "Same interface"),
        MigrationEntry::new(
            "scipy.linalg.cholesky",
            "scirs2.linalg.cholesky_py",
            "Lower-triangular",
        ),
        MigrationEntry::new("scipy.linalg.eig", "scirs2.linalg.eig_py", "Same interface"),
        MigrationEntry::new(
            "scipy.linalg.eigh",
            "scirs2.linalg.eigh_py",
            "Symmetric only",
        ),
        MigrationEntry::new(
            "scipy.linalg.expm",
            "scirs2.linalg.expm_py",
            "Matrix exponential",
        ),
        MigrationEntry::new(
            "scipy.linalg.logm",
            "scirs2.linalg.logm_py",
            "Matrix logarithm",
        ),
        MigrationEntry::new(
            "scipy.linalg.norm",
            "scirs2.linalg.matrix_norm_py",
            "ord='fro'|'1'|'2'|'inf'",
        ),
        MigrationEntry::new(
            "scipy.linalg.kron",
            "scirs2.linalg.kron_py",
            "Same interface",
        ),
        MigrationEntry::new(
            "scipy.linalg.schur",
            "scirs2.linalg.schur_py",
            "Real Schur form",
        ),
    ];
    header("Linear Algebra  scipy.linalg → scirs2.linalg");
    for e in linalg {
        row(e);
    }
    println!("{border}");
    println!();

    // ── FFT ───────────────────────────────────────────────────────────────────
    let fft: &[MigrationEntry] = &[
        MigrationEntry::new("scipy.fft.fft", "scirs2.fft.fft_py", "Same interface"),
        MigrationEntry::new("scipy.fft.ifft", "scirs2.fft.ifft_py", "Same interface"),
        MigrationEntry::new("scipy.fft.rfft", "scirs2.fft.rfft_py", "Same interface"),
        MigrationEntry::new("scipy.fft.irfft", "scirs2.fft.irfft_py", "Same interface"),
        MigrationEntry::new("scipy.fft.fft2", "scirs2.fft.fft2_py", "2-D DFT"),
        MigrationEntry::new("scipy.fft.dct", "scirs2.fft.dct_py", "Types 1–4 supported"),
        MigrationEntry::new(
            "scipy.fft.fftfreq",
            "scirs2.fft.fftfreq_py",
            "Same interface",
        ),
        MigrationEntry::new(
            "scipy.fft.rfftfreq",
            "scirs2.fft.rfftfreq_py",
            "Same interface",
        ),
        MigrationEntry::new(
            "scipy.fft.fftshift",
            "scirs2.fft.fftshift_py",
            "Same interface",
        ),
        MigrationEntry::new(
            "scipy.fft.next_fast_len",
            "scirs2.fft.next_fast_len_py",
            "Same interface",
        ),
    ];
    header("FFT  scipy.fft → scirs2.fft");
    for e in fft {
        row(e);
    }
    println!("{border}");
    println!();

    // ── Statistics ────────────────────────────────────────────────────────────
    let stats: &[MigrationEntry] = &[
        MigrationEntry::new(
            "scipy.stats.describe",
            "scirs2.stats.describe_py",
            "Returns dict",
        ),
        MigrationEntry::new(
            "scipy.stats.ttest_1samp",
            "scirs2.stats.ttest_1samp_py",
            "Returns (stat, p)",
        ),
        MigrationEntry::new(
            "scipy.stats.ttest_ind",
            "scirs2.stats.ttest_ind_py",
            "Returns (stat, p)",
        ),
        MigrationEntry::new(
            "scipy.stats.norm.pdf",
            "scirs2.stats.NormDist.pdf",
            "Instance method",
        ),
        MigrationEntry::new(
            "scipy.stats.norm.cdf",
            "scirs2.stats.NormDist.cdf",
            "Instance method",
        ),
        MigrationEntry::new(
            "scipy.stats.norm.ppf",
            "scirs2.stats.NormDist.ppf",
            "Percent-point fn",
        ),
        MigrationEntry::new(
            "scipy.stats.norm.rvs",
            "scirs2.stats.NormDist.rvs",
            "Random variates",
        ),
        MigrationEntry::new("scipy.stats.skew", "scirs2.stats.skew_py", "Same interface"),
        MigrationEntry::new(
            "scipy.stats.kurtosis",
            "scirs2.stats.kurtosis_py",
            "Fisher excess kurtosis",
        ),
        MigrationEntry::new("scipy.stats.iqr", "scirs2.stats.iqr_py", "Same interface"),
    ];
    header("Statistics  scipy.stats → scirs2.stats");
    for e in stats {
        row(e);
    }
    println!("{border}");
    println!();

    // ── Optimize ──────────────────────────────────────────────────────────────
    let optimize: &[MigrationEntry] = &[
        MigrationEntry::new(
            "scipy.optimize.minimize",
            "scirs2.optimize.minimize_py",
            "method: 'L-BFGS-B' etc.",
        ),
        MigrationEntry::new(
            "scipy.optimize.brentq",
            "scirs2.optimize.brentq_py",
            "Same interface",
        ),
        MigrationEntry::new(
            "scipy.optimize.fsolve",
            "scirs2.optimize.fsolve_py",
            "Same interface",
        ),
        MigrationEntry::new(
            "scipy.optimize.curve_fit",
            "scirs2.optimize.curve_fit_py",
            "Returns (popt, pcov)",
        ),
        MigrationEntry::new(
            "scipy.optimize.linprog",
            "scirs2.optimize.linprog_py",
            "Simplex/revised simplex",
        ),
    ];
    header("Optimize  scipy.optimize → scirs2.optimize");
    for e in optimize {
        row(e);
    }
    println!("{border}");
    println!();

    // ── Integrate ─────────────────────────────────────────────────────────────
    let integrate: &[MigrationEntry] = &[
        MigrationEntry::new(
            "scipy.integrate.quad",
            "scirs2.integrate.quad_py",
            "Adaptive quadrature",
        ),
        MigrationEntry::new(
            "scipy.integrate.trapezoid",
            "scirs2.integrate.trapezoid_array_py",
            "Same interface",
        ),
        MigrationEntry::new(
            "scipy.integrate.simpson",
            "scirs2.integrate.simpson_array_py",
            "Odd # of points",
        ),
        MigrationEntry::new(
            "scipy.integrate.cumtrapz",
            "scirs2.integrate.cumulative_trapezoid_py",
            "Same interface",
        ),
        MigrationEntry::new(
            "scipy.integrate.romberg",
            "scirs2.integrate.romberg_array_py",
            "Uniform grid only",
        ),
        MigrationEntry::new(
            "scipy.integrate.solve_ivp",
            "scirs2.integrate.solve_ivp_py",
            "RK45/DOP853/etc.",
        ),
    ];
    header("Integrate  scipy.integrate → scirs2.integrate");
    for e in integrate {
        row(e);
    }
    println!("{border}");
    println!();

    // ── Signal ────────────────────────────────────────────────────────────────
    let signal: &[MigrationEntry] = &[
        MigrationEntry::new(
            "scipy.signal.butter",
            "scirs2.signal.butter_py",
            "Returns (b, a)",
        ),
        MigrationEntry::new(
            "scipy.signal.sosfilt",
            "scirs2.signal.sosfilt_py",
            "Same interface",
        ),
        MigrationEntry::new(
            "scipy.signal.filtfilt",
            "scirs2.signal.filtfilt_py",
            "Zero-phase filter",
        ),
        MigrationEntry::new(
            "scipy.signal.convolve",
            "scirs2.signal.convolve_py",
            "1-D only",
        ),
        MigrationEntry::new(
            "scipy.signal.correlate",
            "scirs2.signal.correlate_py",
            "Same interface",
        ),
        MigrationEntry::new(
            "scipy.signal.periodogram",
            "scirs2.signal.periodogram_py",
            "Returns (f, Pxx)",
        ),
        MigrationEntry::new(
            "scipy.signal.welch",
            "scirs2.signal.welch_py",
            "Returns (f, Pxx)",
        ),
        MigrationEntry::new(
            "scipy.signal.windows.hann",
            "scirs2.signal.hann_window_py",
            "Same result",
        ),
    ];
    header("Signal  scipy.signal → scirs2.signal");
    for e in signal {
        row(e);
    }
    println!("{border}");
    println!();

    // ── Interpolate ───────────────────────────────────────────────────────────
    let interp: &[MigrationEntry] = &[
        MigrationEntry::new(
            "scipy.interpolate.interp1d",
            "scirs2.interpolate.interp1d_py",
            "kind: linear/cubic/…",
        ),
        MigrationEntry::new(
            "scipy.interpolate.CubicSpline",
            "scirs2.interpolate.cubic_spline_py",
            "Same interface",
        ),
        MigrationEntry::new(
            "scipy.interpolate.RBFInterpolator",
            "scirs2.interpolate.rbf_py",
            "Radial basis fns",
        ),
        MigrationEntry::new(
            "scipy.interpolate.griddata",
            "scirs2.interpolate.griddata_py",
            "Scattered data",
        ),
    ];
    header("Interpolate  scipy.interpolate → scirs2.interpolate");
    for e in interp {
        row(e);
    }
    println!("{border}");
    println!();

    println!(
        "Total entries: {}",
        linalg.len()
            + fft.len()
            + stats.len()
            + optimize.len()
            + integrate.len()
            + signal.len()
            + interp.len()
    );
    println!();
    println!("For detailed per-function documentation, run:");
    println!("  python -c \"import scirs2; help(scirs2.<module>.<function>)\"");
    println!("  cargo doc -p scirs2-python --open");
}
