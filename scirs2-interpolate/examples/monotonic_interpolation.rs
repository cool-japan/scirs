use scirs2_core::ndarray::{array, Array1};
use scirs2_interpolate::interp1d::monotonic::{
    hyman_interpolate, modified_akima_interpolate, monotonic_interpolate, steffen_interpolate,
    MonotonicMethod,
};
use scirs2_interpolate::pchip_interpolate;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Monotonic Interpolation Methods Examples");
    println!("=======================================\n");

    // Example 1: Monotonic increasing data
    println!("Example 1: Monotonic increasing data");
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = array![0.0, 1.0, 2.0, 4.0, 8.0];

    // Create a finer grid for evaluation
    let x_fine = Array1::<f64>::linspace(0.0, 4.0, 41);

    // Compare different monotonic interpolation methods
    let y_pchip = pchip_interpolate(&x.view(), &y.view(), &x_fine.view(), false)?;
    let y_hyman = hyman_interpolate(&x.view(), &y.view(), &x_fine.view(), false)?;
    let y_steffen = steffen_interpolate(&x.view(), &y.view(), &x_fine.view(), false)?;
    let y_akima = modified_akima_interpolate(&x.view(), &y.view(), &x_fine.view(), false)?;

    println!("  Data points: x = {:?}", x);
    println!("               y = {:?}", y);
    println!("\n  Evaluated at x = 1.5:");
    println!("  PCHIP: {:.6}", y_pchip[15]);
    println!("  Hyman: {:.6}", y_hyman[15]);
    println!("  Steffen: {:.6}", y_steffen[15]);
    println!("  Modified Akima: {:.6}", y_akima[15]);

    // Calculate forward differences to check monotonicity
    check_monotonicity(
        "Example 1 (increasing)",
        &x_fine,
        &y_pchip,
        &y_hyman,
        &y_steffen,
        &y_akima,
    );

    // Example 2: Data with a flat section
    println!("\nExample 2: Data with a flat section");
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let x_fine = Array1::<f64>::linspace(0.0, 5.0, 51);

    let y_pchip = pchip_interpolate(&x.view(), &y.view(), &x_fine.view(), false)?;
    let y_hyman = hyman_interpolate(&x.view(), &y.view(), &x_fine.view(), false)?;
    let y_steffen = steffen_interpolate(&x.view(), &y.view(), &x_fine.view(), false)?;
    let y_akima = modified_akima_interpolate(&x.view(), &y.view(), &x_fine.view(), false)?;

    println!("  Data points: x = {:?}", x);
    println!("               y = {:?}", y);
    println!("\n  Evaluated at x = 2.5:");
    println!("  PCHIP: {:.6}", y_pchip[25]);
    println!("  Hyman: {:.6}", y_hyman[25]);
    println!("  Steffen: {:.6}", y_steffen[25]);
    println!("  Modified Akima: {:.6}", y_akima[25]);

    check_monotonicity(
        "Example 2 (step function)",
        &x_fine,
        &y_pchip,
        &y_hyman,
        &y_steffen,
        &y_akima,
    );

    // Example 3: Data with changing monotonicity
    println!("\nExample 3: Data with changing monotonicity");
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![0.0, 1.0, 0.5, 0.0, 0.5, 2.0];

    let x_fine = Array1::<f64>::linspace(0.0, 5.0, 51);

    let y_pchip = pchip_interpolate(&x.view(), &y.view(), &x_fine.view(), false)?;
    let y_hyman = hyman_interpolate(&x.view(), &y.view(), &x_fine.view(), false)?;
    let y_steffen = steffen_interpolate(&x.view(), &y.view(), &x_fine.view(), false)?;
    let y_akima = modified_akima_interpolate(&x.view(), &y.view(), &x_fine.view(), false)?;

    println!("  Data points: x = {:?}", x);
    println!("               y = {:?}", y);
    println!("\n  Evaluated at various points:");
    println!("  At x = 0.5:");
    println!("    PCHIP: {:.6}", y_pchip[5]);
    println!("    Hyman: {:.6}", y_hyman[5]);
    println!("    Steffen: {:.6}", y_steffen[5]);
    println!("    Modified Akima: {:.6}", y_akima[5]);

    println!("  At x = 1.5:");
    println!("    PCHIP: {:.6}", y_pchip[15]);
    println!("    Hyman: {:.6}", y_hyman[15]);
    println!("    Steffen: {:.6}", y_steffen[15]);
    println!("    Modified Akima: {:.6}", y_akima[15]);

    // Check monotonicity in segments
    // In this example, we can check each segment separately
    check_segment_monotonicity(
        "Segment 0-1 (increasing)",
        &x_fine,
        0,
        10,
        &y_pchip,
        &y_hyman,
        &y_steffen,
        &y_akima,
    );
    check_segment_monotonicity(
        "Segment 1-2 (decreasing)",
        &x_fine,
        10,
        20,
        &y_pchip,
        &y_hyman,
        &y_steffen,
        &y_akima,
    );
    check_segment_monotonicity(
        "Segment 2-3 (decreasing)",
        &x_fine,
        20,
        30,
        &y_pchip,
        &y_hyman,
        &y_steffen,
        &y_akima,
    );
    check_segment_monotonicity(
        "Segment 3-4 (increasing)",
        &x_fine,
        30,
        40,
        &y_pchip,
        &y_hyman,
        &y_steffen,
        &y_akima,
    );
    check_segment_monotonicity(
        "Segment 4-5 (increasing)",
        &x_fine,
        40,
        50,
        &y_pchip,
        &y_hyman,
        &y_steffen,
        &y_akima,
    );

    // Example 4: Sharply varying data
    println!("\nExample 4: Sharply varying data");
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = array![0.0, 2.0, 1.0, 4.0, 3.0, 6.0, 5.0];

    let x_fine = Array1::<f64>::linspace(0.0, 6.0, 61);

    // Use the generic interface with different methods
    let y_pchip = monotonic_interpolate(
        &x.view(),
        &y.view(),
        &x_fine.view(),
        MonotonicMethod::Pchip,
        false,
    )?;
    let y_hyman = monotonic_interpolate(
        &x.view(),
        &y.view(),
        &x_fine.view(),
        MonotonicMethod::Hyman,
        false,
    )?;
    let y_steffen = monotonic_interpolate(
        &x.view(),
        &y.view(),
        &x_fine.view(),
        MonotonicMethod::Steffen,
        false,
    )?;
    let y_akima = monotonic_interpolate(
        &x.view(),
        &y.view(),
        &x_fine.view(),
        MonotonicMethod::ModifiedAkima,
        false,
    )?;

    println!("  Data points: x = {:?}", x);
    println!("               y = {:?}", y);
    println!("\n  Evaluated at various points:");
    println!("  At x = 0.5:");
    println!("    PCHIP: {:.6}", y_pchip[5]);
    println!("    Hyman: {:.6}", y_hyman[5]);
    println!("    Steffen: {:.6}", y_steffen[5]);
    println!("    Modified Akima: {:.6}", y_akima[5]);

    println!("  At x = 2.5:");
    println!("    PCHIP: {:.6}", y_pchip[25]);
    println!("    Hyman: {:.6}", y_hyman[25]);
    println!("    Steffen: {:.6}", y_steffen[25]);
    println!("    Modified Akima: {:.6}", y_akima[25]);

    // Check for overshoot
    check_extrema(
        "Example 4 (zig-zag)",
        &y,
        &y_pchip,
        &y_hyman,
        &y_steffen,
        &y_akima,
    );

    Ok(())
}

/// Check if an interpolation is monotonic
#[allow(dead_code)]
fn check_monotonicity(
    label: &str,
    x: &Array1<f64>,
    y_pchip: &Array1<f64>,
    y_hyman: &Array1<f64>,
    y_steffen: &Array1<f64>,
    y_akima: &Array1<f64>,
) {
    println!("\nChecking monotonicity for {}:", label);

    // Calculate forward differences to detect sign changes
    let mut pchip_monotonic = true;
    let mut hyman_monotonic = true;
    let mut steffen_monotonic = true;
    let mut akima_monotonic = true;

    for i in 1..x.len() {
        let dx = x[i] - x[i - 1];

        let d_pchip = (y_pchip[i] - y_pchip[i - 1]) / dx;
        let d_hyman = (y_hyman[i] - y_hyman[i - 1]) / dx;
        let d_steffen = (y_steffen[i] - y_steffen[i - 1]) / dx;
        let d_akima = (y_akima[i] - y_akima[i - 1]) / dx;

        // Check if previous segment was increasing or decreasing
        if i > 1 {
            let prev_d_pchip = (y_pchip[i - 1] - y_pchip[i - 2]) / (x[i - 1] - x[i - 2]);
            let prev_d_hyman = (y_hyman[i - 1] - y_hyman[i - 2]) / (x[i - 1] - x[i - 2]);
            let prev_d_steffen = (y_steffen[i - 1] - y_steffen[i - 2]) / (x[i - 1] - x[i - 2]);
            let prev_d_akima = (y_akima[i - 1] - y_akima[i - 2]) / (x[i - 1] - x[i - 2]);

            // Sign changes indicate non-monotonicity
            if (prev_d_pchip > 0.0 && d_pchip < 0.0) || (prev_d_pchip < 0.0 && d_pchip > 0.0) {
                pchip_monotonic = false;
            }
            if (prev_d_hyman > 0.0 && d_hyman < 0.0) || (prev_d_hyman < 0.0 && d_hyman > 0.0) {
                hyman_monotonic = false;
            }
            if (prev_d_steffen > 0.0 && d_steffen < 0.0)
                || (prev_d_steffen < 0.0 && d_steffen > 0.0)
            {
                steffen_monotonic = false;
            }
            if (prev_d_akima > 0.0 && d_akima < 0.0) || (prev_d_akima < 0.0 && d_akima > 0.0) {
                akima_monotonic = false;
            }
        }
    }

    // Print results
    println!(
        "  PCHIP: {}",
        if pchip_monotonic {
            "Monotonic"
        } else {
            "Not monotonic"
        }
    );
    println!(
        "  Hyman: {}",
        if hyman_monotonic {
            "Monotonic"
        } else {
            "Not monotonic"
        }
    );
    println!(
        "  Steffen: {}",
        if steffen_monotonic {
            "Monotonic"
        } else {
            "Not monotonic"
        }
    );
    println!(
        "  Modified Akima: {}",
        if akima_monotonic {
            "Monotonic"
        } else {
            "Not monotonic"
        }
    );
}

/// Check if a specific segment is monotonic
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn check_segment_monotonicity(
    label: &str,
    x: &Array1<f64>,
    start_idx: usize,
    end_idx: usize,
    y_pchip: &Array1<f64>,
    y_hyman: &Array1<f64>,
    y_steffen: &Array1<f64>,
    y_akima: &Array1<f64>,
) {
    println!(
        "\nChecking monotonicity for {} (x = {:.1} to {:.1}):",
        label, x[start_idx], x[end_idx]
    );

    // Detect if the segment is increasing or decreasing
    let pchip_increasing = y_pchip[end_idx] >= y_pchip[start_idx];
    let hyman_increasing = y_hyman[end_idx] >= y_hyman[start_idx];
    let steffen_increasing = y_steffen[end_idx] >= y_steffen[start_idx];
    let akima_increasing = y_akima[end_idx] >= y_akima[start_idx];

    // Check if the entire segment is monotonic
    let mut pchip_monotonic = true;
    let mut hyman_monotonic = true;
    let mut steffen_monotonic = true;
    let mut akima_monotonic = true;

    for i in start_idx + 1..=end_idx {
        if (pchip_increasing && y_pchip[i] < y_pchip[i - 1])
            || (!pchip_increasing && y_pchip[i] > y_pchip[i - 1])
        {
            pchip_monotonic = false;
        }

        if (hyman_increasing && y_hyman[i] < y_hyman[i - 1])
            || (!hyman_increasing && y_hyman[i] > y_hyman[i - 1])
        {
            hyman_monotonic = false;
        }

        if (steffen_increasing && y_steffen[i] < y_steffen[i - 1])
            || (!steffen_increasing && y_steffen[i] > y_steffen[i - 1])
        {
            steffen_monotonic = false;
        }

        if (akima_increasing && y_akima[i] < y_akima[i - 1])
            || (!akima_increasing && y_akima[i] > y_akima[i - 1])
        {
            akima_monotonic = false;
        }
    }

    // Print results
    println!(
        "  PCHIP: {}",
        if pchip_monotonic {
            if pchip_increasing {
                "Monotonically increasing"
            } else {
                "Monotonically decreasing"
            }
        } else {
            "Not monotonic"
        }
    );

    println!(
        "  Hyman: {}",
        if hyman_monotonic {
            if hyman_increasing {
                "Monotonically increasing"
            } else {
                "Monotonically decreasing"
            }
        } else {
            "Not monotonic"
        }
    );

    println!(
        "  Steffen: {}",
        if steffen_monotonic {
            if steffen_increasing {
                "Monotonically increasing"
            } else {
                "Monotonically decreasing"
            }
        } else {
            "Not monotonic"
        }
    );

    println!(
        "  Modified Akima: {}",
        if akima_monotonic {
            if akima_increasing {
                "Monotonically increasing"
            } else {
                "Monotonically decreasing"
            }
        } else {
            "Not monotonic"
        }
    );
}

/// Check if interpolation avoids overshooting by comparing with data extrema
#[allow(dead_code)]
fn check_extrema(
    label: &str,
    y_data: &Array1<f64>,
    y_pchip: &Array1<f64>,
    y_hyman: &Array1<f64>,
    y_steffen: &Array1<f64>,
    y_akima: &Array1<f64>,
) {
    // Find min and max of original _data
    let data_min = y_data.iter().fold(f64::INFINITY, |a, &b| f64::min(a, b));
    let data_max = y_data
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b));

    // Find min and max of interpolated _data
    let pchip_min = y_pchip.iter().fold(f64::INFINITY, |a, &b| f64::min(a, b));
    let pchip_max = y_pchip
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b));

    let hyman_min = y_hyman.iter().fold(f64::INFINITY, |a, &b| f64::min(a, b));
    let hyman_max = y_hyman
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b));

    let steffen_min = y_steffen.iter().fold(f64::INFINITY, |a, &b| f64::min(a, b));
    let steffen_max = y_steffen
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b));

    let akima_min = y_akima.iter().fold(f64::INFINITY, |a, &b| f64::min(a, b));
    let akima_max = y_akima
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b));

    println!("\nChecking for overshooting in {}:", label);
    println!("  Data range: min = {:.6}, max = {:.6}", data_min, data_max);

    println!("  PCHIP: min = {:.6}, max = {:.6}", pchip_min, pchip_max);
    if pchip_min < data_min || pchip_max > data_max {
        println!("    Overshooting detected!");
    } else {
        println!("    No overshooting");
    }

    println!("  Hyman: min = {:.6}, max = {:.6}", hyman_min, hyman_max);
    if hyman_min < data_min || hyman_max > data_max {
        println!("    Overshooting detected!");
    } else {
        println!("    No overshooting");
    }

    println!(
        "  Steffen: min = {:.6}, max = {:.6}",
        steffen_min, steffen_max
    );
    if steffen_min < data_min || steffen_max > data_max {
        println!("    Overshooting detected!");
    } else {
        println!("    No overshooting");
    }

    println!(
        "  Modified Akima: min = {:.6}, max = {:.6}",
        akima_min, akima_max
    );
    if akima_min < data_min || akima_max > data_max {
        println!("    Overshooting detected!");
    } else {
        println!("    No overshooting");
    }
}
