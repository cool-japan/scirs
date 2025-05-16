use scirs2_special::{pbdv, pbdv_seq, pbvv, pbvv_seq, pbwa};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Parabolic Cylinder Functions Example");
    println!("===================================\n");

    // Evaluate parabolic cylinder D function at several points
    let test_points = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
    let test_orders = [0, 1, 2, 3, -1, -2];

    println!("Parabolic Cylinder Function D_v(x):\n");
    println!(
        "{:^8} | {:^8} | {:^15} | {:^15}",
        "v", "x", "D_v(x)", "D_v'(x)"
    );
    println!("{:-^8} | {:-^8} | {:-^15} | {:-^15}", "", "", "", "");

    for &v in &test_orders {
        for &x in &test_points {
            let (d, dp) = pbdv(v as f64, x)?;

            println!("{:^8} | {:^8.2} | {:^15.6e} | {:^15.6e}", v, x, d, dp);
        }
        println!("{:-^8} | {:-^8} | {:-^15} | {:-^15}", "", "", "", "");
    }

    // Evaluate parabolic cylinder V function
    println!("\nParabolic Cylinder Function V_v(x):\n");
    println!(
        "{:^8} | {:^8} | {:^15} | {:^15}",
        "v", "x", "V_v(x)", "V_v'(x)"
    );
    println!("{:-^8} | {:-^8} | {:-^15} | {:-^15}", "", "", "", "");

    for &v in &test_orders[0..4] {
        // Using positive orders
        for &x in &test_points {
            let (v_val, vp_val) = pbvv(v as f64, x)?;

            println!(
                "{:^8} | {:^8.2} | {:^15.6e} | {:^15.6e}",
                v, x, v_val, vp_val
            );
        }
        println!("{:-^8} | {:-^8} | {:-^15} | {:-^15}", "", "", "", "");
    }

    // Demonstrate sequence computation
    println!("\nSequence of Parabolic Cylinder Functions D_v(x) for v = 0,1,2,3:\n");

    for &x in &test_points {
        let (d_values, dp_values) = pbdv_seq(3, x)?;

        println!("x = {:.2}:", x);
        for v in 0..=3 {
            println!(
                "  D_{}({:.2}) = {:.6e}, D_{}'({:.2}) = {:.6e}",
                v, x, d_values[v], v, x, dp_values[v]
            );
        }
        println!();
    }

    // Demonstrate V sequence computation
    println!("Sequence of Parabolic Cylinder Functions V_v(x) for v = 0,1,2,3:\n");

    for &x in &test_points[1..4] {
        // Using a subset of test points
        let (v_values, vp_values) = pbvv_seq(3, x)?;

        println!("x = {:.2}:", x);
        for v in 0..=3 {
            println!(
                "  V_{}({:.2}) = {:.6e}, V_{}'({:.2}) = {:.6e}",
                v, x, v_values[v], v, x, vp_values[v]
            );
        }
        println!();
    }

    // Demonstrate W function
    println!("Parabolic Cylinder Function W(a,x):\n");
    println!(
        "{:^8} | {:^8} | {:^15} | {:^15}",
        "a", "x", "W(a,x)", "W'(a,x)"
    );
    println!("{:-^8} | {:-^8} | {:-^15} | {:-^15}", "", "", "", "");

    for a in [0.0, 1.0, 2.0, -1.0] {
        for &x in &test_points[1..4] {
            // Using a subset of test points
            let (w, wp) = pbwa(a, x)?;

            println!("{:^8.2} | {:^8.2} | {:^15.6e} | {:^15.6e}", a, x, w, wp);
        }
        println!("{:-^8} | {:-^8} | {:-^15} | {:-^15}", "", "", "", "");
    }

    // Validate recurrence relations
    println!("\nValidating Recurrence Relations:\n");

    let x = 1.5;
    let v = 2.0;

    // Verify the recurrence relation: D_{v+1}(x) = x*D_v(x) - v*D_{v-1}(x)
    let (d_v_minus_1, _) = pbdv(v - 1.0, x)?;
    let (d_v, _) = pbdv(v, x)?;
    let (d_v_plus_1, _) = pbdv(v + 1.0, x)?;

    let recurrence_value = x * d_v - v * d_v_minus_1;
    println!("Recurrence relation for D_v:");
    println!(
        "  D_{:.0}({:.1}) from recurrence: {:.8e}",
        v + 1.0,
        x,
        recurrence_value
    );
    println!("  D_{:.0}({:.1}) direct: {:.8e}", v + 1.0, x, d_v_plus_1);
    println!(
        "  Difference: {:.2e}",
        (recurrence_value - d_v_plus_1).abs()
    );

    // Verify the asymptotic behavior of D_v(x) for large x
    println!("\nAsymptotic Behavior of D_v(x) for Large x:");

    let large_x = 100.0;
    let (d_large_x, _) = pbdv(v, large_x)?;

    // Asymptotic approximation: D_v(x) ≈ x^v * e^(-x²/4) for large x
    let asymptotic_approx = large_x.powf(v) * (-large_x * large_x / 4.0).exp();

    println!("  D_{:.0}({:.1}) direct: {:.8e}", v, large_x, d_large_x);
    println!("  Asymptotic approximation: {:.8e}", asymptotic_approx);
    println!("  Ratio: {:.8}", d_large_x / asymptotic_approx);

    Ok(())
}
