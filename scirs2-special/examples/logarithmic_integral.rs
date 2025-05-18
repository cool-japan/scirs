use num_complex::Complex64;
use scirs2_special::{e1, expint, li, li_complex};

fn main() {
    println!("Logarithmic Integral Function Examples");
    println!("=====================================");

    // Real logarithmic integral examples
    println!("\nReal Logarithmic Integral (Li(x)):");
    for x in [0.5, 1.5, 2.0, 5.0, 10.0, 100.0] {
        match li(x) {
            Ok(result) => println!("li({:.1}) = {:.10}", x, result),
            Err(e) => println!("li({:.1}) = Error: {}", x, e),
        }
    }

    // Error case - singularity at x=1
    match li(1.0) {
        Ok(result) => println!("li(1.0) = {:.10}", result),
        Err(e) => println!("li(1.0) = Error: {}", e),
    }

    // Complex logarithmic integral examples
    println!("\nComplex Logarithmic Integral (Li(z)):");
    let complex_values = [
        Complex64::new(2.0, 0.0),
        Complex64::new(2.0, 1.0),
        Complex64::new(0.5, 0.5),
        Complex64::new(-1.0, 0.0),
    ];

    for z in complex_values {
        match li_complex(z) {
            Ok(result) => println!(
                "li({:.1}+{:.1}i) = {:.6}+{:.6}i",
                z.re, z.im, result.re, result.im
            ),
            Err(e) => println!("li({:.1}+{:.1}i) = Error: {}", z.re, z.im, e),
        }
    }

    // Exponential integral examples
    println!("\nExponential Integral E₁(x):");
    for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
        match e1(x) {
            Ok(result) => println!("E₁({:.1}) = {:.10}", x, result),
            Err(e) => println!("E₁({:.1}) = Error: {}", x, e),
        }
    }

    // Exponential integral of different orders
    println!("\nExponential Integral Eₙ(x) for different n:");
    for n in 1..=5 {
        match expint(n, 1.0) {
            Ok(result) => println!("E₍{}₎(1.0) = {:.10}", n, result),
            Err(e) => println!("E₍{}₎(1.0) = Error: {}", n, e),
        }
    }

    // Verification: Li(x) relation to Ei(ln(x))
    println!("\nVerification of Li(x) relation to Ei(ln(x)):");
    for x in [2.0, 5.0, 10.0] {
        if let Ok(li_x) = li(x) {
            println!("For x = {:.1}:", x);
            println!("  li(x)             = {:.10}", li_x);
            println!(
                "  Ei(ln(x)) - γ     = {:.10}",
                (x.ln().exp_integral() - 0.577_215_664_901_532_9)
            );
            println!(
                "  Relative difference: {:.10e}",
                ((li_x - (x.ln().exp_integral() - 0.577_215_664_901_532_9)).abs() / li_x.abs())
            );
        }
    }
}

// Helper trait to calculate the exponential integral directly
trait ExpIntegral {
    fn exp_integral(self) -> f64;
    fn neg_exp_integral(self) -> f64;
}

impl ExpIntegral for f64 {
    fn exp_integral(self) -> f64 {
        if self <= 0.0 {
            return -(-self).neg_exp_integral();
        }

        // For small x, use series expansion
        if self < 6.0 {
            let mut sum; // Will be set below
            let mut term = self;
            let mut k = 1.0;
            let mut factorial = 1.0;

            // Series: Ei(x) = γ + ln|x| + sum_{k=1}^∞ x^k/(k·k!)
            let euler_mascheroni = 0.577_215_664_901_532_9;
            sum = euler_mascheroni + self.ln();

            while k < 30.0 {
                term *= self / k;
                factorial *= k;
                let contribution = term / factorial / k;
                sum += contribution;

                if contribution.abs() < 1e-15 * sum.abs() {
                    break;
                }

                k += 1.0;
            }

            sum
        } else {
            // For large x, use asymptotic expansion
            let mut sum; // Will be set below
            let mut term = 1.0;
            let mut k = 1.0;

            // Series: Ei(x) ~ e^x/x · (1 + sum_{k=1}^∞ k!/x^k)
            sum = 1.0;

            while k < 30.0 {
                term *= k / self;
                sum += term;

                if term.abs() < 1e-15 * sum.abs() {
                    break;
                }

                k += 1.0;
            }

            sum * self.exp() / self
        }
    }

    fn neg_exp_integral(self) -> f64 {
        if self <= 0.0 {
            panic!("E₁(x) is only defined for x > 0");
        }

        // For small x, use series expansion
        if self < 1.0 {
            let mut sum; // Will be set below
            let mut term = -1.0;
            let mut k = 1.0;

            // Series: E₁(x) = -γ - ln(x) - sum_{k=1}^∞ (-1)^k x^k/(k·k!)
            let euler_mascheroni = 0.577_215_664_901_532_9;
            sum = -euler_mascheroni - self.ln();

            let mut factorial = 1.0;
            while k < 30.0 {
                term *= -self / k;
                factorial *= k;
                let contribution = term / factorial / k;
                sum -= contribution;

                if contribution.abs() < 1e-15 * sum.abs() {
                    break;
                }

                k += 1.0;
            }

            -sum
        } else {
            // For large x, use continued fraction
            let mut a; // Will be initialized in the loop
            let mut b = self + 1.0;
            let mut c = 1.0;
            let mut d = 1.0 / b;
            let mut h = d;

            for i in 1..100 {
                let i_f64 = i as f64;
                a = -i_f64 * i_f64;
                b += 2.0;
                c = b + a / c;
                if c.abs() < 1e-15 {
                    c = 1e-15;
                }
                d = 1.0 / (b + a * d);
                if d.abs() < 1e-15 {
                    d = 1e-15;
                }
                let del = c * d;
                h *= del;

                if (del - 1.0).abs() < 1e-15 {
                    break;
                }
            }

            h * (-self).exp()
        }
    }
}
