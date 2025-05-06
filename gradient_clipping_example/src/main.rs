use ndarray::{ArrayD, IxDyn, ScalarOperand};
use num_traits::Float;
use std::fmt::{Debug, Display};

// Enum to define different gradient clipping methods
#[derive(Debug, Clone, Copy)]
pub enum GradientClippingMethod {
    ClipByGlobalNorm,
    ClipByValue,
}

// Structure for gradient clipping configuration
#[derive(Debug, Clone)]
pub struct GradientClipping<F: Float + Debug + ScalarOperand + Display> {
    pub max_norm: F,
    pub method: GradientClippingMethod,
    pub log_stats: bool,
    clipping_applied: bool,
    clipping_ratio: Option<F>,
}

// Implementation of gradient clipping methods
impl<F: Float + Debug + ScalarOperand + Display> GradientClipping<F> {
    // Factory method to create a new GradientClipping with global norm method
    pub fn by_global_norm(max_norm: F, log_stats: bool) -> Self {
        GradientClipping {
            max_norm,
            method: GradientClippingMethod::ClipByGlobalNorm,
            log_stats,
            clipping_applied: false,
            clipping_ratio: None,
        }
    }

    // Factory method to create a new GradientClipping with value clipping method
    pub fn by_value(max_value: F, log_stats: bool) -> Self {
        GradientClipping {
            max_norm: max_value,
            method: GradientClippingMethod::ClipByValue,
            log_stats,
            clipping_applied: false,
            clipping_ratio: None,
        }
    }

    // Method to clip gradients using the configured clipping method
    pub fn clip_gradients(&mut self, gradients: &mut [ArrayD<F>]) -> Result<(), String> {
        match self.method {
            GradientClippingMethod::ClipByGlobalNorm => self.clip_by_global_norm(gradients),
            GradientClippingMethod::ClipByValue => self.clip_by_value(gradients),
        }
    }

    // Implementation of gradient clipping by global norm
    fn clip_by_global_norm(&mut self, gradients: &mut [ArrayD<F>]) -> Result<(), String> {
        // Calculate the global norm of all gradients
        let mut global_norm_sq = F::zero();
        for grad in gradients.iter() {
            let norm_sq = grad.iter().fold(F::zero(), |acc, &x| acc + x * x);
            global_norm_sq = global_norm_sq + norm_sq;
        }
        let global_norm = global_norm_sq.sqrt();

        // Check if clipping is needed
        if global_norm > self.max_norm {
            let scale_factor = self.max_norm / global_norm;
            self.clipping_ratio = Some(scale_factor);
            self.clipping_applied = true;

            // Apply scaling to gradients
            for grad in gradients.iter_mut() {
                *grad = grad.mapv(|x| x * scale_factor);
            }

            if self.log_stats {
                println!(
                    "Gradients clipped by global norm: {} -> {}. Scale factor: {}",
                    global_norm, self.max_norm, scale_factor
                );
            }
        } else {
            self.clipping_applied = false;
            self.clipping_ratio = None;
        }

        Ok(())
    }

    // Implementation of gradient clipping by value
    fn clip_by_value(&mut self, gradients: &mut [ArrayD<F>]) -> Result<(), String> {
        let mut any_clipped = false;
        let max_value = self.max_norm;
        let min_value = -max_value;

        // Clip each gradient value individually
        for grad in gradients.iter_mut() {
            let orig_grad = grad.clone();
            *grad = grad.mapv(|x| {
                if x > max_value {
                    any_clipped = true;
                    max_value
                } else if x < min_value {
                    any_clipped = true;
                    min_value
                } else {
                    x
                }
            });

            if self.log_stats && any_clipped {
                let max_before =
                    orig_grad
                        .iter()
                        .fold(F::neg_infinity(), |acc, &x| if x > acc { x } else { acc });
                let min_before =
                    orig_grad
                        .iter()
                        .fold(F::infinity(), |acc, &x| if x < acc { x } else { acc });
                let max_after = grad
                    .iter()
                    .fold(F::neg_infinity(), |acc, &x| if x > acc { x } else { acc });
                let min_after = grad
                    .iter()
                    .fold(F::infinity(), |acc, &x| if x < acc { x } else { acc });

                println!(
                    "Gradient values clipped: [{}, {}] -> [{}, {}]",
                    min_before, max_before, min_after, max_after
                );
            }
        }

        self.clipping_applied = any_clipped;
        self.clipping_ratio = None;

        Ok(())
    }

    // Method to get metrics about the clipping operation
    pub fn get_metrics(&self) -> Option<(F, bool)> {
        self.clipping_ratio
            .map(|ratio| (ratio, self.clipping_applied))
    }
}

fn main() {
    // Example of gradient clipping usage
    println!("Gradient Clipping Example");
    println!("========================\n");

    // Create sample gradients with large values that will trigger clipping
    let mut gradients1 = vec![
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![10.0, -8.0, 7.0, -12.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![9.0, -7.0, 8.0, -11.0]).unwrap(),
    ];

    // Create a copy for the second example
    let mut gradients2 = gradients1.clone();

    // Print original gradients
    println!("Original gradients:");
    for (i, grad) in gradients1.iter().enumerate() {
        println!("Gradient {}: {:?}", i + 1, grad);
    }

    println!("\n1. Clipping by Global Norm");
    println!("------------------------");

    // Calculate the global norm manually for demonstration
    let global_norm_sq: f64 = gradients1.iter().fold(0.0, |acc, grad| {
        acc + grad.iter().fold(0.0, |sum, &x| sum + (x * x))
    });
    let global_norm = global_norm_sq.sqrt();
    println!("Original global norm: {:.4}", global_norm);

    // Create a gradient clipper with global norm method
    let max_norm = 5.0;
    let mut clipper_norm = GradientClipping::by_global_norm(max_norm, true);

    // Apply clipping
    println!(
        "Applying global norm clipping with max_norm = {}...",
        max_norm
    );
    clipper_norm.clip_gradients(&mut gradients1).unwrap();

    // Print clipped gradients
    println!("Clipped gradients:");
    for (i, grad) in gradients1.iter().enumerate() {
        println!("Gradient {}: {:?}", i + 1, grad);
    }

    // Calculate the new global norm for verification
    let new_global_norm_sq: f64 = gradients1.iter().fold(0.0, |acc, grad| {
        acc + grad.iter().fold(0.0, |sum, &x| sum + (x * x))
    });
    let new_global_norm = new_global_norm_sq.sqrt();
    println!("New global norm: {:.4}", new_global_norm);

    println!("\n2. Clipping by Value");
    println!("------------------");

    // Print original gradients
    println!("Original gradients:");
    for (i, grad) in gradients2.iter().enumerate() {
        println!("Gradient {}: {:?}", i + 1, grad);
    }

    // Create a gradient clipper with value method
    let max_value = 5.0;
    let mut clipper_value = GradientClipping::by_value(max_value, true);

    // Apply clipping
    println!("Applying value clipping with max_value = {}...", max_value);
    clipper_value.clip_gradients(&mut gradients2).unwrap();

    // Print clipped gradients
    println!("Clipped gradients:");
    for (i, grad) in gradients2.iter().enumerate() {
        println!("Gradient {}: {:?}", i + 1, grad);
    }

    println!("\nGradient Clipping Example Complete");
}
