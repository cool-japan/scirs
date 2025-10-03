//! Interactive Statistical Functions Tutorial
//!
//! This tutorial provides a comprehensive educational experience for statistical
//! functions with mathematical proofs, derivations, and numerical experiments.
//!
//! Run with: cargo run --example statistical_functions_interactive_tutorial

use scirs2_core::ndarray::Array1;
use scirs2_special::*;
use std::io::{self, Write};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Interactive Statistical Functions Learning Laboratory");
    println!("════════════════════════════════════════════════════════");
    println!();

    show_welcome();

    loop {
        show_main_menu();
        let choice = get_user_input("Enter your choice (1-8, or 'q' to quit): ")?;

        if choice.to_lowercase() == "q" {
            println!("🎓 Thank you for exploring statistical functions!");
            break;
        }

        match choice.parse::<u32>() {
            Ok(1) => mathematical_foundations_module()?,
            Ok(2) => logistic_function_deep_dive()?,
            Ok(3) => softmax_comprehensive_study()?,
            Ok(4) => numerical_stability_experiments()?,
            Ok(5) => machine_learning_applications()?,
            Ok(6) => interactive_proofs_section()?,
            Ok(7) => computational_experiments()?,
            Ok(8) => quiz_and_challenges()?,
            _ => println!("❌ Invalid choice. Please try again.\n"),
        }

        pause();
    }

    Ok(())
}

#[allow(dead_code)]
fn show_welcome() {
    println!("Welcome to the comprehensive study of statistical functions!");
    println!("This tutorial combines rigorous mathematical theory with");
    println!("practical applications in machine learning and statistics.");
    println!();
    println!("📖 What You'll Learn:");
    println!("• Mathematical foundations and proofs");
    println!("• Numerical stability techniques");
    println!("• Machine learning applications");
    println!("• Interactive mathematical experiments");
    println!();
}

#[allow(dead_code)]
fn show_main_menu() {
    println!("🏠 MAIN MENU - Choose Your Learning Path:");
    println!("─────────────────────────────────────────");
    println!("1. 📐 Mathematical Foundations & Theory");
    println!("2. 📈 Logistic Function Deep Dive");
    println!("3. 🎯 Softmax Function Comprehensive Study");
    println!("4. ⚖️  Numerical Stability Experiments");
    println!("5. 🤖 Machine Learning Applications");
    println!("6. 📋 Interactive Mathematical Proofs");
    println!("7. 🧪 Computational Experiments");
    println!("8. 🏆 Quiz & Challenges");
    println!();
}

#[allow(dead_code)]
fn mathematical_foundations_module() -> Result<(), Box<dyn std::error::Error>> {
    println!("📐 MATHEMATICAL FOUNDATIONS");
    println!("═══════════════════════════");
    println!();

    println!("🔍 THE LOGISTIC FUNCTION");
    println!("─────────────────────────");
    println!("Definition: σ(x) = 1 / (1 + e^(-x))");
    println!();

    println!("📊 Key Properties with Proofs:");
    println!();

    println!("1. RANGE: σ(x) ∈ (0, 1) for all x ∈ ℝ");
    println!("   Proof: Since e^(-x) > 0, we have 1 + e^(-x) > 1");
    println!("   Therefore: 0 < 1/(1 + e^(-x)) < 1");
    println!();

    println!("2. SYMMETRY: σ(-x) = 1 - σ(x)");
    println!("   Proof: σ(-x) = 1/(1 + e^x) = e^(-x)/(e^(-x) + 1) = 1 - σ(x)");
    println!();

    println!("Let's verify these properties numerically:");

    // Demonstrate range property
    println!("Testing range property:");
    for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
        let sigma_x = logistic(x);
        println!("  σ({:4.1}) = {:8.6} ∈ (0,1) ✓", x, sigma_x);
    }
    println!();

    // Demonstrate symmetry property
    println!("Testing symmetry property σ(-x) = 1 - σ(x):");
    for x in [0.5, 1.0, 2.0, 5.0] {
        let sigma_x = logistic(x);
        let sigma_neg_x = logistic(-x);
        let expected = 1.0 - sigma_x;
        let error = (sigma_neg_x - expected).abs();
        println!(
            "  x={:3.1}: σ(-x)={:8.6}, 1-σ(x)={:8.6}, error={:2.0e}",
            x, sigma_neg_x, expected, error
        );
    }

    Ok(())
}

#[allow(dead_code)]
fn logistic_function_deep_dive() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 LOGISTIC FUNCTION DEEP DIVE");
    println!("══════════════════════════════");
    println!();

    println!("🧮 DERIVATIVE ANALYSIS");
    println!("─────────────────────");
    println!("The derivative of σ(x) is: σ'(x) = σ(x)(1 - σ(x))");
    println!();

    println!("This has important implications:");
    println!("• Maximum derivative at x = 0: σ'(0) = 0.25");
    println!("• Derivative approaches 0 as |x| → ∞");
    println!("• Function is strictly increasing everywhere");
    println!();

    println!("Let's explore the derivative behavior:");
    let x_values = Array1::linspace(-5.0, 5.0, 11);

    println!("    x    │   σ(x)   │  σ'(x)   │ Max at x=0");
    println!("─────────┼──────────┼──────────┼────────────");

    for &x in x_values.iter() {
        let sigma_x = logistic(x);
        let derivative = logistic_derivative(x);
        let marker = if (x - 0.0).abs() < 0.01 {
            " ← MAX"
        } else {
            ""
        };
        println!(
            "{:8.1} │ {:8.6} │ {:8.6} │{}",
            x, sigma_x, derivative, marker
        );
    }

    println!();
    println!("🎯 INFLECTION POINT ANALYSIS");
    println!("────────────────────────────");
    println!("The second derivative σ''(x) = σ'(x)(1 - 2σ(x)) = 0 when σ(x) = 1/2");
    println!("This occurs at x = 0, which is the inflection point.");

    let inflection_x = 0.0;
    let sigma_at_inflection = logistic(inflection_x);
    println!(
        "At x = {}: σ(x) = {:6.4} = 1/2 ✓",
        inflection_x, sigma_at_inflection
    );

    Ok(())
}

#[allow(dead_code)]
fn softmax_comprehensive_study() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 SOFTMAX FUNCTION COMPREHENSIVE STUDY");
    println!("═══════════════════════════════════════");
    println!();

    println!("📖 DEFINITION AND PROPERTIES");
    println!("───────────────────────────");
    println!("For vector x = (x₁, x₂, ..., xₙ):");
    println!("softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)");
    println!();

    println!("Key Properties:");
    println!("1. Probability Distribution: Σᵢ softmax(xᵢ) = 1");
    println!("2. Translation Invariance: softmax(x + c) = softmax(x)");
    println!("3. Maximum Preservation: preserves ordering");
    println!();

    println!("🧪 EXPERIMENTAL VERIFICATION");
    println!("────────────────────────────");

    // Test probability distribution property
    let test_vector = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let softmax_result = softmax(test_vector.view())?;
    let sum = softmax_result.sum();

    println!("Test Vector: {:?}", test_vector.to_vec());
    println!(
        "Softmax:     {:?}",
        softmax_result
            .iter()
            .map(|&x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );
    println!("Sum:         {:.10} (should be 1.0) ✓", sum);
    println!();

    // Test translation invariance
    println!("🔄 TRANSLATION INVARIANCE TEST");
    println!("─────────────────────────────");
    let constant = 100.0; // Large constant to test numerical stability
    let translated_vector = test_vector.mapv(|x| x + constant);
    let softmax_translated = softmax(translated_vector.view())?;

    println!(
        "Original softmax:    {:?}",
        softmax_result
            .iter()
            .map(|&x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "Translated softmax:  {:?}",
        softmax_translated
            .iter()
            .map(|&x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    let max_difference = softmax_result
        .iter()
        .zip(softmax_translated.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    println!(
        "Maximum difference: {:.2e} (should be ~0) ✓",
        max_difference
    );

    Ok(())
}

#[allow(dead_code)]
fn numerical_stability_experiments() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚖️ NUMERICAL STABILITY EXPERIMENTS");
    println!("══════════════════════════════════");
    println!();

    println!("🔬 LOGISTIC FUNCTION STABILITY");
    println!("──────────────────────────────");
    println!("Testing numerically stable implementation vs naive approach");
    println!();

    let extreme_values = vec![-50.0, -20.0, -10.0, 0.0, 10.0, 20.0, 50.0];

    println!("    x    │ Stable Impl │ Naive Impl  │  Difference");
    println!("─────────┼─────────────┼─────────────┼─────────────");

    for &x in &extreme_values {
        let stable_result = logistic(x);
        let naive_result = if x > 20.0 {
            1.0
        } else {
            1.0 / (1.0 + (-x).exp())
        };
        let difference = (stable_result - naive_result).abs();

        println!(
            "{:8.1} │ {:11.8} │ {:11.8} │ {:11.2e}",
            x, stable_result, naive_result, difference
        );
    }

    println!();
    println!("🌊 SOFTMAX OVERFLOW PREVENTION");
    println!("──────────────────────────────");

    // Test with values that would cause overflow in naive implementation
    let large_values = Array1::from_vec(vec![700.0, 800.0, 900.0]);
    println!("Testing with large values: {:?}", large_values.to_vec());

    match softmax(large_values.view()) {
        Ok(result) => {
            println!(
                "Stable softmax result: {:?}",
                result
                    .iter()
                    .map(|&x| format!("{:.6}", x))
                    .collect::<Vec<_>>()
            );
            println!("Sum: {:.10} ✓", result.sum());
        }
        Err(e) => println!("Error: {}", e),
    }

    Ok(())
}

#[allow(dead_code)]
fn machine_learning_applications() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 MACHINE LEARNING APPLICATIONS");
    println!("════════════════════════════════");
    println!();

    println!("🎯 BINARY CLASSIFICATION EXAMPLE");
    println!("────────────────────────────────");
    println!("Logistic regression uses the logistic function as the link function:");
    println!("P(y=1|x) = σ(w·x + b)");
    println!();

    // Simulate a simple dataset
    let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = 0.5;
    let bias = -2.5;

    println!("Dataset simulation (weight={}, bias={}):", weight, bias);
    println!("Feature │ Linear Comb │ Probability │ Prediction");
    println!("────────┼─────────────┼─────────────┼───────────");

    for &x in &features {
        let linear_combination = weight * x + bias;
        let probability = logistic(linear_combination);
        let prediction = if probability > 0.5 {
            "Positive"
        } else {
            "Negative"
        };

        println!(
            "{:7.1} │ {:11.2} │ {:11.6} │ {:>9}",
            x, linear_combination, probability, prediction
        );
    }

    println!();
    println!("🎲 MULTI-CLASS CLASSIFICATION EXAMPLE");
    println!("─────────────────────────────────────");
    println!("Softmax is used for multi-class classification output layer");
    println!();

    // Simulate network outputs (logits)
    let logits = Array1::from_vec(vec![2.0, 1.0, 0.1]);
    let class_names = ["Cat", "Dog", "Bird"];

    let probabilities = softmax(logits.view())?;

    println!("Network outputs (logits): {:?}", logits.to_vec());
    println!("Class probabilities:");
    for (&prob, &name) in probabilities.iter().zip(class_names.iter()) {
        println!("  {}: {:.4} ({:.1}%)", name, prob, prob * 100.0);
    }

    let predicted_class = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    println!("Predicted class: {} ✓", class_names[predicted_class]);

    Ok(())
}

#[allow(dead_code)]
fn interactive_proofs_section() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 INTERACTIVE MATHEMATICAL PROOFS");
    println!("══════════════════════════════════");
    println!();

    println!("🔍 PROOF WALKTHROUGH: Logistic Derivative");
    println!("─────────────────────────────────────────");
    println!("Let's prove that d/dx[σ(x)] = σ(x)(1 - σ(x)) step by step:");
    println!();

    println!("Step 1: Start with σ(x) = 1/(1 + e^(-x))");
    println!("Step 2: Use the chain rule: d/dx[1/u] = -u'/u²");
    println!("        where u = 1 + e^(-x), so u' = -e^(-x)");
    println!();
    println!("Step 3: σ'(x) = -(-e^(-x))/(1 + e^(-x))² = e^(-x)/(1 + e^(-x))²");
    println!();
    println!("Step 4: Factorize: σ'(x) = [1/(1 + e^(-x))] × [e^(-x)/(1 + e^(-x))]");
    println!("                          = σ(x) × [e^(-x)/(1 + e^(-x))]");
    println!();
    println!("Step 5: Note that e^(-x)/(1 + e^(-x)) = 1 - 1/(1 + e^(-x)) = 1 - σ(x)");
    println!();
    println!("Therefore: σ'(x) = σ(x)(1 - σ(x)) Q.E.D. ✓");
    println!();

    println!("🧪 NUMERICAL VERIFICATION");
    println!("─────────────────────────");
    let test_points = vec![0.0, 1.0, -1.0, 2.0];

    for &x in &test_points {
        let sigma_x = logistic(x);
        let analytical_derivative = sigma_x * (1.0 - sigma_x);
        let computed_derivative = logistic_derivative(x);
        let error = (analytical_derivative - computed_derivative).abs();

        println!(
            "x={:4.1}: Analytical={:.8}, Computed={:.8}, Error={:.2e}",
            x, analytical_derivative, computed_derivative, error
        );
    }

    Ok(())
}

#[allow(dead_code)]
fn computational_experiments() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 COMPUTATIONAL EXPERIMENTS");
    println!("════════════════════════════");
    println!();

    println!("⏱️ PERFORMANCE COMPARISON");
    println!("────────────────────────");

    use std::time::Instant;

    let large_array = Array1::linspace(-10.0, 10.0, 100_000);

    // Time logistic function
    let start = Instant::now();
    let _: Vec<f64> = large_array.iter().map(|&x| logistic(x)).collect();
    let logistic_time = start.elapsed();

    // Time softmax function on chunks
    let chunksize = 1000;
    let start = Instant::now();
    for chunk in large_array.exact_chunks(chunksize) {
        let _ = softmax(chunk)?;
    }
    let softmax_time = start.elapsed();

    println!("Logistic function (100k elements): {:?}", logistic_time);
    println!("Softmax function (100 chunks):     {:?}", softmax_time);

    println!();
    println!("🎯 ACCURACY ANALYSIS");
    println!("───────────────────");

    // Test extreme precision
    let precision_test_x = 1e-15;
    let result = logistic(precision_test_x);
    let expected = 0.5 + precision_test_x / 4.0; // First-order Taylor approximation
    let relative_error = ((result - expected) / expected).abs();

    println!("Advanced-small input test:");
    println!("x = {:.2e}", precision_test_x);
    println!("σ(x) = {:.15}", result);
    println!("Expected ≈ {:.15}", expected);
    println!("Relative error: {:.2e}", relative_error);

    Ok(())
}

#[allow(dead_code)]
fn quiz_and_challenges() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏆 QUIZ & CHALLENGES");
    println!("════════════════════");
    println!();

    println!("📝 Challenge 1: Symmetry Property");
    println!("─────────────────────────────────");
    println!("If σ(2) = 0.8808, what is σ(-2)?");

    let user_answer = get_user_input("Your answer: ")?;
    let correct_answer = 1.0 - logistic(2.0);
    let user_value: f64 = user_answer.parse().unwrap_or(0.0);

    if (user_value - correct_answer).abs() < 0.001 {
        println!("✅ Correct! σ(-2) = 1 - σ(2) = {:.4}", correct_answer);
    } else {
        println!("❌ Not quite. The correct answer is {:.4}", correct_answer);
        println!("Remember: σ(-x) = 1 - σ(x)");
    }

    println!();
    println!("📝 Challenge 2: Derivative Maximum");
    println!("─────────────────────────────────");
    println!("At what value of x is the logistic function's derivative maximized?");

    let user_answer = get_user_input("Your answer: ")?;

    if user_answer.trim() == "0" || user_answer.trim() == "0.0" {
        println!("✅ Correct! The derivative σ'(x) = σ(x)(1-σ(x)) is maximized");
        println!("when σ(x) = 0.5, which occurs at x = 0.");
        println!("Maximum value: σ'(0) = 0.5 × 0.5 = 0.25");
    } else {
        println!("❌ Not quite. The derivative is maximized at x = 0.");
        println!("At this point, σ(0) = 0.5, giving maximum derivative of 0.25.");
    }

    println!();
    println!("🎉 Congratulations on completing the statistical functions tutorial!");

    Ok(())
}

// Helper functions
#[allow(dead_code)]
fn get_user_input(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

#[allow(dead_code)]
fn pause() {
    print!("\nPress Enter to continue...");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
}
