//! Advanced Interactive Tutor for Special Functions
//!
//! This example provides a comprehensive, gamified learning experience with:
//! - Adaptive difficulty based on user performance
//! - Real-time hints and explanations

#![allow(clippy::all)]
//! - Mathematical proof walkthroughs
//! - Interactive problem-solving sessions
//! - Visual concept demonstrations
//! - Achievement system and progress tracking
//! - Personalized learning paths
//!
//! Run with: cargo run --example advanced_interactive_tutor

use scirs2_core::Complex64;
use scirs2_special::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::io::{self, Write};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
struct UserProfile {
    name: String,
    level: u32,
    experience_points: u32,
    achievements: Vec<Achievement>,
    learning_preferences: LearningPreferences,
    performance_history: Vec<PerformanceRecord>,
    current_streak: u32,
    mastery_scores: HashMap<String, f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Achievement {
    id: String,
    name: String,
    description: String,
    earned_at: Option<std::time::SystemTime>,
    difficulty: AchievementDifficulty,
}

#[derive(Debug, Clone)]
enum AchievementDifficulty {
    Bronze,
    Silver,
    Gold,
    Platinum,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct LearningPreferences {
    visual_learner: bool,
    theoretical_focus: bool,
    practical_applications: bool,
    difficulty_preference: DifficultyLevel,
    preferred_domains: Vec<String>,
}

#[derive(Debug, Clone)]
enum DifficultyLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct PerformanceRecord {
    topic: String,
    score: f64,
    time_taken: Duration,
    hints_used: u32,
    timestamp: std::time::SystemTime,
}

impl UserProfile {
    fn new(name: String) -> Self {
        Self {
            name,
            level: 1,
            experience_points: 0,
            achievements: Vec::new(),
            learning_preferences: LearningPreferences::default(),
            performance_history: Vec::new(),
            current_streak: 0,
            mastery_scores: HashMap::new(),
        }
    }

    fn add_experience(&mut self, points: u32) {
        self.experience_points += points;
        let new_level = (self.experience_points / 100) + 1;
        if new_level > self.level {
            self.level = new_level;
            println!("🎉 Congratulations! You've reached level {}!", self.level);
        }
    }

    fn earn_achievement(&mut self, achievement: Achievement) {
        if !self.achievements.iter().any(|a| a.id == achievement.id) {
            println!("🏆 Achievement Unlocked: {}", achievement.name);
            println!("   {}", achievement.description);
            self.achievements.push(achievement);
        }
    }

    fn update_mastery(&mut self, topic: &str, score: f64) {
        let current_mastery = self.mastery_scores.get(topic).unwrap_or(&0.0);
        let new_mastery = (current_mastery * 0.7) + (score * 0.3); // Weighted average
        self.mastery_scores.insert(topic.to_string(), new_mastery);
    }
}

impl Default for LearningPreferences {
    fn default() -> Self {
        Self {
            visual_learner: false,
            theoretical_focus: true,
            practical_applications: true,
            difficulty_preference: DifficultyLevel::Intermediate,
            preferred_domains: vec!["mathematics".to_string()],
        }
    }
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎓 Advanced Interactive Special Functions Tutor");
    println!("===============================================\n");

    // Setup user profile
    let mut profile = setup_userprofile()?;

    loop {
        display_personalized_menu(&profile);
        let choice = get_user_input("Enter your choice: ")?;

        if choice.to_lowercase() == "q" {
            display_session_summary(&profile);
            break;
        }

        match choice.parse::<u32>() {
            Ok(1) => adaptive_learning_session(&mut profile)?,
            Ok(2) => mathematical_proof_walkthrough(&mut profile)?,
            Ok(3) => interactive_problem_solver(&mut profile)?,
            Ok(4) => visual_function_explorer(&mut profile)?,
            Ok(5) => real_world_applications_lab(&mut profile)?,
            Ok(6) => mastery_challenge(&mut profile)?,
            Ok(7) => peer_comparison_mode(&mut profile)?,
            Ok(8) => profile_settings(&mut profile)?,
            _ => println!("❌ Invalid choice. Please try again.\n"),
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn setup_userprofile() -> Result<UserProfile, Box<dyn std::error::Error>> {
    println!("👋 Welcome to the Advanced Special Functions Tutor!");
    let name = get_user_input("What's your name? ")?;
    let mut profile = UserProfile::new(name);

    println!("\n📊 Let's customize your learning experience:");

    // Learning style assessment
    let visual = get_yes_noinput("Do you prefer visual explanations and graphs? (y/n): ")?;
    let theoretical =
        get_yes_noinput("Are you interested in mathematical proofs and theory? (y/n): ")?;
    let practical = get_yes_noinput("Do you want to see practical applications? (y/n): ")?;

    profile.learning_preferences.visual_learner = visual;
    profile.learning_preferences.theoretical_focus = theoretical;
    profile.learning_preferences.practical_applications = practical;

    // Difficulty assessment
    println!("\n🎯 Quick assessment to determine your starting level:");
    let assessment_score = run_quick_assessment()?;
    profile.learning_preferences.difficulty_preference = match assessment_score {
        0..=3 => DifficultyLevel::Beginner,
        4..=6 => DifficultyLevel::Intermediate,
        7..=8 => DifficultyLevel::Advanced,
        _ => DifficultyLevel::Expert,
    };

    println!("\n✅ Profile setup complete! Your learning journey begins now.");
    println!(
        "Starting difficulty: {:?}",
        profile.learning_preferences.difficulty_preference
    );

    Ok(profile)
}

#[allow(dead_code)]
fn run_quick_assessment() -> Result<u32, Box<dyn std::error::Error>> {
    let mut score = 0;

    println!("Answer these questions to help us assess your level:");

    // Question 1: Basic gamma function
    println!("\n1. What is Γ(4)?");
    println!("   a) 4");
    println!("   b) 6");
    println!("   c) 24");
    println!("   d) 12");
    let answer = get_user_input("Your answer (a/b/c/d): ")?;
    if answer.to_lowercase() == "b" {
        score += 1;
        println!("✅ Correct! Γ(n) = (n-1)! for positive integers.");
    } else {
        println!("❌ Incorrect. Γ(4) = 3! = 6");
    }

    // Question 2: Bessel function properties
    println!("\n2. Bessel functions J_n(x) are solutions to which type of differential equation?");
    println!("   a) Linear homogeneous");
    println!("   b) Nonlinear");
    println!("   c) Partial differential");
    println!("   d) Integral equation");
    let answer = get_user_input("Your answer (a/b/c/d): ")?;
    if answer.to_lowercase() == "a" {
        score += 1;
        println!("✅ Correct! Bessel's equation is a second-order linear homogeneous ODE.");
    } else {
        println!(
            "❌ Incorrect. Bessel functions satisfy a linear homogeneous differential equation."
        );
    }

    // Question 3: Error function asymptotic behavior
    println!("\n3. As x → ∞, erf(x) approaches:");
    println!("   a) 0");
    println!("   b) 1");
    println!("   c) π/2");
    println!("   d) ∞");
    let answer = get_user_input("Your answer (a/b/c/d): ")?;
    if answer.to_lowercase() == "b" {
        score += 1;
        println!("✅ Correct! erf(∞) = 1");
    } else {
        println!("❌ Incorrect. erf(x) → 1 as x → ∞");
    }

    // Add more sophisticated questions based on responses
    if score >= 2 {
        // Advanced questions
        println!("\n4. The Wright function W(z) = Σ z^n/(n! Γ(αn+β)) generalizes:");
        println!("   a) Exponential function");
        println!("   b) Bessel functions");
        println!("   c) Both exponential and Mittag-Leffler functions");
        println!("   d) Gamma function");
        let answer = get_user_input("Your answer (a/b/c/d): ")?;
        if answer.to_lowercase() == "c" {
            score += 2;
            println!("✅ Excellent! Wright functions are very general.");
        } else {
            println!(
                "❌ Wright functions generalize both exponential and Mittag-Leffler functions."
            );
        }

        println!("\n5. Spherical harmonics Y_l^m(θ,φ) are eigenfunctions of:");
        println!("   a) Laplacian operator");
        println!("   b) Angular momentum operator");
        println!("   c) Hamiltonian operator");
        println!("   d) Gradient operator");
        let answer = get_user_input("Your answer (a/b/c/d): ")?;
        if answer.to_lowercase() == "b" {
            score += 2;
            println!("✅ Perfect! They're eigenfunctions of L² operator.");
        } else {
            println!(
                "❌ Spherical harmonics are eigenfunctions of the angular momentum operator L²."
            );
        }
    }

    println!("\nAssessment complete! Score: {}/10", score);
    Ok(score)
}

#[allow(dead_code)]
fn adaptive_learning_session(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧠 Adaptive Learning Session");
    println!("============================\n");

    // Select topic based on mastery scores and preferences
    let topic = select_adaptive_topic(profile);
    println!("📚 Today's focus: {}", topic);

    match topic.as_str() {
        "Gamma Functions" => gamma_adaptive_session(profile),
        "Bessel Functions" => bessel_adaptive_session(profile),
        "Error Functions" => error_function_adaptive_session(profile),
        "Orthogonal Polynomials" => orthogonal_polynomial_adaptive_session(profile),
        _ => advanced_topic_session(profile, &topic),
    }
}

#[allow(dead_code)]
fn gamma_adaptive_session(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎲 Gamma Function Deep Dive");
    println!("===========================\n");

    let start_time = Instant::now();
    let mut score = 0.0;
    let mut hints_used = 0;

    // Adaptive content based on user level
    match profile.learning_preferences.difficulty_preference {
        DifficultyLevel::Beginner => {
            println!("📖 The gamma function Γ(z) is defined as:");
            println!("   Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt");
            println!("\n🔑 Key property: Γ(n+1) = n! for non-negative integers n");

            // Interactive calculation
            println!("\n🧮 Let's calculate some values:");
            let test_value = 5.0;
            let correct_answer = gamma(test_value);

            println!("Calculate Γ({}):", test_value);
            if profile.learning_preferences.visual_learner {
                display_gamma_visualization(test_value);
            }

            let user_answer: f64 = loop {
                match get_user_input("Your answer (or 'hint' for help): ")?.parse() {
                    Ok(val) => break val,
                    Err(_) => {
                        hints_used += 1;
                        println!("💡 Hint: For integer n, Γ(n) = (n-1)!");
                        println!("   So Γ(5) = 4! = 4 × 3 × 2 × 1 = ?");
                    }
                }
            };

            let error = (user_answer - correct_answer).abs() / correct_answer;
            if error < 0.01 {
                println!("✅ Excellent! Γ({}) = {:.6}", test_value, correct_answer);
                score += 1.0;
            } else {
                println!("❌ Close, but the correct answer is {:.6}", correct_answer);
                println!(
                    "   Your answer: {:.6} (error: {:.2}%)",
                    user_answer,
                    error * 100.0
                );
            }
        }

        DifficultyLevel::Intermediate => {
            println!("📈 Advanced gamma function properties:");
            println!("   • Recurrence: Γ(z+1) = z·Γ(z)");
            println!("   • Reflection: Γ(z)Γ(1-z) = π/sin(πz)");
            println!("   • Duplication: Γ(z)Γ(z+1/2) = √π/2^(2z-1) Γ(2z)");

            // Complex gamma function exercise
            println!("\n🔢 Complex gamma function exercise:");
            let z = Complex64::new(1.5, 0.5);
            let correct_answer = gamma_complex(z);

            println!("Calculate Γ(1.5 + 0.5i)");
            println!("Express your answer as a + bi");

            let real_part: f64 = get_user_input("Real part: ")?
                .parse()
                .map_err(|_| "Invalid number")?;
            let imag_part: f64 = get_user_input("Imaginary part: ")?
                .parse()
                .map_err(|_| "Invalid number")?;

            let user_answer = Complex64::new(real_part, imag_part);
            let error = (user_answer - correct_answer).norm() / correct_answer.norm();

            if error < 0.05 {
                println!("✅ Excellent! Γ(1.5 + 0.5i) = {:.4}", correct_answer);
                score += 1.0;
            } else {
                println!("❌ The correct answer is {:.4}", correct_answer);
                score += 0.5; // Partial credit
            }
        }

        _ => {
            // Advanced/Expert level
            println!("🎯 Expert-level gamma function challenge:");
            println!("Prove that Γ(1/2) = √π using the integral definition.");
            println!("\nHint: Use the substitution t = u² and the Gaussian integral.");

            let steps = vec![
                "Start with Γ(1/2) = ∫₀^∞ t^(-1/2) e^(-t) dt",
                "Substitute t = u², dt = 2u du",
                "Get Γ(1/2) = ∫₀^∞ u^(-1) e^(-u²) · 2u du = 2∫₀^∞ e^(-u²) du",
                "The Gaussian integral ∫_{-∞}^∞ e^(-u²) du = √π",
                "Therefore ∫₀^∞ e^(-u²) du = √π/2",
                "Conclude Γ(1/2) = 2 · √π/2 = √π",
            ];

            println!("\n📝 Walk through the proof steps:");
            for (i, step) in steps.iter().enumerate() {
                println!("Step {}: {}", i + 1, step);
                get_user_input("Press Enter to continue...")?;
            }

            score = 1.0; // Full credit for expert level engagement
        }
    }

    // Record performance
    let duration = start_time.elapsed();
    let record = PerformanceRecord {
        topic: "Gamma Functions".to_string(),
        score,
        time_taken: duration,
        hints_used,
        timestamp: std::time::SystemTime::now(),
    };

    profile.performance_history.push(record);
    profile.update_mastery("Gamma Functions", score);

    let xp_gained = ((score * 50.0) as u32).max(10);
    profile.add_experience(xp_gained);

    // Check for achievements
    check_gamma_achievements(profile, score, hints_used);

    println!("\n📊 Session complete!");
    println!("Score: {:.1}/1.0", score);
    println!("Time: {:.1}s", duration.as_secs_f64());
    println!("XP gained: {}", xp_gained);

    Ok(())
}

#[allow(dead_code)]
fn bessel_adaptive_session(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 Bessel Function Mastery");
    println!("==========================\n");

    println!("📖 Bessel functions are solutions to Bessel's differential equation:");
    println!("   x²y'' + xy' + (x² - ν²)y = 0");
    println!("\n🔑 Key types:");
    println!("   • J_ν(x): Bessel functions of the first kind");
    println!("   • Y_ν(x): Bessel functions of the second kind");
    println!("   • I_ν(x): Modified Bessel functions of the first kind");
    println!("   • K_ν(x): Modified Bessel functions of the second kind");

    // Interactive exploration
    println!("\n🧮 Interactive Bessel function exploration:");

    let x = 5.0;
    println!("For x = {}", x);
    println!("J₀({}) = {:.6}", x, j0(x));
    println!("J₁({}) = {:.6}", x, j1(x));
    println!("Y₀({}) = {:.6}", x, y0(x));
    println!("Y₁({}) = {:.6}", x, y1(x));

    if profile.learning_preferences.visual_learner {
        display_bessel_visualization(x);
    }

    // Practical application
    if profile.learning_preferences.practical_applications {
        println!("\n🔬 Physical Application: Vibrating Circular Membrane");
        println!("The displacement u(r,θ,t) of a circular drumhead is:");
        println!("u(r,θ,t) = J_n(k_mn·r/R)·cos(nθ)·cos(ω_mn·t)");
        println!("where k_mn are zeros of J_n(x).");

        println!("\nFirst 5 zeros of J₀(x):");
        for i in 1..=5 {
            let zero = j0_zeros::<f64>(i).unwrap();
            println!("  α₀,{} = {:.6}", i, zero);
        }
    }

    // Quiz
    println!("\n❓ Quick Quiz:");
    println!("What is the relationship between J_ν(x) and I_ν(x)?");
    println!("a) I_ν(x) = i^(-ν) J_ν(ix)");
    println!("b) I_ν(x) = J_ν(ix)");
    println!("c) I_ν(x) = J_ν(-x)");

    let answer = get_user_input("Your answer (a/b/c): ")?;
    let correct = answer.to_lowercase() == "a";

    if correct {
        println!("✅ Correct! Modified Bessel functions are related to regular Bessel functions through I_ν(x) = i^(-ν) J_ν(ix)");
        profile.add_experience(25);
    } else {
        println!("❌ Incorrect. The correct relationship is I_ν(x) = i^(-ν) J_ν(ix)");
    }

    profile.update_mastery("Bessel Functions", if correct { 1.0 } else { 0.0 });

    Ok(())
}

#[allow(dead_code)]
fn error_function_adaptive_session(
    profile: &mut UserProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Error Function Deep Dive");
    println!("===========================\n");

    println!("📖 The error function is defined as:");
    println!("   erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt");
    println!("\n🔑 Key properties:");
    println!("   • erf(-x) = -erf(x) (odd function)");
    println!("   • erf(∞) = 1, erf(-∞) = -1");
    println!("   • erfc(x) = 1 - erf(x) (complementary error function)");

    // Interactive calculation with step-by-step guidance
    println!("\n🧮 Let's explore the error function:");

    let test_values = vec![0.0, 0.5, 1.0, 2.0];
    for &x in &test_values {
        let erf_val = erf(x);
        let erfc_val = erfc(x);
        println!("erf({}) = {:.6}, erfc({}) = {:.6}", x, erf_val, x, erfc_val);

        // Verify complementary relationship
        let sum = erf_val + erfc_val;
        println!(
            "  Verification: erf({}) + erfc({}) = {:.6} ≈ 1.0",
            x, x, sum
        );
    }

    // Probability connection
    if profile.learning_preferences.practical_applications {
        println!("\n📈 Connection to Normal Distribution:");
        println!("For a normal distribution N(μ, σ²), the CDF is:");
        println!("Φ(x) = (1/2)[1 + erf((x-μ)/(σ√2))]");

        let mu = 0.0;
        let sigma = 1.0;
        let x = 1.0;
        let prob = 0.5 * (1.0 + erf((x - mu) / (sigma * 2.0_f64.sqrt())));
        println!("\nP(X ≤ 1) for N(0,1) = {:.4} = {:.1}%", prob, prob * 100.0);
    }

    // Advanced topic for higher levels
    if matches!(
        profile.learning_preferences.difficulty_preference,
        DifficultyLevel::Advanced | DifficultyLevel::Expert
    ) {
        println!("\n🎯 Advanced Topic: Faddeeva Function");
        println!("The Faddeeva function w(z) = e^(-z²) erfc(-iz) appears in:");
        println!("  • Plasma physics");
        println!("  • Spectroscopy");
        println!("  • Astrophysics");

        let z = Complex64::new(1.0, 0.5);
        let w_val = faddeeva_complex(z);
        println!("w(1 + 0.5i) = {:.6}", w_val);
    }

    profile.update_mastery("Error Functions", 1.0);
    profile.add_experience(30);

    Ok(())
}

#[allow(dead_code)]
fn orthogonal_polynomial_adaptive_session(
    profile: &mut UserProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Orthogonal Polynomials Mastery");
    println!("==================================\n");

    println!("📖 Orthogonal polynomials are polynomial sequences {{p_n(x)}} such that:");
    println!("   ∫_a^b p_m(x) p_n(x) w(x) dx = δ_mn · h_n");
    println!("where w(x) is a weight function and δ_mn is the Kronecker delta.");

    println!("\n🔑 Important families:");
    println!("   • Legendre: P_n(x) on [-1,1] with w(x) = 1");
    println!("   • Chebyshev: T_n(x) on [-1,1] with w(x) = 1/√(1-x²)");
    println!("   • Hermite: H_n(x) on (-∞,∞) with w(x) = e^(-x²)");
    println!("   • Laguerre: L_n(x) on [0,∞) with w(x) = e^(-x)");

    // Interactive polynomial generation
    println!("\n🧮 Let's generate some polynomials:");

    let x = 0.5;
    println!("For x = {}:", x);

    // Legendre polynomials
    println!("\nLegendre polynomials:");
    for n in 0..=4 {
        let p_val = legendre(n, x);
        println!("  P_{}({}) = {:.6}", n, x, p_val);
    }

    // Chebyshev polynomials
    println!("\nChebyshev polynomials:");
    for n in 0..=4 {
        let t_val = chebyshev(n, x, true);
        println!("  T_{}({}) = {:.6}", n, x, t_val);
    }

    // Orthogonality demonstration
    if profile.learning_preferences.theoretical_focus {
        println!("\n🎯 Orthogonality Verification:");
        println!("Let's verify ∫₋₁¹ P_m(x) P_n(x) dx = 2/(2n+1) δ_mn");

        // Numerical integration example
        let n = 2;
        let m = 3;
        let integral = numerical_orthogonality_check(n, m);
        println!("∫₋₁¹ P_{}(x) P_{}(x) dx ≈ {:.8}", n, m, integral);
        println!("Expected: 0 (since n ≠ m)");
    }

    // Applications
    if profile.learning_preferences.practical_applications {
        println!("\n🔬 Applications:");
        println!("• Legendre: Spherical harmonics, potential theory");
        println!("• Chebyshev: Approximation theory, numerical analysis");
        println!("• Hermite: Quantum harmonic oscillator, probability");
        println!("• Laguerre: Quantum hydrogen atom, statistics");
    }

    // Interactive quiz
    println!("\n❓ Challenge Question:");
    println!("Which orthogonal polynomial family is used in Gaussian quadrature");
    println!("for integrating functions over [-1, 1]?");
    println!("a) Hermite");
    println!("b) Laguerre");
    println!("c) Legendre");
    println!("d) Chebyshev");

    let answer = get_user_input("Your answer (a/b/c/d): ")?;
    let correct = answer.to_lowercase() == "c";

    if correct {
        println!("✅ Correct! Legendre polynomials are used in Gauss-Legendre quadrature.");
        profile.add_experience(35);
        profile.update_mastery("Orthogonal Polynomials", 1.0);
    } else {
        println!(
            "❌ Incorrect. Legendre polynomials are used in Gauss-Legendre quadrature for [-1,1]."
        );
        profile.update_mastery("Orthogonal Polynomials", 0.5);
    }

    Ok(())
}

#[allow(dead_code)]
fn advanced_topic_session(
    profile: &mut UserProfile,
    topic: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎓 Advanced Topic: {}", topic);
    println!("==================={}=", "=".repeat(topic.len()));

    match topic {
        "Wright Functions" => wright_functions_session(profile),
        "Spheroidal Functions" => spheroidal_functions_session(profile),
        "Mathieu Functions" => mathieu_functions_session(profile),
        _ => {
            println!("This advanced topic is under development.");
            println!("Stay tuned for more content!");
            Ok(())
        }
    }
}

#[allow(dead_code)]
fn wright_functions_session(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("📖 Wright functions are generalizations defined by:");
    println!("   W(α,β;z) = Σ_{{n=0}}^∞ z^n / (n! Γ(αn + β))");
    println!("\n🔑 Special cases:");
    println!("   • α=0: Exponential function");
    println!("   • β=1: Mittag-Leffler function");

    // Demonstrate Wright Omega function
    println!("\n🧮 Wright Omega Function:");
    println!("The Wright omega function ω(z) satisfies: ω e^ω = z");

    let z = 2.0;
    let omega_val = wright_omega_real(z, 1e-12).unwrap();
    println!("ω({}) = {:.6}", z, omega_val);

    // Verification
    let verification = omega_val * omega_val.exp();
    println!(
        "Verification: ω({}) × e^ω({}) = {:.6} ≈ {}",
        z, z, verification, z
    );

    profile.update_mastery("Wright Functions", 1.0);
    profile.add_experience(50);

    Ok(())
}

#[allow(dead_code)]
fn spheroidal_functions_session(
    profile: &mut UserProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📖 Spheroidal wave functions are solutions to the spheroidal wave equation");
    println!("in prolate and oblate spheroidal coordinates.");
    println!("\n🔑 Applications:");
    println!("   • Electromagnetic scattering by spheroids");
    println!("   • Acoustic problems");
    println!("   • Quantum mechanics");

    // Example calculation
    let c = 1.0;
    let m = 1;
    let n = 1;

    println!("\n🧮 Example: Prolate spheroidal functions");
    println!("Parameters: c = {}, m = {}, n = {}", c, m, n);

    let cv = pro_cv(m, n, c).unwrap();
    println!("Characteristic value: λ_{}^{} = {:.6}", m, n, cv);

    profile.update_mastery("Spheroidal Functions", 1.0);
    profile.add_experience(45);

    Ok(())
}

#[allow(dead_code)]
fn mathieu_functions_session(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("📖 Mathieu functions are solutions to Mathieu's differential equation:");
    println!("   y'' + (a - 2q cos(2z))y = 0");
    println!("\n🔑 Applications:");
    println!("   • Vibrating elliptical membranes");
    println!("   • Radio wave propagation");
    println!("   • Quantum mechanics in periodic potentials");

    // Example calculation
    let m = 1;
    let q = 2.0;

    println!("\n🧮 Example: Mathieu characteristic values");
    println!("Parameters: m = {}, q = {}", m, q);

    let a_val = mathieu_a(m, q).unwrap();
    let b_val = mathieu_b(m, q).unwrap();
    println!("a_{}({}) = {:.6}", m, q, a_val);
    println!("b_{}({}) = {:.6}", m, q, b_val);

    profile.update_mastery("Mathieu Functions", 1.0);
    profile.add_experience(40);

    Ok(())
}

#[allow(dead_code)]
fn mathematical_proof_walkthrough(
    profile: &mut UserProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📝 Mathematical Proof Walkthrough");
    println!("=================================\n");

    println!("Choose a proof to explore:");
    println!("1. Γ(1/2) = √π");
    println!("2. Stirling's approximation derivation");
    println!("3. Bessel function orthogonality");
    println!("4. Error function series expansion");

    let choice = get_user_input("Enter your choice (1-4): ")?;

    match choice.parse::<u32>() {
        Ok(1) => gamma_half_proof(profile),
        Ok(2) => stirling_approximation_proof(profile),
        Ok(3) => bessel_orthogonality_proof(profile),
        Ok(4) => error_function_series_proof(profile),
        _ => {
            println!("Invalid choice.");
            Ok(())
        }
    }
}

#[allow(dead_code)]
fn gamma_half_proof(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Proof that Γ(1/2) = √π");
    println!("===========================\n");

    let steps = vec![
        (
            "Step 1: Definition",
            "Start with the definition: Γ(1/2) = ∫₀^∞ t^(-1/2) e^(-t) dt",
        ),
        ("Step 2: Substitution", "Let t = u², then dt = 2u du"),
        (
            "Step 3: Transform",
            "Γ(1/2) = ∫₀^∞ (u²)^(-1/2) e^(-u²) · 2u du = 2∫₀^∞ e^(-u²) du",
        ),
        (
            "Step 4: Gaussian integral",
            "We know that ∫₋∞^∞ e^(-u²) du = √π",
        ),
        ("Step 5: Symmetry", "By symmetry, ∫₀^∞ e^(-u²) du = (1/2)√π"),
        (
            "Step 6: Conclusion",
            "Therefore, Γ(1/2) = 2 · (1/2)√π = √π ✓",
        ),
    ];

    for (title, content) in steps {
        println!("📖 {}", title);
        println!("   {}", content);
        get_user_input("Press Enter to continue...")?;
        println!();
    }

    // Verification
    let computed_value = gamma(0.5);
    let expected_value = PI.sqrt();
    println!("🔍 Numerical verification:");
    println!("Γ(1/2) = {:.10}", computed_value);
    println!("√π     = {:.10}", expected_value);
    println!("Error  = {:.2e}", (computed_value - expected_value).abs());

    profile.add_experience(40);
    profile.earn_achievement(Achievement {
        id: "proof_master".to_string(),
        name: "Proof Master".to_string(),
        description: "Completed a mathematical proof walkthrough".to_string(),
        earned_at: Some(std::time::SystemTime::now()),
        difficulty: AchievementDifficulty::Silver,
    });

    Ok(())
}

#[allow(dead_code)]
fn stirling_approximation_proof(
    profile: &mut UserProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Stirling's Approximation: Γ(z) ≈ √(2π/z) (z/e)^z");
    println!("========================================================\n");

    println!("📖 This proof uses the method of steepest descent (saddle-point method)");
    println!("on the integral representation of the gamma function.\n");

    let steps = vec![
        "Start with Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt",
        "Rewrite as Γ(z) = ∫₀^∞ e^((z-1)ln(t) - t) dt",
        "For large z, the integrand has a sharp maximum (saddle point)",
        "Find the saddle point by setting d/dt[(z-1)ln(t) - t] = 0",
        "This gives (z-1)/t - 1 = 0, so t₀ = z-1",
        "Expand around t₀: (z-1)ln(t) - t ≈ (z-1)ln(z-1) - (z-1) - (t-t₀)²/2t₀",
        "The integral becomes approximately Gaussian around t₀",
        "Evaluating gives Γ(z) ≈ √(2π/(z-1)) ((z-1)/e)^(z-1)",
        "For large z, (z-1) ≈ z, yielding Stirling's formula",
    ];

    for (i, step) in steps.iter().enumerate() {
        println!("Step {}: {}", i + 1, step);
        get_user_input("Press Enter to continue...")?;
    }

    // Numerical demonstration
    println!("\n🔍 Numerical verification for large values:");
    let test_values = vec![10.0, 50.0, 100.0];

    for &z in &test_values {
        let exact = gamma(z);
        let stirling = (2.0 * PI / z).sqrt() * (z / std::f64::consts::E).powf(z);
        let error = ((exact - stirling) / exact * 100.0).abs();

        println!(
            "z = {}: Γ(z) = {:.6e}, Stirling ≈ {:.6e}, Error: {:.3}%",
            z, exact, stirling, error
        );
    }

    profile.add_experience(60);
    Ok(())
}

#[allow(dead_code)]
fn bessel_orthogonality_proof(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Bessel Function Orthogonality");
    println!("=================================\n");

    println!("📖 We'll prove that ∫₀¹ x J_ν(α_νm x) J_ν(α_νn x) dx = (δ_mn/2)[J_(ν+1)(α_νm)]²");
    println!("where α_νm are the zeros of J_ν(x).\n");

    println!("This proof uses:");
    println!("1. The differential equation satisfied by Bessel functions");
    println!("2. Integration by parts");
    println!("3. Boundary conditions at x = 0 and x = 1");

    get_user_input("Press Enter to continue with the detailed proof...")?;

    // Numerical demonstration
    println!("\n🔍 Numerical verification:");
    let nu = 0;
    let zeros = vec![
        j0_zeros::<f64>(1).unwrap(),
        j0_zeros::<f64>(2).unwrap(),
        j0_zeros::<f64>(3).unwrap(),
    ];

    for i in 0..zeros.len() {
        for j in i..zeros.len() {
            let integral = numerical_bessel_orthogonality(nu, zeros[i], zeros[j]);
            if i == j {
                let expected = 0.5 * j1(zeros[i]).powi(2);
                println!(
                    "∫₀¹ x J_{}({:.3}x) J_{}({:.3}x) dx = {:.6} (expected: {:.6})",
                    nu, zeros[i], nu, zeros[j], integral, expected
                );
            } else {
                println!(
                    "∫₀¹ x J_{}({:.3}x) J_{}({:.3}x) dx = {:.6} (expected: 0)",
                    nu, zeros[i], nu, zeros[j], integral
                );
            }
        }
    }

    profile.add_experience(55);
    Ok(())
}

#[allow(dead_code)]
fn error_function_series_proof(
    profile: &mut UserProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Error Function Series Expansion");
    println!("===================================\n");

    println!("📖 Prove that erf(x) = (2/√π) Σ_{{n=0}}^∞ (-1)ⁿ x^(2n+1) / (n!(2n+1))");

    let steps = vec![
        "Start with the definition: erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt",
        "Expand e^(-t²) using the power series: e^(-t²) = Σ_{n=0}^∞ (-t²)ⁿ/n!",
        "This gives: e^(-t²) = Σ_{n=0}^∞ (-1)ⁿ t^(2n) / n!",
        "Integrate term by term: ∫₀ˣ t^(2n) dt = x^(2n+1) / (2n+1)",
        "Combine: erf(x) = (2/√π) Σ_{n=0}^∞ (-1)ⁿ x^(2n+1) / (n!(2n+1))",
    ];

    for (i, step) in steps.iter().enumerate() {
        println!("Step {}: {}", i + 1, step);
        get_user_input("Press Enter to continue...")?;
    }

    // Numerical verification
    println!("\n🔍 Series convergence demonstration:");
    let x: f64 = 1.0;
    let exact = erf(x);

    println!("erf({}) = {:.10} (exact)", x, exact);

    let mut series_sum = 0.0;
    let sqrt_pi = PI.sqrt();

    for n in 0..=20 {
        let term = (-1.0_f64).powi(n as i32) * x.powi(2 * n + 1)
            / (factorial(n as u64) as f64 * (2 * n + 1) as f64);
        series_sum += term;
        let series_val = 2.0 / sqrt_pi * series_sum;

        if n % 5 == 4 {
            let error = (series_val - exact).abs();
            println!(
                "n = {:2}: series = {:.10}, error = {:.2e}",
                n, series_val, error
            );
        }
    }

    profile.add_experience(50);
    Ok(())
}

#[allow(dead_code)]
fn interactive_problem_solver(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Interactive Problem Solver");
    println!("=============================\n");

    println!("Choose a problem category:");
    println!("1. 📊 Probability and Statistics");
    println!("2. ⚛️  Physics Applications");
    println!("3. 🔧 Engineering Problems");
    println!("4. 💰 Financial Mathematics");
    println!("5. 🎯 Custom Problem");

    let choice = get_user_input("Enter your choice (1-5): ")?;

    match choice.parse::<u32>() {
        Ok(1) => solve_probability_problems(profile),
        Ok(2) => solve_physics_problems(profile),
        Ok(3) => solve_engineering_problems(profile),
        Ok(4) => solve_financial_problems(profile),
        Ok(5) => solve_custom_problem(profile),
        _ => {
            println!("Invalid choice.");
            Ok(())
        }
    }
}

#[allow(dead_code)]
fn solve_probability_problems(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Probability Problem Solver");
    println!("==============================\n");

    println!("Problem: Quality Control");
    println!("A factory produces items with a 2% defect rate.");
    println!("In a batch of 1000 items, what's the probability of finding:");
    println!("a) Exactly 20 defects?");
    println!("b) At most 25 defects?");
    println!("c) More than 15 defects?");

    let n = 1000;
    let p = 0.02;
    let lambda = n as f64 * p; // Poisson approximation parameter

    println!("\n💡 Solution approach:");
    println!("Since n is large and p is small, we can use Poisson approximation:");
    println!("λ = np = {} × {} = {}", n, p, lambda);

    // Part (a)
    let k = 20;
    let prob_exactly = poisson_pmf(k, lambda);
    println!("\na) P(X = {}) = {:.6}", k, prob_exactly);

    // Part (b)
    let kmax = 25;
    let prob_at_most = poisson_cdf(kmax, lambda);
    println!("b) P(X ≤ {}) = {:.6}", kmax, prob_at_most);

    // Part (c)
    let kmin = 15;
    let prob_more_than = 1.0 - poisson_cdf(kmin, lambda);
    println!("c) P(X > {}) = {:.6}", kmin, prob_more_than);

    println!("\n🎯 Interactive exploration:");
    let user_k = get_user_input("Enter a value k to compute P(X = k): ")?
        .parse::<u32>()
        .map_err(|_| "Invalid number")?;

    let user_prob = poisson_pmf(user_k, lambda);
    println!("P(X = {}) = {:.6}", user_k, user_prob);

    profile.add_experience(35);
    Ok(())
}

#[allow(dead_code)]
fn solve_physics_problems(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚛️ Physics Problem Solver");
    println!("=========================\n");

    println!("Problem: Quantum Harmonic Oscillator");
    println!("Find the first few energy eigenvalues and plot the wavefunctions.");

    println!("\n📖 Background:");
    println!("The time-independent Schrödinger equation for the harmonic oscillator is:");
    println!("[-ℏ²/(2m) d²/dx² + (1/2)mω²x²] ψ(x) = E ψ(x)");
    println!("\nSolutions involve Hermite polynomials:");
    println!("ψ_n(x) = (mω/πℏ)^(1/4) / √(2ⁿn!) × H_n(√(mω/ℏ)x) × exp(-mωx²/2ℏ)");
    println!("E_n = ℏω(n + 1/2)");

    // Calculate energy levels
    let hbar = 1.0; // Reduced Planck constant (in natural units)
    let m = 1.0; // Mass
    let omega = 1.0; // Angular frequency

    println!("\n🔢 Energy levels (in units of ℏω):");
    for n in 0..=4 {
        let energy = (n as f64 + 0.5) * hbar * omega;
        println!("E_{} = {:.1} ℏω", n, energy);
    }

    // Calculate wavefunction values
    println!("\n🌊 Wavefunction values at x = 0:");
    let x = 0.0;
    let alpha = (m * omega / hbar).sqrt();
    let normalization = (alpha / PI.sqrt()).sqrt();

    for n in 0..=3 {
        let hermite_val = hermite(n, alpha.sqrt() * x);
        let exponential = (-0.5 * alpha * x * x).exp();
        let wavefunction = normalization
            / (2.0_f64.powi(n as i32) * factorial(n as u64) as f64).sqrt()
            * hermite_val
            * exponential;
        println!("ψ_{}(0) = {:.6}", n, wavefunction);
    }

    // Interactive exploration
    println!("\n🎯 Interactive:");
    let user_n: u32 = get_user_input("Enter quantum number n: ")?
        .parse()
        .map_err(|_| "Invalid number")?;
    let user_x: f64 = get_user_input("Enter position x: ")?
        .parse()
        .map_err(|_| "Invalid number")?;

    let hermite_val = hermite(user_n as usize, alpha.sqrt() * user_x);
    let exponential = (-0.5 * alpha * user_x * user_x).exp();
    let wavefunction = normalization
        / (2.0_f64.powi(user_n as i32) * factorial(user_n as u64) as f64).sqrt()
        * hermite_val
        * exponential;

    println!("ψ_{}({}) = {:.6}", user_n, user_x, wavefunction);

    profile.add_experience(45);
    Ok(())
}

#[allow(dead_code)]
fn solve_engineering_problems(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Engineering Problem Solver");
    println!("==============================\n");

    println!("Problem: Heat Conduction in a Circular Fin");
    println!("A circular fin with radius R is used for heat dissipation.");
    println!("The temperature distribution involves Bessel functions.");

    println!("\n📖 Background:");
    println!("The heat equation in cylindrical coordinates gives:");
    println!("T(r) = C₁ I₀(mr) + C₂ K₀(mr)");
    println!("where m² = h/(kδ), h = heat transfer coefficient,");
    println!("k = thermal conductivity, δ = fin thickness");

    // Example parameters
    let h = 25.0; // W/(m²·k)
    let k = 200.0; // W/(m·k) for aluminum
    let delta = 0.003; // 3 mm thickness
    let r = 0.05; // 5 cm radius
    let t_base = 100.0; // °C
    let t_inf = 20.0; // °C

    let m: f64 = ((h / (k * delta)) as f64).sqrt();
    println!("\n🔢 Parameters:");
    println!("m = √(h/kδ) = {:.2} m⁻¹", m);

    // Boundary conditions: T(0) = finite, T(R) = T_inf (simplified)
    // This gives C₂ = 0 and determines C₁
    let m_r = m * r;
    let c1 = (t_base - t_inf) / i0(m_r);

    println!("C₁ = {:.2} °C", c1);

    // Temperature profile
    println!("\n🌡️  Temperature distribution:");
    let r_values = vec![0.0, 0.01, 0.02, 0.03, 0.04, 0.05];
    for &r in &r_values {
        let t = t_inf + c1 * i0(m * r);
        println!("T({:.3} m) = {:.1} °C", r, t);
    }

    // Heat transfer rate
    let q = -k * delta * 2.0 * PI * r * c1 * m * i1(m_r);
    println!("\n🔥 Heat transfer rate: {:.1} W", q);

    profile.add_experience(50);
    Ok(())
}

#[allow(dead_code)]
fn solve_financial_problems(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("💰 Financial Mathematics Problem");
    println!("=================================\n");

    println!("Problem: Black-Scholes Option Pricing");
    println!("Calculate the price of a European call option using the Black-Scholes formula.");

    println!("\n📖 Black-Scholes Formula:");
    println!("C = S₀N(d₁) - Ke⁻ʳᵀN(d₂)");
    println!("where:");
    println!("d₁ = [ln(S₀/k) + (r + σ²/2)T] / (σ√T)");
    println!("d₂ = d₁ - σ√T");
    println!("N(x) = standard normal CDF = (1/2)[1 + erf(x/√2)]");

    // Example parameters
    let s0: f64 = 100.0; // Current stock price
    let k: f64 = 105.0; // Strike price
    let r: f64 = 0.05; // Risk-free rate
    let t: f64 = 0.25; // Time to expiration (3 months)
    let sigma: f64 = 0.2; // Volatility

    println!("\n🔢 Parameters:");
    println!("S₀ = ${:.2}", s0);
    println!("K = ${:.2}", k);
    println!("r = {:.1}%", r * 100.0);
    println!("T = {:.2} years", t);
    println!("σ = {:.1}%", sigma * 100.0);

    // Calculate d1 and d2
    let d1 = ((s0 / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    println!("\n🧮 Calculations:");
    println!("d₁ = {:.4}", d1);
    println!("d₂ = {:.4}", d2);

    // Calculate normal CDFs using error function
    let n_d1 = 0.5 * (1.0 + erf(d1 / 2.0_f64.sqrt()));
    let n_d2 = 0.5 * (1.0 + erf(d2 / 2.0_f64.sqrt()));

    println!("N(d₁) = {:.4}", n_d1);
    println!("N(d₂) = {:.4}", n_d2);

    // Calculate option price
    let call_price = s0 * n_d1 - k * (-r * t).exp() * n_d2;

    println!("\n💰 Call Option Price: ${:.2}", call_price);

    // Sensitivity analysis
    println!("\n📊 Greeks:");
    let delta = n_d1;
    let gamma = (-(d1 * d1) / 2.0).exp() / (s0 * sigma * t.sqrt() * (2.0 * PI).sqrt());

    println!("Delta (∂C/∂S) = {:.4}", delta);
    println!("Gamma (∂²C/∂S²) = {:.6}", gamma);

    profile.add_experience(55);
    Ok(())
}

#[allow(dead_code)]
fn solve_custom_problem(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Custom Problem Solver");
    println!("========================\n");

    println!("Describe your problem and I'll help you identify the relevant special functions.");

    let problem_description = get_user_input("Problem description: ")?;

    // Simple keyword matching for demonstration
    let keywords = problem_description.to_lowercase();

    if keywords.contains("probability") || keywords.contains("statistics") {
        println!("\n📊 This sounds like a probability/statistics problem.");
        println!("Relevant functions: erf, erfc, gamma, beta, incomplete gamma");
    } else if keywords.contains("oscillator") || keywords.contains("vibration") {
        println!("\n🌊 This involves oscillatory behavior.");
        println!("Relevant functions: Bessel functions, Mathieu functions");
    } else if keywords.contains("heat") || keywords.contains("diffusion") {
        println!("\n🌡️ This is a heat/diffusion problem.");
        println!("Relevant functions: Bessel functions, error function");
    } else if keywords.contains("quantum") || keywords.contains("wave") {
        println!("\n⚛️ This is a quantum mechanics problem.");
        println!("Relevant functions: Hermite polynomials, spherical harmonics, hypergeometric functions");
    } else {
        println!("\n🔍 Please provide more details about your specific problem.");
        println!("What type of differential equation or physical system are you working with?");
    }

    profile.add_experience(25);
    Ok(())
}

#[allow(dead_code)]
fn visual_function_explorer(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Visual Function Explorer");
    println!("===========================\n");

    if !profile.learning_preferences.visual_learner {
        println!("💡 Tip: Visual representations can help build intuition for special functions!");
        println!("Consider enabling visual learning in your preferences.");
    }

    println!("Choose a function family to explore:");
    println!("1. 🎲 Gamma function");
    println!("2. 🌊 Bessel functions");
    println!("3. 📊 Error function");
    println!("4. 📐 Orthogonal polynomials");

    let choice = get_user_input("Enter your choice (1-4): ")?;

    match choice.parse::<u32>() {
        Ok(1) => explore_gamma_visual(profile),
        Ok(2) => explore_bessel_visual(profile),
        Ok(3) => explore_error_visual(profile),
        Ok(4) => explore_polynomial_visual(profile),
        _ => {
            println!("Invalid choice.");
            Ok(())
        }
    }
}

#[allow(dead_code)]
fn explore_gamma_visual(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎲 Gamma Function Visualization");
    println!("===============================\n");

    println!("📊 ASCII plot of Γ(x) for x ∈ [0.1, 5]:");
    ascii_plot_gamma();

    println!("\n🔍 Key features:");
    println!("• Pole at x = 0 (Γ(x) → +∞ as x → 0⁺)");
    println!("• Minimum at x ≈ 1.46 with Γ(1.46) ≈ 0.886");
    println!("• Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(4) = 6, ...");
    println!("• Rapid growth for large x");

    // Interactive exploration
    println!("\n🎯 Interactive exploration:");
    let x: f64 = get_user_input("Enter x value to evaluate Γ(x): ")?
        .parse()
        .map_err(|_| "Invalid number")?;

    if x > 0.0 {
        let gamma_val = gamma(x);
        println!("Γ({}) = {:.6}", x, gamma_val);

        // Show relationship to factorial if close to integer
        if (x - x.round()).abs() < 0.01 && x >= 1.0 {
            let n = x.round() as u64;
            if n <= 20 {
                let factorial_val = factorial(n - 1);
                println!("Note: Γ({}) = {}! = {}", n, n - 1, factorial_val);
            }
        }
    } else {
        println!("⚠️ Gamma function is not defined for x ≤ 0 (has poles at negative integers)");
    }

    profile.add_experience(30);
    Ok(())
}

#[allow(dead_code)]
fn explore_bessel_visual(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 Bessel Function Visualization");
    println!("=================================\n");

    println!("📊 ASCII plot of J₀(x), J₁(x), Y₀(x) for x ∈ [0, 20]:");
    ascii_plot_bessel();

    println!("\n🔍 Key features:");
    println!("• J₀(0) = 1, J₁(0) = 0");
    println!("• Y₀(x) → -∞ as x → 0⁺ (singular at origin)");
    println!("• Oscillatory behavior for large x with decreasing amplitude");
    println!("• Zeros occur at regular intervals");

    // Show zeros
    println!("\n🎯 Zeros of Bessel functions:");
    let j0_zeros_vec = j0_zeros::<f64>(5);
    let j1_zeros_vec = j1_zeros::<f64>(5);

    println!("First 5 zeros of J₀(x):");
    for (i, &zero) in j0_zeros_vec.iter().enumerate() {
        println!("  α₀,{} = {:.6}", i + 1, zero);
    }

    println!("First 5 zeros of J₁(x):");
    for (i, &zero) in j1_zeros_vec.iter().enumerate() {
        println!("  α₁,{} = {:.6}", i + 1, zero);
    }

    profile.add_experience(35);
    Ok(())
}

#[allow(dead_code)]
fn explore_error_visual(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Error Function Visualization");
    println!("===============================\n");

    println!("📊 ASCII plot of erf(x) and erfc(x) for x ∈ [-3, 3]:");
    ascii_plot_error_function();

    println!("\n🔍 Key features:");
    println!("• erf(0) = 0, erfc(0) = 1");
    println!("• erf(-x) = -erf(x) (odd function)");
    println!("• erf(x) + erfc(x) = 1 for all x");
    println!("• erf(∞) = 1, erfc(∞) = 0");
    println!("• Inflection point at x = 0");

    // Connection to normal distribution
    println!("\n📈 Connection to standard normal distribution:");
    println!("If Z ~ N(0,1), then P(Z ≤ x) = (1/2)[1 + erf(x/√2)]");

    let x: f64 = get_user_input("Enter x to find P(Z ≤ x) for standard normal: ")?
        .parse()
        .map_err(|_| "Invalid number")?;

    let prob = 0.5 * (1.0 + erf(x / 2.0_f64.sqrt()));
    println!("P(Z ≤ {}) = {:.6} = {:.2}%", x, prob, prob * 100.0);

    profile.add_experience(30);
    Ok(())
}

#[allow(dead_code)]
fn explore_polynomial_visual(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("📐 Orthogonal Polynomial Visualization");
    println!("======================================\n");

    println!("Choose polynomial family:");
    println!("1. Legendre polynomials P_n(x)");
    println!("2. Chebyshev polynomials T_n(x)");
    println!("3. Hermite polynomials H_n(x)");

    let choice = get_user_input("Enter choice (1-3): ")?;

    match choice.parse::<u32>() {
        Ok(1) => {
            println!("\n📊 Legendre polynomials P_n(x) on [-1, 1]:");
            ascii_plot_legendre();

            println!("\n🔍 Properties:");
            println!("• P_0(x) = 1");
            println!("• P_1(x) = x");
            println!("• P_2(x) = (3x² - 1)/2");
            println!("• Orthogonal: ∫₋₁¹ P_m(x)P_n(x) dx = 2δ_mn/(2n+1)");
        }
        Ok(2) => {
            println!("\n📊 Chebyshev polynomials T_n(x) on [-1, 1]:");
            ascii_plot_chebyshev();

            println!("\n🔍 Properties:");
            println!("• T_0(x) = 1");
            println!("• T_1(x) = x");
            println!("• T_2(x) = 2x² - 1");
            println!("• |T_n(x)| ≤ 1 for x ∈ [-1, 1]");
        }
        Ok(3) => {
            println!("\n📊 Hermite polynomials H_n(x):");
            ascii_plot_hermite();

            println!("\n🔍 Properties:");
            println!("• H_0(x) = 1");
            println!("• H_1(x) = 2x");
            println!("• H_2(x) = 4x² - 2");
            println!("• Used in quantum harmonic oscillator");
        }
        _ => println!("Invalid choice."),
    }

    profile.add_experience(25);
    Ok(())
}

#[allow(dead_code)]
fn real_world_applications_lab(
    profile: &mut UserProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Real-World Applications Lab");
    println!("==============================\n");

    println!("Choose an application domain:");
    println!("1. 🌐 Signal Processing & Communications");
    println!("2. 🎮 Computer Graphics & Animation");
    println!("3. 🧬 Bioinformatics & Computational Biology");
    println!("4. 🌍 Climate Science & Meteorology");
    println!("5. 🚀 Aerospace Engineering");

    let choice = get_user_input("Enter your choice (1-5): ")?;

    match choice.parse::<u32>() {
        Ok(1) => signal_processing_lab(profile),
        Ok(2) => computer_graphics_lab(profile),
        Ok(3) => bioinformatics_lab(profile),
        Ok(4) => climate_science_lab(profile),
        Ok(5) => aerospace_lab(profile),
        _ => {
            println!("Invalid choice.");
            Ok(())
        }
    }
}

#[allow(dead_code)]
fn signal_processing_lab(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 Signal Processing Laboratory");
    println!("===============================\n");

    println!("Application: Bessel Filter Design");
    println!("Bessel filters provide optimal step response with minimal overshoot.");

    println!("\n📖 Background:");
    println!("Bessel filter transfer function:");
    println!("H(s) = B_n(0) / B_n(s/ω_c)");
    println!("where B_n(s) are Bessel polynomials");

    // Generate Bessel filter coefficients
    let _order = 3;
    let cutoff_freq = 1000.0; // Hz

    println!("\n🔢 3rd-order Bessel filter:");
    println!("Cutoff frequency: {} Hz", cutoff_freq);

    // Bessel polynomial coefficients for n=3: s³ + 6s² + 15s + 15
    let coeffs = vec![1.0, 6.0, 15.0, 15.0];
    println!("B₃(s) = s³ + 6s² + 15s + 15");

    // Calculate group delay (approximately constant for Bessel filters)
    let group_delay = coeffs[1] / (2.0 * PI * cutoff_freq);
    println!("Group delay ≈ {:.6} seconds", group_delay);

    // Demonstrate filter response
    println!("\n📊 Filter characteristics:");
    let frequencies = vec![100.0, 500.0, 1000.0, 2000.0, 5000.0];

    for &freq in &frequencies {
        let normalized_freq = freq / cutoff_freq;
        // Simplified magnitude response approximation
        let magnitude_db = -10.0 * (1.0 + normalized_freq.powi(6)).log10();
        println!("f = {} Hz: |H(jω)| ≈ {:.1} dB", freq, magnitude_db);
    }

    profile.add_experience(40);
    Ok(())
}

#[allow(dead_code)]
fn computer_graphics_lab(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎮 Computer Graphics Laboratory");
    println!("===============================\n");

    println!("Application: Spherical Harmonic Lighting");
    println!("Used for ambient lighting and global illumination in real-time graphics.");

    println!("\n📖 Background:");
    println!("Spherical harmonics represent functions on the sphere:");
    println!("f(θ,φ) = Σ_l Σ_m c_lm Y_l^m(θ,φ)");
    println!("where Y_l^m are spherical harmonic basis functions");

    // Example: lighting representation
    println!("\n💡 Lighting representation:");
    println!("For l=0,1,2 (9 coefficients total):");

    let theta = PI / 4.0; // 45 degrees
    let phi = PI / 3.0; // 60 degrees

    // Calculate first few spherical harmonics
    for l in 0..=2 {
        for m in -(l as i32)..=(l as i32) {
            let y_lm = sph_harm(l, m, theta, phi);
            println!(
                "Y_{}^{}({:.2}, {:.2}) = {:.6}",
                l,
                m,
                theta,
                phi,
                y_lm.unwrap()
            );
        }
    }

    println!("\n🎨 Applications:");
    println!("• Precomputed radiance transfer (PRT)");
    println!("• Environment map representation");
    println!("• Subsurface scattering");
    println!("• Irradiance maps");

    // Interactive component
    let user_theta: f64 = get_user_input("Enter theta (0 to π): ")?
        .parse()
        .map_err(|_| "Invalid number")?;
    let user_phi: f64 = get_user_input("Enter phi (0 to 2π): ")?
        .parse()
        .map_err(|_| "Invalid number")?;

    let y_00 = sph_harm(0, 0, user_theta, user_phi);
    let y_11 = sph_harm(1, 1, user_theta, user_phi);

    println!(
        "Y_0^0({:.2}, {:.2}) = {:.6}",
        user_theta,
        user_phi,
        y_00.unwrap()
    );
    println!(
        "Y_1^1({:.2}, {:.2}) = {:.6}",
        user_theta,
        user_phi,
        y_11.unwrap()
    );

    profile.add_experience(45);
    Ok(())
}

#[allow(dead_code)]
fn bioinformatics_lab(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧬 Bioinformatics Laboratory");
    println!("============================\n");

    println!("Application: Protein Folding Energy Landscapes");
    println!("Statistical mechanics of protein folding involves partition functions.");

    println!("\n📖 Background:");
    println!("The partition function Z relates to the probability of conformations:");
    println!("Z = Σ_i exp(-E_i/kT)");
    println!("where E_i are energy levels and T is temperature");

    // Example: Two-state folding model
    println!("\n🧮 Two-state folding model:");
    let k_b: f64 = 1.381e-23; // Boltzmann constant (J/k)
    let t: f64 = 300.0; // Temperature (k)
    let delta_g: f64 = -20.0 * 1000.0 * 4.184; // -20 kcal/mol to J/mol
    let n_a: f64 = 6.022e23; // Avogadro's number

    let delta_g_per_molecule: f64 = delta_g / n_a;
    let equilibrium_constant: f64 = (-delta_g_per_molecule / (k_b * t)).exp();

    println!("ΔG = {:.1} kJ/mol", delta_g / 1000.0);
    println!("K_eq = exp(-ΔG/RT) = {:.2e}", equilibrium_constant);

    // Folding probability
    let p_folded = equilibrium_constant / (1.0 + equilibrium_constant);
    let p_unfolded = 1.0 / (1.0 + equilibrium_constant);

    println!("P(folded) = {:.4}", p_folded);
    println!("P(unfolded) = {:.4}", p_unfolded);

    // Temperature dependence
    println!("\n🌡️ Temperature dependence:");
    let temperatures = vec![250.0, 275.0, 300.0, 325.0, 350.0];

    for &temp in &temperatures {
        let k_eq: f64 = (-delta_g_per_molecule / (k_b * temp)).exp();
        let p_fold = k_eq / (1.0 + k_eq);
        println!("T = {} K: P(folded) = {:.4}", temp, p_fold);
    }

    profile.add_experience(50);
    Ok(())
}

#[allow(dead_code)]
fn climate_science_lab(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🌍 Climate Science Laboratory");
    println!("=============================\n");

    println!("Application: Radiative Transfer in Earth's Atmosphere");
    println!("Atmospheric radiation involves integral equations with special functions.");

    println!("\n📖 Background:");
    println!("The radiative transfer equation:");
    println!("dI/dτ = I - (1-ω₀)/2 ∫₋₁¹ P(μ,μ')I(μ') dμ'");
    println!("where I is intensity, τ is optical depth, ω₀ is single-scattering albedo");

    // Example: Rayleigh scattering
    println!("\n🌅 Rayleigh scattering phase function:");
    println!("P(cosθ) = (3/4)(1 + cos²θ)");

    let angles = vec![0.0, 30.0, 60.0, 90.0, 120.0, 180.0];

    for &angle_deg in &angles {
        let angle_rad = angle_deg * PI / 180.0;
        let cos_theta = angle_rad.cos();
        let phase_function = 0.75 * (1.0 + cos_theta * cos_theta);
        println!("θ = {}°: P(cosθ) = {:.4}", angle_deg, phase_function);
    }

    // Atmospheric absorption
    println!("\n🌫️ Atmospheric absorption:");
    println!("Beer-Lambert law: I = I₀ exp(-τ)");

    let i_0 = 1360.0; // Solar constant (W/m²)
    let optical_depths = vec![0.0, 0.1, 0.5, 1.0, 2.0];

    for &tau in &optical_depths {
        let intensity: f64 = i_0 * ((-tau) as f64).exp();
        let attenuation = (1.0 - intensity / i_0) * 100.0;
        println!(
            "τ = {}: I = {:.1} W/m² ({:.1}% attenuated)",
            tau, intensity, attenuation
        );
    }

    profile.add_experience(45);
    Ok(())
}

#[allow(dead_code)]
fn aerospace_lab(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Aerospace Engineering Laboratory");
    println!("===================================\n");

    println!("Application: Orbital Mechanics and Spacecraft Trajectories");
    println!("Elliptic integrals appear in orbital mechanics calculations.");

    println!("\n📖 Background:");
    println!("Kepler's equation relates orbital position to time:");
    println!("M = E - e sin(E)");
    println!("where M is mean anomaly, E is eccentric anomaly, e is eccentricity");

    // Example orbit parameters
    let e = 0.3; // Eccentricity
    let a: f64 = 42164.0; // Semi-major axis (km) - geostationary orbit
    let mu = 3.986e5; // Earth's gravitational parameter (km³/s²)

    println!("\n🛰️ Orbit parameters:");
    println!("Semi-major axis: {} km", a);
    println!("Eccentricity: {}", e);

    // Orbital period
    let period = 2.0 * PI * (a.powi(3) / mu).sqrt();
    println!("Orbital period: {:.2} hours", period / 3600.0);

    // Elliptic integral for orbit arc length
    let k = e; // Modulus for elliptic integral
    let elliptic_k = elliptic_k(k);
    let elliptic_e = elliptic_e(k);

    println!("\n📐 Elliptic integrals:");
    println!("K({}) = {:.6}", k, elliptic_k);
    println!("E({}) = {:.6}", k, elliptic_e);

    // Orbit circumference approximation
    let circumference = 4.0 * a * elliptic_e;
    println!("Orbit circumference ≈ {:.0} km", circumference);

    // Velocity at periapsis and apoapsis
    let r_peri = a * (1.0 - e);
    let r_apo = a * (1.0 + e);
    let v_peri = (mu * (2.0 / r_peri - 1.0 / a)).sqrt();
    let v_apo = (mu * (2.0 / r_apo - 1.0 / a)).sqrt();

    println!("\n🚀 Orbital velocities:");
    println!("Periapsis: r = {:.0} km, v = {:.2} km/s", r_peri, v_peri);
    println!("Apoapsis:  r = {:.0} km, v = {:.2} km/s", r_apo, v_apo);

    profile.add_experience(55);
    Ok(())
}

#[allow(dead_code)]
fn mastery_challenge(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🏆 Mastery Challenge");
    println!("====================\n");

    println!("Test your knowledge with progressively difficult challenges!");
    println!("Current level: {}", profile.level);

    let challenge_level = determine_challenge_level(profile);

    match challenge_level {
        1 => basic_mastery_challenge(profile),
        2 => intermediate_mastery_challenge(profile),
        3 => advanced_mastery_challenge(profile),
        _ => expert_mastery_challenge(profile),
    }
}

#[allow(dead_code)]
fn determine_challenge_level(profile: &UserProfile) -> u32 {
    let avg_mastery: f64 = if profile.mastery_scores.is_empty() {
        0.5
    } else {
        profile.mastery_scores.values().sum::<f64>() / profile.mastery_scores.len() as f64
    };

    match profile.level {
        1..=3 => 1,
        4..=7 => {
            if avg_mastery > 0.7 {
                2
            } else {
                1
            }
        }
        8..=15 => {
            if avg_mastery > 0.8 {
                3
            } else {
                2
            }
        }
        _ => {
            if avg_mastery > 0.9 {
                4
            } else {
                3
            }
        }
    }
}

#[allow(dead_code)]
fn basic_mastery_challenge(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Basic Mastery Challenge");
    println!("==========================\n");

    let mut score = 0;
    let total_questions = 5;

    // Question 1
    println!("Question 1/5:");
    println!("What is the value of Γ(3)?");
    let answer1: f64 = get_user_input("Answer: ")?.parse().unwrap_or(0.0);
    if (answer1 - 2.0).abs() < 0.01 {
        println!("✅ Correct!");
        score += 1;
    } else {
        println!("❌ Incorrect. Γ(3) = 2! = 2");
    }

    // Question 2
    println!("\nQuestion 2/5:");
    println!("What is J₀(0)?");
    let answer2: f64 = get_user_input("Answer: ")?.parse().unwrap_or(0.0);
    if (answer2 - 1.0).abs() < 0.01 {
        println!("✅ Correct!");
        score += 1;
    } else {
        println!("❌ Incorrect. J₀(0) = 1");
    }

    // Question 3
    println!("\nQuestion 3/5:");
    println!("What is erf(0)?");
    let answer3: f64 = get_user_input("Answer: ")?.parse().unwrap_or(1.0);
    if answer3.abs() < 0.01 {
        println!("✅ Correct!");
        score += 1;
    } else {
        println!("❌ Incorrect. erf(0) = 0");
    }

    // Question 4
    println!("\nQuestion 4/5:");
    println!("What is P₀(x) for all x? (Legendre polynomial)");
    let answer4: f64 = get_user_input("Answer: ")?.parse().unwrap_or(0.0);
    if (answer4 - 1.0).abs() < 0.01 {
        println!("✅ Correct!");
        score += 1;
    } else {
        println!("❌ Incorrect. P₀(x) = 1 for all x");
    }

    // Question 5
    println!("\nQuestion 5/5:");
    println!("What is the relationship between erf(x) and erfc(x)?");
    println!("a) erf(x) = 1 - erfc(x)");
    println!("b) erf(x) = erfc(x) - 1");
    println!("c) erf(x) = 2 * erfc(x)");
    let answer5 = get_user_input("Answer (a/b/c): ")?;
    if answer5.to_lowercase() == "a" {
        println!("✅ Correct!");
        score += 1;
    } else {
        println!("❌ Incorrect. erf(x) = 1 - erfc(x)");
    }

    println!("\n📊 Challenge Results:");
    println!("Score: {}/{}", score, total_questions);
    let percentage = (score as f64 / total_questions as f64) * 100.0;
    println!("Percentage: {:.1}%", percentage);

    let xp_earned = score * 20;
    profile.add_experience(xp_earned as u32);

    if score == total_questions {
        profile.earn_achievement(Achievement {
            id: "perfect_basic".to_string(),
            name: "Perfect Scholar".to_string(),
            description: "Scored 100% on basic mastery challenge".to_string(),
            earned_at: Some(std::time::SystemTime::now()),
            difficulty: AchievementDifficulty::Bronze,
        });
    }

    Ok(())
}

#[allow(dead_code)]
fn intermediate_mastery_challenge(
    profile: &mut UserProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Intermediate Mastery Challenge");
    println!("=================================\n");

    let mut score = 0;
    let total_questions = 4;

    // Question 1: Complex gamma function
    println!("Question 1/4:");
    println!("Using the reflection formula Γ(z)Γ(1-z) = π/sin(πz),");
    println!("what is Γ(1/2)?");
    let answer1: f64 = get_user_input("Answer: ")?.parse().unwrap_or(0.0);
    let expected = PI.sqrt();
    if (answer1 - expected).abs() / expected < 0.05 {
        println!("✅ Correct! Γ(1/2) = √π ≈ {:.4}", expected);
        score += 1;
    } else {
        println!("❌ Incorrect. Γ(1/2) = √π ≈ {:.4}", expected);
    }

    // Question 2: Bessel function zeros
    println!("\nQuestion 2/4:");
    println!("The first zero of J₀(x) is approximately:");
    println!("a) 2.405");
    println!("b) 3.832");
    println!("c) 5.520");
    let answer2 = get_user_input("Answer (a/b/c): ")?;
    if answer2.to_lowercase() == "a" {
        println!("✅ Correct! First zero ≈ 2.4048");
        score += 1;
    } else {
        println!("❌ Incorrect. First zero of J₀(x) ≈ 2.4048");
    }

    // Question 3: Orthogonal polynomials
    println!("\nQuestion 3/4:");
    println!("Which polynomial family satisfies the recurrence:");
    println!("P_{{n+1}}(x) = (2n+1)xP_n(x) - nP_{{n-1}}(x)");
    println!("a) Chebyshev");
    println!("b) Legendre");
    println!("c) Hermite");
    let answer3 = get_user_input("Answer (a/b/c): ")?;
    if answer3.to_lowercase() == "b" {
        println!("✅ Correct! This is the Legendre recurrence relation");
        score += 1;
    } else {
        println!("❌ Incorrect. This is the Legendre polynomial recurrence");
    }

    // Question 4: Asymptotic approximation
    println!("\nQuestion 4/4:");
    println!("For large n, n! is approximately (Stirling's approximation):");
    println!("a) √(2πn) (n/e)ⁿ");
    println!("b) √(πn) (n/e)ⁿ");
    println!("c) 2√(πn) (n/e)ⁿ");
    let answer4 = get_user_input("Answer (a/b/c): ")?;
    if answer4.to_lowercase() == "a" {
        println!("✅ Correct! Stirling's approximation");
        score += 1;
    } else {
        println!("❌ Incorrect. Stirling: n! ≈ √(2πn) (n/e)ⁿ");
    }

    println!("\n📊 Challenge Results:");
    println!("Score: {}/{}", score, total_questions);
    let percentage = (score as f64 / total_questions as f64) * 100.0;
    println!("Percentage: {:.1}%", percentage);

    let xp_earned = score * 30;
    profile.add_experience(xp_earned as u32);

    if score >= 3 {
        profile.earn_achievement(Achievement {
            id: "intermediate_master".to_string(),
            name: "Intermediate Master".to_string(),
            description: "Scored 75%+ on intermediate challenge".to_string(),
            earned_at: Some(std::time::SystemTime::now()),
            difficulty: AchievementDifficulty::Silver,
        });
    }

    Ok(())
}

#[allow(dead_code)]
fn advanced_mastery_challenge(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Advanced Mastery Challenge");
    println!("=============================\n");

    println!("🧠 This challenge tests deep understanding of special function theory.");

    let mut score = 0;
    let total_questions = 3;

    // Question 1: Wright functions
    println!("Question 1/3:");
    println!("The Wright function W_α,β(z) generalizes which functions when α=0?");
    println!("a) Exponential functions");
    println!("b) Bessel functions");
    println!("c) Gamma functions");
    let answer1 = get_user_input("Answer (a/b/c): ")?;
    if answer1.to_lowercase() == "a" {
        println!("✅ Correct! When α=0, Wright functions reduce to exponential functions");
        score += 1;
    } else {
        println!("❌ Incorrect. Wright functions generalize exponential functions when α=0");
    }

    // Question 2: Spherical harmonics
    println!("\nQuestion 2/3:");
    println!("Spherical harmonics Y_l^m(θ,φ) are eigenfunctions of which operator?");
    println!("a) Laplacian ∇²");
    println!("b) Angular momentum L²");
    println!("c) Gradient ∇");
    let answer2 = get_user_input("Answer (a/b/c): ")?;
    if answer2.to_lowercase() == "b" {
        println!("✅ Correct! They are eigenfunctions of the angular momentum operator L²");
        score += 1;
    } else {
        println!("❌ Incorrect. Spherical harmonics are eigenfunctions of L²");
    }

    // Question 3: Asymptotic analysis
    println!("\nQuestion 3/3:");
    println!("For large |z|, the Wright Bessel function J_ρ,β(z) with ρ > 0 exhibits:");
    println!("a) Exponential decay");
    println!("b) Polynomial growth");
    println!("c) Exponential growth");
    let answer3 = get_user_input("Answer (a/b/c): ")?;
    if answer3.to_lowercase() == "c" {
        println!("✅ Correct! Wright Bessel functions show exponential growth for large |z|");
        score += 1;
    } else {
        println!("❌ Incorrect. Wright Bessel functions grow exponentially for large arguments");
    }

    println!("\n📊 Advanced Challenge Results:");
    println!("Score: {}/{}", score, total_questions);
    let percentage = (score as f64 / total_questions as f64) * 100.0;
    println!("Percentage: {:.1}%", percentage);

    let xp_earned = score * 50;
    profile.add_experience(xp_earned as u32);

    if score == total_questions {
        profile.earn_achievement(Achievement {
            id: "advanced_expert".to_string(),
            name: "Advanced Expert".to_string(),
            description: "Perfect score on advanced mastery challenge".to_string(),
            earned_at: Some(std::time::SystemTime::now()),
            difficulty: AchievementDifficulty::Gold,
        });
    }

    Ok(())
}

#[allow(dead_code)]
fn expert_mastery_challenge(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Expert Mastery Challenge");
    println!("===========================\n");

    println!("🏆 Expert level: Open-ended problems requiring deep mathematical insight");

    println!("\nExpert Problem:");
    println!("Derive the connection between the Riemann zeta function ζ(s) and");
    println!("the gamma function for the functional equation:");
    println!("π^(-s/2) Γ(s/2) ζ(s) = π^(-(1-s)/2) Γ((1-s)/2) ζ(1-s)");

    println!("\n📝 This is a research-level problem. Discuss your approach:");
    let approach = get_user_input("Your approach (or press Enter to see hints): ")?;

    if approach.trim().is_empty() {
        println!("\n💡 Hints:");
        println!("1. Start with the Mellin transform of the theta function");
        println!("2. Use Poisson summation formula");
        println!("3. Apply the functional equation for theta functions");
        println!("4. Relate to the completed zeta function Ξ(s)");
    } else {
        println!("\n🎓 Expert insight recorded. This demonstrates advanced understanding.");
    }

    println!("\n🏆 Expert level engagement recognized!");
    profile.add_experience(100);

    profile.earn_achievement(Achievement {
        id: "expert_researcher".to_string(),
        name: "Expert Researcher".to_string(),
        description: "Engaged with expert-level mathematical problems".to_string(),
        earned_at: Some(std::time::SystemTime::now()),
        difficulty: AchievementDifficulty::Platinum,
    });

    Ok(())
}

#[allow(dead_code)]
fn peer_comparison_mode(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n👥 Peer Comparison Mode");
    println!("========================\n");

    println!("📊 Your Performance vs. Typical Learners");

    // Simulate peer data for demonstration
    let peer_levels = vec![2, 3, 5, 4, 6, 3, 7, 4, 5, 8];
    let peer_avg_level = peer_levels.iter().sum::<u32>() as f64 / peer_levels.len() as f64;

    println!("Your level: {}", profile.level);
    println!("Peer average level: {:.1}", peer_avg_level);

    if profile.level as f64 > peer_avg_level {
        println!("🌟 You're ahead of the curve! Great job!");
    } else if profile.level as f64 > peer_avg_level * 0.8 {
        println!("📈 You're doing well, close to average!");
    } else {
        println!("💪 Room for improvement - keep practicing!");
    }

    // Mastery comparison
    println!("\n📚 Mastery Scores Comparison:");
    let subjects = [
        "Gamma Functions",
        "Bessel Functions",
        "Error Functions",
        "Orthogonal Polynomials",
    ];
    let peer_mastery = [0.72, 0.68, 0.75, 0.65]; // Simulated peer averages

    for (i, subject) in subjects.iter().enumerate() {
        let user_mastery = profile.mastery_scores.get(*subject).unwrap_or(&0.0);
        let peer_avg = peer_mastery[i];

        println!(
            "{}: You: {:.2}, Peers: {:.2}",
            subject, user_mastery, peer_avg
        );

        if user_mastery > &peer_avg {
            println!("  ✅ Above average");
        } else {
            println!("  📖 Opportunity to improve");
        }
    }

    // Recommendations
    println!("\n💡 Personalized Recommendations:");
    let weakest_subject = profile
        .mastery_scores
        .iter()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(subject, _score)| subject.clone());

    if let Some(subject) = weakest_subject {
        println!("Focus on: {}", subject);
        println!("Recommended next session: {}", subject);
    } else {
        println!("Take a fundamentals review to build strong foundations");
    }

    profile.add_experience(20);
    Ok(())
}

#[allow(dead_code)]
fn profile_settings(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚙️ Profile Settings");
    println!("===================\n");

    println!("Current profile for: {}", profile.name);
    println!("Level: {}", profile.level);
    println!("Experience Points: {}", profile.experience_points);
    println!("Current Streak: {}", profile.current_streak);

    println!("\n📊 Learning Preferences:");
    println!(
        "Visual learner: {}",
        profile.learning_preferences.visual_learner
    );
    println!(
        "Theoretical focus: {}",
        profile.learning_preferences.theoretical_focus
    );
    println!(
        "Practical applications: {}",
        profile.learning_preferences.practical_applications
    );
    println!(
        "Difficulty preference: {:?}",
        profile.learning_preferences.difficulty_preference
    );

    println!("\n🏆 Achievements ({}):", profile.achievements.len());
    for achievement in &profile.achievements {
        println!("  🏅 {} - {}", achievement.name, achievement.description);
    }

    println!("\n📈 Performance History:");
    let recent_sessions = profile.performance_history.len().min(5);
    for record in profile
        .performance_history
        .iter()
        .rev()
        .take(recent_sessions)
    {
        println!(
            "  {} - Score: {:.2} - Time: {:.1}s",
            record.topic,
            record.score,
            record.time_taken.as_secs_f64()
        );
    }

    // Settings modification
    println!("\nModify settings? (y/n)");
    let modify = get_user_input("")?;

    if modify.to_lowercase() == "y" {
        println!("\n🔧 Settings Modification:");

        let new_visual = get_yes_noinput("Enable visual learning? (y/n): ")?;
        let new_theoretical = get_yes_noinput("Focus on theory? (y/n): ")?;
        let new_practical = get_yes_noinput("Include practical applications? (y/n): ")?;

        profile.learning_preferences.visual_learner = new_visual;
        profile.learning_preferences.theoretical_focus = new_theoretical;
        profile.learning_preferences.practical_applications = new_practical;

        println!("✅ Settings updated!");
    }

    Ok(())
}

// Helper functions for adaptive learning
#[allow(dead_code)]
fn select_adaptive_topic(profile: &UserProfile) -> String {
    // Find the topic with lowest mastery score
    if profile.mastery_scores.is_empty() {
        return "Gamma Functions".to_string();
    }

    let weakest_topic = profile
        .mastery_scores
        .iter()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(topic, _score)| topic.clone())
        .unwrap_or_else(|| "Gamma Functions".to_string());

    // Introduce new topics based on level
    match profile.level {
        1..=3 => {
            let basics = ["Gamma Functions", "Bessel Functions", "Error Functions"];
            let covered = basics
                .iter()
                .filter(|&&topic| profile.mastery_scores.contains_key(topic))
                .count();

            if covered < basics.len() {
                basics[covered].to_string()
            } else {
                weakest_topic
            }
        }
        4..=7 => {
            let intermediate = [
                "Orthogonal Polynomials",
                "Hypergeometric Functions",
                "Elliptic Functions",
            ];
            let covered = intermediate
                .iter()
                .filter(|&&topic| profile.mastery_scores.contains_key(topic))
                .count();

            if covered < intermediate.len() {
                intermediate[covered].to_string()
            } else {
                weakest_topic
            }
        }
        _ => {
            let advanced = [
                "Wright Functions",
                "Spheroidal Functions",
                "Mathieu Functions",
            ];
            let covered = advanced
                .iter()
                .filter(|&&topic| profile.mastery_scores.contains_key(topic))
                .count();

            if covered < advanced.len() {
                advanced[covered].to_string()
            } else {
                weakest_topic
            }
        }
    }
}

#[allow(dead_code)]
fn check_gamma_achievements(profile: &mut UserProfile, score: f64, hintsused: u32) {
    if score >= 1.0 && hintsused == 0 {
        profile.earn_achievement(Achievement {
            id: "gamma_perfectionist".to_string(),
            name: "Gamma Perfectionist".to_string(),
            description: "Perfect gamma function score without hints".to_string(),
            earned_at: Some(std::time::SystemTime::now()),
            difficulty: AchievementDifficulty::Gold,
        });
    }

    if profile
        .mastery_scores
        .get("Gamma Functions")
        .unwrap_or(&0.0)
        >= &0.9
    {
        profile.earn_achievement(Achievement {
            id: "gamma_master".to_string(),
            name: "Gamma Master".to_string(),
            description: "Achieved 90% mastery in gamma functions".to_string(),
            earned_at: Some(std::time::SystemTime::now()),
            difficulty: AchievementDifficulty::Silver,
        });
    }
}

// Display functions
#[allow(dead_code)]
fn display_personalized_menu(profile: &UserProfile) {
    println!(
        "\n🎓 Welcome back, {}! (Level {}, {} XP)",
        profile.name, profile.level, profile.experience_points
    );

    if profile.current_streak > 0 {
        println!("🔥 Current streak: {} days", profile.current_streak);
    }

    println!("\n📚 Choose your learning path:");
    println!("1. 🧠 Adaptive Learning Session (Recommended)");
    println!("2. 📝 Mathematical Proof Walkthrough");
    println!("3. 🎯 Interactive Problem Solver");
    println!("4. 📈 Visual Function Explorer");
    println!("5. 🔬 Real-World Applications Lab");
    println!("6. 🏆 Mastery Challenge");
    println!("7. 👥 Peer Comparison Mode");
    println!("8. ⚙️ Profile Settings");
    println!("9. Press 'q' to quit");
}

#[allow(dead_code)]
fn display_session_summary(profile: &UserProfile) {
    println!("\n📊 Session Summary");
    println!("==================");
    println!("Name: {}", profile.name);
    println!("Final Level: {}", profile.level);
    println!("Total XP: {}", profile.experience_points);
    println!("Achievements Earned: {}", profile.achievements.len());
    println!("Topics Studied: {}", profile.mastery_scores.len());

    if !profile.mastery_scores.is_empty() {
        let avg_mastery: f64 =
            profile.mastery_scores.values().sum::<f64>() / profile.mastery_scores.len() as f64;
        println!("Average Mastery: {:.1}%", avg_mastery * 100.0);
    }

    println!("\n🎓 Keep exploring the fascinating world of special functions!");
    println!("Remember: Mathematics is not about memorizing formulas,");
    println!("but about understanding the beautiful patterns that govern our universe.");
}

// Visualization functions (ASCII art representations)
#[allow(dead_code)]
fn display_gamma_visualization(x: f64) {
    println!("\n📊 Gamma function visualization around x = {}", x);
    println!("     |");
    println!("  Γ(x)");
    println!("     |   *");
    println!("     |  / \\");
    println!("     | /   \\");
    println!("     |/     \\");
    println!("-----+-------*----- x");
    println!("     |        \\");
    println!("     |         \\");
    println!("Note: Γ(x) has a minimum around x ≈ 1.46");
}

#[allow(dead_code)]
fn display_bessel_visualization(x: f64) {
    println!("\n📊 Bessel function behavior at x = {}", x);
    println!("  J_n(x)");
    println!("     |");
    println!("   1 +     ~~~");
    println!("     |    /   \\");
    println!("   0 +---+-----+----> x");
    println!("     |    \\   /");
    println!("  -1 +     ~~~");
    println!("     |");
    println!("Note: Oscillatory with decreasing amplitude");
}

#[allow(dead_code)]
fn ascii_plot_gamma() {
    println!("  Γ(x)");
    println!("  10 +");
    println!("     |    *");
    println!("   5 +   *");
    println!("     |  *");
    println!("   2 + *");
    println!("   1 +*");
    println!("     |*");
    println!("   0 +--+--+--+--+-> x");
    println!("      1  2  3  4  5");
}

#[allow(dead_code)]
fn ascii_plot_bessel() {
    println!("  J_n(x)");
    println!("   1 +");
    println!("     |*\\  /\\  /");
    println!("   0 +--*--*--*----> x");
    println!("     |   \\/  \\/");
    println!("  -1 +");
    println!("      0    10   20");
    println!("  * = J₀(x), / = J₁(x)");
}

#[allow(dead_code)]
fn ascii_plot_error_function() {
    println!("  erf(x)");
    println!("   1 +      ****");
    println!("     |    **");
    println!("   0 +  **");
    println!("     |**");
    println!("  -1 +****");
    println!("     +--+--+--+--+-> x");
    println!("    -3 -1  0  1  3");
}

#[allow(dead_code)]
fn ascii_plot_legendre() {
    println!("  P_n(x)");
    println!("   1 +  ****    P₀");
    println!("     | /    \\");
    println!("   0 +*------*-> x  P₁");
    println!("     | \\    /");
    println!("  -1 +  ****    P₂");
    println!("     +--+--+--+");
    println!("    -1  0  1");
}

#[allow(dead_code)]
fn ascii_plot_chebyshev() {
    println!("  T_n(x)");
    println!("   1 +*\\  /*\\  /*");
    println!("     | \\/  \\/");
    println!("   0 +--*--*--*-> x");
    println!("     |  /\\  /\\");
    println!("  -1 +*/  \\*/  \\*");
    println!("     +--+--+--+");
    println!("    -1  0  1");
}

#[allow(dead_code)]
fn ascii_plot_hermite() {
    println!("  H_n(x)");
    println!("  10 +    *     *");
    println!("     |   / \\   /");
    println!("   0 +--*---*-*----> x");
    println!("     |       /");
    println!(" -10 +      *");
    println!("     +--+--+--+");
    println!("    -2  0  2");
}

// Utility functions
#[allow(dead_code)]
fn get_user_input(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

#[allow(dead_code)]
fn get_yes_noinput(prompt: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let input = get_user_input(prompt)?;
    Ok(input.to_lowercase().starts_with('y'))
}

// Mathematical utility functions
#[allow(dead_code)]
fn factorial(n: u64) -> u64 {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

#[allow(dead_code)]
fn poisson_pmf(k: u32, lambda: f64) -> f64 {
    (lambda.powi(k as i32) * (-lambda).exp()) / factorial(k as u64) as f64
}

#[allow(dead_code)]
fn poisson_cdf(k: u32, lambda: f64) -> f64 {
    (0..=k).map(|i| poisson_pmf(i, lambda)).sum()
}

#[allow(dead_code)]
fn numerical_orthogonality_check(n: u32, m: u32) -> f64 {
    // Numerical integration of Legendre polynomial orthogonality
    let num_points = 1000;
    let dx = 2.0 / num_points as f64;
    let mut integral = 0.0;

    for i in 0..num_points {
        let x = -1.0 + (i as f64 + 0.5) * dx;
        integral += legendre(n as usize, x) * legendre(m as usize, x) * dx;
    }

    integral
}

#[allow(dead_code)]
fn numerical_bessel_orthogonality(nu: u32, alpha_m: f64, alphan: f64) -> f64 {
    // Numerical integration for Bessel function orthogonality
    let num_points = 1000;
    let dx = 1.0 / num_points as f64;
    let mut integral = 0.0;

    for i in 1..=num_points {
        let x = (i as f64) * dx;
        integral += x * jn(nu, alpha_m * x) * jn(nu, alphan * x) * dx;
    }

    integral
}

// Placeholder for jn function (Bessel function of order nu)
#[allow(dead_code)]
fn jn(nu: u32, x: f64) -> f64 {
    match nu {
        0 => j0(x),
        1 => j1(x),
        _ => {
            // Use recurrence relation for higher orders
            let mut j_prev = j0(x);
            let mut j_curr = j1(x);

            for _ in 2..=nu {
                let j_next = (2.0 * (nu as f64 - 1.0) / x) * j_curr - j_prev;
                j_prev = j_curr;
                j_curr = j_next;
            }

            j_curr
        }
    }
}
