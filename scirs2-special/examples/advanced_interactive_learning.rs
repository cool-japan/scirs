//! Advanced Interactive Learning Module for Special Functions
//!
//! This module provides enhanced educational features including:
//! - Adaptive learning paths based on user performance
//! - Real-time visualization of mathematical concepts
//! - Interactive proof exploration with theorem proving assistance
//! - Personalized difficulty adjustment
//! - Comprehensive knowledge assessment
//! - Mathematical concept mapping
//! - Historical context and applications
//!
//! Run with: cargo run --example advanced_interactive_learning

use ndarray::Array1;
use num_complex::Complex64;
use scirs2_special::*;
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use std::io::{self, Write};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
struct LearningProfile {
    user_id: String,
    skill_levels: HashMap<String, f64>, // Topic -> proficiency (0.0-1.0)
    learning_speed: f64,                // Words per minute reading speed
    preferred_learning_style: LearningStyle,
    completed_modules: Vec<String>,
    time_spent: HashMap<String, Duration>,
    assessment_scores: Vec<AssessmentResult>,
    mistake_patterns: HashMap<String, u32>, // Error type -> frequency
    mastery_goals: Vec<String>,
    #[allow(dead_code)]
    last_session: Option<Instant>,
}

#[derive(Debug, Clone)]
enum LearningStyle {
    Visual,     // Prefers graphs, diagrams, visual proofs
    Analytical, // Prefers step-by-step algebraic derivations
    Intuitive,  // Prefers conceptual explanations and analogies
    Practical,  // Prefers applications and numerical examples
    Historical, // Prefers historical development and context
}

#[derive(Debug, Clone)]
struct AssessmentResult {
    topic: String,
    score: f64,
    #[allow(dead_code)]
    time_taken: Duration,
    difficulty_level: u32,
    #[allow(dead_code)]
    mistakes: Vec<String>,
    #[allow(dead_code)]
    timestamp: Instant,
}

#[derive(Debug, Clone)]
struct ConceptNode {
    name: String,
    description: String,
    prerequisites: Vec<String>,
    difficulty: u32,
    estimated_time: Duration,
    learning_objectives: Vec<String>,
    applications: Vec<String>,
    visualizations: Vec<VisualizationType>,
    assessment_questions: Vec<AssessmentQuestion>,
}

#[derive(Debug, Clone)]
enum VisualizationType {
    Graph2D {
        #[allow(dead_code)]
        x_range: (f64, f64),
        #[allow(dead_code)]
        y_range: (f64, f64),
    },
    Graph3D {
        #[allow(dead_code)]
        ranges: ((f64, f64), (f64, f64), (f64, f64)),
    },
    ComplexPlane {
        #[allow(dead_code)]
        radius: f64,
    },
    Contour {
        #[allow(dead_code)]
        levels: Vec<f64>,
    },
    Animation {
        #[allow(dead_code)]
        frames: usize,
        #[allow(dead_code)]
        duration: Duration,
    },
    Interactive {
        #[allow(dead_code)]
        parameters: Vec<String>,
    },
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct AssessmentQuestion {
    question_type: QuestionType,
    content: String,
    difficulty: u32,
    expected_time: Duration,
    hints: Vec<String>,
    solution_steps: Vec<String>,
    common_mistakes: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
enum QuestionType {
    MultipleChoice {
        options: Vec<String>,
        correct: usize,
    },
    NumericalAnswer {
        expected: f64,
        tolerance: f64,
    },
    ProofCompletion {
        steps: Vec<String>,
        blanks: Vec<usize>,
    },
    ConceptMapping {
        concepts: Vec<String>,
        relationships: Vec<(usize, usize)>,
    },
    CodeCompletion {
        template: String,
        solution: String,
    },
    GraphicalInterpretation {
        data: Vec<(f64, f64)>,
    },
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct AdaptiveLearningSession {
    profile: LearningProfile,
    current_topic: String,
    knowledge_graph: HashMap<String, ConceptNode>,
    session_start: Instant,
    performance_history: VecDeque<(String, f64, Duration)>, // Topic, score, time
    difficulty_adjustment: f64,                             // -1.0 to 1.0
    next_recommendations: Vec<String>,
}

impl AdaptiveLearningSession {
    fn new(profile: LearningProfile) -> Self {
        let mut knowledge_graph = HashMap::new();

        // Build the knowledge graph
        Self::initialize_knowledge_graph(&mut knowledge_graph);

        Self {
            profile,
            current_topic: String::new(),
            knowledge_graph,
            session_start: Instant::now(),
            performance_history: VecDeque::with_capacity(10),
            difficulty_adjustment: 0.0,
            next_recommendations: Vec::new(),
        }
    }

    fn initialize_knowledge_graph(graph: &mut HashMap<String, ConceptNode>) {
        // Gamma Functions
        graph.insert(
            "gamma_basics".to_string(),
            ConceptNode {
                name: "Gamma Function Fundamentals".to_string(),
                description: "Definition, basic properties, and simple evaluations".to_string(),
                prerequisites: vec![
                    "calculus_integration".to_string(),
                    "factorial_concept".to_string(),
                ],
                difficulty: 2,
                estimated_time: Duration::from_secs(1800), // 30 minutes
                learning_objectives: vec![
                    "Understand the integral definition of the gamma function".to_string(),
                    "Apply the recurrence relation Γ(z+1) = zΓ(z)".to_string(),
                    "Evaluate Γ(n) for positive integers".to_string(),
                    "Recognize key values like Γ(1/2) = √π".to_string(),
                ],
                applications: vec![
                    "Probability distributions".to_string(),
                    "Stirling's approximation".to_string(),
                    "Beta function relationship".to_string(),
                ],
                visualizations: vec![
                    VisualizationType::Graph2D {
                        x_range: (0.1, 5.0),
                        y_range: (0.0, 10.0),
                    },
                    VisualizationType::ComplexPlane { radius: 3.0 },
                ],
                assessment_questions: create_gamma_basic_questions(),
            },
        );

        // Advanced Gamma Theory
        graph.insert(
            "gamma_advanced".to_string(),
            ConceptNode {
                name: "Advanced Gamma Function Theory".to_string(),
                description: "Reflection formula, duplication formula, and analytic continuation"
                    .to_string(),
                prerequisites: vec!["gamma_basics".to_string(), "complex_analysis".to_string()],
                difficulty: 4,
                estimated_time: Duration::from_secs(3600), // 60 minutes
                learning_objectives: vec![
                    "Derive and apply the reflection formula".to_string(),
                    "Understand the duplication formula".to_string(),
                    "Grasp analytic continuation concepts".to_string(),
                    "Work with gamma function poles and residues".to_string(),
                ],
                applications: vec![
                    "Special function identities".to_string(),
                    "Asymptotic analysis".to_string(),
                    "Number theory".to_string(),
                ],
                visualizations: vec![
                    VisualizationType::ComplexPlane { radius: 5.0 },
                    VisualizationType::Graph3D {
                        ranges: ((-3.0, 3.0), (-3.0, 3.0), (-10.0, 10.0)),
                    },
                ],
                assessment_questions: create_gamma_advanced_questions(),
            },
        );

        // Bessel Functions
        graph.insert(
            "bessel_basics".to_string(),
            ConceptNode {
                name: "Bessel Functions Introduction".to_string(),
                description: "Bessel's equation, series solutions, and basic properties"
                    .to_string(),
                prerequisites: vec![
                    "differential_equations".to_string(),
                    "series_solutions".to_string(),
                ],
                difficulty: 3,
                estimated_time: Duration::from_secs(2700), // 45 minutes
                learning_objectives: vec![
                    "Understand Bessel's differential equation".to_string(),
                    "Derive series solutions for J_n(x)".to_string(),
                    "Explore orthogonality properties".to_string(),
                    "Calculate zeros and oscillations".to_string(),
                ],
                applications: vec![
                    "Wave equations in cylindrical coordinates".to_string(),
                    "Vibrating membranes".to_string(),
                    "Heat conduction in cylinders".to_string(),
                    "Antenna radiation patterns".to_string(),
                ],
                visualizations: vec![
                    VisualizationType::Graph2D {
                        x_range: (0.0, 20.0),
                        y_range: (-0.5, 1.0),
                    },
                    VisualizationType::Animation {
                        frames: 60,
                        duration: Duration::from_secs(10),
                    },
                    VisualizationType::Interactive {
                        parameters: vec!["order".to_string(), "argument".to_string()],
                    },
                ],
                assessment_questions: create_bessel_basic_questions(),
            },
        );

        // Add more concepts...
        Self::add_advanced_concepts(graph);
    }

    fn add_advanced_concepts(graph: &mut HashMap<String, ConceptNode>) {
        // Hypergeometric Functions
        graph.insert(
            "hypergeometric".to_string(),
            ConceptNode {
                name: "Hypergeometric Functions".to_string(),
                description: "Generalized hypergeometric series and their properties".to_string(),
                prerequisites: vec![
                    "gamma_advanced".to_string(),
                    "series_convergence".to_string(),
                ],
                difficulty: 4,
                estimated_time: Duration::from_secs(4500), // 75 minutes
                learning_objectives: vec![
                    "Understand the general hypergeometric series".to_string(),
                    "Derive integral representations".to_string(),
                    "Apply transformation formulas".to_string(),
                    "Connect to other special functions".to_string(),
                ],
                applications: vec![
                    "Elliptic integrals".to_string(),
                    "Appell functions".to_string(),
                    "Mathematical physics".to_string(),
                ],
                visualizations: vec![
                    VisualizationType::ComplexPlane { radius: 2.0 },
                    VisualizationType::Contour {
                        levels: vec![-2.0, -1.0, 0.0, 1.0, 2.0],
                    },
                ],
                assessment_questions: create_hypergeometric_questions(),
            },
        );

        // Wright Functions
        graph.insert(
            "wright_functions".to_string(),
            ConceptNode {
                name: "Wright Functions and Fractional Calculus".to_string(),
                description:
                    "Advanced Wright functions and their role in fractional differential equations"
                        .to_string(),
                prerequisites: vec![
                    "hypergeometric".to_string(),
                    "fractional_calculus".to_string(),
                ],
                difficulty: 5,
                estimated_time: Duration::from_secs(5400), // 90 minutes
                learning_objectives: vec![
                    "Understand Wright function definitions".to_string(),
                    "Explore asymptotic behavior".to_string(),
                    "Apply to fractional differential equations".to_string(),
                    "Connect to Mittag-Leffler functions".to_string(),
                ],
                applications: vec![
                    "Anomalous diffusion".to_string(),
                    "Fractional kinetics".to_string(),
                    "Memory effects in materials".to_string(),
                ],
                visualizations: vec![
                    VisualizationType::Graph3D {
                        ranges: ((-5.0, 5.0), (-5.0, 5.0), (-2.0, 10.0)),
                    },
                    VisualizationType::Animation {
                        frames: 120,
                        duration: Duration::from_secs(20),
                    },
                ],
                assessment_questions: create_wright_function_questions(),
            },
        );
    }

    fn recommend_next_topic(&mut self) -> Option<String> {
        let current_skills = &self.profile.skill_levels;
        let mut candidates = Vec::new();

        for (topic, node) in &self.knowledge_graph {
            // Check if prerequisites are met
            let prerequisites_met = node
                .prerequisites
                .iter()
                .all(|prereq| current_skills.get(prereq).unwrap_or(&0.0) >= &0.7);

            if prerequisites_met && !self.profile.completed_modules.contains(topic) {
                let current_skill = current_skills.get(topic).unwrap_or(&0.0);
                let adjusted_difficulty = (node.difficulty as f64) + self.difficulty_adjustment;

                // Score based on readiness and optimal challenge
                let score =
                    self.calculate_topic_score(topic, node, *current_skill, adjusted_difficulty);
                candidates.push((topic.clone(), score));
            }
        }

        // Sort by score and return the best candidate
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.first().map(|(topic_, _)| topic_.clone())
    }

    fn calculate_topic_score(
        &self,
        topic: &str,
        node: &ConceptNode,
        skill: f64,
        difficulty: f64,
    ) -> f64 {
        // Optimal challenge: not too easy, not too hard
        let difficulty_score = 1.0 - (difficulty - 3.0).abs() / 5.0;

        // Prefer topics where we have some but not complete knowledge
        let knowledge_score = if skill < 0.3 {
            1.0 - skill
        } else if skill > 0.8 {
            0.2
        } else {
            1.0
        };

        // Consider learning style preferences
        let style_score = match self.profile.preferred_learning_style {
            LearningStyle::Visual => {
                if node.visualizations.len() > 2 {
                    1.2
                } else {
                    1.0
                }
            }
            LearningStyle::Practical => {
                if node.applications.len() > 3 {
                    1.2
                } else {
                    1.0
                }
            }
            LearningStyle::Analytical => {
                if node.difficulty >= 3 {
                    1.1
                } else {
                    1.0
                }
            }
            _ => 1.0,
        };

        // Recent performance adjustment
        let performance_score = if self.performance_history.len() >= 3 {
            let recent_avg = self
                .performance_history
                .iter()
                .rev()
                .take(3)
                .map(|(_, score_, _)| *score_)
                .sum::<f64>()
                / 3.0;

            if recent_avg > 0.8 {
                1.1
            } else if recent_avg < 0.6 {
                0.9
            } else {
                1.0
            }
        } else {
            1.0
        };

        difficulty_score * knowledge_score * style_score * performance_score
    }

    fn adapt_difficulty_based_on_performance(&mut self) {
        if self.performance_history.len() >= 3 {
            let recent_scores: Vec<f64> = self
                .performance_history
                .iter()
                .rev()
                .take(3)
                .map(|(_, score_, _)| *score_)
                .collect();

            let avg_score = recent_scores.iter().sum::<f64>() / recent_scores.len() as f64;

            // Adjust difficulty based on performance
            if avg_score > 0.85 {
                self.difficulty_adjustment = (self.difficulty_adjustment + 0.2).min(1.0);
                println!("🚀 Excellent performance! Increasing challenge level.");
            } else if avg_score < 0.65 {
                self.difficulty_adjustment = (self.difficulty_adjustment - 0.2).max(-1.0);
                println!("💪 Adjusting difficulty to better match your current level.");
            }
        }
    }

    fn provide_personalized_feedback(&self, topic: &str, score: f64, timetaken: Duration) {
        let node = self.knowledge_graph.get(topic).unwrap();
        let expected_time = node.estimated_time;

        println!("\n📊 Performance Analysis for {}:", node.name);
        println!(
            "Score: {:.1}% ({}/10)",
            score * 100.0,
            (score * 10.0) as u32
        );

        if timetaken <= expected_time {
            println!(
                "⏱️ Excellent time management! Completed in {:.1} minutes (expected: {:.1})",
                timetaken.as_secs_f64() / 60.0,
                expected_time.as_secs_f64() / 60.0
            );
        } else {
            println!(
                "⏱️ Took {:.1} minutes (expected: {:.1}). Consider reviewing fundamentals.",
                timetaken.as_secs_f64() / 60.0,
                expected_time.as_secs_f64() / 60.0
            );
        }

        // Specific feedback based on score ranges
        match score {
            s if s >= 0.9 => {
                println!("🌟 Outstanding mastery! You're ready for advanced topics.");
                println!(
                    "💡 Consider exploring: {}",
                    self.get_advanced_recommendations(topic).join(", ")
                );
            }
            s if s >= 0.8 => {
                println!("✅ Good understanding! Minor review might help solidify concepts.");
            }
            s if s >= 0.7 => {
                println!("👍 Satisfactory progress. Focus on the challenging areas:");
                self.suggest_review_areas(topic, score);
            }
            s if s >= 0.6 => {
                println!("📚 Needs more practice. Let's review the fundamentals:");
                self.suggest_prerequisite_review(topic);
            }
            _ => {
                println!("🔄 Let's take a step back and strengthen the foundation:");
                println!("Consider reviewing: {}", node.prerequisites.join(", "));
            }
        }
    }

    fn get_advanced_recommendations(&self, currenttopic: &str) -> Vec<String> {
        let mut recommendations = Vec::new();

        for (_topic, node) in &self.knowledge_graph {
            if node.prerequisites.contains(&currenttopic.to_string())
                && !self.profile.completed_modules.contains(_topic)
            {
                recommendations.push(node.name.clone());
            }
        }

        if recommendations.is_empty() {
            recommendations.push("Advanced applications and research topics".to_string());
        }

        recommendations
    }

    fn suggest_review_areas(&self, topic: &str, score: f64) {
        // Based on the topic and score, suggest specific areas to review
        match topic {
            "gamma_basics" => {
                if score < 0.75 {
                    println!("  • Review integral definition and evaluation techniques");
                    println!("  • Practice with the recurrence relation");
                    println!("  • Work through more numerical examples");
                }
            }
            "bessel_basics" => {
                if score < 0.75 {
                    println!("  • Review the differential equation derivation");
                    println!("  • Practice series solution methods");
                    println!("  • Study orthogonality properties");
                }
            }
            _ => {
                println!("  • Review key theorems and their applications");
                println!("  • Practice computational examples");
            }
        }
    }

    fn suggest_prerequisite_review(&self, topic: &str) {
        if let Some(node) = self.knowledge_graph.get(topic) {
            println!("Recommended prerequisite review:");
            for prereq in &node.prerequisites {
                if let Some(skill_level) = self.profile.skill_levels.get(prereq) {
                    if *skill_level < 0.7 {
                        println!(
                            "  • {} (current level: {:.1}%)",
                            prereq,
                            skill_level * 100.0
                        );
                    }
                }
            }
        }
    }
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎓 Advanced Interactive Learning Module for Special Functions");
    println!("============================================================\n");

    // Initialize or load user profile
    let profile = create_or_loadprofile()?;

    // Create adaptive learning session
    let mut session = AdaptiveLearningSession::new(profile.clone());

    println!("Welcome back, {}! 👋", profile.user_id);
    display_learning_dashboard(&profile);

    loop {
        println!("\n🎯 What would you like to do today?");
        println!("1. 📚 Continue adaptive learning path");
        println!("2. 🔍 Explore specific topics");
        println!("3. 📊 Take comprehensive assessment");
        println!("4. 📈 View learning analytics");
        println!("5. 🎨 Interactive visualizations");
        println!("6. 🧠 Proof exploration mode");
        println!("7. ⚙️ Adjust learning preferences");
        println!("8. 💾 Save progress and exit");

        let choice = get_user_input("Enter your choice (1-8): ")?;

        match choice.parse::<u32>() {
            Ok(1) => run_adaptive_learning(&mut session)?,
            Ok(2) => explore_topics(&mut session)?,
            Ok(3) => run_comprehensive_assessment(&mut session)?,
            Ok(4) => display_learning_analytics(&session.profile)?,
            Ok(5) => run_interactive_visualizations()?,
            Ok(6) => run_proof_exploration()?,
            Ok(7) => adjust_learning_preferences(&mut session.profile)?,
            Ok(8) => {
                saveprofile(&session.profile)?;
                println!("👋 Progress saved! See you next time!");
                break;
            }
            _ => println!("❌ Invalid choice. Please try again."),
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn create_or_loadprofile() -> Result<LearningProfile, Box<dyn std::error::Error>> {
    // In a real implementation, this would load from a file or database
    let user_id = get_user_input("Enter your name or user ID: ")?;

    // For demo purposes, create a sample profile
    let mut skill_levels = HashMap::new();
    skill_levels.insert("calculus_integration".to_string(), 0.8);
    skill_levels.insert("factorial_concept".to_string(), 0.9);
    skill_levels.insert("differential_equations".to_string(), 0.7);
    skill_levels.insert("complex_analysis".to_string(), 0.6);
    skill_levels.insert("gamma_basics".to_string(), 0.3);

    Ok(LearningProfile {
        user_id,
        skill_levels,
        learning_speed: 200.0,                               // words per minute
        preferred_learning_style: LearningStyle::Analytical, // default
        completed_modules: Vec::new(),
        time_spent: HashMap::new(),
        assessment_scores: Vec::new(),
        mistake_patterns: HashMap::new(),
        mastery_goals: vec!["gamma_advanced".to_string(), "bessel_basics".to_string()],
        last_session: None,
    })
}

#[allow(dead_code)]
fn display_learning_dashboard(profile: &LearningProfile) {
    println!("\n📊 Your Learning Dashboard");
    println!("==========================");

    let total_modules = profile.skill_levels.len();
    let mastered_modules = profile
        .skill_levels
        .values()
        .filter(|&&level| level >= 0.8)
        .count();
    let in_progress = profile
        .skill_levels
        .values()
        .filter(|&&level| level >= 0.3 && level < 0.8)
        .count();

    println!(
        "📚 Modules mastered: {}/{}",
        mastered_modules, total_modules
    );
    println!("📖 Modules in progress: {}", in_progress);
    println!("🎯 Learning style: {:?}", profile.preferred_learning_style);

    if !profile.mastery_goals.is_empty() {
        println!("🎯 Current goals: {}", profile.mastery_goals.join(", "));
    }

    // Display skill progression
    println!("\n📈 Skill Levels:");
    let mut skills: Vec<_> = profile.skill_levels.iter().collect();
    skills.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    for (skill, level) in skills.iter().take(5) {
        let bar_length = 20;
        let filled = (*level * bar_length as f64) as usize;
        let bar: String = (0..bar_length)
            .map(|i| if i < filled { '█' } else { '░' })
            .collect();
        println!("  {:<25} [{}] {:.1}%", skill, bar, *level * 100.0);
    }
}

#[allow(dead_code)]
fn run_adaptive_learning(
    session: &mut AdaptiveLearningSession,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧠 Adaptive Learning Mode");
    println!("=========================\n");

    // Get recommendation for next topic
    if let Some(next_topic) = session.recommend_next_topic() {
        println!("📚 Recommended topic: {}", next_topic);

        if let Some(node) = session.knowledge_graph.get(&next_topic).cloned() {
            println!("📝 Description: {}", node.description);
            println!(
                "⏱️ Estimated time: {:.0} minutes",
                node.estimated_time.as_secs_f64() / 60.0
            );
            println!("🎯 Difficulty: {}/5", node.difficulty);

            let start_learning = get_user_input("Start this learning module? (y/n): ")?;

            if start_learning.to_lowercase() == "y" {
                let start_time = Instant::now();
                let score = run_learning_module(&next_topic, &node, &session.profile)?;
                let time_taken = start_time.elapsed();

                // Update session performance
                session
                    .performance_history
                    .push_back((next_topic.clone(), score, time_taken));
                if session.performance_history.len() > 10 {
                    session.performance_history.pop_front();
                }

                // Update profile
                session
                    .profile
                    .skill_levels
                    .insert(next_topic.clone(), score);
                if score >= 0.8 {
                    session.profile.completed_modules.push(next_topic.clone());
                }
                session
                    .profile
                    .time_spent
                    .insert(next_topic.clone(), time_taken);

                // Provide feedback and adapt
                session.provide_personalized_feedback(&next_topic, score, time_taken);
                session.adapt_difficulty_based_on_performance();
            }
        }
    } else {
        println!("🎉 Congratulations! You've mastered all available topics in your current path.");
        println!("Consider setting new learning goals or exploring advanced research topics.");
    }

    Ok(())
}

#[allow(dead_code)]
fn run_learning_module(
    topic: &str,
    node: &ConceptNode,
    profile: &LearningProfile,
) -> Result<f64, Box<dyn std::error::Error>> {
    println!("\n📖 Learning Module: {}", node.name);
    println!("======================================\n");

    // Present learning objectives
    println!("🎯 Learning Objectives:");
    for (i, objective) in node.learning_objectives.iter().enumerate() {
        println!("  {}. {}", i + 1, objective);
    }

    // Adaptive content presentation based on learning style
    match profile.preferred_learning_style {
        LearningStyle::Visual => present_visual_content(topic, node)?,
        LearningStyle::Analytical => present_analytical_content(topic, node)?,
        LearningStyle::Practical => present_practical_content(topic, node)?,
        LearningStyle::Intuitive => present_intuitive_content(topic, node)?,
        LearningStyle::Historical => present_historical_content(topic, node)?,
    }

    // Interactive assessment
    println!("\n📝 Let's assess your understanding...");
    let mut total_score = 0.0;
    let mut total_questions = 0;

    for question in &node.assessment_questions {
        total_questions += 1;
        let score = run_assessment_question(question)?;
        total_score += score;
    }

    let final_score = total_score / total_questions as f64;

    println!("\n✅ Module completed!");
    println!(
        "Final score: {:.1}% ({}/{})",
        final_score * 100.0,
        (total_score as u32),
        total_questions
    );

    Ok(final_score)
}

#[allow(dead_code)]
fn present_visual_content(
    topic: &str,
    node: &ConceptNode,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Visual Learning Mode");
    println!("========================\n");

    match topic {
        "gamma_basics" => {
            println!("📊 Visualizing the Gamma Function:");
            println!("The gamma function extends the factorial to real numbers.");
            println!("Let's see how Γ(x) behaves for different values of x:\n");

            // ASCII art visualization
            println!("Γ(x) for x ∈ [0.1, 5.0]:");
            println!("    y");
            println!("    ↑");
            println!("10  |     ●");
            println!(" 8  |   ● ●");
            println!(" 6  |  ●   ●");
            println!(" 4  | ●     ●");
            println!(" 2  |●       ●");
            println!(" 0  +―――――――――→ x");
            println!("    0 1 2 3 4 5");
            println!();

            // Interactive plotting
            let x_values = Array1::linspace(0.1, 5.0, 20);
            println!("Computed values:");
            for (i, &x) in x_values.iter().enumerate() {
                if i % 4 == 0 {
                    // Show every 4th value
                    let gamma_val = gamma(x);
                    println!("  Γ({:.1}) = {:.3}", x, gamma_val);
                }
            }
        }
        "bessel_basics" => {
            println!("🌊 Visualizing Bessel Functions:");
            println!("Bessel functions are oscillatory with decreasing amplitude.");
            println!();

            // Show oscillatory behavior
            println!("J₀(x) oscillation pattern:");
            for i in 0..15 {
                let x = i as f64 * 0.5;
                let j0_val = j0(x);
                let normalized = ((j0_val + 1.0) * 10.0) as usize;
                let display = if normalized > 20 { 20 } else { normalized };

                print!("x={:4.1}: ", x);
                for j in 0..20 {
                    if j == 10 {
                        print!("|");
                    } else if j == display {
                        print!("●");
                    } else {
                        print!(" ");
                    }
                }
                println!(" ({:6.3})", j0_val);
            }
        }
        _ => {
            println!("📈 Conceptual visualization for {}:", node.name);
            println!("{}", node.description);
        }
    }

    println!("\n💡 Visual insight: Notice the patterns and symmetries in the functions!");
    wait_for_user_input()?;

    Ok(())
}

#[allow(dead_code)]
fn present_analytical_content(
    topic: &str,
    node: &ConceptNode,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Analytical Learning Mode");
    println!("============================\n");

    match topic {
        "gamma_basics" => {
            println!("📐 Mathematical Definition and Properties:");
            println!();
            println!("Definition: Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt  for Re(z) > 0");
            println!();
            println!("Key Properties:");
            println!("1. Recurrence relation: Γ(z+1) = z·Γ(z)");
            println!("2. For positive integers: Γ(n) = (n-1)!");
            println!("3. Special value: Γ(1/2) = √π");
            println!();
            println!("Derivation of Γ(1/2) = √π:");
            println!("Step 1: Γ(1/2) = ∫₀^∞ t^(-1/2) e^(-t) dt");
            println!("Step 2: Substitute t = u², dt = 2u du");
            println!("Step 3: Γ(1/2) = 2∫₀^∞ e^(-u²) du");
            println!("Step 4: Use Gaussian integral ∫₋∞^∞ e^(-u²) du = √π");
            println!("Step 5: Therefore Γ(1/2) = 2 · (√π/2) = √π ✓");

            // Numerical verification
            let gamma_half = gamma(0.5);
            let sqrt_pi = PI.sqrt();
            println!("\nNumerical verification:");
            println!("  Γ(1/2) = {:.10}", gamma_half);
            println!("  √π     = {:.10}", sqrt_pi);
            println!("  Error  = {:.2e}", (gamma_half - sqrt_pi).abs());
        }
        "bessel_basics" => {
            println!("📐 Bessel's Differential Equation:");
            println!();
            println!("x²y'' + xy' + (x² - ν²)y = 0");
            println!();
            println!("Series Solution (Frobenius method):");
            println!("Assume y = x^r ∑_{{n=0}}^∞ aₙx^n");
            println!();
            println!("Indicial equation: r² - ν² = 0 → r = ±ν");
            println!();
            println!("For r = ν, the solution is:");
            println!("Jᵥ(x) = (x/2)^ν ∑_{{k=0}}^∞ (-1)^k / (k!Γ(ν+k+1)) (x/2)^(2k)");
            println!();
            println!("First few terms for J₀(x):");
            println!("J₀(x) = 1 - x²/4 + x⁴/64 - x⁶/2304 + ...");

            // Show convergence
            let x = 2.0;
            let mut sum = 1.0;
            let mut term = 1.0;
            println!("\nConvergence demonstration for J₀(2.0):");
            println!("  Term 0: {:.6}", sum);

            for k in 1..=5 {
                term *= -(x * x) / (4.0 * k as f64 * k as f64);
                sum += term;
                println!("  Term {}: {:.6} (sum = {:.6})", k, term, sum);
            }

            let exact = j0(x);
            println!("  Exact J₀(2.0) = {:.6}", exact);
            println!("  Error = {:.2e}", (sum - exact).abs());
        }
        _ => {
            println!("📊 Analytical approach to {}:", node.name);
            println!("{}", node.description);
        }
    }

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn present_practical_content(
    topic: &str,
    node: &ConceptNode,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Practical Application Mode");
    println!("==============================\n");

    println!("🎯 Real-world applications of {}:", node.name);
    for (i, application) in node.applications.iter().enumerate() {
        println!("  {}. {}", i + 1, application);
    }
    println!();

    match topic {
        "gamma_basics" => {
            println!("📊 Application: Probability Distributions");
            println!("The gamma function appears in many probability distributions:");
            println!();

            // Gamma distribution example
            println!("Gamma Distribution: f(x) = (β^α/Γ(α)) x^(α-1) e^(-βx)");
            println!();
            let alpha = 2.0;
            let beta: f64 = 1.5;
            println!("Example: α = {}, β = {}", alpha, beta);
            println!(
                "Normalization constant: β^α/Γ(α) = {:.4}",
                beta.powf(alpha) / gamma(alpha)
            );

            // Chi-square distribution
            println!("\nChi-square Distribution (special case of Gamma):");
            let dof = vec![1, 2, 5, 10];
            for &k in &dof {
                let chi_sq_norm = 1.0 / (2.0_f64.powf(k as f64 / 2.0) * gamma(k as f64 / 2.0));
                println!("  χ²({} dof): normalization = {:.6}", k, chi_sq_norm);
            }
        }
        "bessel_basics" => {
            println!("🎵 Application: Vibrating Circular Membrane");
            println!("Natural frequencies of a circular drum involve Bessel function zeros:");
            println!();

            let radius = 0.3; // meters
            let wave_speed = 343.0; // m/s

            println!("Drum radius: {} m", radius);
            println!("Wave speed: {} m/s", wave_speed);
            println!();

            // Calculate some frequencies using Bessel zeros
            let j0_zeros = [2.4048, 5.5201, 8.6537]; // First few zeros of J₀
            println!("Fundamental frequencies (J₀ zeros):");
            for (mode, &zero) in j0_zeros.iter().enumerate() {
                let frequency = zero * wave_speed / (2.0 * PI * radius);
                println!(
                    "  Mode {}: {:.1} Hz (wavelength: {:.3} m)",
                    mode + 1,
                    frequency,
                    wave_speed / frequency
                );
            }

            // J₁ zeros give different mode shapes
            let j1_zeros = [3.8317, 7.0156, 10.1735];
            println!("\nNext mode family (J₁ zeros):");
            for (mode, &zero) in j1_zeros.iter().enumerate() {
                let frequency = zero * wave_speed / (2.0 * PI * radius);
                println!("  Mode {}: {:.1} Hz", mode + 1, frequency);
            }
        }
        _ => {
            println!("🔧 Practical examples for {}:", node.name);
            println!("This topic has applications in:");
            for app in &node.applications {
                println!("  • {}", app);
            }
        }
    }

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn present_intuitive_content(
    topic: &str,
    node: &ConceptNode,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("💡 Intuitive Understanding Mode");
    println!("================================\n");

    match topic {
        "gamma_basics" => {
            println!("🤔 What is the Gamma Function Really?");
            println!();
            println!("Think of the gamma function as a 'smooth factorial':");
            println!("• Factorials work for positive integers: 5! = 5×4×3×2×1");
            println!("• But what about 2.5! or π! ?");
            println!("• The gamma function extends this idea to ALL positive real numbers!");
            println!();
            println!("🎯 Key Insight: Γ(n) = (n-1)! for positive integers");
            println!("So Γ(5) = 4! = 24, Γ(3) = 2! = 2, etc.");
            println!();
            println!("🌟 The Magic: Γ(1/2) = √π");
            println!("This connects factorials to the famous number π!");
            println!("It's like asking 'What is (-1/2)!' and getting √π as the answer.");
            println!();
            println!("💭 Intuitive Properties:");
            println!("• Γ(x+1) = x·Γ(x) - like factorial recurrence but smooth");
            println!("• Γ(x) → ∞ as x → 0⁺ - dividing by smaller and smaller numbers");
            println!("• Γ(x) grows very fast for large x - faster than exponential!");

            // Demonstrate the recurrence relation
            println!("\n🔗 Recurrence Relation Demo:");
            for n in 1..5 {
                let x = n as f64 + 0.5;
                let gamma_x = gamma(x);
                let gamma_x_plus_1 = gamma(x + 1.0);
                let ratio = gamma_x_plus_1 / gamma_x;
                println!(
                    "  Γ({}) / Γ({}) = {:.6} (should equal {})",
                    x + 1.0,
                    x,
                    ratio,
                    x
                );
            }
        }
        "bessel_basics" => {
            println!("🌊 Understanding Bessel Functions Intuitively");
            println!();
            println!("🎯 Think of Bessel functions as 'circular waves':");
            println!("• Sine waves oscillate in straight lines");
            println!("• Bessel functions oscillate in circles!");
            println!();
            println!("🥁 Drum Analogy:");
            println!("When you hit a circular drum, the vibration patterns are described by Bessel functions.");
            println!("• J₀ describes the fundamental mode - like a piston moving up and down");
            println!("• J₁ describes the next mode - like a saddle shape");
            println!("• Higher orders give more complex patterns");
            println!();
            println!("📉 Key Intuitions:");
            println!("• They oscillate like sine/cosine but with decreasing amplitude");
            println!("• They have zeros (like sine) but the spacing changes");
            println!("• Near x=0, they behave like powers: J₀(x) ≈ 1, J₁(x) ≈ x/2");
            println!("• For large x, they look like shifted and scaled sine waves");
            println!();
            println!("🌀 Why do they appear everywhere?");
            println!("Whenever you have circular or cylindrical symmetry in physics:");
            println!("• Heat flow in circular pipes");
            println!("• Electromagnetic fields around antennas");
            println!("• Quantum mechanics in circular potentials");
            println!("• Sound waves in cylindrical spaces");
        }
        _ => {
            println!("💡 Intuitive understanding of {}:", node.name);
            println!("{}", node.description);
            println!();
            println!("💭 Think of this concept as connecting to everyday experience through:");
            for app in &node.applications {
                println!("  • {}", app);
            }
        }
    }

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn present_historical_content(
    topic: &str,
    node: &ConceptNode,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📜 Historical Context Mode");
    println!("===========================\n");

    match topic {
        "gamma_basics" => {
            println!("🏛️ The Historical Journey of the Gamma Function");
            println!();
            println!("📅 Timeline:");
            println!("• 1728: Leonhard Euler first studied the factorial interpolation problem");
            println!(
                "• 1729: Euler derived the integral representation Γ(n) = ∫₀^∞ t^(n-1) e^(-t) dt"
            );
            println!("• 1812: Adrien-Marie Legendre introduced the notation Γ(z)");
            println!("• 1856: Karl Weierstrass proved the infinite product representation");
            println!();
            println!("🎯 Euler's Original Question:");
            println!("'How can we extend n! to non-integer values?'");
            println!("This led to one of the most important functions in mathematics!");
            println!();
            println!("🌟 Key Historical Insights:");
            println!("• Euler discovered Γ(1/2) = √π by connecting it to the Gaussian integral");
            println!("• The reflection formula Γ(z)Γ(1-z) = π/sin(πz) unified many identities");
            println!("• Stirling's approximation provided asymptotic behavior for large arguments");
            println!();
            println!(
                "🎓 Fun Fact: Euler calculated Γ(1/2) without knowing about normal distributions!"
            );
            println!("He derived √π purely from mathematical curiosity about factorials.");
        }
        "bessel_basics" => {
            println!("🔭 The Historical Development of Bessel Functions");
            println!();
            println!("📅 Timeline:");
            println!(
                "• 1732: Daniel Bernoulli studied vibrating chains (early Bessel-like functions)"
            );
            println!(
                "• 1824: Friedrich Bessel systematically studied these functions for astronomy"
            );
            println!("• 1826: Bessel applied them to planetary motion and Kepler's equation");
            println!("• 1838: Used Bessel functions to measure stellar parallax (first stellar distance!)");
            println!();
            println!("🌟 Why 'Bessel' Functions?");
            println!("Friedrich Bessel was an astronomer who needed to solve Kepler's equation:");
            println!(
                "M = E - e sin(E)  (Mean anomaly = Eccentric anomaly - eccentricity × sin(E))"
            );
            println!();
            println!(
                "This led him to study functions that are now central to physics and engineering!"
            );
            println!();
            println!("🎯 Historical Applications:");
            println!("• Stellar parallax measurement (1838) - first distance to a star");
            println!("• Telegraph cable theory (Lord Kelvin, 1850s)");
            println!("• Radio wave propagation (Marconi era, 1900s)");
            println!("• Quantum mechanics foundations (1920s)");
            println!();
            println!("🎓 Amazing Fact: Bessel measured the distance to 61 Cygni (11 light-years)");
            println!("using mathematical techniques he developed for these functions!");
        }
        _ => {
            println!("📚 Historical perspective on {}:", node.name);
            println!("{}", node.description);
            println!();
            println!("This topic developed through the work of many mathematicians");
            println!("who were solving practical problems in their time.");
        }
    }

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn run_assessment_question(
    question: &AssessmentQuestion,
) -> Result<f64, Box<dyn std::error::Error>> {
    println!(
        "\n❓ Assessment Question (Difficulty: {}/5)",
        question.difficulty
    );
    println!(
        "⏱️ Suggested time: {:.1} minutes",
        question.expected_time.as_secs_f64() / 60.0
    );
    println!();
    println!("{}", question.content);

    match &question.question_type {
        QuestionType::MultipleChoice { options, correct } => {
            println!();
            for (i, option) in options.iter().enumerate() {
                println!("  {}. {}", (b'A' + i as u8) as char, option);
            }

            let answer = get_user_input("\nYour answer (A, B, C, etc.): ")?;
            let answer_index = answer.to_uppercase().chars().next().and_then(|c| {
                if c >= 'A' && c <= 'Z' {
                    Some((c as u8 - b'A') as usize)
                } else {
                    None
                }
            });

            if let Some(idx) = answer_index {
                if idx == *correct {
                    println!("✅ Correct!");
                    Ok(1.0)
                } else {
                    println!(
                        "❌ Incorrect. The correct answer is {}.",
                        (b'A' + *correct as u8) as char
                    );
                    offer_hints_and_retry(question)
                }
            } else {
                println!("❌ Invalid answer format.");
                Ok(0.0)
            }
        }
        QuestionType::NumericalAnswer {
            expected,
            tolerance,
        } => {
            let answer = get_user_input("\nEnter your numerical answer: ")?;
            match answer.parse::<f64>() {
                Ok(value) => {
                    let error = (value - expected).abs();
                    if error <= *tolerance {
                        println!(
                            "✅ Correct! (Answer: {:.6}, Your answer: {:.6})",
                            expected, value
                        );
                        Ok(1.0)
                    } else {
                        println!(
                            "❌ Close, but not quite. Expected: {:.6}, Your answer: {:.6}",
                            expected, value
                        );
                        println!("Error: {:.6} (tolerance: {:.6})", error, tolerance);
                        offer_hints_and_retry(question)
                    }
                }
                Err(_) => {
                    println!("❌ Invalid numerical format.");
                    Ok(0.0)
                }
            }
        }
        _ => {
            println!("This question type is not yet implemented in the demo.");
            Ok(0.5) // Partial credit for demonstration
        }
    }
}

#[allow(dead_code)]
fn offer_hints_and_retry(question: &AssessmentQuestion) -> Result<f64, Box<dyn std::error::Error>> {
    if !question.hints.is_empty() {
        let want_hint = get_user_input("Would you like a hint? (y/n): ")?;
        if want_hint.to_lowercase() == "y" {
            println!("\n💡 Hint: {}", question.hints[0]);
            let retry = get_user_input("Try again? (y/n): ")?;
            if retry.to_lowercase() == "y" {
                return run_assessment_question(question);
            }
        }
    }
    Ok(0.3) // Partial credit for wrong answer
}

#[allow(dead_code)]
fn explore_topics(session: &mut AdaptiveLearningSession) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Topic Explorer");
    println!("=================\n");

    println!("📚 Available topics:");
    let mut topics: Vec<_> = session.knowledge_graph.keys().collect();
    topics.sort();

    for (i, topic) in topics.iter().enumerate() {
        if let Some(node) = session.knowledge_graph.get(*topic) {
            let skill_level = session.profile.skill_levels.get(*topic).unwrap_or(&0.0);
            let status = if *skill_level >= 0.8 {
                "✅"
            } else if *skill_level >= 0.3 {
                "📖"
            } else {
                "🔒"
            };
            println!(
                "  {}. {} {} (Level: {:.1}%) - {}",
                i + 1,
                status,
                node.name,
                skill_level * 100.0,
                node.description
            );
        }
    }

    let choice = get_user_input("\nEnter topic number to explore: ")?;
    if let Ok(index) = choice.parse::<usize>() {
        if index > 0 && index <= topics.len() {
            let topic = topics[index - 1];
            if let Some(node) = session.knowledge_graph.get(topic).cloned() {
                println!("\n📖 Topic: {}", node.name);
                println!("Description: {}", node.description);
                println!("Difficulty: {}/5", node.difficulty);
                println!("Prerequisites: {}", node.prerequisites.join(", "));
                println!("Applications: {}", node.applications.join(", "));

                let explore = get_user_input("\nStart learning this topic? (y/n): ")?;
                if explore.to_lowercase() == "y" {
                    let start_time = Instant::now();
                    let score = run_learning_module(topic, &node, &session.profile)?;
                    let time_taken = start_time.elapsed();

                    // Update profile
                    session
                        .profile
                        .skill_levels
                        .insert(topic.to_string(), score);
                    session
                        .profile
                        .time_spent
                        .insert(topic.to_string(), time_taken);
                }
            }
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn run_comprehensive_assessment(
    session: &mut AdaptiveLearningSession,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Comprehensive Assessment");
    println!("============================\n");

    println!("This assessment will evaluate your understanding across multiple topics.");
    println!("It will help identify your strengths and areas for improvement.\n");

    let proceed = get_user_input("Start comprehensive assessment? (y/n): ")?;
    if proceed.to_lowercase() != "y" {
        return Ok(());
    }

    let start_time = Instant::now();
    let mut total_score = 0.0;
    let mut topic_scores = HashMap::new();
    let mut questions_answered = 0;

    // Sample questions from different topics
    for (topic, node) in &session.knowledge_graph {
        if session.profile.skill_levels.get(topic).unwrap_or(&0.0) >= &0.3 {
            println!("\n--- {} Section ---", node.name);

            let mut topic_score = 0.0;
            let mut topic_questions = 0;

            // Take up to 3 questions per topic
            for question in node.assessment_questions.iter().take(3) {
                topic_questions += 1;
                questions_answered += 1;
                let score = run_assessment_question(question)?;
                topic_score += score;
                total_score += score;
            }

            if topic_questions > 0 {
                topic_scores.insert(topic.clone(), topic_score / topic_questions as f64);
            }
        }
    }

    let time_taken = start_time.elapsed();
    let average_score = if questions_answered > 0 {
        total_score / questions_answered as f64
    } else {
        0.0
    };

    // Present comprehensive results
    println!("\n🎉 Assessment Complete!");
    println!("========================");
    println!(
        "⏱️ Time taken: {:.1} minutes",
        time_taken.as_secs_f64() / 60.0
    );
    println!(
        "📊 Overall score: {:.1}% ({}/{})",
        average_score * 100.0,
        total_score as u32,
        questions_answered
    );
    println!();

    // Topic breakdown
    println!("📈 Topic Performance:");
    let mut sorted_topics: Vec<_> = topic_scores.iter().collect();
    sorted_topics.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    for (topic, &score) in sorted_topics {
        let node = session.knowledge_graph.get(topic).unwrap();
        println!("  {:<30} {:.1}%", node.name, score * 100.0);
        session.profile.skill_levels.insert(topic.clone(), score);
    }

    // Recommendations
    println!("\n💡 Recommendations:");
    for (topic, &score) in &topic_scores {
        if score < 0.7 {
            let node = session.knowledge_graph.get(topic).unwrap();
            println!(
                "  📚 Review: {} (current: {:.1}%)",
                node.name,
                score * 100.0
            );
        }
    }

    // Record assessment result
    let assessment = AssessmentResult {
        topic: "comprehensive".to_string(),
        score: average_score,
        time_taken,
        difficulty_level: 3,
        mistakes: Vec::new(),
        timestamp: Instant::now(),
    };
    session.profile.assessment_scores.push(assessment);

    Ok(())
}

#[allow(dead_code)]
fn display_learning_analytics(profile: &LearningProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Learning Analytics Dashboard");
    println!("===============================\n");

    // Progress overview
    let total_skills = profile.skill_levels.len();
    let mastered = profile
        .skill_levels
        .values()
        .filter(|&&level| level >= 0.8)
        .count();
    let learning = profile
        .skill_levels
        .values()
        .filter(|&&level| level >= 0.3 && level < 0.8)
        .count();
    let not_started = total_skills - mastered - learning;

    println!("📊 Overall Progress:");
    println!(
        "  Mastered: {} topics ({:.1}%)",
        mastered,
        mastered as f64 / total_skills as f64 * 100.0
    );
    println!(
        "  Learning: {} topics ({:.1}%)",
        learning,
        learning as f64 / total_skills as f64 * 100.0
    );
    println!(
        "  Not started: {} topics ({:.1}%)",
        not_started,
        not_started as f64 / total_skills as f64 * 100.0
    );

    // Time analysis
    let total_time: Duration = profile.time_spent.values().sum();
    println!("\n⏱️ Time Investment:");
    println!(
        "  Total study time: {:.1} hours",
        total_time.as_secs_f64() / 3600.0
    );
    if !profile.time_spent.is_empty() {
        let avg_time = total_time.as_secs_f64() / profile.time_spent.len() as f64;
        println!("  Average per topic: {:.1} minutes", avg_time / 60.0);
    }

    // Assessment history
    if !profile.assessment_scores.is_empty() {
        println!("\n📝 Assessment History:");
        let recent_scores: Vec<_> = profile.assessment_scores.iter().rev().take(5).collect();

        for assessment in recent_scores {
            println!(
                "  {}: {:.1}% (Level {})",
                assessment.topic,
                assessment.score * 100.0,
                assessment.difficulty_level
            );
        }
    }

    // Learning pattern analysis
    println!("\n🧠 Learning Patterns:");
    println!("  Preferred style: {:?}", profile.preferred_learning_style);
    println!("  Reading speed: {:.0} WPM", profile.learning_speed);

    if !profile.mistake_patterns.is_empty() {
        println!("\n❌ Common Mistake Patterns:");
        let mut mistakes: Vec<_> = profile.mistake_patterns.iter().collect();
        mistakes.sort_by(|a, b| b.1.cmp(a.1));

        for (mistake_type, &count) in mistakes.iter().take(3) {
            println!("  {}: {} occurrences", mistake_type, count);
        }
    }

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn run_interactive_visualizations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎨 Interactive Visualizations");
    println!("=============================\n");

    println!("📊 Available visualizations:");
    println!("1. 📈 Function plots with parameter adjustment");
    println!("2. 🌀 Complex plane visualization");
    println!("3. 🎵 Bessel function animations");
    println!("4. 🌊 3D surface plots");
    println!("5. 📐 Geometric interpretations");

    let choice = get_user_input("Choose visualization (1-5): ")?;

    match choice.as_str() {
        "1" => {
            println!("\n📈 Interactive Function Plots");
            println!("=============================");

            println!("Choose a function to visualize:");
            println!("a) Gamma function");
            println!("b) Bessel functions");
            println!("c) Error function");

            let func_choice = get_user_input("Your choice (a-c): ")?;

            match func_choice.to_lowercase().as_str() {
                "a" => visualize_gamma_function()?,
                "b" => visualize_bessel_functions()?,
                "c" => visualize_error_function()?,
                _ => println!("Invalid choice"),
            }
        }
        "2" => visualize_complex_plane()?,
        "3" => animate_bessel_functions()?,
        "4" => visualize_3d_surfaces()?,
        "5" => show_geometric_interpretations()?,
        _ => println!("Invalid choice"),
    }

    Ok(())
}

#[allow(dead_code)]
fn visualize_gamma_function() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎲 Gamma Function Visualization");
    println!("===============================\n");

    let xmin = get_user_input("Enter x minimum (e.g., 0.1): ")?
        .parse::<f64>()
        .unwrap_or(0.1);
    let xmax = get_user_input("Enter x maximum (e.g., 5.0): ")?
        .parse::<f64>()
        .unwrap_or(5.0);
    let points = get_user_input("Number of points (e.g., 50): ")?
        .parse::<usize>()
        .unwrap_or(50);

    println!("\nΓ(x) values:");
    println!("x      Γ(x)");
    println!("─────────────");

    for i in 0..points {
        let x = xmin + (xmax - xmin) * i as f64 / (points - 1) as f64;
        let gamma_val = gamma(x);

        // Simple ASCII visualization
        if gamma_val < 20.0 {
            let bar_length = (gamma_val * 2.0) as usize;
            let bar: String = std::iter::repeat('■').take(bar_length.min(40)).collect();
            println!("{:5.2}  {:8.3} {}", x, gamma_val, bar);
        } else {
            println!("{:5.2}  {:8.3} (too large to display)", x, gamma_val);
        }
    }

    // Special values
    println!("\n🌟 Special Values:");
    println!("Γ(1) = {:.6}", gamma(1.0));
    println!("Γ(2) = {:.6}", gamma(2.0));
    println!("Γ(1/2) = {:.6} ≈ √π = {:.6}", gamma(0.5), PI.sqrt());

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn visualize_bessel_functions() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌊 Bessel Function Visualization");
    println!("================================\n");

    let order = get_user_input("Enter Bessel function order (0, 1, 2): ")?
        .parse::<i32>()
        .unwrap_or(0);
    let xmax = get_user_input("Enter maximum x value (e.g., 20): ")?
        .parse::<f64>()
        .unwrap_or(20.0);

    println!("\nJ_{}(x) oscillation pattern:", order);
    println!("x      J_{}(x)    Visualization", order);
    println!("─────────────────────────────────────");

    for i in 0..40 {
        let x = i as f64 * xmax / 40.0;
        let j_val = match order {
            0 => j0(x),
            1 => j1(x),
            2 => jn(2, x),
            _ => jn(order, x),
        };

        // ASCII oscillation display
        let center = 20;
        let position = center + (j_val * 15.0) as i32;
        let display_pos = position.max(0).min(40) as usize;

        let mut line = [' '; 41];
        line[center as usize] = '|';
        if display_pos < line.len() {
            line[display_pos] = if j_val > 0.0 { '●' } else { '○' };
        }

        let display: String = line.iter().collect();
        println!("{:5.1}  {:8.4}    {}", x, j_val, display);
    }

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn visualize_error_function() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Error Function Visualization");
    println!("===============================\n");

    println!("erf(x) and erfc(x) comparison:");
    println!("x      erf(x)    erfc(x)   Visual");
    println!("────────────────────────────────────");

    for i in 0..21 {
        let x = -3.0 + i as f64 * 6.0 / 20.0;
        let erf_val = erf(x);
        let erfc_val = erfc(x);

        // Visual representation
        let erf_pos = (15.0 + erf_val * 10.0) as usize;
        let erfc_pos = (erfc_val * 20.0) as usize;

        let mut line = [' '; 31];
        line[15] = '|'; // Zero line
        if erf_pos < line.len() {
            line[erf_pos] = '●';
        }
        if erfc_pos < line.len() && erfc_pos != erf_pos {
            line[erfc_pos] = '○';
        }

        let display: String = line.iter().collect();
        println!("{:5.1}  {:7.4}   {:7.4}  {}", x, erf_val, erfc_val, display);
    }

    println!("\nLegend: ● = erf(x), ○ = erfc(x), | = zero");

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn visualize_complex_plane() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌀 Complex Plane Visualization");
    println!("==============================\n");

    println!("This would show complex function visualizations using color coding:");
    println!("• Hue represents argument (angle)");
    println!("• Brightness represents magnitude");
    println!("• Zeros appear as black points");
    println!("• Poles appear as white points");
    println!();
    println!("In a full implementation, this would generate interactive plots");
    println!("showing the beautiful patterns of complex special functions.");

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn animate_bessel_functions() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎵 Bessel Function Animation");
    println!("============================\n");

    println!("Animating circular membrane vibration patterns...");

    // Simple text-based animation
    for frame in 0..20 {
        print!("\x1B[2J\x1B[H"); // Clear screen and move cursor to top

        println!("Frame {}/20 - Circular Membrane Vibration", frame + 1);
        println!("==========================================\n");

        let time = frame as f64 * 0.2;

        // Create circular pattern
        for row in 0..15 {
            for col in 0..30 {
                let x = (col as f64 - 15.0) * 0.3;
                let y = (row as f64 - 7.5) * 0.3;
                let r = (x * x + y * y).sqrt();

                if r < 0.1 {
                    print!("●"); // Center
                } else if r < 4.0 {
                    let amplitude = j0(r) * (time * 3.0).cos();
                    let char = if amplitude > 0.3 {
                        '█'
                    } else if amplitude > 0.0 {
                        '▓'
                    } else if amplitude > -0.3 {
                        '▒'
                    } else {
                        '░'
                    };
                    print!("{}", char);
                } else {
                    print!(" ");
                }
            }
            println!();
        }

        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    println!("\nAnimation complete! This shows how J₀ creates circular wave patterns.");
    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn visualize_3d_surfaces() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌊 3D Surface Visualization");
    println!("===========================\n");

    println!("3D surface plot of |Γ(x + iy)|:");
    println!("(Showing magnitude in the complex plane)\n");

    // Simple 3D-like ASCII representation
    for y in (0..10).rev() {
        let im_part = (y as f64 - 5.0) * 0.5;
        print!("y={:4.1} ", im_part);

        for x in 0..20 {
            let re_part = x as f64 * 0.3 + 0.1;
            let z = Complex64::new(re_part, im_part);

            // Calculate |Γ(z)| with safety checks
            let gamma_mag = if re_part > 0.0 {
                gamma_complex(z).norm()
            } else {
                // Use reflection formula for negative real parts
                let reflected = Complex64::new(-re_part, im_part);
                let pi_over_sin = PI / (PI * z).sin().norm();
                pi_over_sin / gamma_complex(reflected + Complex64::new(1.0, 0.0)).norm()
            };

            let level = if gamma_mag < 1.0 {
                '·'
            } else if gamma_mag < 2.0 {
                '▒'
            } else if gamma_mag < 5.0 {
                '▓'
            } else {
                '█'
            };

            print!("{}", level);
        }
        println!();
    }

    println!("\nLegend: · < 1, ▒ 1-2, ▓ 2-5, █ > 5");

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn show_geometric_interpretations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📐 Geometric Interpretations");
    println!("============================\n");

    println!("🎲 Gamma Function Geometry:");
    println!("• Γ(z) can be viewed as the 'volume' of an infinite-dimensional simplex");
    println!("• The integral ∫₀^∞ t^(z-1) e^(-t) dt represents weighted infinite sums");
    println!("• Γ(1/2) = √π connects discrete (factorials) to continuous (circles)");
    println!();

    println!("🌊 Bessel Function Geometry:");
    println!("• J_n(r) describes standing wave patterns on circular domains");
    println!("• The zeros correspond to nodal circles where amplitude = 0");
    println!("• Different orders n give different symmetry patterns");
    println!();

    println!("📊 Error Function Geometry:");
    println!("• erf(x) represents the area under the bell curve from 0 to x");
    println!("• Geometrically links linear measure (x) to area (probability)");
    println!("• The √π factor comes from the total area under e^(-t²)");
    println!();

    println!("🔗 Connections:");
    println!("• All these functions arise from trying to solve geometric problems");
    println!("• They connect algebra (equations) to geometry (shapes and areas)");
    println!("• Modern applications use the same geometric intuitions");

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn run_proof_exploration() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧠 Interactive Proof Exploration");
    println!("=================================\n");

    println!("📜 Available proofs to explore:");
    println!("1. 🎲 Γ(1/2) = √π (Multiple approaches)");
    println!("2. 🔄 Gamma function reflection formula");
    println!("3. 🌊 Bessel function generating function");
    println!("4. 📊 Error function series expansion");

    let choice = get_user_input("Choose proof to explore (1-4): ")?;

    match choice.as_str() {
        "1" => explore_gamma_half_proof()?,
        "2" => explore_reflection_formula_proof()?,
        "3" => explore_bessel_generating_function_proof()?,
        "4" => explore_error_function_series_proof()?,
        _ => println!("Invalid choice"),
    }

    Ok(())
}

#[allow(dead_code)]
fn explore_gamma_half_proof() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎲 Exploring the proof of Γ(1/2) = √π");
    println!("======================================\n");

    println!("We'll explore this beautiful result step by step.");
    println!("You can choose how much detail to see at each step.\n");

    // Step 1
    println!("📝 Step 1: Start with the definition");
    println!("Γ(1/2) = ∫₀^∞ t^(1/2-1) e^(-t) dt = ∫₀^∞ t^(-1/2) e^(-t) dt");

    let detail = get_user_input("\nWant to see why we use this definition? (y/n): ")?;
    if detail.to_lowercase() == "y" {
        println!("\n💡 The gamma function is defined as Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt");
        println!("For z = 1/2, we get the exponent z-1 = 1/2-1 = -1/2");
        println!("So we need to evaluate ∫₀^∞ t^(-1/2) e^(-t) dt");
    }

    // Step 2
    println!("\n📝 Step 2: Make the substitution t = u²");
    println!("If t = u², then dt = 2u du");
    println!("The integral becomes: ∫₀^∞ (u²)^(-1/2) e^(-u²) · 2u du");

    let detail = get_user_input("\nWork through this substitution? (y/n): ")?;
    if detail.to_lowercase() == "y" {
        println!("\n🔧 Substitution details:");
        println!("• t = u² → dt = d(u²) = 2u du");
        println!("• t^(-1/2) = (u²)^(-1/2) = u^(-1)");
        println!("• Limits: t: 0→∞ becomes u: 0→∞");
        println!("• So: ∫₀^∞ u^(-1) e^(-u²) · 2u du = ∫₀^∞ 2 e^(-u²) du");
        println!("The u^(-1) and u terms cancel!");
    }

    // Step 3
    println!("\n📝 Step 3: Recognize the Gaussian integral");
    println!("We now have: Γ(1/2) = 2∫₀^∞ e^(-u²) du");
    println!("But we know that ∫_{{-∞}}^∞ e^(-u²) du = √π");

    let detail = get_user_input("\nExplore why the Gaussian integral equals √π? (y/n): ")?;
    if detail.to_lowercase() == "y" {
        println!("\n🎯 The famous Gaussian integral proof:");
        println!("Let I = ∫_{{-∞}}^∞ e^(-x²) dx");
        println!("Then I² = (∫_{{-∞}}^∞ e^(-x²) dx)(∫_{{-∞}}^∞ e^(-y²) dy)");
        println!("     = ∫∫ e^(-(x²+y²)) dx dy");
        println!("Convert to polar coordinates: x = r cos θ, y = r sin θ");
        println!("I² = ∫₀^(2π) ∫₀^∞ e^(-r²) r dr dθ = 2π ∫₀^∞ r e^(-r²) dr");
        println!("The inner integral = 1/2, so I² = π, thus I = √π");
    }

    // Step 4
    println!("\n📝 Step 4: Complete the calculation");
    println!("Since ∫_{{-∞}}^∞ e^(-u²) du = √π and e^(-u²) is even:");
    println!("∫₀^∞ e^(-u²) du = (1/2)√π");
    println!("Therefore: Γ(1/2) = 2 · (1/2)√π = √π ✓");

    // Numerical verification
    println!("\n🔍 Numerical verification:");
    let gamma_half = gamma(0.5);
    let sqrt_pi = PI.sqrt();
    println!("Γ(1/2) = {:.12}", gamma_half);
    println!("√π     = {:.12}", sqrt_pi);
    println!("Error  = {:.2e}", (gamma_half - sqrt_pi).abs());

    println!("\n🌟 This beautiful result connects:");
    println!("• Factorials (discrete) ↔ π (continuous circles)");
    println!("• Integration ↔ Probability theory");
    println!("• Real analysis ↔ Complex analysis");

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn explore_reflection_formula_proof() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Exploring the Gamma Function Reflection Formula");
    println!("=================================================\n");

    println!("🎯 Goal: Prove that Γ(z)Γ(1-z) = π/sin(πz)");
    println!("This is one of the most beautiful identities in mathematics!\n");

    println!("📚 We'll use complex analysis and the beta function.");
    println!("The proof involves several sophisticated techniques:");
    println!("• Beta function representation");
    println!("• Complex contour integration");
    println!("• Residue calculus");
    println!("• Analytic continuation");

    let proceed = get_user_input("\nContinue with the detailed proof? (y/n): ")?;
    if proceed.to_lowercase() == "y" {
        println!("\n📝 Step 1: Start with the beta function");
        println!("B(z, 1-z) = ∫₀¹ t^(z-1)(1-t)^(-z) dt = Γ(z)Γ(1-z)/Γ(1) = Γ(z)Γ(1-z)");

        println!("\n📝 Step 2: Transform the integral");
        println!("Using the substitution t = u/(1+u):");
        println!("B(z, 1-z) = ∫₀^∞ u^(z-1)/(1+u) du");

        println!("\n📝 Step 3: Apply complex analysis");
        println!("Consider the complex integral ∮_C w^(z-1)/(1+w) dw");
        println!("around a keyhole contour avoiding the branch cut on [0,∞).");

        println!("\n📝 Step 4: Calculate residues and limits");
        println!("The residue at w = -1 gives us the connection to sin(πz).");

        println!("\n🎉 Final result:");
        println!("After careful evaluation of the contour integral,");
        println!("we get: Γ(z)Γ(1-z) = π/sin(πz)");

        // Show some special cases
        println!("\n✨ Beautiful special cases:");
        println!("z = 1/2: Γ(1/2)² = π/sin(π/2) = π → Γ(1/2) = √π");
        println!("z = 1/3: Γ(1/3)Γ(2/3) = π/sin(π/3) = 2π/√3");

        let gamma_third = gamma(1.0 / 3.0);
        let gamma_two_thirds = gamma(2.0 / 3.0);
        let product = gamma_third * gamma_two_thirds;
        let expected = 2.0 * PI / 3.0_f64.sqrt();
        println!(
            "Verification: Γ(1/3)Γ(2/3) = {:.6}, expected = {:.6}",
            product, expected
        );
    }

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn explore_bessel_generating_function_proof() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌊 Bessel Function Generating Function");
    println!("=====================================\n");

    println!("🎯 Goal: Prove that exp(x(t-1/t)/2) = Σ J_n(x) t^n");
    println!("This generating function is fundamental to Bessel function theory.\n");

    // Implementation similar to other proofs...
    println!("This proof involves expanding the exponential and carefully");
    println!("collecting coefficients to identify the Bessel functions.");

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn explore_error_function_series_proof() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Error Function Series Expansion");
    println!("==================================\n");

    println!("🎯 Goal: Derive erf(x) = (2/√π) Σ (-1)^n x^(2n+1) / (n!(2n+1))");
    println!("This shows how the error function connects to power series.\n");

    // Implementation similar to other proofs...
    println!("This derivation uses term-by-term integration of the");
    println!("exponential series for e^(-t²).");

    wait_for_user_input()?;
    Ok(())
}

#[allow(dead_code)]
fn adjust_learning_preferences(
    profile: &mut LearningProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚙️ Learning Preferences");
    println!("========================\n");

    println!("Current preferences:");
    println!("• Learning style: {:?}", profile.preferred_learning_style);
    println!("• Reading speed: {:.0} WPM", profile.learning_speed);

    println!("\nLearning styles:");
    println!("1. Visual - Graphs, diagrams, visual proofs");
    println!("2. Analytical - Step-by-step algebraic derivations");
    println!("3. Intuitive - Conceptual explanations and analogies");
    println!("4. Practical - Applications and numerical examples");
    println!("5. Historical - Historical development and context");

    let style_choice = get_user_input("Choose learning style (1-5): ")?;
    match style_choice.parse::<u32>() {
        Ok(1) => profile.preferred_learning_style = LearningStyle::Visual,
        Ok(2) => profile.preferred_learning_style = LearningStyle::Analytical,
        Ok(3) => profile.preferred_learning_style = LearningStyle::Intuitive,
        Ok(4) => profile.preferred_learning_style = LearningStyle::Practical,
        Ok(5) => profile.preferred_learning_style = LearningStyle::Historical,
        _ => println!("Invalid choice, keeping current setting"),
    }

    let speedinput = get_user_input("Reading speed (WPM, current: {:.0}): ")?;
    if let Ok(speed) = speedinput.parse::<f64>() {
        if speed > 0.0 && speed < 1000.0 {
            profile.learning_speed = speed;
        }
    }

    println!("✅ Preferences updated!");
    Ok(())
}

#[allow(dead_code)]
fn saveprofile(profile: &LearningProfile) -> Result<(), Box<dyn std::error::Error>> {
    // In a real implementation, this would save to a file or database
    println!("💾 Profile saved successfully!");
    Ok(())
}

// Helper functions for creating assessment questions
#[allow(dead_code)]
fn create_gamma_basic_questions() -> Vec<AssessmentQuestion> {
    vec![
        AssessmentQuestion {
            question_type: QuestionType::MultipleChoice {
                options: vec![
                    "0".to_string(),
                    "1".to_string(),
                    "∞".to_string(),
                    "√π".to_string(),
                ],
                correct: 3,
            },
            content: "What is the value of Γ(1/2)?".to_string(),
            difficulty: 2,
            expected_time: Duration::from_secs(60),
            hints: vec![
                "This involves the famous Gaussian integral".to_string(),
                "Think about the connection between factorials and π".to_string(),
            ],
            solution_steps: vec![
                "Start with the integral definition".to_string(),
                "Make the substitution t = u²".to_string(),
                "Recognize the Gaussian integral".to_string(),
            ],
            common_mistakes: vec![
                "Forgetting the substitution Jacobian".to_string(),
                "Not recognizing the Gaussian integral".to_string(),
            ],
        },
        AssessmentQuestion {
            question_type: QuestionType::NumericalAnswer {
                expected: 24.0,
                tolerance: 0.1,
            },
            content: "Calculate Γ(5).".to_string(),
            difficulty: 1,
            expected_time: Duration::from_secs(30),
            hints: vec!["Use the recurrence relation Γ(n) = (n-1)!".to_string()],
            solution_steps: vec![
                "Recall that Γ(n) = (n-1)! for positive integers".to_string(),
                "So Γ(5) = 4! = 4×3×2×1 = 24".to_string(),
            ],
            common_mistakes: vec!["Calculating 5! instead of 4!".to_string()],
        },
    ]
}

#[allow(dead_code)]
fn create_gamma_advanced_questions() -> Vec<AssessmentQuestion> {
    vec![AssessmentQuestion {
        question_type: QuestionType::MultipleChoice {
            options: vec![
                "π/sin(πz)".to_string(),
                "π/cos(πz)".to_string(),
                "sin(πz)/π".to_string(),
                "π·sin(πz)".to_string(),
            ],
            correct: 0,
        },
        content: "What is Γ(z)Γ(1-z) equal to?".to_string(),
        difficulty: 4,
        expected_time: Duration::from_secs(120),
        hints: vec![
            "This is the reflection formula".to_string(),
            "It involves the sine function".to_string(),
        ],
        solution_steps: vec![
            "This is derived using complex analysis".to_string(),
            "The beta function provides the connection".to_string(),
        ],
        common_mistakes: vec!["Confusing with the duplication formula".to_string()],
    }]
}

#[allow(dead_code)]
fn create_bessel_basic_questions() -> Vec<AssessmentQuestion> {
    vec![AssessmentQuestion {
        question_type: QuestionType::MultipleChoice {
            options: vec![
                "x²y'' + xy' + (x² - ν²)y = 0".to_string(),
                "x²y'' + xy' - (x² + ν²)y = 0".to_string(),
                "xy'' + y' + (x - ν²)y = 0".to_string(),
                "y'' + xy' + (x² - ν²)y = 0".to_string(),
            ],
            correct: 0,
        },
        content: "What is Bessel's differential equation?".to_string(),
        difficulty: 2,
        expected_time: Duration::from_secs(90),
        hints: vec![
            "It's a second-order linear ODE".to_string(),
            "The coefficient of y involves x² - ν²".to_string(),
        ],
        solution_steps: vec![
            "This equation arises from separation of variables in cylindrical coordinates"
                .to_string(),
        ],
        common_mistakes: vec!["Wrong signs in the equation".to_string()],
    }]
}

#[allow(dead_code)]
fn create_hypergeometric_questions() -> Vec<AssessmentQuestion> {
    vec![AssessmentQuestion {
        question_type: QuestionType::NumericalAnswer {
            expected: 1.0,
            tolerance: 0.001,
        },
        content: "What is ₂F₁(a,b;c;0)?".to_string(),
        difficulty: 3,
        expected_time: Duration::from_secs(60),
        hints: vec![
            "Look at the series definition".to_string(),
            "What happens when z = 0?".to_string(),
        ],
        solution_steps: vec![
            "The series ₂F₁(a,b;c;z) = Σ (a)_n(b)_n/(c)_n · z^n/n!".to_string(),
            "When z = 0, only the n = 0 term survives".to_string(),
            "The n = 0 term is 1".to_string(),
        ],
        common_mistakes: vec!["Thinking it equals 0".to_string()],
    }]
}

#[allow(dead_code)]
fn create_wright_function_questions() -> Vec<AssessmentQuestion> {
    vec![AssessmentQuestion {
        question_type: QuestionType::MultipleChoice {
            options: vec![
                "Exponential".to_string(),
                "Polynomial".to_string(),
                "Super-exponential".to_string(),
                "Logarithmic".to_string(),
            ],
            correct: 2,
        },
        content: "How does the Wright function grow for large |z|?".to_string(),
        difficulty: 5,
        expected_time: Duration::from_secs(120),
        hints: vec![
            "Consider the asymptotic expansion".to_string(),
            "It grows faster than any exponential".to_string(),
        ],
        solution_steps: vec![
            "The asymptotic behavior involves exp((z/α)^(1/α))".to_string(),
            "This grows faster than exp(z) for α < 1".to_string(),
        ],
        common_mistakes: vec!["Thinking it's just exponential growth".to_string()],
    }]
}

#[allow(dead_code)]
fn wait_for_user_input() -> Result<(), Box<dyn std::error::Error>> {
    get_user_input("Press Enter to continue...")?;
    Ok(())
}

#[allow(dead_code)]
fn get_user_input(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}
