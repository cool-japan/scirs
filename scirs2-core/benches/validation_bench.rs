//! Benchmarks for validation system performance

#[cfg(feature = "data_validation")]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(feature = "data_validation")]
use std::hint::black_box;

#[cfg(feature = "data_validation")]
use ndarray::{Array1, Array2};

#[cfg(feature = "data_validation")]
use scirs2_core::validation::data::{
    ArrayValidationConstraints, Constraint, ConstraintBuilder, DataType, ValidationConfig,
    ValidationSchema, Validator,
};

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn bench_simple_validation(c: &mut Criterion) {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    let schema = ValidationSchema::new()
        .require_field(name, DataType::String)
        .require_field("age", DataType::Integer)
        .add_constraint(
            "age",
            Constraint::Range {
                min: 0.0,
                max: 150.0,
            },
        );

    c.bench_function("simple_validation", |b| {
        b.iter(|| {
            let data = serde_json::json!({
                name: black_box("John Doe"),
                "age": black_box(30)
            });
            let _ = validator.validate(&data, &schema);
        })
    });
}

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn bench_complex_constraints(c: &mut Criterion) {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create complex nested constraints
    let complex_constraint = Constraint::And(vec![
        Constraint::Or(vec![
            Constraint::Range {
                min: 0.0,
                max: 50.0,
            },
            Constraint::Range {
                min: 100.0,
                max: 150.0,
            },
        ]),
        Constraint::Not(Box::new(Constraint::Range {
            min: 25.0,
            max: 30.0,
        })),
    ]);

    let schema = ValidationSchema::new()
        .require_field(value, DataType::Float64)
        .add_constraint(value, complex_constraint);

    c.bench_function("complex_constraints", |b| {
        b.iter(|| {
            let data = serde_json::json!({
                value: black_box(42.0)
            });
            let _ = validator.validate(&data, &schema);
        })
    });
}

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn bench_array_validation(c: &mut Criterion) {
    let config = ValidationConfig::default();
    let validator = Validator::new(config.clone()).unwrap();

    let mut group = c.benchmark_group(array_validation);

    for size in [100, 1000, 10000].iter() {
        let data = Array2::<f64>::zeros((*size, 10));
        let constraints = ArrayValidationConstraints::new()
            .withshape(vec![*size, 10])
            .check_numeric_quality();

        group.bench_with_input(BenchmarkId::new("array_size", size), size, |b_| {
            b.iter(|| {
                let _ = validator.validate_ndarray(&data, &constraints, &config);
            })
        });
    }

    group.finish();
}

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn bench_pattern_matching(c: &mut Criterion) {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    let schema = ValidationSchema::new()
        .require_field("email", DataType::String)
        .add_constraint(
            "email",
            Constraint::Pattern(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$".to_string()),
        );

    c.bench_function("pattern_matching", |b| {
        b.iter(|| {
            let data = serde_json::json!({
                "email": black_box("test.user@example.com")
            });
            let _ = validator.validate(&data, &schema);
        })
    });
}

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn bench_constraint_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group(constraint_builder);

    group.bench_function("build_simple", |b| {
        b.iter(|| {
            let _ = ConstraintBuilder::new()
                .range(black_box(0.0), black_box(100.0))
                .not_null()
                .and();
        })
    });

    group.bench_function("build_complex", |b| {
        b.iter(|| {
            let _ = ConstraintBuilder::new()
                .range(black_box(0.0), black_box(100.0))
                .pattern(black_box("^[A-Z]+$"))
                .length(black_box(5), black_box(20))
                .not_null()
                .and();
        })
    });

    group.finish();
}

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn bench_large_or_constraint(c: &mut Criterion) {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    let mut group = c.benchmark_group(large_or_constraint);

    for size in [10, 50, 100].iter() {
        let patterns: Vec<Constraint> = (0..*size)
            .map(|i| Constraint::Pattern(format!("pattern{}", i)))
            .collect();

        let schema = ValidationSchema::new()
            .require_field("text", DataType::String)
            .add_constraint("text", Constraint::Or(patterns));

        group.bench_with_input(BenchmarkId::new("or_size", size), size, |b_| {
            b.iter(|| {
                let data = serde_json::json!({
                    "text": black_box(pattern42)
                });
                let _ = validator.validate(&data, &schema);
            })
        });
    }

    group.finish();
}

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn bench_cache_performance(c: &mut Criterion) {
    // Note: Cache configuration would be done through default config
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    let schema = ValidationSchema::new()
        .require_field(value, DataType::Float64)
        .add_constraint(
            value,
            Constraint::Range {
                min: 0.0,
                max: 100.0,
            },
        );

    let mut group = c.benchmark_group(cache_performance);

    // First run - cache miss
    group.bench_function("cache_miss", |b| {
        b.iter(|| {
            {
                let data = serde_json::json!({
                    value: black_box(50.0)
                });
                let _ = validator.validate(&data, &schema);
                let _ = validator.clear_cache(); // Clear cache to ensure miss
            }
        })
    });

    // Warm up cache

    {
        let data = serde_json::json!({ value: 50.0 });
        let _ = validator.validate(&data, &schema);
    }

    // Subsequent runs - cache hit
    group.bench_function("cache_hit", |b| {
        b.iter(|| {
            let data = serde_json::json!({
                value: black_box(50.0)
            });
            let _ = validator.validate(&data, &schema);
        })
    });

    group.finish();
}

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn bench_quality_report_generation(c: &mut Criterion) {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    let mut group = c.benchmark_group(quality_report);

    for size in [100, 1000].iter() {
        let data = Array1::<f64>::from_vec(
            (0..*size)
                .map(|i| i as f64 + 0.1 * (i as f64).sin())
                .collect(),
        );

        group.bench_with_input(BenchmarkId::new("array_size", size), size, |b_| {
            b.iter(|| {
                let _ = validator.generate_quality_report(&data, "test_field");
            })
        });
    }

    group.finish();
}

#[cfg(feature = "data_validation")]
criterion_group!(
    benches,
    bench_simple_validation,
    bench_complex_constraints,
    bench_array_validation,
    bench_pattern_matching,
    bench_constraint_builder,
    bench_large_or_constraint,
    bench_cache_performance,
    bench_quality_report_generation
);

#[cfg(feature = "data_validation")]
criterion_main!(benches);

#[cfg(not(feature = "data_validation"))]
#[allow(dead_code)]
fn main() {
    // No benchmarks to run without data_validation feature
}
