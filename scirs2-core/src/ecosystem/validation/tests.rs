//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use crate::apiversioning::Version;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;

    /// Build a minimal `ModuleInfo` with the given dependencies (as
    /// `(name, version_requirement)` pairs) for use in validator tests.
    fn make_module(name: &str, version: &str, deps: &[(&str, &str)]) -> ModuleInfo {
        ModuleInfo {
            name: name.to_string(),
            version: version.to_string(),
            dependencies: deps
                .iter()
                .map(|(dep_name, requirement)| DependencyInfo {
                    name: (*dep_name).to_string(),
                    version_requirement: VersionRequirement::new(requirement),
                    optional: false,
                })
                .collect(),
            apisurface: ApiSurface {
                public_apis: Vec::new(),
                deprecated_apis: Vec::new(),
            },
            features: Vec::new(),
            metadata: ModuleMetadata {
                author: "test".to_string(),
                description: "test module".to_string(),
                license: "MIT".to_string(),
                repository: None,
                build_time: None,
            },
        }
    }

    fn api(name: &str, signature: &str) -> ApiInfo {
        ApiInfo {
            name: name.to_string(),
            signature: signature.to_string(),
            documentation: "doc".to_string(),
            since_version: Some(Version::new(1, 0, 0)),
            stability: ApiStability::Stable,
        }
    }

    #[test]
    fn test_validator_creation() {
        let validator = EcosystemValidator::new().expect("Operation failed");
        // Basic functionality test
    }

    #[test]
    fn test_module_registration() {
        let validator = EcosystemValidator::new().expect("Operation failed");
        let module = create_core_module_info();

        validator.register_module(module).expect("Operation failed");

        let result = validator
            .validate_module("scirs2-core")
            .expect("Operation failed");
        assert!(result.is_valid());
    }

    #[test]
    fn test_ecosystem_validation() {
        let validator = EcosystemValidator::new().expect("Operation failed");
        validator
            .register_module(create_core_module_info())
            .expect("Operation failed");

        let result = validator.validate_ecosystem().expect("Operation failed");
        assert!(result.is_valid());
    }

    #[test]
    fn test_version_requirement() {
        let req = VersionRequirement::new("1.0.0");
        let version = Version::new(1, 0, 0);

        assert!(req.version(&version));
    }

    // --- VersionRequirement: real semver matching (was exact-string-equality) ---

    #[test]
    fn test_version_requirement_caret_default_matches_compatible_range() {
        let req = VersionRequirement::new("1.2.0"); // bare requirement = caret
        assert!(req.version(&Version::new(1, 2, 0)));
        assert!(req.version(&Version::new(1, 2, 5)));
        assert!(req.version(&Version::new(1, 9, 9)));
        assert!(
            !req.version(&Version::new(1, 1, 9)),
            "below the required minor.patch must not match"
        );
        assert!(
            !req.version(&Version::new(2, 0, 0)),
            "a different major version is a breaking change under caret"
        );
    }

    #[test]
    fn test_version_requirement_caret_zero_major_locks_minor() {
        let req = VersionRequirement::new("^0.2.3");
        assert!(req.version(&Version::new(0, 2, 3)));
        assert!(req.version(&Version::new(0, 2, 9)));
        assert!(
            !req.version(&Version::new(0, 3, 0)),
            "0.x caret requirements lock the minor version (Cargo semantics)"
        );
    }

    #[test]
    fn test_version_requirement_tilde_locks_minor_allows_patch() {
        let req = VersionRequirement::new("~1.4.2");
        assert!(req.version(&Version::new(1, 4, 2)));
        assert!(req.version(&Version::new(1, 4, 9)));
        assert!(!req.version(&Version::new(1, 5, 0)));
        assert!(!req.version(&Version::new(1, 4, 1)));
    }

    #[test]
    fn test_version_requirement_comparison_operators() {
        assert!(VersionRequirement::new(">=1.0.0").version(&Version::new(1, 5, 0)));
        assert!(!VersionRequirement::new(">=1.0.0").version(&Version::new(0, 9, 0)));
        assert!(VersionRequirement::new("<2.0.0").version(&Version::new(1, 9, 9)));
        assert!(!VersionRequirement::new("<2.0.0").version(&Version::new(2, 0, 0)));
        assert!(VersionRequirement::new("=1.2.3").version(&Version::new(1, 2, 3)));
        assert!(!VersionRequirement::new("=1.2.3").version(&Version::new(1, 2, 4)));
    }

    #[test]
    fn test_version_requirement_compound_range() {
        let req = VersionRequirement::new(">=1.2.0, <2.0.0");
        assert!(req.version(&Version::new(1, 2, 0)));
        assert!(req.version(&Version::new(1, 9, 9)));
        assert!(!req.version(&Version::new(2, 0, 0)));
        assert!(!req.version(&Version::new(1, 1, 9)));
    }

    #[test]
    fn test_version_requirement_wildcard_matches_anything() {
        let req = VersionRequirement::new("*");
        assert!(req.version(&Version::new(0, 0, 1)));
        assert!(req.version(&Version::new(99, 99, 99)));
    }

    #[test]
    fn test_version_requirement_rejects_unparsable_clause() {
        let req = VersionRequirement::new("not-a-version");
        assert!(
            !req.version(&Version::new(1, 0, 0)),
            "an unparsable requirement must never be silently satisfied"
        );
    }

    // --- has_circular_dependency: real DFS (was hardcoded `false`) ---

    #[test]
    fn test_has_circular_dependency_detects_real_cycle() {
        let validator = EcosystemValidator::new().expect("construct validator");
        let mut registry = ModuleRegistry::new();
        // Existing edges: a -> b -> c
        registry
            .register(make_module("a", "1.0.0", &[("b", "*")]))
            .expect("register a");
        registry
            .register(make_module("b", "1.0.0", &[("c", "*")]))
            .expect("register b");
        registry
            .register(make_module("c", "1.0.0", &[]))
            .expect("register c");
        registry
            .register(make_module("d", "1.0.0", &[]))
            .expect("register d");

        assert!(
            validator.has_circular_dependency(&registry, "c", "a"),
            "c already transitively depends on a (via b); c -> a must close a cycle"
        );
        assert!(
            !validator.has_circular_dependency(&registry, "a", "d"),
            "d has no dependencies; a -> d cannot create a cycle"
        );
        assert!(
            validator.has_circular_dependency(&registry, "a", "a"),
            "a direct self-dependency is trivially a cycle"
        );
    }

    #[test]
    fn test_validate_module_detects_real_circular_dependency() {
        let validator = EcosystemValidator::new().expect("construct validator");
        // cyc-a -> cyc-b -> cyc-a: a genuine cycle.
        validator
            .register_module(make_module("cyc-a", "1.0.0", &[("cyc-b", "*")]))
            .expect("register cyc-a");
        validator
            .register_module(make_module("cyc-b", "1.0.0", &[("cyc-a", "*")]))
            .expect("register cyc-b");

        let result = validator.validate_module("cyc-a").expect("validate cyc-a");
        assert!(
            !result.is_valid(),
            "a real circular dependency must be reported as invalid"
        );
        assert!(
            result.errors.iter().any(|e| e.message.contains("cyc-b")),
            "expected a dependency error mentioning the offending dependency; got: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_validate_module_does_not_flag_acyclic_chain_as_circular() {
        let validator = EcosystemValidator::new().expect("construct validator");
        validator
            .register_module(make_module("chain-a", "1.0.0", &[("chain-b", "*")]))
            .expect("register chain-a");
        validator
            .register_module(make_module("chain-b", "1.0.0", &[("chain-c", "*")]))
            .expect("register chain-b");
        validator
            .register_module(make_module("chain-c", "1.0.0", &[]))
            .expect("register chain-c");

        let result = validator
            .validate_module("chain-a")
            .expect("validate chain-a");
        assert!(
            result.is_valid(),
            "a simple acyclic dependency chain must not be flagged circular; errors: {:?}",
            result.errors
        );
    }

    // --- Dependency version-requirement enforcement now uses real semver ---

    #[test]
    fn test_validate_module_accepts_real_semver_range_satisfied_by_dependency() {
        let validator = EcosystemValidator::new().expect("construct validator");
        validator
            .register_module(make_module("consumer", "1.0.0", &[("provider", "^1.2.0")]))
            .expect("register consumer");
        validator
            .register_module(make_module("provider", "1.7.3", &[]))
            .expect("register provider");

        let result = validator
            .validate_module("consumer")
            .expect("validate consumer");
        assert!(
            result.is_valid(),
            "provider 1.7.3 satisfies ^1.2.0; expected no dependency error, got: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_validate_module_flags_real_semver_range_violation() {
        let validator = EcosystemValidator::new().expect("construct validator");
        validator
            .register_module(make_module(
                "consumer2",
                "1.0.0",
                &[("provider2", "^2.0.0")],
            ))
            .expect("register consumer2");
        validator
            .register_module(make_module("provider2", "1.7.3", &[]))
            .expect("register provider2");

        let result = validator
            .validate_module("consumer2")
            .expect("validate consumer2");
        assert!(
            !result.is_valid(),
            "provider2 1.7.3 does not satisfy ^2.0.0; expected a dependency error"
        );
    }

    // --- areversions_compatible: real semver comparison (was hardcoded `true`) ---

    #[test]
    fn test_areversions_compatible_respects_strict_policy() {
        let validator = EcosystemValidator::new().expect("construct validator");
        let v1_0_0 = Version::new(1, 0, 0);
        let v1_5_2 = Version::new(1, 5, 2);
        let v2_0_0 = Version::new(2, 0, 0);

        let lenient = ValidationPolicies {
            strict_version_matching: false,
            ..ValidationPolicies::default()
        };
        assert!(
            validator.areversions_compatible(&v1_0_0, &v1_5_2, &lenient),
            "same major version is compatible under lenient policy"
        );
        assert!(
            !validator.areversions_compatible(&v1_0_0, &v2_0_0, &lenient),
            "different major version is incompatible even under lenient policy"
        );

        let strict = ValidationPolicies {
            strict_version_matching: true,
            ..ValidationPolicies::default()
        };
        assert!(
            !validator.areversions_compatible(&v1_0_0, &v1_5_2, &strict),
            "different versions are incompatible under strict policy despite sharing a major"
        );
        assert!(validator.areversions_compatible(&v1_0_0, &v1_0_0, &strict));
    }

    // --- are_apis_compatible: real symbol-conflict check (was hardcoded `true`) ---

    #[test]
    fn test_are_apis_compatible_detects_symbol_conflict() {
        let validator = EcosystemValidator::new().expect("construct validator");
        let surface_v1 = ApiSurface {
            public_apis: vec![api("compute", "fn compute(x: f64) -> f64")],
            deprecated_apis: Vec::new(),
        };
        let surface_conflicting = ApiSurface {
            public_apis: vec![api("compute", "fn compute(x: f64, y: f64) -> f64")],
            deprecated_apis: Vec::new(),
        };
        let surface_disjoint = ApiSurface {
            public_apis: vec![api("other_symbol", "fn other_symbol()")],
            deprecated_apis: Vec::new(),
        };

        assert!(
            !validator.are_apis_compatible(&surface_v1, &surface_conflicting),
            "same-named API with a different signature must be flagged incompatible"
        );
        assert!(
            validator.are_apis_compatible(&surface_v1, &surface_v1.clone()),
            "identical API surfaces must be compatible"
        );
        assert!(
            validator.are_apis_compatible(&surface_v1, &surface_disjoint),
            "disjoint API surfaces (no shared names) must be compatible"
        );
    }

    #[test]
    fn test_validate_ecosystem_flags_real_api_symbol_conflict_between_modules() {
        let validator = EcosystemValidator::new().expect("construct validator");
        let mut mod_x = make_module("conflict-x", "1.0.0", &[]);
        mod_x.apisurface = ApiSurface {
            public_apis: vec![api("shared_symbol", "fn shared_symbol(x: i32) -> i32")],
            deprecated_apis: Vec::new(),
        };
        let mut mod_y = make_module("conflict-y", "1.0.0", &[]);
        mod_y.apisurface = ApiSurface {
            public_apis: vec![api(
                "shared_symbol",
                "fn shared_symbol(x: i32, y: i32) -> i32",
            )],
            deprecated_apis: Vec::new(),
        };

        validator.register_module(mod_x).expect("register x");
        validator.register_module(mod_y).expect("register y");

        let result = validator.validate_ecosystem().expect("validate ecosystem");
        assert!(
            !result.compatibilityresult.incompatibilities.is_empty(),
            "two modules declaring the same symbol with conflicting signatures must be flagged"
        );
    }

    #[test]
    fn test_validate_ecosystem_does_not_flag_unrelated_modules_incompatible() {
        let validator = EcosystemValidator::new().expect("construct validator");
        validator
            .register_module(make_module("solo-p", "1.0.0", &[]))
            .expect("register p");
        validator
            .register_module(make_module("solo-q", "1.0.0", &[]))
            .expect("register q");

        let result = validator.validate_ecosystem().expect("validate ecosystem");
        assert!(
            result.compatibilityresult.incompatibilities.is_empty(),
            "unrelated modules with no conflicting APIs must not be flagged incompatible; got: {:?}",
            result.compatibilityresult.incompatibilities
        );
    }

    // --- are_features_compatible: real policy-blacklist check (was hardcoded `true`) ---

    #[test]
    fn test_are_features_compatible_honors_policy_blacklist() {
        let validator = EcosystemValidator::new().expect("construct validator");
        let mut policies = ValidationPolicies::default();
        policies
            .incompatible_features
            .insert("legacy-abi".to_string());

        let clean_a = vec!["serde".to_string()];
        let clean_b = vec!["simd".to_string()];
        assert!(validator.are_features_compatible(&clean_a, &clean_b, &policies));

        let tainted = vec!["legacy-abi".to_string()];
        assert!(!validator.are_features_compatible(&tainted, &clean_b, &policies));
        assert!(!validator.are_features_compatible(&clean_a, &tainted, &policies));
    }

    // --- check_api_stability: real removed/changed-signature diff (was hardcoded stable) ---

    #[test]
    fn test_check_api_stability_detects_removed_and_changed_apis() {
        let validator = EcosystemValidator::new().expect("construct validator");

        let previous = ApiSurface {
            public_apis: vec![
                api("old_only", "fn old_only()"),
                api("changed", "fn changed(x: i32)"),
            ],
            deprecated_apis: Vec::new(),
        };
        let current = ApiSurface {
            // "old_only" removed; "changed" has a different signature.
            public_apis: vec![api("changed", "fn changed(x: i64)")],
            deprecated_apis: Vec::new(),
        };

        let check = validator.check_api_stability(Some(&previous), &current);
        assert!(
            !check.is_stable(),
            "removing an API and changing a signature must not be stable"
        );
        assert!(check
            .breakingchanges()
            .iter()
            .any(|m| m.contains("old_only") && m.contains("removed")));
        assert!(check
            .breakingchanges()
            .iter()
            .any(|m| m.contains("changed") && m.contains("signature")));

        // With no previous version on record, there is nothing to have broken.
        let no_history = validator.check_api_stability(None, &current);
        assert!(no_history.is_stable());
        assert!(no_history.breakingchanges().is_empty());
    }

    // --- is_api_properly_versioned: real check (was hardcoded `true`) ---

    #[test]
    fn test_is_api_properly_versioned_flags_missing_since_version() {
        let validator = EcosystemValidator::new().expect("construct validator");
        let versioned = ApiSurface {
            public_apis: vec![api("a", "fn a()")],
            deprecated_apis: Vec::new(),
        };
        assert!(validator.is_api_properly_versioned(&versioned));

        let mut unversioned = versioned.clone();
        unversioned.public_apis[0].since_version = None;
        assert!(!validator.is_api_properly_versioned(&unversioned));
    }

    // --- has_known_vulnerabilities: honest Unsupported (was hardcoded `false`) ---

    #[test]
    fn test_has_known_vulnerabilities_is_honestly_unsupported_not_fabricated_safe() {
        let validator = EcosystemValidator::new().expect("construct validator");
        assert_eq!(
            validator.has_known_vulnerabilities("some-dependency"),
            VulnerabilityStatus::Unsupported
        );
    }

    #[test]
    fn test_security_result_distinguishes_unchecked_from_confirmed_secure() {
        let mut result = SecurityValidationResult::new("m".to_string());
        assert!(result.is_secure());
        assert!(result.is_fully_verified());

        result.add_unchecked_dependency("dep-a".to_string());
        assert!(
            result.is_secure(),
            "an unchecked dependency is not a confirmed problem"
        );
        assert!(
            !result.is_fully_verified(),
            "but it must not silently count as fully verified either"
        );
    }

    #[test]
    fn test_ecosystem_health() {
        let mut result = EcosystemValidationResult::new();
        result.add_moduleresult(
            "test".to_string(),
            ModuleValidationResult::new("test".to_string()),
        );

        let health = EcosystemHealth::from_validationresult(&result);
        assert_eq!(health.overall_status, HealthStatus::Excellent);
    }

    // --- validate_apisurface tests ---

    /// Validation passes (is_stable=true, breakingchanges empty) when all expected
    /// stable APIs are present and well-formed.
    #[test]
    fn test_validate_apisurface_passes_when_all_apis_valid() {
        let surface = ApiSurface {
            public_apis: vec![
                ApiInfo {
                    name: "compute_mean".to_string(),
                    signature: "fn compute_mean(data: &[f64]) -> f64".to_string(),
                    documentation: "Computes the arithmetic mean of a slice.".to_string(),
                    since_version: Some(Version::new(1, 0, 0)),
                    stability: ApiStability::Stable,
                },
                ApiInfo {
                    name: "MatrixSolver".to_string(),
                    signature: "struct MatrixSolver".to_string(),
                    documentation: "Solves linear matrix equations.".to_string(),
                    since_version: Some(Version::new(1, 0, 0)),
                    stability: ApiStability::Stable,
                },
            ],
            deprecated_apis: Vec::new(),
        };

        let check = validate_apisurface(&surface);
        assert!(
            check.is_valid(),
            "Expected valid surface; breaking changes: {:?}",
            check.breakingchanges
        );
        assert!(check.breakingchanges.is_empty());
        assert!(check.is_stable);
    }

    /// Validation fails and reports missing `since_version` when a stable API
    /// omits that field (simulating an "absent" required annotation).
    #[test]
    fn test_validate_apisurface_fails_missing_since_version() {
        let surface = ApiSurface {
            public_apis: vec![ApiInfo {
                name: "load_dataset".to_string(),
                signature: "fn load_dataset(path: &str) -> Result<Dataset>".to_string(),
                documentation: "Loads a dataset from disk.".to_string(),
                since_version: None, // intentionally absent
                stability: ApiStability::Stable,
            }],
            deprecated_apis: Vec::new(),
        };

        let check = validate_apisurface(&surface);
        assert!(
            !check.is_stable,
            "Should be unstable due to missing since_version"
        );
        assert!(
            check
                .breakingchanges
                .iter()
                .any(|m| m.contains("since_version")),
            "Expected a message about missing since_version; got: {:?}",
            check.breakingchanges
        );
    }

    /// An empty `public_apis` list should be treated as valid — the module simply
    /// exposes no public API surface yet.
    #[test]
    fn test_validate_apisurface_handles_empty_apis_gracefully() {
        let surface = ApiSurface {
            public_apis: Vec::new(),
            deprecated_apis: Vec::new(),
        };

        let check = validate_apisurface(&surface);
        assert!(
            check.is_valid(),
            "Empty API surface should pass; got: {:?}",
            check.breakingchanges
        );
    }

    /// Validation detects invalid symbol names: a function that uses CamelCase
    /// instead of the required snake_case.
    #[test]
    fn test_validate_apisurface_detects_invalid_symbol_name() {
        let surface = ApiSurface {
            public_apis: vec![ApiInfo {
                name: "ComputeMean".to_string(), // should be snake_case for a fn
                signature: "fn ComputeMean(data: &[f64]) -> f64".to_string(),
                documentation: "Computes the mean.".to_string(),
                since_version: Some(Version::new(1, 0, 0)),
                stability: ApiStability::Stable,
            }],
            deprecated_apis: Vec::new(),
        };

        let check = validate_apisurface(&surface);
        assert!(
            !check.is_stable,
            "Should detect invalid snake_case violation"
        );
        assert!(
            check
                .breakingchanges
                .iter()
                .any(|m| m.contains("snake_case")),
            "Expected snake_case message; got: {:?}",
            check.breakingchanges
        );
    }

    /// The `ApiStabilityCheck` result fields are populated correctly:
    /// `is_stable` reflects validity, and `breakingchanges` contains
    /// one entry per problem found.
    #[test]
    fn test_api_stability_check_fields_populated_correctly() {
        // Surface with two problems: duplicate name and missing documentation
        let api_entry = ApiInfo {
            name: "shared_name".to_string(),
            signature: "fn shared_name()".to_string(),
            documentation: String::new(), // missing documentation → 1 error
            since_version: Some(Version::new(1, 0, 0)),
            stability: ApiStability::Stable,
        };
        let duplicate = api_entry.clone(); // duplicate name → 1 error

        let surface = ApiSurface {
            public_apis: vec![api_entry, duplicate],
            deprecated_apis: Vec::new(),
        };

        let check = validate_apisurface(&surface);

        // is_stable must be false
        assert!(!check.is_stable, "Expected is_stable=false");
        // breakingchanges must not be empty
        assert!(
            !check.breakingchanges.is_empty(),
            "breakingchanges must be non-empty"
        );
        // At least the duplicate should be reported
        assert!(
            check
                .breakingchanges
                .iter()
                .any(|m| m.contains("Duplicate")),
            "Expected a duplicate-name entry; got: {:?}",
            check.breakingchanges
        );
        // At least the missing-documentation error should be reported
        assert!(
            check
                .breakingchanges
                .iter()
                .any(|m| m.contains("documentation")),
            "Expected a missing-documentation entry; got: {:?}",
            check.breakingchanges
        );
    }

    /// Signatures with visibility prefix (`pub fn`, `async fn`, `const fn`, etc.)
    /// are still classified as functions and validated as snake_case.
    #[test]
    fn test_validate_apisurface_pub_fn_is_snake_case() {
        let surface = ApiSurface {
            public_apis: vec![
                ApiInfo {
                    name: "compute_sum".to_string(),
                    signature: "pub fn compute_sum(data: &[f64]) -> f64".to_string(),
                    documentation: "Computes the sum.".to_string(),
                    since_version: Some(Version::new(1, 0, 0)),
                    stability: ApiStability::Stable,
                },
                ApiInfo {
                    name: "async_fetch".to_string(),
                    signature: "async fn async_fetch() -> Result<Data>".to_string(),
                    documentation: "Fetches data asynchronously.".to_string(),
                    since_version: Some(Version::new(1, 0, 0)),
                    stability: ApiStability::Stable,
                },
            ],
            deprecated_apis: Vec::new(),
        };

        let check = validate_apisurface(&surface);
        assert!(
            check.is_valid(),
            "pub fn / async fn should pass snake_case check; got: {:?}",
            check.breakingchanges
        );
    }

    /// Deprecated API name that collides with a current public API name is flagged.
    #[test]
    fn test_validate_apisurface_detects_deprecated_collision() {
        let surface = ApiSurface {
            public_apis: vec![ApiInfo {
                name: "my_function".to_string(),
                signature: "fn my_function() -> u32".to_string(),
                documentation: "Does something.".to_string(),
                since_version: Some(Version::new(1, 0, 0)),
                stability: ApiStability::Stable,
            }],
            deprecated_apis: vec![DeprecatedApiInfo {
                name: "my_function".to_string(), // same name as public API → collision
                deprecated_since: Version::new(0, 9, 0),
                removal_version: Some(Version::new(2, 0, 0)),
                migration_path: Some("Use my_function v2".to_string()),
            }],
        };

        let check = validate_apisurface(&surface);
        assert!(!check.is_stable, "Collision should mark surface unstable");
        assert!(
            check.breakingchanges.iter().any(|m| m.contains("collides")),
            "Expected collision message; got: {:?}",
            check.breakingchanges
        );
    }
}
