//! Ordinary Differential Equation solvers
//!
//! This module provides numerical solvers for ordinary differential equations (ODEs).
//! It includes a variety of methods for solving initial value problems (IVPs).
//!
//! Features:
//! - Multiple methods including explicit and implicit solvers
//! - Automatic stiffness detection and method switching
//! - Dense output for continuous solution approximation
//! - Event detection and handling for detecting specific conditions
//! - Support for different error control schemes

// Public types module
pub mod types;

// Public modules
pub mod chemical;
pub mod chemical_equilibrium;
pub mod enzyme_kinetics;
pub mod mechanical;
pub mod methods;
pub mod multirate;
pub mod solver;
pub mod utils;

// Re-export core types
pub use self::types::{MassMatrix, MassMatrixType, ODEMethod, ODEOptions, ODEResult};

// Re-export chemical kinetics types
pub use self::chemical::{
    systems as chemical_systems, ChemicalConfig, ChemicalIntegrator, ChemicalProperties,
    ChemicalState, ChemicalSystemType, Reaction, ReactionType,
};

// Re-export enzyme kinetics types
pub use self::enzyme_kinetics::{
    pathways as metabolic_pathways, EnzymeDefinition, EnzymeMechanism, EnzymeParameters,
    MetabolicPathway, PathwayAnalysis, RegulationType,
};

// Re-export chemical equilibrium types
pub use self::chemical_equilibrium::{
    systems as equilibrium_systems, ActivityModel, EquilibriumCalculator, EquilibriumResult,
    EquilibriumType, ThermoData,
};

// Re-export mechanical systems types
pub use self::mechanical::{
    systems as mechanical_systems, MechanicalConfig, MechanicalIntegrator, MechanicalProperties,
    MechanicalSystemType, RigidBodyState,
};

// Re-export solver functions
pub use self::solver::{solve_ivp, solve_ivp_with_events};

// Re-export event detection types
pub use self::utils::events::{
    terminal_event, EventAction, EventDirection, EventSpec, ODEOptionsWithEvents,
    ODEResultWithEvents,
};

// Re-export multirate types
pub use self::multirate::{MultirateMethod, MultirateOptions, MultirateSolver, MultirateSystem};
