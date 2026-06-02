//! Geometric integration methods
//!
//! This module provides structure-preserving numerical integrators for systems
//! with geometric properties such as:
//! - Lie group structure
//! - Volume preservation (divergence-free flows)
//! - Energy conservation (Hamiltonian systems)
//! - Momentum conservation (Lagrangian systems)

pub mod lie_group;
pub mod port_hamiltonian;
pub mod structure_preserving;
pub mod volume_preserving;

pub use lie_group::{
    ExponentialMap, GLn, Gln, HeisenbergAlgebra, HeisenbergGroup, LieAlgebra, LieGroupIntegrator,
    LieGroupMethod, SE3Integrator, SLn, SO3Integrator, Se3, Sln, So3, Sp2n, SE3, SO3,
};
pub use structure_preserving::{
    invariants::{AngularMomentumInvariant2D, EnergyInvariant, LinearMomentumInvariant},
    ConservationChecker, ConstrainedIntegrator, EnergyMomentumIntegrator, EnergyPreservingMethod,
    GeometricInvariant, MomentumPreservingMethod, MultiSymplecticIntegrator, SplittingIntegrator,
    StructurePreservingIntegrator, StructurePreservingMethod,
};
pub use volume_preserving::{
    gauss_legendre_quadrature, ABCFlow, CircularFlow2D, DiscreteGradientIntegrator,
    DivergenceFreeFlow, DoubleGyre, HamiltonianFlow, IncompressibleFlow,
    ModifiedMidpointIntegrator, StreamFunction, StuartVortex, TaylorGreenVortex,
    VariationalIntegrator, VolumeChecker, VolumePreservingIntegrator, VolumePreservingMethod,
};
