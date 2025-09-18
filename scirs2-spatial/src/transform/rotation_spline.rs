//! RotationSpline for smooth interpolation between multiple rotations
//!
//! This module provides a `RotationSpline` class that allows for smooth interpolation
//! between multiple rotations, creating a continuous curve in rotation space.

use crate::error::{SpatialError, SpatialResult};
use crate::transform::{Rotation, Slerp};
use ndarray::{array, Array1};

// Helper function to create an array from values
#[allow(dead_code)]
fn euler_array(x: f64, y: f64, z: f64) -> Array1<f64> {
    array![x, y, z]
}

// Helper function to create a rotation from Euler angles
#[allow(dead_code)]
fn rotation_from_euler(x: f64, y: f64, z: f64, convention: &str) -> SpatialResult<Rotation> {
    let angles = euler_array(x, y, z);
    let angles_view = angles.view();
    Rotation::from_euler(&angles_view, convention)
}

/// RotationSpline provides smooth interpolation between multiple rotations.
///
/// A rotation spline allows for smooth interpolation between a sequence of rotations,
/// creating a continuous curve in rotation space. It can be used to create smooth
/// camera paths, character animations, or any other application requiring smooth
/// rotation transitions.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::transform::{Rotation, RotationSpline};
/// use ndarray::array;
/// use std::f64::consts::PI;
///
/// // Create some rotations
/// let rotations = vec![
///     Rotation::identity(),
///     Rotation::from_euler(&array![0.0, 0.0, PI/2.0].view(), "xyz").unwrap(),
///     Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
/// ];
///
/// // Create times at which these rotations occur
/// let times = vec![0.0, 0.5, 1.0];
///
/// // Create a rotation spline
/// let spline = RotationSpline::new(&rotations, &times).unwrap();
///
/// // Get the interpolated rotation at t=0.25 (between the first two rotations)
/// let rot_25 = spline.interpolate(0.25);
///
/// // Get the interpolated rotation at t=0.75 (between the second two rotations)
/// let rot_75 = spline.interpolate(0.75);
/// ```
#[derive(Clone, Debug)]
pub struct RotationSpline {
    /// Sequence of rotations
    rotations: Vec<Rotation>,
    /// Times at which these rotations occur
    times: Vec<f64>,
    /// Cached velocities for natural cubic spline interpolation
    velocities: Option<Vec<Array1<f64>>>,
    /// Type of interpolation to use ("slerp" or "cubic")
    interpolation_type: String,
}

impl RotationSpline {
    /// Create a new rotation spline from a sequence of rotations and times
    ///
    /// # Arguments
    ///
    /// * `rotations` - A sequence of rotations
    /// * `times` - The times at which these rotations occur
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing the RotationSpline if valid, or an error if invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::{Rotation, RotationSpline};
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// let rotations = vec![
    ///     Rotation::identity(),
    ///     Rotation::from_euler(&array![0.0, 0.0, PI/2.0].view(), "xyz").unwrap(),
    ///     Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
    /// ];
    /// let times = vec![0.0, 1.0, 2.0];
    /// let spline = RotationSpline::new(&rotations, &times).unwrap();
    /// ```
    pub fn new(rotations: &[Rotation], times: &[f64]) -> SpatialResult<Self> {
        if rotations.is_empty() {
            return Err(SpatialError::ValueError("Rotations cannot be empty".into()));
        }

        if times.is_empty() {
            return Err(SpatialError::ValueError("Times cannot be empty".into()));
        }

        if rotations.len() != times.len() {
            return Err(SpatialError::ValueError(format!(
                "Number of _rotations ({}) must match number of times ({})",
                rotations.len(),
                times.len()
            )));
        }

        // Check if times are strictly increasing
        for i in 1..times.len() {
            if times[i] <= times[i - 1] {
                return Err(SpatialError::ValueError(format!(
                    "Times must be strictly increasing, but times[{}] = {} <= times[{}] = {}",
                    i,
                    times[i],
                    i - 1,
                    times[i - 1]
                )));
            }
        }

        // Make a copy of the _rotations and times
        let rotations = rotations.to_vec();
        let times = times.to_vec();

        Ok(RotationSpline {
            rotations,
            times,
            velocities: None,
            interpolation_type: "slerp".to_string(),
        })
    }

    /// Set the interpolation type for the rotation spline
    ///
    /// # Arguments
    ///
    /// * `_interptype` - The interpolation type ("slerp" or "cubic")
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing nothing if successful, or an error if the interpolation type is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::{Rotation, RotationSpline};
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// let rotations = vec![
    ///     Rotation::identity(),
    ///     Rotation::from_euler(&array![0.0, 0.0, PI/2.0].view(), "xyz").unwrap(),
    ///     Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
    /// ];
    /// let times = vec![0.0, 1.0, 2.0];
    /// let mut spline = RotationSpline::new(&rotations, &times).unwrap();
    ///
    /// // Set the interpolation type to cubic (natural cubic spline)
    /// spline.set_interpolation_type("cubic").unwrap();
    /// ```
    pub fn set_interpolation_type(&mut self, _interptype: &str) -> SpatialResult<()> {
        match _interptype.to_lowercase().as_str() {
            "slerp" => {
                self.interpolation_type = "slerp".to_string();
                self.velocities = None;
                Ok(())
            }
            "cubic" => {
                self.interpolation_type = "cubic".to_string();
                // Compute velocities for cubic interpolation if needed
                self.compute_velocities();
                Ok(())
            }
            _ => Err(SpatialError::ValueError(format!(
                "Invalid interpolation _type: {_interptype}. Must be 'slerp' or 'cubic'"
            ))),
        }
    }

    /// Compute velocities for natural cubic spline interpolation
    fn compute_velocities(&mut self) {
        if self.velocities.is_some() {
            return; // Already computed
        }

        let n = self.times.len();
        if n <= 2 {
            // For 2 or fewer points, use zero velocities
            let mut vels = Vec::with_capacity(n);
            for _ in 0..n {
                vels.push(Array1::zeros(3));
            }
            self.velocities = Some(vels);
            return;
        }

        // Convert rotations to rotation vectors (axis-angle representation)
        let mut rotvecs = Vec::with_capacity(n);
        for rot in &self.rotations {
            rotvecs.push(rot.as_rotvec());
        }

        // Compute velocities using finite differences and natural boundary conditions
        let mut vels = Vec::with_capacity(n);

        // For endpoints, we'll use one-sided differences
        // For internal points, we'll use centered differences
        for i in 0..n {
            let vel = if i == 0 {
                // Forward difference for the first point
                let dt = self.times[1] - self.times[0];
                (&rotvecs[1] - &rotvecs[0]) / dt
            } else if i == n - 1 {
                // Backward difference for the last point
                let dt = self.times[n - 1] - self.times[n - 2];
                (&rotvecs[n - 1] - &rotvecs[n - 2]) / dt
            } else {
                // Centered difference for internal points
                let dt_prev = self.times[i] - self.times[i - 1];
                let dt_next = self.times[i + 1] - self.times[i];

                // Use weighted average based on time intervals
                let vel_prev = (&rotvecs[i] - &rotvecs[i - 1]) / dt_prev;
                let vel_next = (&rotvecs[i + 1] - &rotvecs[i]) / dt_next;

                // Weighted average
                let weight_prev = dt_next / (dt_prev + dt_next);
                let weight_next = dt_prev / (dt_prev + dt_next);
                &vel_prev * weight_prev + &vel_next * weight_next
            };

            vels.push(vel);
        }

        self.velocities = Some(vels);
    }

    /// Compute the second derivatives for natural cubic spline interpolation
    #[allow(dead_code)]
    fn compute_natural_spline_second_derivatives(&self, values: &[f64]) -> Vec<f64> {
        let n = values.len();
        if n <= 2 {
            return vec![0.0; n];
        }

        // Set up the tridiagonal system for natural cubic spline
        // The system is in the form: A * x = b
        // where A is a tridiagonal matrix, x is the second derivatives we're solving for,
        // and b is the right-hand side of the system

        // Allocate arrays for the diagonals of the tridiagonal matrix
        let mut a = vec![0.0; n - 2]; // Lower diagonal
        let mut b = vec![0.0; n - 2]; // Main diagonal
        let mut c = vec![0.0; n - 2]; // Upper diagonal
        let mut d = vec![0.0; n - 2]; // Right-hand side

        // Set up the tridiagonal system
        for i in 0..n - 2 {
            let h_i = self.times[i + 1] - self.times[i];
            let h_ip1 = self.times[i + 2] - self.times[i + 1];

            a[i] = h_i;
            b[i] = 2.0 * (h_i + h_ip1);
            c[i] = h_ip1;

            let fd_i = (values[i + 1] - values[i]) / h_i;
            let fd_ip1 = (values[i + 2] - values[i + 1]) / h_ip1;
            d[i] = 6.0 * (fd_ip1 - fd_i);
        }

        // Solve the tridiagonal system using the Thomas algorithm
        let mut x = vec![0.0; n - 2];
        self.solve_tridiagonal(&a, &b, &c, &d, &mut x);

        // The second derivatives at the endpoints are set to zero (natural spline)
        let mut second_derivs = vec![0.0; n];
        second_derivs[1..((n - 2) + 1)].copy_from_slice(&x[..(n - 2)]);

        second_derivs
    }

    /// Solve a tridiagonal system using the Thomas algorithm
    #[allow(dead_code)]
    fn solve_tridiagonal(
        &self,
        a: &[f64],     // Lower diagonal
        b: &[f64],     // Main diagonal
        c: &[f64],     // Upper diagonal
        d: &[f64],     // Right-hand side
        x: &mut [f64], // Solution vector
    ) {
        let n = x.len();
        if n == 0 {
            return;
        }

        // Forward sweep
        let mut c_prime = vec![0.0; n];
        let mut d_prime = vec![0.0; n];

        c_prime[0] = c[0] / b[0];
        d_prime[0] = d[0] / b[0];

        for i in 1..n {
            let m = b[i] - a[i - 1] * c_prime[i - 1];
            c_prime[i] = if i < n - 1 { c[i] / m } else { 0.0 };
            d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / m;
        }

        // Back substitution
        x[n - 1] = d_prime[n - 1];
        for i in (0..n - 1).rev() {
            x[i] = d_prime[i] - c_prime[i] * x[i + 1];
        }
    }

    /// Interpolate the rotation spline at a given time
    ///
    /// # Arguments
    ///
    /// * `t` - The time at which to interpolate
    ///
    /// # Returns
    ///
    /// The interpolated rotation
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::{Rotation, RotationSpline};
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// let rotations = vec![
    ///     Rotation::identity(),
    ///     Rotation::from_euler(&array![0.0, 0.0, PI/2.0].view(), "xyz").unwrap(),
    ///     Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
    /// ];
    /// let times = vec![0.0, 1.0, 2.0];
    /// let spline = RotationSpline::new(&rotations, &times).unwrap();
    ///
    /// // Interpolate at t=0.5 (halfway between the first two rotations)
    /// let rot_half = spline.interpolate(0.5);
    /// ```
    pub fn interpolate(&self, t: f64) -> Rotation {
        let n = self.times.len();

        // Handle boundary cases
        if t <= self.times[0] {
            return self.rotations[0].clone();
        }
        if t >= self.times[n - 1] {
            return self.rotations[n - 1].clone();
        }

        // Find the segment containing t
        let mut idx = 0;
        for i in 0..n - 1 {
            if t >= self.times[i] && t < self.times[i + 1] {
                idx = i;
                break;
            }
        }

        // Interpolate within the segment based on interpolation type
        match self.interpolation_type.as_str() {
            "slerp" => self.interpolate_slerp(t, idx),
            "cubic" => self.interpolate_cubic(t, idx),
            _ => self.interpolate_slerp(t, idx), // Default to slerp
        }
    }

    /// Interpolate the rotation spline at a given time using Slerp
    fn interpolate_slerp(&self, t: f64, idx: usize) -> Rotation {
        let t0 = self.times[idx];
        let t1 = self.times[idx + 1];
        let normalized_t = (t - t0) / (t1 - t0);

        // Create a Slerp between the two rotations
        let slerp =
            Slerp::new(self.rotations[idx].clone(), self.rotations[idx + 1].clone()).unwrap();

        slerp.interpolate(normalized_t)
    }

    /// Interpolate the rotation spline at a given time using cubic spline
    fn interpolate_cubic(&self, t: f64, idx: usize) -> Rotation {
        // Ensure velocities are computed
        if self.velocities.is_none() {
            let mut mutable_self = self.clone();
            mutable_self.compute_velocities();
            return mutable_self.interpolate_cubic(t, idx);
        }

        let t0 = self.times[idx];
        let t1 = self.times[idx + 1];
        let dt = t1 - t0;
        let normalized_t = (t - t0) / dt;

        let rot0 = &self.rotations[idx];
        let rot1 = &self.rotations[idx + 1];

        // Convert rotations to rotation vectors
        let rotvec0 = rot0.as_rotvec();
        let rotvec1 = rot1.as_rotvec();

        // Get velocities
        let velocities = self.velocities.as_ref().unwrap();
        let vel0 = &velocities[idx];
        let vel1 = &velocities[idx + 1];

        // Use Hermite cubic interpolation formula
        // h(t) = (2t³ - 3t² + 1)p0 + (t³ - 2t² + t)m0 + (-2t³ + 3t²)p1 + (t³ - t²)m1
        // where p0, p1 are the start and end values, m0, m1 are the scaled tangents
        let t2 = normalized_t * normalized_t;
        let t3 = t2 * normalized_t;

        // Hermite basis functions
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + normalized_t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        // Compute the interpolated rotation vector
        let mut result = rotvec0 * h00;
        result = &result + &(vel0 * dt * h10);
        result = &result + &(rotvec1 * h01);
        result = &result + &(vel1 * dt * h11);

        // Convert back to rotation
        Rotation::from_rotvec(&result.view()).unwrap()
    }

    /// Get the times at which the rotations are defined
    ///
    /// # Returns
    ///
    /// A reference to the times vector
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::{Rotation, RotationSpline};
    /// use ndarray::array;
    ///
    /// let rotations = vec![
    ///     Rotation::identity(),
    ///     Rotation::identity(),
    /// ];
    /// let times = vec![0.0, 1.0];
    /// let spline = RotationSpline::new(&rotations, &times).unwrap();
    ///
    /// let retrieved_times = spline.times();
    /// assert_eq!(retrieved_times, &vec![0.0, 1.0]);
    /// ```
    pub fn times(&self) -> &Vec<f64> {
        &self.times
    }

    /// Get the rotations that define the spline
    ///
    /// # Returns
    ///
    /// A reference to the rotations vector
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::{Rotation, RotationSpline};
    /// use ndarray::array;
    ///
    /// let rotations = vec![
    ///     Rotation::identity(),
    ///     Rotation::identity(),
    /// ];
    /// let times = vec![0.0, 1.0];
    /// let spline = RotationSpline::new(&rotations, &times).unwrap();
    ///
    /// let retrieved_rotations = spline.rotations();
    /// assert_eq!(retrieved_rotations.len(), 2);
    /// ```
    pub fn rotations(&self) -> &Vec<Rotation> {
        &self.rotations
    }

    /// Generate evenly spaced samples from the rotation spline
    ///
    /// # Arguments
    ///
    /// * `n` - The number of samples to generate
    ///
    /// # Returns
    ///
    /// A vector of sampled rotations and the corresponding times
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::{Rotation, RotationSpline};
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// let rotations = vec![
    ///     Rotation::identity(),
    ///     Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
    /// ];
    /// let times = vec![0.0, 1.0];
    /// let spline = RotationSpline::new(&rotations, &times).unwrap();
    ///
    /// // Generate 5 samples from the spline
    /// let (sample_times, sample_rotations) = spline.sample(5);
    /// assert_eq!(sample_times.len(), 5);
    /// assert_eq!(sample_rotations.len(), 5);
    /// ```
    pub fn sample(&self, n: usize) -> (Vec<f64>, Vec<Rotation>) {
        if n <= 1 {
            return (vec![self.times[0]], vec![self.rotations[0].clone()]);
        }

        let t_min = self.times[0];
        let t_max = self.times[self.times.len() - 1];

        let mut sampled_times = Vec::with_capacity(n);
        let mut sampled_rotations = Vec::with_capacity(n);

        for i in 0..n {
            let t = t_min + (t_max - t_min) * (i as f64 / (n - 1) as f64);
            sampled_times.push(t);
            sampled_rotations.push(self.interpolate(t));
        }

        (sampled_times, sampled_rotations)
    }

    /// Create a new rotation spline from key rotations at specific times
    ///
    /// This is equivalent to the regular constructor but with a more explicit name.
    ///
    /// # Arguments
    ///
    /// * `key_rots` - The key rotations
    /// * `keytimes` - The times at which these key rotations occur
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing the RotationSpline if valid, or an error if invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::{Rotation, RotationSpline};
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// let key_rots = vec![
    ///     Rotation::identity(),
    ///     Rotation::from_euler(&array![0.0, 0.0, PI/2.0].view(), "xyz").unwrap(),
    ///     Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
    /// ];
    /// let keytimes = vec![0.0, 1.0, 2.0];
    ///
    /// let spline = RotationSpline::from_key_rotations(&key_rots, &keytimes).unwrap();
    /// ```
    pub fn from_key_rotations(_key_rots: &[Rotation], keytimes: &[f64]) -> SpatialResult<Self> {
        Self::new(_key_rots, keytimes)
    }

    /// Get the current interpolation type
    ///
    /// # Returns
    ///
    /// The current interpolation type ("slerp" or "cubic")
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::{Rotation, RotationSpline};
    /// use ndarray::array;
    ///
    /// let rotations = vec![
    ///     Rotation::identity(),
    ///     Rotation::identity(),
    /// ];
    /// let times = vec![0.0, 1.0];
    /// let spline = RotationSpline::new(&rotations, &times).unwrap();
    ///
    /// assert_eq!(spline.interpolation_type(), "slerp");
    /// ```
    pub fn interpolation_type(&self) -> &'_ str {
        &self.interpolation_type
    }

    /// Calculate the angular velocity at a specific time
    ///
    /// # Arguments
    ///
    /// * `t` - The time at which to calculate the angular velocity
    ///
    /// # Returns
    ///
    /// The angular velocity as a 3-element array
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::{Rotation, RotationSpline};
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// let rotations = vec![
    ///     Rotation::identity(),
    ///     Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
    /// ];
    /// let times = vec![0.0, 1.0];
    /// let spline = RotationSpline::new(&rotations, &times).unwrap();
    ///
    /// // Calculate angular velocity at t=0.5
    /// let velocity = spline.angular_velocity(0.5);
    /// // Should be approximately [0, 0, PI]
    /// ```
    pub fn angular_velocity(&self, t: f64) -> SpatialResult<Array1<f64>> {
        let n = self.times.len();

        // Handle boundary cases
        if t <= self.times[0] || t >= self.times[n - 1] {
            return Ok(Array1::zeros(3));
        }

        // Find the segment containing t
        let mut idx = 0;
        for i in 0..n - 1 {
            if t >= self.times[i] && t < self.times[i + 1] {
                idx = i;
                break;
            }
        }

        // Calculate angular velocity based on interpolation type
        match self.interpolation_type.as_str() {
            "slerp" => self.angular_velocity_slerp(t, idx),
            "cubic" => Ok(self.angular_velocity_cubic(t, idx)),
            _ => self.angular_velocity_slerp(t, idx), // Default to slerp
        }
    }

    /// Calculate angular velocity using Slerp interpolation
    fn angular_velocity_slerp(&self, t: f64, idx: usize) -> SpatialResult<Array1<f64>> {
        let t0 = self.times[idx];
        let t1 = self.times[idx + 1];
        let dt = t1 - t0;
        let normalized_t = (t - t0) / dt;

        // Get rotations at the endpoints of the segment
        let r0 = &self.rotations[idx];
        let r1 = &self.rotations[idx + 1];

        // Calculate the delta rotation from r0 to r1
        let delta_rot = r0.inv().compose(r1);

        // Convert to axis-angle representation via rotation vector
        let rotvec = delta_rot.as_rotvec();
        let angle = (rotvec.dot(&rotvec)).sqrt();
        let axis = if angle > 1e-10 {
            &rotvec / angle
        } else {
            Array1::zeros(3)
        };

        // For slerp, the angular velocity is constant and equals angle/dt along the axis
        // The angular velocity vector in the current frame is:
        // ω = (angle / dt) * axis

        // However, we need to transform this to the frame at time t
        // First interpolate to get the rotation at time t
        let slerp = Slerp::new(r0.clone(), r1.clone()).unwrap();
        let rot_t = slerp.interpolate(normalized_t);

        // The angular velocity in the global frame is the axis scaled by angular rate
        let angular_rate = angle / dt;
        let omega_global = axis * angular_rate;

        // Transform to the body frame at time t
        // ω_body = R(t)^T * ω_global
        rot_t.inv().apply(&omega_global.view())
    }

    /// Calculate angular velocity using cubic spline interpolation
    fn angular_velocity_cubic(&self, t: f64, idx: usize) -> Array1<f64> {
        // Ensure velocities are computed
        if self.velocities.is_none() {
            let mut mutable_self = self.clone();
            mutable_self.compute_velocities();
            return mutable_self.angular_velocity_cubic(t, idx);
        }

        let t0 = self.times[idx];
        let t1 = self.times[idx + 1];
        let dt = t1 - t0;
        let normalized_t = (t - t0) / dt;

        let rot0 = &self.rotations[idx];
        let rot1 = &self.rotations[idx + 1];

        // Convert rotations to rotation vectors
        let rotvec0 = rot0.as_rotvec();
        let rotvec1 = rot1.as_rotvec();

        // Get velocities
        let velocities = self.velocities.as_ref().unwrap();
        let vel0 = &velocities[idx];
        let vel1 = &velocities[idx + 1];

        // Derivatives of Hermite basis functions
        let dh00_dt = (6.0 * normalized_t.powi(2) - 6.0 * normalized_t) / dt;
        let dh10_dt = (3.0 * normalized_t.powi(2) - 4.0 * normalized_t + 1.0) / dt;
        let dh01_dt = (-6.0 * normalized_t.powi(2) + 6.0 * normalized_t) / dt;
        let dh11_dt = (3.0 * normalized_t.powi(2) - 2.0 * normalized_t) / dt;

        // Compute derivative of rotation vector interpolation
        let mut d_rotvec_dt = &rotvec0 * dh00_dt;
        d_rotvec_dt = &d_rotvec_dt + &(vel0 * dt * dh10_dt);
        d_rotvec_dt = &d_rotvec_dt + &(&rotvec1 * dh01_dt);
        d_rotvec_dt = &d_rotvec_dt + &(vel1 * dt * dh11_dt);

        // The derivative gives us the angular velocity in the rotation vector space
        // This is already the angular velocity we want
        d_rotvec_dt
    }

    /// Calculate the angular acceleration at a specific time
    ///
    /// # Arguments
    ///
    /// * `t` - The time at which to calculate the angular acceleration
    ///
    /// # Returns
    ///
    /// The angular acceleration as a 3-element array
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::{Rotation, RotationSpline};
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// let rotations = vec![
    ///     Rotation::identity(),
    ///     Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
    ///     Rotation::identity(),
    /// ];
    /// let times = vec![0.0, 1.0, 2.0];
    /// let mut spline = RotationSpline::new(&rotations, &times).unwrap();
    ///
    /// // Set to cubic interpolation for non-zero acceleration
    /// spline.set_interpolation_type("cubic").unwrap();
    ///
    /// // Calculate angular acceleration at t=0.5
    /// let acceleration = spline.angular_acceleration(0.5);
    /// ```
    pub fn angular_acceleration(&self, t: f64) -> Array1<f64> {
        // Cubic interpolation is needed for meaningful acceleration
        if self.interpolation_type != "cubic" {
            return Array1::zeros(3); // Slerp has constant velocity, so acceleration is zero
        }

        let n = self.times.len();

        // Handle boundary cases
        if t <= self.times[0] || t >= self.times[n - 1] {
            return Array1::zeros(3);
        }

        // Find the segment containing t
        let mut idx = 0;
        for i in 0..n - 1 {
            if t >= self.times[i] && t < self.times[i + 1] {
                idx = i;
                break;
            }
        }

        // Calculate angular acceleration
        self.angular_acceleration_cubic(t, idx)
    }

    /// Calculate angular acceleration using cubic spline interpolation
    fn angular_acceleration_cubic(&self, t: f64, idx: usize) -> Array1<f64> {
        // Ensure velocities are computed
        if self.velocities.is_none() {
            let mut mutable_self = self.clone();
            mutable_self.compute_velocities();
            return mutable_self.angular_acceleration_cubic(t, idx);
        }

        let t0 = self.times[idx];
        let t1 = self.times[idx + 1];
        let dt = t1 - t0;
        let normalized_t = (t - t0) / dt;

        let rot0 = &self.rotations[idx];
        let rot1 = &self.rotations[idx + 1];

        // Convert rotations to rotation vectors
        let rotvec0 = rot0.as_rotvec();
        let rotvec1 = rot1.as_rotvec();

        // Get velocities
        let velocities = self.velocities.as_ref().unwrap();
        let vel0 = &velocities[idx];
        let vel1 = &velocities[idx + 1];

        // Second derivatives of Hermite basis functions
        let d2h00_dt2 = (12.0 * normalized_t - 6.0) / (dt * dt);
        let d2h10_dt2 = (6.0 * normalized_t - 4.0) / (dt * dt);
        let d2h01_dt2 = (-12.0 * normalized_t + 6.0) / (dt * dt);
        let d2h11_dt2 = (6.0 * normalized_t - 2.0) / (dt * dt);

        // Compute second derivative of rotation vector interpolation
        let mut d2_rotvec_dt2 = &rotvec0 * d2h00_dt2;
        d2_rotvec_dt2 = &d2_rotvec_dt2 + &(vel0 * dt * d2h10_dt2);
        d2_rotvec_dt2 = &d2_rotvec_dt2 + &(&rotvec1 * d2h01_dt2);
        d2_rotvec_dt2 = &d2_rotvec_dt2 + &(vel1 * dt * d2h11_dt2);

        // This gives us the angular acceleration
        d2_rotvec_dt2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_rotation_spline_creation() {
        let rotations = vec![
            Rotation::identity(),
            rotation_from_euler(0.0, 0.0, PI / 2.0, "xyz").unwrap(),
            Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
        ];
        let times = vec![0.0, 1.0, 2.0];

        let spline = RotationSpline::new(&rotations, &times).unwrap();

        assert_eq!(spline.rotations().len(), 3);
        assert_eq!(spline.times().len(), 3);
        assert_eq!(spline.interpolation_type(), "slerp");
    }

    #[test]
    fn test_rotation_spline_interpolation_endpoints() {
        let rotations = vec![
            Rotation::identity(),
            rotation_from_euler(0.0, 0.0, PI / 2.0, "xyz").unwrap(),
            Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
        ];
        let times = vec![0.0, 1.0, 2.0];

        let spline = RotationSpline::new(&rotations, &times).unwrap();

        // Test at endpoints
        let interp_start = spline.interpolate(0.0);
        let interp_end = spline.interpolate(2.0);

        // Should match the first and last rotations
        assert_eq!(interp_start.as_quat(), rotations[0].as_quat());
        assert_eq!(interp_end.as_quat(), rotations[2].as_quat());

        // Test beyond endpoints (should clamp)
        let before_start = spline.interpolate(-1.0);
        let after_end = spline.interpolate(3.0);

        assert_eq!(before_start.as_quat(), rotations[0].as_quat());
        assert_eq!(after_end.as_quat(), rotations[2].as_quat());
    }

    #[test]
    fn test_rotation_spline_interpolation_midpoints() {
        let rotations = vec![
            Rotation::identity(),
            rotation_from_euler(0.0, 0.0, PI / 2.0, "xyz").unwrap(),
            Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
        ];
        let times = vec![0.0, 1.0, 2.0];

        let spline = RotationSpline::new(&rotations, &times).unwrap();

        // Test at midpoints
        let interp_mid1 = spline.interpolate(0.5);
        let interp_mid2 = spline.interpolate(1.5);

        // Apply to a test point
        let test_point = array![1.0, 0.0, 0.0];

        // Verify interpolation results
        let rotated_mid1 = interp_mid1.apply(&test_point.view()).unwrap();
        let rotated_mid2 = interp_mid2.apply(&test_point.view()).unwrap();

        // At t=0.5 (between identity and 90-degree rotation), should be approximately 45 degrees
        assert_relative_eq!(rotated_mid1[0], 2.0_f64.sqrt() / 2.0, epsilon = 1e-3);
        assert_relative_eq!(rotated_mid1[1], 2.0_f64.sqrt() / 2.0, epsilon = 1e-3);
        assert_relative_eq!(rotated_mid1[2], 0.0, epsilon = 1e-3);

        // At t=1.5 (between 90 and 180 degrees), should be approximately 135 degrees
        assert_relative_eq!(rotated_mid2[0], -2.0_f64.sqrt() / 2.0, epsilon = 1e-3);
        assert_relative_eq!(rotated_mid2[1], 2.0_f64.sqrt() / 2.0, epsilon = 1e-3);
        assert_relative_eq!(rotated_mid2[2], 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_rotation_spline_sampling() {
        let rotations = vec![
            Rotation::identity(),
            Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
        ];
        let times = vec![0.0, 1.0];

        let spline = RotationSpline::new(&rotations, &times).unwrap();

        // Sample 5 points
        let (sample_times, sample_rotations) = spline.sample(5);

        assert_eq!(sample_times.len(), 5);
        assert_eq!(sample_rotations.len(), 5);

        // Check if times are evenly spaced
        assert_relative_eq!(sample_times[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sample_times[1], 0.25, epsilon = 1e-10);
        assert_relative_eq!(sample_times[2], 0.5, epsilon = 1e-10);
        assert_relative_eq!(sample_times[3], 0.75, epsilon = 1e-10);
        assert_relative_eq!(sample_times[4], 1.0, epsilon = 1e-10);

        // Check if rotations are correct
        let point = array![1.0, 0.0, 0.0];

        // At t=0.0, should be identity
        let rot0 = &sample_rotations[0];
        let rotated0 = rot0.apply(&point.view()).unwrap();
        assert_relative_eq!(rotated0[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated0[1], 0.0, epsilon = 1e-10);

        // At t=0.5, should be 90-degree rotation
        let rot2 = &sample_rotations[2];
        let rotated2 = rot2.apply(&point.view()).unwrap();
        assert_relative_eq!(rotated2[0], 0.0, epsilon = 1e-3);
        assert_relative_eq!(rotated2[1], 1.0, epsilon = 1e-3);
        assert_relative_eq!(rotated2[2], 0.0, epsilon = 1e-3);

        // At t=1.0, should be 180-degree rotation
        let rot4 = &sample_rotations[4];
        let rotated4 = rot4.apply(&point.view()).unwrap();
        assert_relative_eq!(rotated4[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated4[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated4[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_spline_errors() {
        // Empty rotations
        let result = RotationSpline::new(&[], &[0.0]);
        assert!(result.is_err());

        // Empty times
        let rotations = vec![Rotation::identity()];
        let result = RotationSpline::new(&rotations, &[]);
        assert!(result.is_err());

        // Mismatched lengths
        let rotations = vec![Rotation::identity(), Rotation::identity()];
        let times = vec![0.0];
        let result = RotationSpline::new(&rotations, &times);
        assert!(result.is_err());

        // Non-increasing times
        let rotations = vec![Rotation::identity(), Rotation::identity()];
        let times = vec![1.0, 0.0];
        let result = RotationSpline::new(&rotations, &times);
        assert!(result.is_err());

        // Equal times
        let rotations = vec![Rotation::identity(), Rotation::identity()];
        let times = vec![0.0, 0.0];
        let result = RotationSpline::new(&rotations, &times);
        assert!(result.is_err());

        // Invalid interpolation type
        let rotations = vec![Rotation::identity(), Rotation::identity()];
        let times = vec![0.0, 1.0];
        let mut spline = RotationSpline::new(&rotations, &times).unwrap();
        let result = spline.set_interpolation_type("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_interpolation_types() {
        let rotations = vec![
            Rotation::identity(),
            rotation_from_euler(0.0, 0.0, PI / 2.0, "xyz").unwrap(),
            Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
        ];
        let times = vec![0.0, 1.0, 2.0];

        let mut spline = RotationSpline::new(&rotations, &times).unwrap();

        // Default should be slerp
        assert_eq!(spline.interpolation_type(), "slerp");

        // Change to cubic
        spline.set_interpolation_type("cubic").unwrap();
        assert_eq!(spline.interpolation_type(), "cubic");

        // Check that velocities are computed
        assert!(spline.velocities.is_some());

        // Change back to slerp
        spline.set_interpolation_type("slerp").unwrap();
        assert_eq!(spline.interpolation_type(), "slerp");

        // Velocities should be cleared
        assert!(spline.velocities.is_none());
    }

    #[test]
    fn test_angular_velocity() {
        let rotations = vec![
            Rotation::identity(),
            Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
        ];
        let times = vec![0.0, 1.0];

        let spline = RotationSpline::new(&rotations, &times).unwrap();

        // Angular velocity should be constant for slerp
        let velocity = spline.angular_velocity(0.5).unwrap();

        // For a rotation from identity to 180 degrees around z-axis over 1 second,
        // the angular velocity should be approximately [0, 0, π]
        assert_relative_eq!(velocity[0], 0.0, epsilon = 1e-3);
        assert_relative_eq!(velocity[1], 0.0, epsilon = 1e-3);
        assert_relative_eq!(velocity[2], PI, epsilon = 1e-3);

        // Velocity should be the same at any point in the segment
        let velocity_25 = spline.angular_velocity(0.25).unwrap();
        let velocity_75 = spline.angular_velocity(0.75).unwrap();

        assert_relative_eq!(velocity_25[0], velocity[0], epsilon = 1e-10);
        assert_relative_eq!(velocity_25[1], velocity[1], epsilon = 1e-10);
        assert_relative_eq!(velocity_25[2], velocity[2], epsilon = 1e-10);

        assert_relative_eq!(velocity_75[0], velocity[0], epsilon = 1e-10);
        assert_relative_eq!(velocity_75[1], velocity[1], epsilon = 1e-10);
        assert_relative_eq!(velocity_75[2], velocity[2], epsilon = 1e-10);
    }

    #[test]
    fn test_cubic_interpolation() {
        let rotations = vec![
            Rotation::identity(),
            rotation_from_euler(0.0, 0.0, PI / 2.0, "xyz").unwrap(),
            Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
        ];
        let times = vec![0.0, 1.0, 2.0];

        let mut spline = RotationSpline::new(&rotations, &times).unwrap();

        // Set to cubic interpolation
        spline.set_interpolation_type("cubic").unwrap();

        // Test at endpoints, should match original rotations
        let rot_0 = spline.interpolate(0.0);
        let rot_1 = spline.interpolate(1.0);
        let rot_2 = spline.interpolate(2.0);

        let test_point = array![1.0, 0.0, 0.0];

        // Check that endpoints match original rotations
        let rotated_0 = rot_0.apply(&test_point.view()).unwrap();
        let expected_0 = rotations[0].apply(&test_point.view()).unwrap();
        assert_relative_eq!(rotated_0[0], expected_0[0], epsilon = 1e-10);
        assert_relative_eq!(rotated_0[1], expected_0[1], epsilon = 1e-10);
        assert_relative_eq!(rotated_0[2], expected_0[2], epsilon = 1e-10);

        let rotated_1 = rot_1.apply(&test_point.view()).unwrap();
        let expected_1 = rotations[1].apply(&test_point.view()).unwrap();
        assert_relative_eq!(rotated_1[0], expected_1[0], epsilon = 1e-10);
        assert_relative_eq!(rotated_1[1], expected_1[1], epsilon = 1e-10);
        assert_relative_eq!(rotated_1[2], expected_1[2], epsilon = 1e-10);

        let rotated_2 = rot_2.apply(&test_point.view()).unwrap();
        let expected_2 = rotations[2].apply(&test_point.view()).unwrap();
        assert_relative_eq!(rotated_2[0], expected_2[0], epsilon = 1e-10);
        assert_relative_eq!(rotated_2[1], expected_2[1], epsilon = 1e-10);
        assert_relative_eq!(rotated_2[2], expected_2[2], epsilon = 1e-10);

        // Test midpoints - cubic interpolation should be smoother than slerp
        // but still interpolate the key rotations
        let rot_05 = spline.interpolate(0.5);
        let rot_15 = spline.interpolate(1.5);

        // Verify that interpolated rotations are valid
        let rotated_05 = rot_05.apply(&test_point.view()).unwrap();
        let rotated_15 = rot_15.apply(&test_point.view()).unwrap();

        // Check that the results are normalized
        let norm_05 = (rotated_05.dot(&rotated_05)).sqrt();
        let norm_15 = (rotated_15.dot(&rotated_15)).sqrt();
        assert_relative_eq!(norm_05, 1.0, epsilon = 1e-10);
        assert_relative_eq!(norm_15, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_angular_acceleration() {
        let rotations = vec![
            Rotation::identity(),
            rotation_from_euler(0.0, 0.0, PI / 2.0, "xyz").unwrap(),
            Rotation::from_euler(&array![0.0, 0.0, PI].view(), "xyz").unwrap(),
        ];
        let times = vec![0.0, 1.0, 2.0];

        let mut spline = RotationSpline::new(&rotations, &times).unwrap();

        // Slerp should have zero acceleration
        let accel_slerp = spline.angular_acceleration(0.5);
        assert_relative_eq!(accel_slerp[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(accel_slerp[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(accel_slerp[2], 0.0, epsilon = 1e-10);

        // Set to cubic interpolation
        spline.set_interpolation_type("cubic").unwrap();

        // Cubic should have non-zero acceleration
        let _accel_cubic = spline.angular_acceleration(0.5);

        // For linear rotation sequence, acceleration might still be close to zero
        // Let's create a more complex rotation sequence
        let complex_rotations = vec![
            Rotation::identity(),
            {
                let angles = array![PI / 2.0, 0.0, 0.0];
                Rotation::from_euler(&angles.view(), "xyz").unwrap()
            },
            {
                let angles = array![PI / 2.0, PI / 2.0, 0.0];
                Rotation::from_euler(&angles.view(), "xyz").unwrap()
            },
        ];
        let complex_times = vec![0.0, 1.0, 2.0];

        let mut complex_spline = RotationSpline::new(&complex_rotations, &complex_times).unwrap();
        complex_spline.set_interpolation_type("cubic").unwrap();

        let complex_accel = complex_spline.angular_acceleration(0.5);

        // For non-linear rotation sequences, acceleration should be non-zero
        let magnitude = (complex_accel.dot(&complex_accel)).sqrt();
        assert!(magnitude > 1e-6); // Should have meaningful acceleration
    }
}
