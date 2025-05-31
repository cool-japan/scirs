//! Rotation class for 3D rotations
//!
//! This module provides a `Rotation` class that supports various rotation representations,
//! including quaternions, rotation matrices, Euler angles, and axis-angle representations.

use crate::error::{SpatialError, SpatialResult};
use ndarray::{array, Array1, Array2, ArrayView1, ArrayView2};
use num_traits::Float;
use std::f64::consts::PI;
use std::fmt;

/// Supported Euler angle conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EulerConvention {
    /// Intrinsic rotation around X, then Y, then Z
    Xyz,
    /// Intrinsic rotation around Z, then Y, then X
    Zyx,
    /// Intrinsic rotation around X, then Y, then X
    Xyx,
    /// Intrinsic rotation around X, then Z, then X
    Xzx,
    /// Intrinsic rotation around Y, then X, then Y
    Yxy,
    /// Intrinsic rotation around Y, then Z, then Y
    Yzy,
    /// Intrinsic rotation around Z, then X, then Z
    Zxz,
    /// Intrinsic rotation around Z, then Y, then Z
    Zyz,
}

impl fmt::Display for EulerConvention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EulerConvention::Xyz => write!(f, "xyz"),
            EulerConvention::Zyx => write!(f, "zyx"),
            EulerConvention::Xyx => write!(f, "xyx"),
            EulerConvention::Xzx => write!(f, "xzx"),
            EulerConvention::Yxy => write!(f, "yxy"),
            EulerConvention::Yzy => write!(f, "yzy"),
            EulerConvention::Zxz => write!(f, "zxz"),
            EulerConvention::Zyz => write!(f, "zyz"),
        }
    }
}

impl EulerConvention {
    /// Parse a string into an EulerConvention
    pub fn from_str(s: &str) -> SpatialResult<Self> {
        match s.to_lowercase().as_str() {
            "xyz" => Ok(EulerConvention::Xyz),
            "zyx" => Ok(EulerConvention::Zyx),
            "xyx" => Ok(EulerConvention::Xyx),
            "xzx" => Ok(EulerConvention::Xzx),
            "yxy" => Ok(EulerConvention::Yxy),
            "yzy" => Ok(EulerConvention::Yzy),
            "zxz" => Ok(EulerConvention::Zxz),
            "zyz" => Ok(EulerConvention::Zyz),
            _ => Err(SpatialError::ValueError(format!(
                "Invalid Euler convention: {}",
                s
            ))),
        }
    }
}

/// Rotation representation for 3D rotations
///
/// This class provides a convenient way to represent and manipulate 3D rotations.
/// It supports multiple representations including quaternions, rotation matrices,
/// Euler angles (in various conventions), and axis-angle representation.
///
/// Rotations can be composed, inverted, and applied to vectors.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::transform::Rotation;
/// use ndarray::array;
/// use std::f64::consts::PI;
///
/// // Create a rotation from a quaternion [w, x, y, z]
/// let quat = array![std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, 0.0, 0.0]; // 90 degree rotation around X
/// let rot = Rotation::from_quat(&quat.view()).unwrap();
///
/// // Apply the rotation to a vector
/// let vec = array![0.0, 1.0, 0.0];
/// let rotated = rot.apply(&vec.view());
///
/// // Verify the result (should be approximately [0, 0, 1])
/// assert!((rotated[0]).abs() < 1e-10);
/// assert!((rotated[1]).abs() < 1e-10);
/// assert!((rotated[2] - 1.0).abs() < 1e-10);
///
/// // Convert to other representations
/// let matrix = rot.as_matrix();
/// let euler = rot.as_euler("xyz").unwrap();
/// let axis_angle = rot.as_rotvec();
/// ```
#[derive(Clone, Debug)]
pub struct Rotation {
    /// Quaternion representation [w, x, y, z] where w is the scalar part
    quat: Array1<f64>,
}

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

impl Rotation {
    /// Create a new rotation from a quaternion [w, x, y, z]
    ///
    /// # Arguments
    ///
    /// * `quat` - A quaternion in the format [w, x, y, z] where w is the scalar part
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing the rotation if valid, or an error if the quaternion is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::Rotation;
    /// use ndarray::array;
    ///
    /// // Create a quaternion for a 90-degree rotation around the x-axis
    /// let quat = array![std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, 0.0, 0.0];
    /// let rot = Rotation::from_quat(&quat.view()).unwrap();
    /// ```
    pub fn from_quat(quat: &ArrayView1<f64>) -> SpatialResult<Self> {
        if quat.len() != 4 {
            return Err(SpatialError::DimensionError(format!(
                "Quaternion must have 4 elements, got {}",
                quat.len()
            )));
        }

        // Normalize the quaternion
        let norm = (quat.iter().map(|&x| x * x).sum::<f64>()).sqrt();

        if norm < 1e-10 {
            return Err(SpatialError::ComputationError(
                "Quaternion has near-zero norm".into(),
            ));
        }

        let normalized_quat = quat.to_owned() / norm;

        Ok(Rotation {
            quat: normalized_quat,
        })
    }

    /// Create a rotation from a rotation matrix
    ///
    /// # Arguments
    ///
    /// * `matrix` - A 3x3 orthogonal rotation matrix
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing the rotation if valid, or an error if the matrix is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::Rotation;
    /// use ndarray::array;
    ///
    /// // Create a rotation matrix for a 90-degree rotation around the z-axis
    /// let matrix = array![
    ///     [0.0, -1.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 0.0, 1.0]
    /// ];
    /// let rot = Rotation::from_matrix(&matrix.view()).unwrap();
    /// ```
    pub fn from_matrix(matrix: &ArrayView2<f64>) -> SpatialResult<Self> {
        if matrix.shape() != [3, 3] {
            return Err(SpatialError::DimensionError(format!(
                "Matrix must be 3x3, got {:?}",
                matrix.shape()
            )));
        }

        // Check if the matrix is approximately orthogonal
        let det = matrix[[0, 0]]
            * (matrix[[1, 1]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 1]])
            - matrix[[0, 1]] * (matrix[[1, 0]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 0]])
            + matrix[[0, 2]] * (matrix[[1, 0]] * matrix[[2, 1]] - matrix[[1, 1]] * matrix[[2, 0]]);

        if (det - 1.0).abs() > 1e-6 {
            return Err(SpatialError::ValueError(format!(
                "Matrix is not orthogonal, determinant = {}",
                det
            )));
        }

        // Convert rotation matrix to quaternion using method from:
        // http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

        let mut quat = Array1::zeros(4);
        let m = matrix;

        let trace = m[[0, 0]] + m[[1, 1]] + m[[2, 2]];

        if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            quat[0] = 0.25 / s;
            quat[1] = (m[[2, 1]] - m[[1, 2]]) * s;
            quat[2] = (m[[0, 2]] - m[[2, 0]]) * s;
            quat[3] = (m[[1, 0]] - m[[0, 1]]) * s;
        } else if m[[0, 0]] > m[[1, 1]] && m[[0, 0]] > m[[2, 2]] {
            let s = 2.0 * (1.0 + m[[0, 0]] - m[[1, 1]] - m[[2, 2]]).sqrt();
            quat[0] = (m[[2, 1]] - m[[1, 2]]) / s;
            quat[1] = 0.25 * s;
            quat[2] = (m[[0, 1]] + m[[1, 0]]) / s;
            quat[3] = (m[[0, 2]] + m[[2, 0]]) / s;
        } else if m[[1, 1]] > m[[2, 2]] {
            let s = 2.0 * (1.0 + m[[1, 1]] - m[[0, 0]] - m[[2, 2]]).sqrt();
            quat[0] = (m[[0, 2]] - m[[2, 0]]) / s;
            quat[1] = (m[[0, 1]] + m[[1, 0]]) / s;
            quat[2] = 0.25 * s;
            quat[3] = (m[[1, 2]] + m[[2, 1]]) / s;
        } else {
            let s = 2.0 * (1.0 + m[[2, 2]] - m[[0, 0]] - m[[1, 1]]).sqrt();
            quat[0] = (m[[1, 0]] - m[[0, 1]]) / s;
            quat[1] = (m[[0, 2]] + m[[2, 0]]) / s;
            quat[2] = (m[[1, 2]] + m[[2, 1]]) / s;
            quat[3] = 0.25 * s;
        }

        // Make w the first element
        quat = array![quat[0], quat[1], quat[2], quat[3]];

        // Normalize the quaternion
        let norm = (quat.iter().map(|&x| x * x).sum::<f64>()).sqrt();
        quat /= norm;

        Ok(Rotation { quat })
    }

    /// Create a rotation from Euler angles
    ///
    /// # Arguments
    ///
    /// * `angles` - A 3-element array of Euler angles (in radians)
    /// * `convention` - The Euler angle convention (e.g., "xyz", "zyx")
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing the rotation if valid, or an error if the angles are invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::Rotation;
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// // Create a rotation using Euler angles in the XYZ convention
    /// let angles = array![PI/2.0, 0.0, 0.0]; // 90 degrees around X
    /// let rot = Rotation::from_euler(&angles.view(), "xyz").unwrap();
    /// ```
    pub fn from_euler(angles: &ArrayView1<f64>, convention: &str) -> SpatialResult<Self> {
        if angles.len() != 3 {
            return Err(SpatialError::DimensionError(format!(
                "Euler angles must have 3 elements, got {}",
                angles.len()
            )));
        }

        let conv = EulerConvention::from_str(convention)?;
        let mut quat = Array1::zeros(4);

        // Compute quaternions for individual rotations
        let angles = angles.as_slice().unwrap();
        let a = angles[0] / 2.0;
        let b = angles[1] / 2.0;
        let c = angles[2] / 2.0;

        let ca = a.cos();
        let sa = a.sin();
        let cb = b.cos();
        let sb = b.sin();
        let cc = c.cos();
        let sc = c.sin();

        // Construct quaternion based on convention
        match conv {
            EulerConvention::Xyz => {
                // Intrinsic rotation around X, then Y, then Z
                // Quaternion multiplication order: Qz * Qy * Qx
                quat[0] = cc * cb * ca + sc * sb * sa;
                quat[1] = cc * cb * sa - sc * sb * ca;
                quat[2] = cc * sb * ca + sc * cb * sa;
                quat[3] = sc * cb * ca - cc * sb * sa;
            }
            EulerConvention::Zyx => {
                // Intrinsic rotation around Z, then Y, then X
                // For ZYX: angles[0] = Z, angles[1] = Y, angles[2] = X
                // So: ca = cos(Z/2), sa = sin(Z/2)
                //     cb = cos(Y/2), sb = sin(Y/2)
                //     cc = cos(X/2), sc = sin(X/2)
                // Quaternion multiplication order: Qx * Qy * Qz
                // Formula: Qx(cc,sc) * Qy(cb,sb) * Qz(ca,sa)
                quat[0] = cc * cb * ca - sc * sb * sa;
                quat[1] = cc * sb * sa + sc * cb * ca;
                quat[2] = cc * sb * ca - sc * cb * sa;
                quat[3] = cc * cb * sa + sc * sb * ca;
            }
            EulerConvention::Xyx => {
                quat[0] = ca * cb * cc - sa * cb * sc;
                quat[1] = sa * cb * cc + ca * cb * sc;
                quat[2] = ca * sb * sc + sa * sb * cc;
                quat[3] = sa * sb * sc - ca * sb * cc;
            }
            EulerConvention::Xzx => {
                quat[0] = ca * cb * cc - sa * cb * sc;
                quat[1] = sa * cb * cc + ca * cb * sc;
                quat[2] = sa * sb * sc - ca * sb * cc;
                quat[3] = ca * sb * sc + sa * sb * cc;
            }
            EulerConvention::Yxy => {
                quat[0] = ca * cb * cc - sa * cb * sc;
                quat[1] = ca * sb * sc + sa * sb * cc;
                quat[2] = sa * cb * cc + ca * cb * sc;
                quat[3] = sa * sb * sc - ca * sb * cc;
            }
            EulerConvention::Yzy => {
                quat[0] = ca * cb * cc - sa * cb * sc;
                quat[1] = sa * sb * sc - ca * sb * cc;
                quat[2] = sa * cb * cc + ca * cb * sc;
                quat[3] = ca * sb * sc + sa * sb * cc;
            }
            EulerConvention::Zxz => {
                quat[0] = ca * cb * cc - sa * cb * sc;
                quat[1] = ca * sb * sc + sa * sb * cc;
                quat[2] = sa * sb * sc - ca * sb * cc;
                quat[3] = sa * cb * cc + ca * cb * sc;
            }
            EulerConvention::Zyz => {
                quat[0] = ca * cb * cc - sa * cb * sc;
                quat[1] = sa * sb * sc - ca * sb * cc;
                quat[2] = ca * sb * sc + sa * sb * cc;
                quat[3] = sa * cb * cc + ca * cb * sc;
            }
        }

        // Normalize the quaternion
        let norm = (quat.iter().map(|&x| x * x).sum::<f64>()).sqrt();
        quat /= norm;

        Ok(Rotation { quat })
    }

    /// Create a rotation from an axis-angle representation (rotation vector)
    ///
    /// # Arguments
    ///
    /// * `rotvec` - A 3-element array representing the rotation axis and angle
    ///   (axis is the unit vector in the direction of the array, and the angle is the
    ///   magnitude of the array in radians)
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing the rotation if valid, or an error if the rotation vector is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::Rotation;
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// // Create a rotation for a 90-degree rotation around the x-axis
    /// let rotvec = array![PI/2.0, 0.0, 0.0];
    /// let rot = Rotation::from_rotvec(&rotvec.view()).unwrap();
    /// ```
    pub fn from_rotvec(rotvec: &ArrayView1<f64>) -> SpatialResult<Self> {
        if rotvec.len() != 3 {
            return Err(SpatialError::DimensionError(format!(
                "Rotation vector must have 3 elements, got {}",
                rotvec.len()
            )));
        }

        // Compute the angle (magnitude of the rotation vector)
        let angle = (rotvec[0] * rotvec[0] + rotvec[1] * rotvec[1] + rotvec[2] * rotvec[2]).sqrt();

        let mut quat = Array1::zeros(4);

        if angle < 1e-10 {
            // For zero rotation, use the identity quaternion
            quat[0] = 1.0;
            quat[1] = 0.0;
            quat[2] = 0.0;
            quat[3] = 0.0;
        } else {
            // Extract the normalized axis
            let x = rotvec[0] / angle;
            let y = rotvec[1] / angle;
            let z = rotvec[2] / angle;

            // Convert axis-angle to quaternion
            let half_angle = angle / 2.0;
            let s = half_angle.sin();

            quat[0] = half_angle.cos();
            quat[1] = x * s;
            quat[2] = y * s;
            quat[3] = z * s;
        }

        Ok(Rotation { quat })
    }

    /// Convert the rotation to a rotation matrix
    ///
    /// # Returns
    ///
    /// A 3x3 rotation matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::Rotation;
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// let angles = array![0.0, 0.0, PI/2.0];
    /// let rot = Rotation::from_euler(&angles.view(), "xyz").unwrap();
    /// let matrix = rot.as_matrix();
    /// // Should approximately be a 90 degree rotation around Z
    /// ```
    pub fn as_matrix(&self) -> Array2<f64> {
        let mut matrix = Array2::zeros((3, 3));

        let q = &self.quat;
        let w = q[0];
        let x = q[1];
        let y = q[2];
        let z = q[3];

        // Fill the rotation matrix using quaternion to matrix conversion
        matrix[[0, 0]] = 1.0 - 2.0 * (y * y + z * z);
        matrix[[0, 1]] = 2.0 * (x * y - w * z);
        matrix[[0, 2]] = 2.0 * (x * z + w * y);

        matrix[[1, 0]] = 2.0 * (x * y + w * z);
        matrix[[1, 1]] = 1.0 - 2.0 * (x * x + z * z);
        matrix[[1, 2]] = 2.0 * (y * z - w * x);

        matrix[[2, 0]] = 2.0 * (x * z - w * y);
        matrix[[2, 1]] = 2.0 * (y * z + w * x);
        matrix[[2, 2]] = 1.0 - 2.0 * (x * x + y * y);

        matrix
    }

    /// Convert the rotation to Euler angles
    ///
    /// # Arguments
    ///
    /// * `convention` - The Euler angle convention to use (e.g., "xyz", "zyx")
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing a 3-element array of Euler angles (in radians),
    /// or an error if the convention is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::Rotation;
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// let rot = Rotation::from_rotvec(&array![PI/2.0, 0.0, 0.0].view()).unwrap();
    /// let angles = rot.as_euler("xyz").unwrap();
    /// // Should be approximately [PI/2, 0, 0]
    /// ```
    pub fn as_euler(&self, convention: &str) -> SpatialResult<Array1<f64>> {
        let conv = EulerConvention::from_str(convention)?;
        let matrix = self.as_matrix();

        let mut angles = Array1::zeros(3);

        match conv {
            EulerConvention::Xyz => {
                // For intrinsic XYZ: R = Rz(c) * Ry(b) * Rx(a)
                // Extract angles using arctan2 to handle singularities
                angles[1] = (-matrix[[2, 0]]).asin();

                if angles[1].cos().abs() > 1e-6 {
                    // Not in gimbal lock
                    angles[0] = matrix[[2, 1]].atan2(matrix[[2, 2]]);
                    angles[2] = matrix[[1, 0]].atan2(matrix[[0, 0]]);
                } else {
                    // Gimbal lock case
                    angles[0] = 0.0;
                    if matrix[[2, 0]] < 0.0 {
                        // sin(beta) = 1, beta = pi/2
                        angles[2] = (-matrix[[0, 1]]).atan2(matrix[[1, 1]]);
                    } else {
                        // sin(beta) = -1, beta = -pi/2
                        angles[2] = matrix[[0, 1]].atan2(matrix[[1, 1]]);
                    }
                }
            }
            EulerConvention::Zyx => {
                // For intrinsic ZYX: First rotate around Z, then around new Y, then around new X
                // This is equivalent to extrinsic rotations in reverse order: Rx * Ry * Rz
                // The combined matrix has elements:
                // R[0,2] = sin(Y)
                // So we need to extract Y from arcsin(R[0,2]), not arcsin(-R[0,2])
                angles[1] = matrix[[0, 2]].asin();

                if angles[1].cos().abs() > 1e-6 {
                    // Not in gimbal lock
                    // R[0,1]/cos(Y) = -sin(Z), R[0,0]/cos(Y) = cos(Z)
                    angles[0] = (-matrix[[0, 1]]).atan2(matrix[[0, 0]]);
                    // R[1,2]/cos(Y) = -sin(X), R[2,2]/cos(Y) = cos(X)
                    angles[2] = (-matrix[[1, 2]]).atan2(matrix[[2, 2]]);
                } else {
                    // Gimbal lock case
                    angles[0] = 0.0;
                    if matrix[[0, 2]] > 0.0 {
                        // sin(Y) = 1, Y = pi/2
                        angles[2] = matrix[[1, 0]].atan2(matrix[[1, 1]]);
                    } else {
                        // sin(Y) = -1, Y = -pi/2
                        angles[2] = (-matrix[[1, 0]]).atan2(matrix[[1, 1]]);
                    }
                }
            }
            EulerConvention::Xyx => {
                angles[1] = (matrix[[0, 0]].abs()).acos();

                if angles[1].abs() < 1e-6 || (angles[1] - PI).abs() < 1e-6 {
                    // Gimbal lock case
                    angles[0] = 0.0;
                    // In gimbal lock, only the sum or difference of angles[0] and angles[2] matters
                    angles[2] = (matrix[[1, 2]]).atan2(matrix[[1, 1]]);
                } else {
                    angles[0] = (matrix[[2, 0]]).atan2(matrix[[1, 0]]);
                    angles[2] = (matrix[[0, 2]]).atan2(-matrix[[0, 1]]);
                }
            }
            EulerConvention::Xzx => {
                angles[1] = (matrix[[0, 0]].abs()).acos();

                if angles[1].abs() < 1e-6 || (angles[1] - PI).abs() < 1e-6 {
                    // Gimbal lock case
                    angles[0] = 0.0;
                    angles[2] = (matrix[[1, 2]]).atan2(matrix[[1, 1]]);
                } else {
                    angles[0] = (matrix[[2, 0]]).atan2(-matrix[[1, 0]]);
                    angles[2] = (matrix[[0, 2]]).atan2(matrix[[0, 1]]);
                }
            }
            EulerConvention::Yxy => {
                angles[1] = (matrix[[1, 1]].abs()).acos();

                if angles[1].abs() < 1e-6 || (angles[1] - PI).abs() < 1e-6 {
                    // Gimbal lock case
                    angles[0] = 0.0;
                    angles[2] = (matrix[[0, 2]]).atan2(matrix[[0, 0]]);
                } else {
                    angles[0] = (matrix[[0, 1]]).atan2(matrix[[2, 1]]);
                    angles[2] = (matrix[[1, 0]]).atan2(-matrix[[1, 2]]);
                }
            }
            EulerConvention::Yzy => {
                angles[1] = (matrix[[1, 1]].abs()).acos();

                if angles[1].abs() < 1e-6 || (angles[1] - PI).abs() < 1e-6 {
                    // Gimbal lock case
                    angles[0] = 0.0;
                    angles[2] = (matrix[[0, 2]]).atan2(matrix[[0, 0]]);
                } else {
                    angles[0] = (matrix[[0, 1]]).atan2(-matrix[[2, 1]]);
                    angles[2] = (matrix[[1, 0]]).atan2(matrix[[1, 2]]);
                }
            }
            EulerConvention::Zxz => {
                angles[1] = (matrix[[2, 2]].abs()).acos();

                if angles[1].abs() < 1e-6 || (angles[1] - PI).abs() < 1e-6 {
                    // Gimbal lock case
                    angles[0] = 0.0;
                    angles[2] = (matrix[[0, 1]]).atan2(matrix[[0, 0]]);
                } else {
                    angles[0] = (matrix[[0, 2]]).atan2(matrix[[1, 2]]);
                    angles[2] = (matrix[[2, 0]]).atan2(-matrix[[2, 1]]);
                }
            }
            EulerConvention::Zyz => {
                angles[1] = (matrix[[2, 2]].abs()).acos();

                if angles[1].abs() < 1e-6 || (angles[1] - PI).abs() < 1e-6 {
                    // Gimbal lock case
                    angles[0] = 0.0;
                    angles[2] = (matrix[[0, 1]]).atan2(matrix[[0, 0]]);
                } else {
                    angles[0] = (matrix[[1, 2]]).atan2(-matrix[[0, 2]]);
                    angles[2] = (matrix[[2, 1]]).atan2(matrix[[2, 0]]);
                }
            }
        }

        Ok(angles)
    }

    /// Convert the rotation to an axis-angle representation (rotation vector)
    ///
    /// # Returns
    ///
    /// A 3-element array representing the rotation axis and angle (axis is the
    /// unit vector in the direction of the array, and the angle is the magnitude
    /// of the array in radians)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::Rotation;
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// let angles = array![PI/2.0, 0.0, 0.0];
    /// let rot = Rotation::from_euler(&angles.view(), "xyz").unwrap();
    /// let rotvec = rot.as_rotvec();
    /// // Should be approximately [PI/2, 0, 0]
    /// ```
    pub fn as_rotvec(&self) -> Array1<f64> {
        let q = &self.quat;
        let angle = 2.0 * q[0].acos();

        let mut rotvec = Array1::zeros(3);

        // Handle the case of zero rotation
        if angle < 1e-10 {
            return rotvec;
        }

        // Extract the axis
        let sin_half_angle = (1.0 - q[0] * q[0]).sqrt();

        if sin_half_angle < 1e-10 {
            // If sin(angle/2) is close to zero, rotation is close to 0 or 2π
            return rotvec;
        }

        // Normalize the axis
        let axis_x = q[1] / sin_half_angle;
        let axis_y = q[2] / sin_half_angle;
        let axis_z = q[3] / sin_half_angle;

        // Compute the rotation vector
        rotvec[0] = axis_x * angle;
        rotvec[1] = axis_y * angle;
        rotvec[2] = axis_z * angle;

        rotvec
    }

    /// Get the quaternion representation [w, x, y, z]
    ///
    /// # Returns
    ///
    /// A 4-element array representing the quaternion (w, x, y, z)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::Rotation;
    /// use ndarray::array;
    ///
    /// let angles = array![0.0, 0.0, 0.0];
    /// let rot = Rotation::from_euler(&angles.view(), "xyz").unwrap();
    /// let quat = rot.as_quat();
    /// // Should be [1, 0, 0, 0] (identity rotation)
    /// ```
    pub fn as_quat(&self) -> Array1<f64> {
        self.quat.clone()
    }

    /// Get the inverse of the rotation
    ///
    /// # Returns
    ///
    /// A new Rotation that is the inverse of this one
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::Rotation;
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// let angles = array![0.0, 0.0, PI/4.0];
    /// let rot = Rotation::from_euler(&angles.view(), "xyz").unwrap();
    /// let rot_inv = rot.inv();
    /// ```
    pub fn inv(&self) -> Rotation {
        // For unit quaternions, the inverse is the conjugate
        let mut quat_inv = self.quat.clone();
        quat_inv[1] = -quat_inv[1];
        quat_inv[2] = -quat_inv[2];
        quat_inv[3] = -quat_inv[3];

        Rotation { quat: quat_inv }
    }

    /// Apply the rotation to a vector or array of vectors
    ///
    /// # Arguments
    ///
    /// * `vec` - A 3-element vector or a 2D array of 3-element vectors to rotate
    ///
    /// # Returns
    ///
    /// The rotated vector or array of vectors
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::Rotation;
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// let angles = array![0.0, 0.0, PI/2.0];
    /// let rot = Rotation::from_euler(&angles.view(), "xyz").unwrap();
    /// let vec = array![1.0, 0.0, 0.0];
    /// let rotated = rot.apply(&vec.view());
    /// // Should be approximately [0, 1, 0]
    /// ```
    pub fn apply(&self, vec: &ArrayView1<f64>) -> Array1<f64> {
        if vec.len() != 3 {
            panic!("Vector must have 3 elements");
        }

        // Convert to matrix and apply
        let matrix = self.as_matrix();
        matrix.dot(vec)
    }

    /// Apply the rotation to multiple vectors
    ///
    /// # Arguments
    ///
    /// * `vecs` - A 2D array of vectors (each row is a 3-element vector)
    ///
    /// # Returns
    ///
    /// A 2D array of rotated vectors
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::Rotation;
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// let angles = array![0.0, 0.0, PI/2.0];
    /// let rot = Rotation::from_euler(&angles.view(), "xyz").unwrap();
    /// let vecs = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    /// let rotated = rot.apply_multiple(&vecs.view());
    /// // First row should be approximately [0, 1, 0]
    /// // Second row should be approximately [-1, 0, 0]
    /// ```
    pub fn apply_multiple(&self, vecs: &ArrayView2<f64>) -> Array2<f64> {
        if vecs.ncols() != 3 {
            panic!("Each vector must have 3 elements");
        }

        let matrix = self.as_matrix();
        vecs.dot(&matrix.t())
    }

    /// Combine this rotation with another (apply the other rotation after this one)
    ///
    /// # Arguments
    ///
    /// * `other` - The other rotation to combine with
    ///
    /// # Returns
    ///
    /// A new rotation that represents the composition of the two rotations
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::Rotation;
    /// use ndarray::array;
    /// use std::f64::consts::PI;
    ///
    /// // Rotate 90 degrees around X, then 90 degrees around Y
    /// let angles1 = array![PI/2.0, 0.0, 0.0];
    /// let rot1 = Rotation::from_euler(&angles1.view(), "xyz").unwrap();
    /// let angles2 = array![0.0, PI/2.0, 0.0];
    /// let rot2 = Rotation::from_euler(&angles2.view(), "xyz").unwrap();
    /// let combined = rot1.compose(&rot2);
    /// ```
    pub fn compose(&self, other: &Rotation) -> Rotation {
        // Quaternion multiplication
        let w1 = self.quat[0];
        let x1 = self.quat[1];
        let y1 = self.quat[2];
        let z1 = self.quat[3];

        let w2 = other.quat[0];
        let x2 = other.quat[1];
        let y2 = other.quat[2];
        let z2 = other.quat[3];

        let result_quat = array![
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ];

        // Normalize the resulting quaternion
        let norm = (result_quat.iter().map(|&x| x * x).sum::<f64>()).sqrt();
        let normalized_quat = result_quat / norm;

        Rotation {
            quat: normalized_quat,
        }
    }

    /// Create an identity rotation (no rotation)
    ///
    /// # Returns
    ///
    /// A new Rotation that represents no rotation
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::Rotation;
    /// use ndarray::array;
    ///
    /// let identity = Rotation::identity();
    /// let vec = array![1.0, 2.0, 3.0];
    /// let rotated = identity.apply(&vec.view());
    /// // Should still be [1.0, 2.0, 3.0]
    /// ```
    pub fn identity() -> Rotation {
        let quat = array![1.0, 0.0, 0.0, 0.0];
        Rotation { quat }
    }

    /// Create a random uniform rotation
    ///
    /// # Returns
    ///
    /// A new random Rotation uniformly distributed over the rotation space
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::transform::Rotation;
    ///
    /// let random_rot = Rotation::random();
    /// ```
    pub fn random() -> Rotation {
        use rand::Rng;
        let mut rng = rand::rng();

        // Generate random quaternion using method from:
        // http://planning.cs.uiuc.edu/node198.html
        let u1 = rng.random_range(0.0..1.0);
        let u2 = rng.random_range(0.0..1.0);
        let u3 = rng.random_range(0.0..1.0);

        let sqrt_u1 = u1.sqrt();
        let sqrt_1_minus_u1 = (1.0 - u1).sqrt();
        let two_pi_u2 = 2.0 * PI * u2;
        let two_pi_u3 = 2.0 * PI * u3;

        let quat = array![
            sqrt_1_minus_u1 * (two_pi_u2 / 2.0).sin(),
            sqrt_1_minus_u1 * (two_pi_u2 / 2.0).cos(),
            sqrt_u1 * (two_pi_u3 / 2.0).sin(),
            sqrt_u1 * (two_pi_u3 / 2.0).cos()
        ];

        Rotation { quat }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;
    use std::f64::consts::PI;

    #[test]
    fn test_rotation_identity() {
        let identity = Rotation::identity();
        let vec = array![1.0, 2.0, 3.0];
        let rotated = identity.apply(&vec.view());

        assert_relative_eq!(rotated[0], vec[0], epsilon = 1e-10);
        assert_relative_eq!(rotated[1], vec[1], epsilon = 1e-10);
        assert_relative_eq!(rotated[2], vec[2], epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_from_quat() {
        // A quaternion for 90 degrees rotation around X axis
        let quat = array![
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
            0.0
        ];
        let rot = Rotation::from_quat(&quat.view()).unwrap();

        let vec = array![0.0, 1.0, 0.0];
        let rotated = rot.apply(&vec.view());

        // Should rotate [0, 1, 0] to [0, 0, 1]
        assert_relative_eq!(rotated[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_from_matrix() {
        // A rotation matrix for 90 degrees rotation around Z axis
        let matrix = array![[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let rot = Rotation::from_matrix(&matrix.view()).unwrap();

        let vec = array![1.0, 0.0, 0.0];
        let rotated = rot.apply(&vec.view());

        // Should rotate [1, 0, 0] to [0, 1, 0]
        assert_relative_eq!(rotated[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_from_euler() {
        // 90 degrees rotation around X axis
        let angles = array![PI / 2.0, 0.0, 0.0];
        let rot = Rotation::from_euler(&angles.view(), "xyz").unwrap();

        let vec = array![0.0, 1.0, 0.0];
        let rotated = rot.apply(&vec.view());

        // Should rotate [0, 1, 0] to [0, 0, 1]
        assert_relative_eq!(rotated[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_from_rotvec() {
        // 90 degrees rotation around X axis
        let rotvec = array![PI / 2.0, 0.0, 0.0];
        let rot = Rotation::from_rotvec(&rotvec.view()).unwrap();

        let vec = array![0.0, 1.0, 0.0];
        let rotated = rot.apply(&vec.view());

        // Should rotate [0, 1, 0] to [0, 0, 1]
        assert_relative_eq!(rotated[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_compose() {
        // 90 degrees rotation around X, then 90 degrees around Y
        let rot_x = rotation_from_euler(PI / 2.0, 0.0, 0.0, "xyz").unwrap();
        let rot_y = rotation_from_euler(0.0, PI / 2.0, 0.0, "xyz").unwrap();

        let composed = rot_x.compose(&rot_y);

        let vec = array![0.0, 0.0, 1.0];
        let rotated = composed.apply(&vec.view());

        // Should rotate [0, 0, 1] to [1, 0, 0]
        assert_relative_eq!(rotated[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_inv() {
        // 45 degrees rotation around Z axis
        let rot = rotation_from_euler(0.0, 0.0, PI / 4.0, "xyz").unwrap();
        let rot_inv = rot.inv();

        let vec = array![1.0, 0.0, 0.0];
        let rotated = rot.apply(&vec.view());
        let rotated_back = rot_inv.apply(&rotated.view());

        // Should get back the original vector
        assert_relative_eq!(rotated_back[0], vec[0], epsilon = 1e-10);
        assert_relative_eq!(rotated_back[1], vec[1], epsilon = 1e-10);
        assert_relative_eq!(rotated_back[2], vec[2], epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_conversions() {
        // Create a rotation and verify conversion between representations
        let angles = array![PI / 4.0, PI / 6.0, PI / 3.0];
        let rot = Rotation::from_euler(&angles.view(), "xyz").unwrap();

        // Convert to matrix and back to Euler
        let matrix = rot.as_matrix();
        let rot_from_matrix = Rotation::from_matrix(&matrix.view()).unwrap();
        let _angles_back = rot_from_matrix.as_euler("xyz").unwrap();

        // Allow for different but equivalent representations
        let vec = array![1.0, 2.0, 3.0];
        let rotated1 = rot.apply(&vec.view());
        let rotated2 = rot_from_matrix.apply(&vec.view());

        assert_relative_eq!(rotated1[0], rotated2[0], epsilon = 1e-10);
        assert_relative_eq!(rotated1[1], rotated2[1], epsilon = 1e-10);
        assert_relative_eq!(rotated1[2], rotated2[2], epsilon = 1e-10);

        // Convert to axis-angle and back
        let rotvec = rot.as_rotvec();
        let rot_from_rotvec = Rotation::from_rotvec(&rotvec.view()).unwrap();
        let rotated3 = rot_from_rotvec.apply(&vec.view());

        assert_relative_eq!(rotated1[0], rotated3[0], epsilon = 1e-10);
        assert_relative_eq!(rotated1[1], rotated3[1], epsilon = 1e-10);
        assert_relative_eq!(rotated1[2], rotated3[2], epsilon = 1e-10);
    }
}
