// Moment-based feature extraction for images

use super::utils::calculate_raw_moment;
use crate::error::SignalResult;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

#[allow(unused_imports)]
/// Extract moment-based features from an image
#[allow(dead_code)]
pub fn extract_moment_features(
    image: &Array2<f64>,
    features: &mut HashMap<String, f64>,
) -> SignalResult<()> {
    let shape = image.shape();

    // Calculate raw moments up to order 3
    let m00 = calculate_raw_moment(image, 0, 0);
    let m01 = calculate_raw_moment(image, 0, 1);
    let m10 = calculate_raw_moment(image, 1, 0);
    let m11 = calculate_raw_moment(image, 1, 1);
    let m02 = calculate_raw_moment(image, 0, 2);
    let m20 = calculate_raw_moment(image, 2, 0);
    let m12 = calculate_raw_moment(image, 1, 2);
    let m21 = calculate_raw_moment(image, 2, 1);
    let m03 = calculate_raw_moment(image, 0, 3);
    let m30 = calculate_raw_moment(image, 3, 0);

    // Store raw moments
    features.insert("moment_m00".to_string(), m00);
    features.insert("moment_m01".to_string(), m01);
    features.insert("moment_m10".to_string(), m10);
    features.insert("moment_m11".to_string(), m11);
    features.insert("moment_m02".to_string(), m02);
    features.insert("moment_m20".to_string(), m20);

    // Calculate central moments
    let x_centroid = m10 / m00;
    let y_centroid = m01 / m00;

    features.insert("centroid_x".to_string(), x_centroid);
    features.insert("centroid_y".to_string(), y_centroid);

    // Calculate central moments
    let mu00 = m00;
    let mu11 = m11 - x_centroid * m01;
    let mu20 = m20 - x_centroid * m10;
    let mu02 = m02 - y_centroid * m01;
    let mu30 = m30 - 3.0 * x_centroid * m20 + 2.0 * x_centroid * x_centroid * m10;
    let mu03 = m03 - 3.0 * y_centroid * m02 + 2.0 * y_centroid * y_centroid * m01;
    let mu21 =
        m21 - 2.0 * x_centroid * m11 - y_centroid * m20 + 2.0 * x_centroid * x_centroid * m01;
    let mu12 =
        m12 - 2.0 * y_centroid * m11 - x_centroid * m02 + 2.0 * y_centroid * y_centroid * m10;

    // Store central moments
    features.insert("central_moment_mu11".to_string(), mu11);
    features.insert("central_moment_mu20".to_string(), mu20);
    features.insert("central_moment_mu02".to_string(), mu02);

    // Normalized central moments
    let norm_factor = mu00.powf(1.0 + 1.0); // 1.0 + 1.0 is exponent for 2nd order moments
    let eta11 = mu11 / norm_factor;
    let eta20 = mu20 / norm_factor;
    let eta02 = mu02 / norm_factor;

    features.insert("norm_central_moment_eta11".to_string(), eta11);
    features.insert("norm_central_moment_eta20".to_string(), eta20);
    features.insert("norm_central_moment_eta02".to_string(), eta02);

    // Hu's seven invariant moments
    let h1 = eta20 + eta02;
    let h2 = (eta20 - eta02).powi(2) + 4.0 * eta11.powi(2);
    let h3 = (mu30 - 3.0 * mu12).powi(2) + (3.0 * mu21 - mu03).powi(2);
    let h4 = (mu30 + mu12).powi(2) + (mu21 + mu03).powi(2);

    features.insert("hu_moment_1".to_string(), h1);
    features.insert("hu_moment_2".to_string(), h2);
    features.insert("hu_moment_3".to_string(), h3);
    features.insert("hu_moment_4".to_string(), h4);

    // Calculate image orientation
    let orientation = 0.5 * (2.0 * mu11 / (mu20 - mu02)).atan();
    features.insert("orientation".to_string(), orientation);

    // Calculate eccentricity
    let common = (mu20 - mu02).powi(2) + 4.0 * mu11.powi(2);
    let major_axis = 2.0 * ((mu20 + mu02 + common.sqrt()) / mu00).sqrt();
    let minor_axis = 2.0 * ((mu20 + mu02 - common.sqrt()) / mu00).sqrt();

    if minor_axis > 1e-10 {
        let eccentricity = (1.0 - (minor_axis / major_axis).powi(2)).sqrt();
        features.insert("eccentricity".to_string(), eccentricity);
    } else {
        features.insert("eccentricity".to_string(), 1.0); // Degenerate case
    }

    // Calculate major and minor axis lengths
    features.insert("major_axis_length".to_string(), major_axis);
    features.insert("minor_axis_length".to_string(), minor_axis);

    Ok(())
}
