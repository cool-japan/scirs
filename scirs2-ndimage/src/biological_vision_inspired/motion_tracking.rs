//! Motion Prediction and Tracking
//!
//! This module implements biological motion prediction mechanisms for robust
//! object tracking and motion extrapolation.

use scirs2_core::ndarray::ArrayView2;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::VecDeque;

use super::config::{BiologicalVisionConfig, MotionTrack};
use crate::error::{NdimageError, NdimageResult};

/// Motion Prediction and Tracking
///
/// Implements biological motion prediction mechanisms for robust
/// object tracking and motion extrapolation.
pub fn bio_motion_prediction_tracking<T>(
    image_sequence: &[ArrayView2<T>],
    initial_targets: &[(usize, usize)],
    config: &BiologicalVisionConfig,
) -> NdimageResult<Vec<MotionTrack>>
where
    T: Float + FromPrimitive + Copy + Send + Sync,
{
    if image_sequence.len() < config.motion_prediction_window {
        return Err(NdimageError::InvalidInput(
            "Insufficient frames for motion prediction".to_string(),
        ));
    }

    let mut motion_tracks = Vec::new();

    // Initialize tracks for each target
    for &target in initial_targets {
        let track = MotionTrack {
            object_id: motion_tracks.len(),
            position: (target.0 as f64, target.1 as f64),
            velocity: (0.0, 0.0),
            positionhistory: VecDeque::from(vec![(target.0 as f64, target.1 as f64)]),
            predicted_positions: Vec::new(),
            confidence: 1.0,
            age: 0,
        };
        motion_tracks.push(track);
    }

    // Process temporal sequence
    for window_start in 0..image_sequence
        .len()
        .saturating_sub(config.motion_prediction_window)
    {
        let window = &image_sequence[window_start..window_start + config.motion_prediction_window];

        for track in &mut motion_tracks {
            // Update motion estimates
            update_motion_estimates(track, window, config)?;

            // Predict future positions
            predict_future_positions(track, config)?;

            // Update confidence based on prediction accuracy
            update_tracking_confidence(track, window, config)?;

            // Update track age
            track.age += 1;
        }

        // Handle track management (creation, deletion, merging)
        manage_tracks(&mut motion_tracks, image_sequence, window_start, config)?;
    }

    Ok(motion_tracks)
}

/// Update motion estimates for a track
pub fn update_motion_estimates<T>(
    track: &mut MotionTrack,
    window: &[ArrayView2<T>],
    config: &BiologicalVisionConfig,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Copy,
{
    if window.len() < 2 {
        return Ok(());
    }

    let (height, width) = window[0].dim();

    // Find the best matching position in the current frame
    let current_position = find_best_match(track, &window[window.len() - 1], (height, width))?;

    // Update position history
    track.positionhistory.push_back(current_position);
    if track.positionhistory.len() > config.motion_prediction_window {
        track.positionhistory.pop_front();
    }

    // Update current position
    track.position = current_position;

    // Estimate velocity from position history
    if track.positionhistory.len() >= 2 {
        let positions: Vec<(f64, f64)> = track.positionhistory.iter().cloned().collect();
        track.velocity = estimate_velocity(&positions)?;
    }

    Ok(())
}

/// Find the best matching position for a track in the current frame
fn find_best_match<T>(
    track: &MotionTrack,
    frame: &ArrayView2<T>,
    frame_size: (usize, usize),
) -> NdimageResult<(f64, f64)>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width) = frame_size;
    let search_radius = 20; // Search within 20 pixels of predicted position

    // Predict current position based on velocity
    let predicted_x = track.position.0 + track.velocity.0;
    let predicted_y = track.position.1 + track.velocity.1;

    let mut best_position = (predicted_x, predicted_y);
    let mut best_score = f64::NEG_INFINITY;

    // Search in a window around the predicted position
    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            let test_x = (predicted_x + dx as f64).max(0.0).min((width - 1) as f64);
            let test_y = (predicted_y + dy as f64).max(0.0).min((height - 1) as f64);

            // Compute matching score based on local image features
            let score = compute_matching_score(frame, (test_x as usize, test_y as usize))?;

            if score > best_score {
                best_score = score;
                best_position = (test_x, test_y);
            }
        }
    }

    Ok(best_position)
}

/// Compute matching score for a position in the frame
fn compute_matching_score<T>(frame: &ArrayView2<T>, position: (usize, usize)) -> NdimageResult<f64>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width) = frame.dim();
    let (y, x) = position;

    if y >= height || x >= width {
        return Ok(0.0);
    }

    // Simple score based on local variance (indicates structure/features)
    let window_size = 5;
    let half_window = window_size / 2;

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0;

    for dy in 0..window_size {
        for dx in 0..window_size {
            let y_offset = dy as isize - half_window as isize;
            let x_offset = dx as isize - half_window as isize;
            let ny = (y as isize + y_offset) as usize;
            let nx = (x as isize + x_offset) as usize;

            if y as isize + y_offset >= 0 && x as isize + x_offset >= 0 && ny < height && nx < width
            {
                let value = frame[(ny, nx)].to_f64().unwrap_or(0.0);
                sum += value;
                sum_sq += value * value;
                count += 1;
            }
        }
    }

    if count > 1 {
        let mean = sum / count as f64;
        let variance = (sum_sq / count as f64) - (mean * mean);
        Ok(variance.sqrt()) // Standard deviation as score
    } else {
        Ok(0.0)
    }
}

/// Estimate velocity from position history
fn estimate_velocity(positions: &[(f64, f64)]) -> NdimageResult<(f64, f64)> {
    if positions.len() < 2 {
        return Ok((0.0, 0.0));
    }

    // Simple linear velocity estimation
    let last_pos = positions[positions.len() - 1];
    let prev_pos = positions[positions.len() - 2];

    let velocity_x = last_pos.0 - prev_pos.0;
    let velocity_y = last_pos.1 - prev_pos.1;

    Ok((velocity_x, velocity_y))
}

/// Predict future positions based on current motion model
pub fn predict_future_positions(
    track: &mut MotionTrack,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()> {
    track.predicted_positions.clear();

    // Predict positions for the next few frames
    let prediction_steps = config.motion_prediction_window;

    for step in 1..=prediction_steps {
        let predicted_x = track.position.0 + track.velocity.0 * step as f64;
        let predicted_y = track.position.1 + track.velocity.1 * step as f64;

        track.predicted_positions.push((predicted_x, predicted_y));
    }

    Ok(())
}

/// Update tracking confidence based on prediction accuracy
pub fn update_tracking_confidence<T>(
    track: &mut MotionTrack,
    window: &[ArrayView2<T>],
    config: &BiologicalVisionConfig,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Copy,
{
    if track.predicted_positions.is_empty() || window.is_empty() {
        return Ok(());
    }

    // Simple confidence update based on prediction accuracy
    if let Some(&predicted_pos) = track.predicted_positions.first() {
        let actual_pos = (track.position.0 as usize, track.position.1 as usize);

        let prediction_error = ((predicted_pos.0 as f64 - actual_pos.0 as f64).powi(2)
            + (predicted_pos.1 as f64 - actual_pos.1 as f64).powi(2))
        .sqrt();

        // Update confidence inversely proportional to error
        let error_threshold = 10.0; // pixels
        let confidence_adjustment =
            (error_threshold - prediction_error.min(error_threshold)) / error_threshold;

        track.confidence = (track.confidence * 0.9 + confidence_adjustment * 0.1)
            .max(0.0)
            .min(1.0);
    }

    Ok(())
}

/// Manage track creation, deletion, and merging
pub fn manage_tracks<T>(
    tracks: &mut Vec<MotionTrack>,
    image_sequence: &[ArrayView2<T>],
    current_time: usize,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Copy,
{
    // Remove tracks with very low confidence
    tracks.retain(|track| track.confidence > 0.1);

    // Remove very old tracks
    tracks.retain(|track| track.age < 100);

    // Simple track merging: remove tracks that are too close to each other
    let mut to_remove = Vec::new();

    for i in 0..tracks.len() {
        for j in (i + 1)..tracks.len() {
            let distance = ((tracks[i].position.0 - tracks[j].position.0).powi(2)
                + (tracks[i].position.1 - tracks[j].position.1).powi(2))
            .sqrt();

            if distance < 5.0 {
                // Very close tracks
                // Keep the one with higher confidence
                if tracks[i].confidence < tracks[j].confidence {
                    to_remove.push(i);
                } else {
                    to_remove.push(j);
                }
            }
        }
    }

    // Remove duplicates and sort in reverse order for safe removal
    to_remove.sort_unstable();
    to_remove.dedup();

    for &index in to_remove.iter().rev() {
        if index < tracks.len() {
            tracks.remove(index);
        }
    }

    Ok(())
}
