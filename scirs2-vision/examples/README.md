# SciRS2 Vision Examples

This directory contains examples demonstrating the various features of the scirs2-vision module.

## Directory Structure

- `input/` - Directory for input images
  - `input.jpg` - Default test image used by examples
  - `input_gray.jpg` - Grayscale version of the test image
- `output/` - Directory for output images (automatically created by examples)
  - `.gitignore` - Ignores all output files except directory structure

## Creating Test Images

Run the test image creation utility to generate sample input images:

```bash
cargo run --example create_test_image
```

This will create two test images in the `input/` directory:
- `input.jpg` - A colored test image with various shapes
- `input_gray.jpg` - A grayscale version

## Running Examples

All examples expect input images to be in `examples/input/` and will save their output to `examples/output/`.

### Edge Detection Examples

1. **Canny Edge Detection**
   ```bash
   cargo run --example canny_edge_detection
   ```
   Demonstrates the Canny edge detector with various parameters.

2. **Edge Detection Comparison**
   ```bash
   cargo run --example edge_detection_comparison
   ```
   Compares different edge detection algorithms: Sobel, Canny, Prewitt, Laplacian, and Laplacian of Gaussian.

### Corner Detection Examples

1. **Corner Detection Comparison**
   ```bash
   cargo run --example corner_detection_comparison
   ```
   Compares different corner detection algorithms: Harris, Shi-Tomasi, and FAST.

2. **Feature Detection**
   ```bash
   cargo run --example feature_detection
   ```
   Demonstrates feature detection and descriptor computation.

### Image Processing Examples

1. **Color Transformations**
   ```bash
   cargo run --example color_transformations
   ```
   Shows color space conversions: RGB ↔ HSV, RGB ↔ LAB.

2. **Image Segmentation**
   ```bash
   cargo run --example image_segmentation
   ```
   Demonstrates thresholding and segmentation techniques.

3. **Morphological Operations**
   ```bash
   cargo run --example morphological_operations
   ```
   Shows erosion, dilation, opening, closing, and other morphological operations.

4. **Non-Rigid Transformations**
   ```bash
   cargo run --example non_rigid_transformations
   ```
   Demonstrates non-rigid transformation methods including thin-plate splines and elastic deformations.

5. **Noise Reduction**
   ```bash
   cargo run --example noise_reduction
   ```
   Demonstrates various noise reduction techniques including Gaussian blur, bilateral filtering, and median filtering.

## Using Your Own Images

To use your own images with the examples:

1. Place your image in the `examples/input/` directory
2. Rename it to `input.jpg` (or modify the example code to use your filename)
3. Run the example

## Output Files

All output files are saved to `examples/output/` with descriptive names indicating the operation performed. The directory is gitignored to prevent committing generated files.