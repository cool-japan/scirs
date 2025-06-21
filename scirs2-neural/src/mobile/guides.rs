//! Documentation and integration guide content
//!
//! This module provides comprehensive documentation templates for mobile
//! platform integration including iOS integration guides, Android setup
//! instructions, and optimization best practices.

/// iOS integration guide content
pub const IOS_INTEGRATION_GUIDE: &str = r#"# iOS Integration Guide

## Prerequisites

- Xcode 14.0 or later
- iOS 12.0 or later
- Swift 5.0 or later

## Installation

### Using CocoaPods

```ruby
pod 'SciRS2Neural', '~> 1.0'
```

### Manual Installation

1. Download the SciRS2Neural.framework
2. Drag and drop it into your Xcode project
3. Ensure "Copy items if needed" is checked
4. Add the framework to "Embedded Binaries"

## Usage

### Swift

```swift
import SciRS2Neural

class ViewController: UIViewController {
    var model: SciRS2Model?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupModel()
    }
    
    func setupModel() {
        model = SciRS2Model()
        
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "mlmodel") else {
            print("Model file not found")
            return
        }
        
        do {
            try model?.loadModel(from: modelPath)
            print("Model loaded successfully")
        } catch {
            print("Failed to load model: \(error)")
        }
    }
    
    func runInference(with inputData: [Float]) {
        guard let model = model else { return }
        
        do {
            // Convert input data to MLMultiArray
            let inputArray = try MLMultiArray(shape: [1, NSNumber(value: inputData.count)], dataType: .float32)
            for (index, value) in inputData.enumerated() {
                inputArray[index] = NSNumber(value: value)
            }
            
            // Run inference
            let output = try model.predict(input: inputArray)
            
            // Process output
            print("Prediction completed")
            
        } catch {
            print("Inference failed: \(error)")
        }
    }
}
```

### Objective-C

```objc
#import "SciRS2Model.h"

@interface ViewController ()
@property (nonatomic, strong) SciRS2Model *model;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    [self setupModel];
}

- (void)setupModel {
    self.model = [[SciRS2Model alloc] init];
    
    NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"model" ofType:@"mlmodel"];
    if (!modelPath) {
        NSLog(@"Model file not found");
        return;
    }
    
    NSError *error;
    [self.model loadModelFromPath:modelPath error:&error];
    if (error) {
        NSLog(@"Failed to load model: %@", error.localizedDescription);
    } else {
        NSLog(@"Model loaded successfully");
    }
}

@end
```

## Performance Optimization

### Memory Management

- Use autorelease pools for batch processing
- Release model resources when not needed
- Monitor memory usage with Instruments

### Metal Performance Shaders

```swift
import MetalPerformanceShaders

// Enable Metal acceleration
let metalDevice = MTLCreateSystemDefaultDevice()
let commandQueue = metalDevice?.makeCommandQueue()
```

### Core ML Optimization

- Use Core ML Tools for model optimization
- Enable compute unit preferences
- Use asynchronous prediction for better performance

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model file is included in the app bundle
2. **Memory issues**: Check model size and available memory
3. **Performance problems**: Profile with Instruments

### Debug Tips

- Enable verbose logging
- Use breakpoints for debugging
- Test on different device types

## Best Practices

1. Load models asynchronously
2. Cache prediction results when appropriate
3. Handle low memory warnings
4. Test on older devices
5. Monitor thermal state
"#;

/// Android integration guide content
pub const ANDROID_INTEGRATION_GUIDE: &str = r#"# Android Integration Guide

## Prerequisites

- Android Studio 4.0 or later
- Android API level 21 or later
- NDK (for native code)

## Installation

### Using Gradle

```gradle
dependencies {
    implementation 'com.scirs2:neural:1.0.0'
}
```

### Manual Installation

1. Download the AAR file
2. Place it in your `libs` directory
3. Add to your `build.gradle`:

```gradle
repositories {
    flatDir {
        dirs 'libs'
    }
}

dependencies {
    implementation(name: 'scirs2-neural', version: '1.0.0', ext: 'aar')
}
```

## Usage

### Kotlin

```kotlin
import com.scirs2.neural.SciRS2Model

class MainActivity : AppCompatActivity() {
    private var model: SciRS2Model? = null
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        setupModel()
    }
    
    private fun setupModel() {
        try {
            model = SciRS2Model(this, "scirs2_model.tflite")
            Log.d("SciRS2", "Model loaded successfully")
        } catch (e: IOException) {
            Log.e("SciRS2", "Failed to load model", e)
        }
    }
    
    private fun runInference(inputData: FloatArray) {
        model?.let { model ->
            try {
                val output = model.predict(inputData)
                Log.d("SciRS2", "Prediction: ${output.contentToString()}")
            } catch (e: Exception) {
                Log.e("SciRS2", "Inference failed", e)
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        model?.close()
    }
}
```

### Java

```java
import com.scirs2.neural.SciRS2Model;

public class MainActivity extends AppCompatActivity {
    private SciRS2Model model;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        setupModel();
    }
    
    private void setupModel() {
        try {
            model = new SciRS2Model(this, "scirs2_model.tflite");
            Log.d("SciRS2", "Model loaded successfully");
        } catch (IOException e) {
            Log.e("SciRS2", "Failed to load model", e);
        }
    }
    
    private void runInference(float[] inputData) {
        if (model != null) {
            try {
                float[] output = model.predict(inputData);
                Log.d("SciRS2", "Prediction completed");
            } catch (Exception e) {
                Log.e("SciRS2", "Inference failed", e);
            }
        }
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (model != null) {
            model.close();
        }
    }
}
```

## Performance Optimization

### NNAPI Acceleration

```kotlin
// Enable NNAPI acceleration
val options = Interpreter.Options()
options.setUseNNAPI(true)
val interpreter = Interpreter(modelBuffer, options)
```

### GPU Delegation

```kotlin
// Enable GPU acceleration
val gpuDelegate = GpuDelegate()
val options = Interpreter.Options()
options.addDelegate(gpuDelegate)
```

### Multi-threading

```kotlin
// Use multiple threads
val options = Interpreter.Options()
options.setNumThreads(4)
```

## ProGuard/R8 Configuration

```proguard
-keep class com.scirs2.neural.** { *; }
-keep class org.tensorflow.lite.** { *; }
-keepclassmembers class * {
    native <methods>;
}
```

## Troubleshooting

### Common Issues

1. **Model loading failed**: Check if model file is in assets
2. **Native library not found**: Ensure NDK is properly configured
3. **Out of memory**: Reduce model size or input batch size

### Performance Issues

- Profile with Android Profiler
- Use systrace for detailed analysis
- Monitor CPU and GPU usage

## Best Practices

1. Load models on background threads
2. Use appropriate delegates for acceleration
3. Handle different screen densities
4. Test on various device configurations
5. Implement proper error handling
"#;

/// Optimization guide content
pub const OPTIMIZATION_GUIDE: &str = r#"# Mobile Optimization Guide

## Overview

This guide covers optimization techniques for deploying neural networks on mobile devices.

## Model Optimization Techniques

### Quantization

#### Post-Training Quantization
- Converts FP32 weights to INT8
- Minimal accuracy loss
- 4x size reduction
- 2-4x speed improvement

#### Quantization-Aware Training
- Training with simulated quantization
- Better accuracy preservation
- Requires retraining

### Pruning

#### Magnitude-Based Pruning
- Remove smallest weights
- Structured or unstructured
- 50-90% sparsity possible

#### Gradient-Based Pruning
- Use gradient information
- More sophisticated importance metrics
- Better accuracy retention

### Knowledge Distillation

#### Teacher-Student Framework
- Large teacher model
- Small student model
- Transfer knowledge through soft targets

### Layer Fusion

#### Common Patterns
- Conv + BatchNorm + ReLU
- Dense + Activation
- Reduces memory bandwidth

## Platform-Specific Optimizations

### iOS Optimizations

#### Core ML
- Automatic optimization
- Hardware-specific acceleration
- Neural Engine utilization

#### Metal Performance Shaders
- GPU acceleration
- Custom kernels
- Memory optimization

### Android Optimizations

#### TensorFlow Lite
- Optimized for mobile
- Multiple acceleration options
- Flexible deployment

#### NNAPI
- Hardware abstraction layer
- Vendor-optimized implementations
- Automatic fallbacks

## Performance Monitoring

### Key Metrics

1. **Latency**: Time per inference
2. **Throughput**: Inferences per second
3. **Memory**: Peak and average usage
4. **Power**: Energy consumption
5. **Thermal**: Temperature impact

### Profiling Tools

#### iOS
- Instruments
- Core ML Performance Reports
- Xcode Energy Gauge

#### Android
- Android Profiler
- Systrace
- GPU Profiler

## Memory Optimization

### Strategies

1. **Model Compression**: Reduce model size
2. **Memory Pooling**: Reuse allocations
3. **Lazy Loading**: Load on demand
4. **Memory Mapping**: Map instead of load

### Implementation

```swift
// iOS Memory Pool
class MemoryPool {
    private var buffers: [MLMultiArray] = []
    
    func getBuffer(shape: [Int]) -> MLMultiArray {
        // Reuse existing buffer or create new one
    }
    
    func returnBuffer(_ buffer: MLMultiArray) {
        // Return buffer to pool
    }
}
```

## Power Management

### Strategies

1. **Adaptive Inference**: Adjust based on battery level
2. **Thermal Throttling**: Reduce performance when hot
3. **Scheduling**: Run during charging
4. **Quality Scaling**: Lower quality for battery saving

### Implementation

```kotlin
// Android Power Management
class PowerManager {
    fun getOptimalInferenceMode(): InferenceMode {
        val batteryLevel = getBatteryLevel()
        val thermalState = getThermalState()
        
        return when {
            batteryLevel < 20 -> InferenceMode.POWER_SAVE
            thermalState == ThermalState.CRITICAL -> InferenceMode.THROTTLED
            else -> InferenceMode.NORMAL
        }
    }
}
```

## Best Practices

### Development

1. **Profile Early**: Start optimization from day one
2. **Target Devices**: Test on representative hardware
3. **Measure Real Usage**: Monitor production metrics
4. **Iterative Optimization**: Gradual improvements

### Deployment

1. **A/B Testing**: Compare optimization variants
2. **Progressive Rollout**: Gradual feature deployment
3. **Monitoring**: Track performance metrics
4. **Fallbacks**: Handle edge cases gracefully

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check for memory leaks
   - Optimize buffer allocation
   - Use memory profiling tools

2. **Poor Performance**
   - Profile inference pipeline
   - Check hardware utilization
   - Optimize model architecture

3. **Battery Drain**
   - Monitor power consumption
   - Implement adaptive strategies
   - Optimize inference frequency

4. **Thermal Issues**
   - Monitor temperature
   - Implement throttling
   - Optimize workload distribution

### Debug Tips

1. Enable detailed logging
2. Use profiling tools extensively
3. Test on multiple devices
4. Monitor real-world usage
"#;
