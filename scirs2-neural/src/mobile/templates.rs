//! Platform-specific code generation templates
//!
//! This module provides template content for generating platform-specific
//! code files including iOS framework components, Android wrappers,
//! and native code interfaces.

/// iOS Info.plist template for framework
pub const IOS_INFO_PLIST: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.scirs2.neural</string>
    <key>CFBundleName</key>
    <string>SciRS2Neural</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>MinimumOSVersion</key>
    <string>12.0</string>
</dict>
</plist>"#;

/// Swift wrapper implementation
pub const SWIFT_WRAPPER: &str = r#"
import Foundation
import CoreML

@objc public class SciRS2Model: NSObject {
    private var model: MLModel?
    
    @objc public override init() {
        super.init()
    }
    
    @objc public func loadModel(from path: String) throws {
        let modelURL = URL(fileURLWithPath: path)
        model = try MLModel(contentsOf: modelURL)
    }
    
    @objc public func predict(input: MLMultiArray) throws -> MLMultiArray {
        guard let model = model else {
            throw NSError(domain: "SciRS2Model", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
        }
        
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input": input])
        let output = try model.prediction(from: provider)
        
        guard let result = output.featureValue(for: "output")?.multiArrayValue else {
            throw NSError(domain: "SciRS2Model", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid output"])
        }
        
        return result
    }
}
"#;

/// Objective-C header template
pub const OBJC_HEADER: &str = r#"
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

@interface SciRS2Model : NSObject

- (instancetype)init;
- (void)loadModelFromPath:(NSString *)path error:(NSError **)error;
- (MLMultiArray *)predictWithInput:(MLMultiArray *)input error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
"#;

/// Objective-C implementation template
pub const OBJC_IMPL: &str = r#"
#import "SciRS2Model.h"

@interface SciRS2Model ()
@property (nonatomic, strong) MLModel *model;
@end

@implementation SciRS2Model

- (instancetype)init {
    self = [super init];
    if (self) {
        // Initialization
    }
    return self;
}

- (void)loadModelFromPath:(NSString *)path error:(NSError **)error {
    NSURL *modelURL = [NSURL fileURLWithPath:path];
    self.model = [MLModel modelWithContentsOfURL:modelURL error:error];
}

- (MLMultiArray *)predictWithInput:(MLMultiArray *)input error:(NSError **)error {
    if (!self.model) {
        if (error) {
            *error = [NSError errorWithDomain:@"SciRS2Model" code:1 userInfo:@{NSLocalizedDescriptionKey: @"Model not loaded"}];
        }
        return nil;
    }
    
    MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"input": input} error:error];
    if (!provider) return nil;
    
    id<MLFeatureProvider> output = [self.model predictionFromFeatures:provider error:error];
    if (!output) return nil;
    
    MLFeatureValue *result = [output featureValueForName:@"output"];
    return result.multiArrayValue;
}

@end
"#;

/// Java wrapper implementation
pub const JAVA_WRAPPER: &str = r#"
package com.scirs2.neural;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class SciRS2Model {
    private Interpreter interpreter;
    
    public SciRS2Model(Context context, String modelPath) throws IOException {
        interpreter = new Interpreter(loadModelFile(context, modelPath));
    }
    
    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    
    public float[] predict(float[] input) {
        float[][] output = new float[1][1]; // Adjust based on model output shape
        interpreter.run(input, output);
        return output[0];
    }
    
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
}
"#;

/// Kotlin wrapper implementation
pub const KOTLIN_WRAPPER: &str = r#"
package com.scirs2.neural

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class SciRS2Model(context: Context, modelPath: String) {
    private var interpreter: Interpreter? = null
    
    init {
        val modelBuffer = loadModelFile(context, modelPath)
        interpreter = Interpreter(modelBuffer)
    }
    
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun predict(input: FloatArray): FloatArray {
        val output = Array(1) { FloatArray(1) } // Adjust based on model output shape
        interpreter?.run(input, output)
        return output[0]
    }
    
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
"#;

/// JNI header template
pub const JNI_HEADER: &str = r#"
#ifndef SCIRS2_JNI_H
#define SCIRS2_JNI_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_scirs2_neural_SciRS2Model_createNativeModel(JNIEnv *env, jobject thiz, jstring model_path);

JNIEXPORT jfloatArray JNICALL
Java_com_scirs2_neural_SciRS2Model_predictNative(JNIEnv *env, jobject thiz, jlong handle, jfloatArray input);

JNIEXPORT void JNICALL
Java_com_scirs2_neural_SciRS2Model_destroyNativeModel(JNIEnv *env, jobject thiz, jlong handle);

#ifdef __cplusplus
}
#endif

#endif // SCIRS2_JNI_H
"#;

/// JNI implementation template
pub const JNI_IMPL: &str = r#"
#include "scirs2_jni.h"
#include <android/log.h>
#include <string>

#define LOG_TAG "SciRS2Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct NativeModel {
    // Model implementation would go here
    int dummy;
};

JNIEXPORT jlong JNICALL
Java_com_scirs2_neural_SciRS2Model_createNativeModel(JNIEnv *env, jobject thiz, jstring model_path) {
    const char *path = env->GetStringUTFChars(model_path, nullptr);
    LOGI("Loading model from: %s", path);
    
    NativeModel *model = new NativeModel();
    model->dummy = 42; // Stub implementation
    
    env->ReleaseStringUTFChars(model_path, path);
    return reinterpret_cast<jlong>(model);
}

JNIEXPORT jfloatArray JNICALL
Java_com_scirs2_neural_SciRS2Model_predictNative(JNIEnv *env, jobject thiz, jlong handle, jfloatArray input) {
    NativeModel *model = reinterpret_cast<NativeModel*>(handle);
    if (!model) {
        LOGE("Invalid model handle");
        return nullptr;
    }
    
    jsize input_length = env->GetArrayLength(input);
    jfloat *input_data = env->GetFloatArrayElements(input, nullptr);
    
    // Stub prediction: copy input to output
    jfloatArray output = env->NewFloatArray(input_length);
    env->SetFloatArrayRegion(output, 0, input_length, input_data);
    
    env->ReleaseFloatArrayElements(input, input_data, JNI_ABORT);
    return output;
}

JNIEXPORT void JNICALL
Java_com_scirs2_neural_SciRS2Model_destroyNativeModel(JNIEnv *env, jobject thiz, jlong handle) {
    NativeModel *model = reinterpret_cast<NativeModel*>(handle);
    if (model) {
        delete model;
    }
}
"#;
