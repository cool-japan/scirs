use ag::tensor_ops as T;
use ndarray::Array2;
use scirs2_autograd as ag;
use std::time::Instant;

fn main() {
    println!("Flexible Gradient Checkpointing Example");
    println!("======================================");
    println!("This example demonstrates the enhanced checkpoint functionality");
    println!("which makes it easier to work with checkpointing for arbitrary segments");
    println!("of a computation graph.");
    println!();

    ag::run::<f32, _, _>(|ctx| {
        // Create some input tensors for our computation
        let a = T::convert_to_tensor(Array2::<f32>::eye(64).into_dyn(), ctx);
        let b = T::convert_to_tensor(Array2::<f32>::ones((64, 64)).into_dyn(), ctx);

        println!("Running a multi-step computation without checkpointing...");
        let start = Instant::now();

        // Regular computation
        let c1 = T::matmul(&a, &b); // Matrix multiplication
        let d1 = T::relu(&c1); // Non-linear activation
        let e1 = T::matmul(&d1, &b); // Another matrix multiplication
        let f1 = T::tanh(&e1); // Another non-linear activation
        let g1 = T::matmul(&f1, &b); // Final matrix multiplication
        let h1 = T::sum_all(&g1); // Reduction to a scalar

        // Compute gradients
        let grad1 = T::grad(&[h1], &[&a])[0];
        let grad1_val = grad1.eval(ctx).unwrap();

        let normal_time = start.elapsed();
        println!("  Normal execution took: {:?}", normal_time);

        println!("\nRunning with manual checkpointing...");
        let start = Instant::now();

        // Manual checkpointing
        let c2 = T::matmul(&a, &b);
        let c2_ckpt = T::checkpoint(&c2);
        let d2 = T::relu(&c2_ckpt);
        let d2_ckpt = T::checkpoint(&d2);
        let e2 = T::matmul(&d2_ckpt, &b);
        let e2_ckpt = T::checkpoint(&e2);
        let f2 = T::tanh(&e2_ckpt);
        let f2_ckpt = T::checkpoint(&f2);
        let g2 = T::matmul(&f2_ckpt, &b);
        let h2 = T::sum_all(&g2);

        // Compute gradients
        let grad2 = T::grad(&[h2], &[&a])[0];
        let grad2_val = grad2.eval(ctx).unwrap();

        let manual_ckpt_time = start.elapsed();
        println!("  Manual checkpointing took: {:?}", manual_ckpt_time);

        println!("\nRunning with checkpoint_segment_flex...");
        let start = Instant::now();

        // Use the new flexible checkpoint segment
        let result3 = T::checkpoint_segment_flex(ctx, &[&a, &b], |inputs| {
            let a = inputs[0];
            let b = inputs[1];

            let c = T::matmul(a, b);
            let d = T::relu(&c);
            let e = T::matmul(&d, b);
            let f = T::tanh(&e);
            let g = T::matmul(&f, b);
            T::sum_all(&g)
        });

        // Compute gradients
        let grad3 = T::grad(&[result3], &[&a])[0];
        let grad3_val = grad3.eval(ctx).unwrap();

        let flex_ckpt_time = start.elapsed();
        println!("  Flexible checkpointing took: {:?}", flex_ckpt_time);

        // Now demonstrate the new CheckpointGroup methods
        println!("\nRunning with CheckpointGroup::checkpoint_fn_flex...");
        let ckpt_group = T::CheckpointGroup::new(ctx);
        let start = Instant::now();

        // Use the new checkpoint_fn_flex method
        let result4 = ckpt_group.checkpoint_fn_flex(&[&a, &b], |inputs| {
            let a = inputs[0];
            let b = inputs[1];

            let c = T::matmul(a, b);
            let d = T::relu(&c);
            let e = T::matmul(&d, b);
            let f = T::tanh(&e);
            let g = T::matmul(&f, b);
            T::sum_all(&g)
        });

        // Compute gradients
        let grad4 = T::grad(&[result4], &[&a])[0];
        let grad4_val = grad4.eval(ctx).unwrap();

        let group_flex_time = start.elapsed();
        println!(
            "  CheckpointGroup::checkpoint_fn_flex took: {:?}",
            group_flex_time
        );

        // Demonstrate checkpoint_fn_flex2 with multiple outputs
        println!("\nRunning with CheckpointGroup::checkpoint_fn_flex2 (multiple outputs)...");
        let start = Instant::now();

        // Use checkpoint_fn_flex2 to get two output tensors
        let (result5a, result5b) = ckpt_group.checkpoint_fn_flex2(&[&a, &b], |inputs| {
            let a = inputs[0];
            let b = inputs[1];

            // Compute two different output paths
            let path1 = {
                let c = T::matmul(a, b);
                let d = T::relu(&c);
                let e = T::matmul(&d, b);
                T::sum_all(&e)
            };

            let path2 = {
                let c = T::matmul(a, b);
                let d = T::tanh(&c);
                let e = T::matmul(&d, b);
                T::sum_all(&e)
            };

            (path1, path2)
        });

        // Compute gradients through both outputs
        let loss_multi = result5a + result5b;
        let grad5 = T::grad(&[loss_multi], &[&a])[0];
        let grad5_val = grad5.eval(ctx).unwrap();

        let multi_output_time = start.elapsed();
        println!(
            "  Multiple output checkpointing took: {:?}",
            multi_output_time
        );

        // Compare results across all methods
        println!("\nResults comparison:");
        println!("  Normal result shape: {:?}", grad1_val.shape());
        println!("  Manual checkpoint result shape: {:?}", grad2_val.shape());
        println!(
            "  Flexible checkpoint result shape: {:?}",
            grad3_val.shape()
        );
        println!(
            "  Group flex checkpoint result shape: {:?}",
            grad4_val.shape()
        );
        println!(
            "  Multi-output checkpoint result shape: {:?}",
            grad5_val.shape()
        );

        // Check if gradients match with the baseline
        let match_count_manual = grad1_val
            .iter()
            .zip(grad2_val.iter())
            .filter(|(a, b)| (*a - *b).abs() < 1e-5)
            .count();

        let match_count_flex = grad1_val
            .iter()
            .zip(grad3_val.iter())
            .filter(|(a, b)| (*a - *b).abs() < 1e-5)
            .count();

        let match_count_group_flex = grad1_val
            .iter()
            .zip(grad4_val.iter())
            .filter(|(a, b)| (*a - *b).abs() < 1e-5)
            .count();

        println!(
            "  Gradient elements that match (manual vs normal): {}/{}",
            match_count_manual,
            grad1_val.len()
        );

        println!(
            "  Gradient elements that match (flex vs normal): {}/{}",
            match_count_flex,
            grad1_val.len()
        );

        println!(
            "  Gradient elements that match (group flex vs normal): {}/{}",
            match_count_group_flex,
            grad1_val.len()
        );

        println!("\nPerformance comparison:");
        println!("  Normal execution: {:?}", normal_time);
        println!("  Manual checkpointing: {:?}", manual_ckpt_time);
        println!("  Flexible checkpointing: {:?}", flex_ckpt_time);
        println!("  Group flex checkpointing: {:?}", group_flex_time);
        println!("  Multi-output checkpointing: {:?}", multi_output_time);

        // Enable memory profiling
        T::CheckpointProfiler::reset_statistics();
        T::CheckpointProfiler::start_tracking();

        // Recompute using checkpointing to measure memory savings
        let _ = T::checkpoint_segment_flex(ctx, &[&a, &b], |inputs| {
            // Recreate the complex computation with explicit checkpoints
            let a = inputs[0];
            let b = inputs[1];

            let c = T::matmul(a, b);
            let c_ckpt = T::checkpoint(&c);
            let d = T::relu(&c_ckpt);
            let d_ckpt = T::checkpoint(&d);
            let e = T::matmul(&d_ckpt, b);
            let e_ckpt = T::checkpoint(&e);
            let f = T::tanh(&e_ckpt);
            let f_ckpt = T::checkpoint(&f);
            let g = T::matmul(&f_ckpt, b);
            T::sum_all(&g)
        });

        // Compute and report memory savings
        let memory_saved = T::CheckpointProfiler::memory_saved();
        let num_checkpoints = T::CheckpointProfiler::checkpoint_count();

        println!("\nMemory usage statistics:");
        println!("  Checkpoints used: {}", num_checkpoints);
        println!("  Memory saved: {} KB", memory_saved / 1024);

        T::CheckpointProfiler::stop_tracking();
    });
}
