use ag::tensor_ops::*;
use scirs2_autograd as ag;

fn main() {
    let (kh, kw) = (2, 2);
    let (xch, ych) = (3, 2);
    let (yh, yw) = (2, 2);
    let batch_size = 2;

    println!("Testing conv2d_transpose:");
    println!("Kernel: {}x{}", kh, kw);
    println!("Input channels: {}", xch);
    println!("Output channels: {}", ych);
    println!("Input height/width: {}x{}", yh, yw);
    println!("Batch size: {}", batch_size);

    let ctx = ag::VariableEnvironment::<f32>::new();

    ctx.run(|graph| {
        // Create weight tensor: [out_channels, in_channels, kernel_h, kernel_w]
        let w = ones(&[ych, xch, kh, kw], graph);
        println!("\nWeight shape: {:?}", shape(w).eval(graph).unwrap());

        // Create input tensor: [batch, channels, height, width]
        let g = ones(&[batch_size, ych, yh, yw], graph);
        println!("Input shape: {:?}", shape(g).eval(graph).unwrap());

        // Perform conv2d_transpose
        println!("\nPerforming conv2d_transpose with pad=0, stride=1");
        let out = conv2d_transpose(g, w, 0, 1);

        // Check shape
        match shape(out).eval(graph) {
            Ok(s) => println!("Output shape tensor: {:?}", s),
            Err(e) => println!("Shape eval error: {:?}", e),
        }

        // Try to evaluate
        match out.eval(graph) {
            Ok(out_val) => {
                println!("Output evaluated successfully!");
                println!("Output shape: {:?}", out_val.shape());
                println!("Expected shape: [2, 3, 3, 3]");
                println!(
                    "First few values: {:?}",
                    &out_val.as_slice().unwrap()[..10.min(out_val.len())]
                );
            }
            Err(e) => {
                println!("Evaluation error: {:?}", e);
            }
        }
    });
}
