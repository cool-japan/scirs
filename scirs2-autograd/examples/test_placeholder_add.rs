use ag::ndarray::array;
use scirs2_autograd as ag;

fn main() {
    ag::run(|ctx| {
        // Test the exact code from the doc test
        let x1 = ctx.placeholder("x", &[-1, 2]);
        let x2 = ctx.placeholder("y", &[-1, 2]);
        let x3 = ctx.placeholder("x", &[-1, 2]);

        println!("x1 id: {}", x1.id());
        println!("x2 id: {}", x2.id());
        println!("x3 id: {}", x3.id());

        let sum = x1 + x2 + x3;

        let arr = &array![[1., 1.]].into_dyn();

        let result = ctx.evaluator()
            .push(&sum)
            .feed("x", arr.view()) // feed for x1 and x3
            .feed("y", arr.view()) // feed for x2
            .feed(x2, arr.view()) // same as .feed("y", ...)
            .run();

        println!("Result: {:?}", result[0]);
        println!("Expected: {:?}", arr + arr + arr);

        // Debug: evaluate each component
        let r1 = ctx.evaluator().push(&x1).feed("x", arr.view()).run();
        println!("x1 result: {:?}", r1[0]);

        let r2 = ctx.evaluator().push(&x2).feed("y", arr.view()).run();
        println!("x2 result: {:?}", r2[0]);

        let r3 = ctx.evaluator().push(&x3).feed("x", arr.view()).run();
        println!("x3 result: {:?}", r3[0]);

        // Test simple addition
        let simple_sum = x1 + x2;
        let rs = ctx
            .evaluator()
            .push(&simple_sum)
            .feed("x", arr.view())
            .feed("y", arr.view())
            .run();
        println!("x1 + x2 result: {:?}", rs[0]);
    });
}
