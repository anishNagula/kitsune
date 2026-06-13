use criterion::{criterion_group, criterion_main, Criterion};
use tensor::Tensor;

fn bench_matmul(c: &mut Criterion) {
    let a = Tensor::ones(vec![128, 128]);
    let b = Tensor::ones(vec![128, 128]);

    c.bench_function("matmul_128x128", |bencher| {
        bencher.iter(|| {
            let _ = a.matmul(&b).unwrap();
        });
    });
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
