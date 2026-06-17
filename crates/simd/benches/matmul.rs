use criterion::{
    criterion_group,
    criterion_main,
    Criterion,
};

use simd::scalar::matmul_scalar;

#[cfg(target_arch = "aarch64")]
use simd::neon::matmul_neon;

fn bench_matmul(c: &mut Criterion) {
    let a = vec![1.0; 128 * 128];
    let b = vec![1.0; 128 * 128];

    c.bench_function(
        "matmul_scalar_128",
        |bench| {
            bench.iter(|| {
                matmul_scalar(
                    &a,
                    &b,
                    128,
                    128,
                    128,
                )
            })
        },
    );

    #[cfg(target_arch = "aarch64")]
    c.bench_function(
        "matmul_neon_128",
        |bench| {
            bench.iter(|| {
                matmul_neon(
                    &a,
                    &b,
                    128,
                    128,
                    128,
                )
            })
        },
    );
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
