use simd::scalar::dot_product_scalar;

#[test]
fn dot_product() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];

    assert_eq!(
        dot_product_scalar(&a, &b),
        32.0
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_matches_scalar() {
    use simd::neon::dot_product_neon;
    use simd::scalar::dot_product_scalar;

    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];

    let scalar = dot_product_scalar(&a, &b);
    let neon = dot_product_neon(&a, &b);

    assert_eq!(scalar, neon);
}

use simd::scalar::matmul_scalar;

#[test]
fn matmul_scalar_2x2() {
    let a = vec![
        1.0, 2.0,
        3.0, 4.0,
    ];

    let b = vec![
        5.0, 6.0,
        7.0, 8.0,
    ];

    let result = matmul_scalar(
        &a,
        &b,
        2,
        2,
        2,
    );

    assert_eq!(
        result,
        vec![
            19.0, 22.0,
            43.0, 50.0,
        ]
    );
}

use simd::scalar::transpose;

#[test]
fn transpose_matrix() {
    let matrix = vec![
        1.0, 2.0,
        3.0, 4.0,
    ];

    let t = transpose(&matrix, 2, 2);

    assert_eq!(
        t,
        vec![
            1.0, 3.0,
            2.0, 4.0,
        ]
    );
}
