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
