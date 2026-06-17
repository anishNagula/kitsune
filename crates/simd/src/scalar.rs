pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

pub fn matmul_scalar(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut result = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;

            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }

            result[i * n + j] = sum;
        }
    }

    result
}

pub fn transpose(
    matrix: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut result = vec![0.0; rows * cols];

    for r in 0..rows {
        for c in 0..cols {
            result[c * rows + r] =
                matrix[r * cols + c];
        }
    }

    result
}
