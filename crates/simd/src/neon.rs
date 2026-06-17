#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
use crate::scalar::transpose;
use crate::types::MatrixView;

#[cfg(target_arch = "aarch64")]
pub fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    unsafe {
        let mut acc = vdupq_n_f32(0.0);

        let chunks = a.len() / 4;

        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));

            acc = vfmaq_f32(acc, va, vb);
        }

        let mut result = vaddvq_f32(acc);

        for i in (chunks * 4)..a.len() {
            result += a[i] * b[i];
        }

        result
    }
}

#[cfg(target_arch = "aarch64")]
pub fn matmul_neon(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let bt = transpose(b, k, n);

    let a_view = MatrixView {
        data: a,
        rows: m,
        cols: k,
    };

    let bt_view = MatrixView {
        data: &bt,
        rows: n,
        cols: k,
    };

    let mut result = vec![0.0; m * n];

    for i in 0..m {
        let a_row = a_view.row(i);

        for j in 0..n {
            let b_row = bt_view.row(j);

            result[i * n + j] =
                dot_product_neon(a_row, b_row);
        }
    }

    result
}
