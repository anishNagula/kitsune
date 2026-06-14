#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

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
