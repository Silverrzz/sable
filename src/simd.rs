mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum SimdBackend {
    Scalar,
    Avx2,
    Neon,
}

static BACKEND: OnceLock<SimdBackend> = OnceLock::new();

fn backend() -> SimdBackend {
    *BACKEND.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                return SimdBackend::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return SimdBackend::Neon;
        }
        SimdBackend::Scalar
    })
}

pub fn apply_feature_delta(accumulator: &mut [i16], weights: &[i16], sign: i32) {
    debug_assert_eq!(accumulator.len(), weights.len());
    match backend() {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => unsafe { avx2::apply_feature_delta(accumulator, weights, sign) },
        #[cfg(not(target_arch = "x86_64"))]
        SimdBackend::Avx2 => unreachable!("AVX2 backend is only selected on x86_64"),
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => unsafe { neon::apply_feature_delta(accumulator, weights, sign) },
        #[cfg(not(target_arch = "aarch64"))]
        SimdBackend::Neon => unreachable!("NEON backend is only selected on aarch64"),
        SimdBackend::Scalar => scalar::apply_feature_delta(accumulator, weights, sign),
    }
}

pub fn dot_product_i32(left: &[i32], right: &[i32]) -> i64 {
    debug_assert_eq!(left.len(), right.len());
    match backend() {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => unsafe { avx2::dot_product_i32(left, right) },
        #[cfg(not(target_arch = "x86_64"))]
        SimdBackend::Avx2 => unreachable!("AVX2 backend is only selected on x86_64"),
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => unsafe { neon::dot_product_i32(left, right) },
        #[cfg(not(target_arch = "aarch64"))]
        SimdBackend::Neon => unreachable!("NEON backend is only selected on aarch64"),
        SimdBackend::Scalar => scalar::dot_product_i32(left, right),
    }
}

pub fn screlu_dot_i16(accumulator: &[i16], weights: &[i16], qa: i16) -> i64 {
    debug_assert_eq!(accumulator.len(), weights.len());
    match backend() {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => unsafe { avx2::screlu_dot_i16(accumulator, weights, qa) },
        #[cfg(not(target_arch = "x86_64"))]
        SimdBackend::Avx2 => unreachable!("AVX2 backend is only selected on x86_64"),
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => unsafe { neon::screlu_dot_i16(accumulator, weights, qa) },
        #[cfg(not(target_arch = "aarch64"))]
        SimdBackend::Neon => unreachable!("NEON backend is only selected on aarch64"),
        SimdBackend::Scalar => scalar::screlu_dot_i16(accumulator, weights, qa),
    }
}

pub fn matrix_vector_i32(
    weights: &[i32],
    input: &[i32],
    rows: usize,
    cols: usize,
    output: &mut [i64],
) {
    debug_assert_eq!(input.len(), cols);
    debug_assert!(output.len() >= rows);
    debug_assert!(weights.len() >= rows.saturating_mul(cols));
    match backend() {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => unsafe { avx2::matrix_vector_i32(weights, input, rows, cols, output) },
        #[cfg(not(target_arch = "x86_64"))]
        SimdBackend::Avx2 => unreachable!("AVX2 backend is only selected on x86_64"),
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => unsafe { neon::matrix_vector_i32(weights, input, rows, cols, output) },
        #[cfg(not(target_arch = "aarch64"))]
        SimdBackend::Neon => unreachable!("NEON backend is only selected on aarch64"),
        SimdBackend::Scalar => scalar::matrix_vector_i32(weights, input, rows, cols, output),
    }
}
