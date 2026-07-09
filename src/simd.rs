mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum SimdBackend {
    Scalar,
    Avx512,
    Avx2,
    Neon,
}

static BACKEND: OnceLock<SimdBackend> = OnceLock::new();

fn backend() -> SimdBackend {
    *BACKEND.get_or_init(|| backend_override().unwrap_or_else(detect_backend))
}

fn backend_override() -> Option<SimdBackend> {
    let requested = std::env::var("SABLE_SIMD_BACKEND")
        .or_else(|_| std::env::var("SABLE_SIMD"))
        .ok()?;
    let requested = requested.trim().to_ascii_lowercase();
    let candidate = match requested.as_str() {
        "" | "auto" | "native" => return None,
        "scalar" | "none" | "off" => SimdBackend::Scalar,
        "avx512" | "avx-512" => SimdBackend::Avx512,
        "avx2" | "avx-2" => SimdBackend::Avx2,
        "neon" => SimdBackend::Neon,
        _ => return None,
    };
    backend_supported(candidate).then_some(candidate)
}

fn detect_backend() -> SimdBackend {
    if backend_supported(SimdBackend::Avx512) {
        return SimdBackend::Avx512;
    }
    if backend_supported(SimdBackend::Avx2) {
        return SimdBackend::Avx2;
    }
    if backend_supported(SimdBackend::Neon) {
        return SimdBackend::Neon;
    }
    SimdBackend::Scalar
}

fn backend_supported(candidate: SimdBackend) -> bool {
    match candidate {
        SimdBackend::Scalar => true,
        SimdBackend::Avx512 => avx512_supported(),
        SimdBackend::Avx2 => avx2_supported(),
        SimdBackend::Neon => neon_supported(),
    }
}

#[cfg(target_arch = "x86_64")]
fn avx512_supported() -> bool {
    std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn avx512_supported() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn avx2_supported() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_supported() -> bool {
    false
}

#[cfg(target_arch = "aarch64")]
fn neon_supported() -> bool {
    true
}

#[cfg(not(target_arch = "aarch64"))]
fn neon_supported() -> bool {
    false
}

pub fn runtime_backend_name() -> &'static str {
    match backend() {
        SimdBackend::Scalar => "scalar",
        SimdBackend::Avx512 => "avx512",
        SimdBackend::Avx2 => "avx2",
        SimdBackend::Neon => "neon",
    }
}

pub fn apply_feature_delta(accumulator: &mut [i16], weights: &[i16], sign: i32) {
    debug_assert_eq!(accumulator.len(), weights.len());
    match backend() {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx512 => unsafe { avx512::apply_feature_delta(accumulator, weights, sign) },
        #[cfg(not(target_arch = "x86_64"))]
        SimdBackend::Avx512 => unreachable!("AVX-512 backend is only selected on x86_64"),
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

pub fn apply_feature_deltas(
    accumulator: &mut [i16],
    feature_weights: &[i16],
    hidden_size: usize,
    features: &[usize],
    signs: &[i32],
) {
    debug_assert_eq!(accumulator.len(), hidden_size);
    debug_assert_eq!(features.len(), signs.len());
    debug_assert!(features.iter().all(|feature| {
        feature_weights.len() >= feature.saturating_add(1).saturating_mul(hidden_size)
    }));
    match backend() {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx512 => unsafe {
            avx512::apply_feature_deltas(
                accumulator,
                feature_weights,
                hidden_size,
                features,
                signs,
            )
        },
        #[cfg(not(target_arch = "x86_64"))]
        SimdBackend::Avx512 => unreachable!("AVX-512 backend is only selected on x86_64"),
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => unsafe {
            avx2::apply_feature_deltas(accumulator, feature_weights, hidden_size, features, signs)
        },
        #[cfg(not(target_arch = "x86_64"))]
        SimdBackend::Avx2 => unreachable!("AVX2 backend is only selected on x86_64"),
        SimdBackend::Neon | SimdBackend::Scalar => {
            scalar::apply_feature_deltas(accumulator, feature_weights, hidden_size, features, signs)
        }
    }
}

pub fn screlu_dot_i16_dual(
    left_accumulator: &[i16],
    left_weights: &[i16],
    right_accumulator: &[i16],
    right_weights: &[i16],
    qa: i16,
) -> i64 {
    debug_assert_eq!(left_accumulator.len(), left_weights.len());
    debug_assert_eq!(right_accumulator.len(), right_weights.len());
    match backend() {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx512 => unsafe {
            avx512::screlu_dot_i16_dual(
                left_accumulator,
                left_weights,
                right_accumulator,
                right_weights,
                qa,
            )
        },
        #[cfg(not(target_arch = "x86_64"))]
        SimdBackend::Avx512 => unreachable!("AVX-512 backend is only selected on x86_64"),
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => unsafe {
            avx2::screlu_dot_i16_dual(
                left_accumulator,
                left_weights,
                right_accumulator,
                right_weights,
                qa,
            )
        },
        #[cfg(not(target_arch = "x86_64"))]
        SimdBackend::Avx2 => unreachable!("AVX2 backend is only selected on x86_64"),
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => unsafe {
            neon::screlu_dot_i16(left_accumulator, left_weights, qa)
                + neon::screlu_dot_i16(right_accumulator, right_weights, qa)
        },
        #[cfg(not(target_arch = "aarch64"))]
        SimdBackend::Neon => unreachable!("NEON backend is only selected on aarch64"),
        SimdBackend::Scalar => scalar::screlu_dot_i16_dual(
            left_accumulator,
            left_weights,
            right_accumulator,
            right_weights,
            qa,
        ),
    }
}
