#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx2")]
pub(super) unsafe fn apply_feature_delta(accumulator: &mut [i16], weights: &[i16], sign: i32) {
    if sign == 0 {
        return;
    }
    unsafe {
        let len = accumulator.len();
        let mut idx = 0_usize;
        let acc_ptr = accumulator.as_mut_ptr();
        let weight_ptr = weights.as_ptr();

        while idx + 32 <= len {
            let weights = _mm512_loadu_si512(weight_ptr.add(idx) as *const __m512i);
            let acc = _mm512_loadu_si512(acc_ptr.add(idx) as *const __m512i);
            let updated = if sign > 0 {
                _mm512_add_epi16(acc, weights)
            } else {
                _mm512_sub_epi16(acc, weights)
            };
            _mm512_storeu_si512(acc_ptr.add(idx) as *mut __m512i, updated);
            idx += 32;
        }

        while idx < len {
            if sign > 0 {
                *acc_ptr.add(idx) += *weight_ptr.add(idx);
            } else {
                *acc_ptr.add(idx) -= *weight_ptr.add(idx);
            }
            idx += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx2")]
pub(super) unsafe fn apply_feature_deltas(
    accumulator: &mut [i16],
    feature_weights: &[i16],
    hidden_size: usize,
    features: &[usize],
    signs: &[i32],
) {
    unsafe {
        let len = accumulator.len();
        let mut idx = 0_usize;
        let acc_ptr = accumulator.as_mut_ptr();
        let weights_ptr = feature_weights.as_ptr();

        while idx + 32 <= len {
            let mut delta = _mm512_setzero_si512();
            for (&feature, &sign) in features.iter().zip(signs.iter()) {
                let weight_ptr = weights_ptr.add(feature * hidden_size + idx);
                let weights = _mm512_loadu_si512(weight_ptr as *const __m512i);
                if sign > 0 {
                    delta = _mm512_add_epi16(delta, weights);
                } else if sign < 0 {
                    delta = _mm512_sub_epi16(delta, weights);
                }
            }

            let acc = _mm512_loadu_si512(acc_ptr.add(idx) as *const __m512i);
            _mm512_storeu_si512(
                acc_ptr.add(idx) as *mut __m512i,
                _mm512_add_epi16(acc, delta),
            );
            idx += 32;
        }

        while idx < len {
            let mut value = i32::from(*acc_ptr.add(idx));
            for (&feature, &sign) in features.iter().zip(signs.iter()) {
                let weight = i32::from(*weights_ptr.add(feature * hidden_size + idx));
                if sign > 0 {
                    value += weight;
                } else if sign < 0 {
                    value -= weight;
                }
            }
            *acc_ptr.add(idx) = value as i16;
            idx += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx2")]
pub(super) unsafe fn copy_feature_delta_pair(
    source: &[i16],
    target: &mut [i16],
    feature_weights: &[i16],
    hidden_size: usize,
    remove: usize,
    add: usize,
) {
    unsafe {
        let len = source.len();
        let mut idx = 0_usize;
        let source_ptr = source.as_ptr();
        let target_ptr = target.as_mut_ptr();
        let remove_ptr = feature_weights.as_ptr().add(remove * hidden_size);
        let add_ptr = feature_weights.as_ptr().add(add * hidden_size);

        while idx + 32 <= len {
            let acc = _mm512_loadu_si512(source_ptr.add(idx) as *const __m512i);
            let removed = _mm512_sub_epi16(
                acc,
                _mm512_loadu_si512(remove_ptr.add(idx) as *const __m512i),
            );
            let updated = _mm512_add_epi16(
                removed,
                _mm512_loadu_si512(add_ptr.add(idx) as *const __m512i),
            );
            _mm512_storeu_si512(target_ptr.add(idx) as *mut __m512i, updated);
            idx += 32;
        }

        while idx < len {
            *target_ptr.add(idx) = (*source_ptr.add(idx))
                .wrapping_sub(*remove_ptr.add(idx))
                .wrapping_add(*add_ptr.add(idx));
            idx += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx2")]
pub(super) unsafe fn copy_feature_delta_triplet(
    source: &[i16],
    target: &mut [i16],
    feature_weights: &[i16],
    hidden_size: usize,
    remove_first: usize,
    remove_second: usize,
    add: usize,
) {
    unsafe {
        let len = source.len();
        let mut idx = 0_usize;
        let source_ptr = source.as_ptr();
        let target_ptr = target.as_mut_ptr();
        let remove_first_ptr = feature_weights.as_ptr().add(remove_first * hidden_size);
        let remove_second_ptr = feature_weights.as_ptr().add(remove_second * hidden_size);
        let add_ptr = feature_weights.as_ptr().add(add * hidden_size);

        while idx + 32 <= len {
            let acc = _mm512_loadu_si512(source_ptr.add(idx) as *const __m512i);
            let removed_first = _mm512_sub_epi16(
                acc,
                _mm512_loadu_si512(remove_first_ptr.add(idx) as *const __m512i),
            );
            let removed_second = _mm512_sub_epi16(
                removed_first,
                _mm512_loadu_si512(remove_second_ptr.add(idx) as *const __m512i),
            );
            let updated = _mm512_add_epi16(
                removed_second,
                _mm512_loadu_si512(add_ptr.add(idx) as *const __m512i),
            );
            _mm512_storeu_si512(target_ptr.add(idx) as *mut __m512i, updated);
            idx += 32;
        }

        while idx < len {
            *target_ptr.add(idx) = (*source_ptr.add(idx))
                .wrapping_sub(*remove_first_ptr.add(idx))
                .wrapping_sub(*remove_second_ptr.add(idx))
                .wrapping_add(*add_ptr.add(idx));
            idx += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx2")]
pub(super) unsafe fn screlu_dot_f32(accumulator: &[i16], weights: &[f32], qa: i16) -> f32 {
    unsafe {
        let zero = _mm512_setzero_ps();
        let limit = _mm512_set1_ps(f32::from(qa));
        let mut sum = _mm512_setzero_ps();
        let mut idx = 0;
        while idx + 16 <= accumulator.len() {
            let values = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_loadu_si256(accumulator.as_ptr().add(idx) as *const __m256i)));
            let clamped = _mm512_min_ps(_mm512_max_ps(values, zero), limit);
            let weight = _mm512_loadu_ps(weights.as_ptr().add(idx));
            sum = _mm512_add_ps(sum, _mm512_mul_ps(_mm512_mul_ps(clamped, clamped), weight));
            idx += 16;
        }
        let mut lanes = [0.0_f32; 16];
        _mm512_storeu_ps(lanes.as_mut_ptr(), sum);
        let mut result = lanes.into_iter().sum::<f32>();
        while idx < accumulator.len() {
            let clamped = f32::from(accumulator[idx].clamp(0, qa));
            result += clamped * clamped * weights[idx];
            idx += 1;
        }
        result / (f32::from(qa) * f32::from(qa))
    }
}
