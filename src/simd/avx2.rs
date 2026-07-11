#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn apply_feature_delta(accumulator: &mut [i16], weights: &[i16], sign: i32) {
    unsafe {
        let len = accumulator.len();
        let mut idx = 0_usize;
        let acc_ptr = accumulator.as_mut_ptr();
        let weight_ptr = weights.as_ptr();

        while idx + 16 <= len {
            let weights = _mm256_loadu_si256(weight_ptr.add(idx) as *const __m256i);
            let acc = _mm256_loadu_si256(acc_ptr.add(idx) as *const __m256i);
            let updated = if sign > 0 {
                _mm256_add_epi16(acc, weights)
            } else if sign < 0 {
                _mm256_sub_epi16(acc, weights)
            } else {
                acc
            };
            _mm256_storeu_si256(acc_ptr.add(idx) as *mut __m256i, updated);
            idx += 16;
        }

        while idx < len {
            if sign > 0 {
                *acc_ptr.add(idx) += *weight_ptr.add(idx);
            } else if sign < 0 {
                *acc_ptr.add(idx) -= *weight_ptr.add(idx);
            }
            idx += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
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

        while idx + 16 <= len {
            let mut delta = _mm256_setzero_si256();
            for (&feature, &sign) in features.iter().zip(signs.iter()) {
                let weight_ptr = weights_ptr.add(feature * hidden_size + idx);
                let weights = _mm256_loadu_si256(weight_ptr as *const __m256i);
                if sign > 0 {
                    delta = _mm256_add_epi16(delta, weights);
                } else if sign < 0 {
                    delta = _mm256_sub_epi16(delta, weights);
                }
            }

            let acc = _mm256_loadu_si256(acc_ptr.add(idx) as *const __m256i);
            _mm256_storeu_si256(
                acc_ptr.add(idx) as *mut __m256i,
                _mm256_add_epi16(acc, delta),
            );
            idx += 16;
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
#[target_feature(enable = "avx2")]
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

        while idx + 16 <= len {
            let acc = _mm256_loadu_si256(source_ptr.add(idx) as *const __m256i);
            let removed = _mm256_sub_epi16(
                acc,
                _mm256_loadu_si256(remove_ptr.add(idx) as *const __m256i),
            );
            let updated = _mm256_add_epi16(
                removed,
                _mm256_loadu_si256(add_ptr.add(idx) as *const __m256i),
            );
            _mm256_storeu_si256(target_ptr.add(idx) as *mut __m256i, updated);
            idx += 16;
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
#[target_feature(enable = "avx2")]
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

        while idx + 16 <= len {
            let acc = _mm256_loadu_si256(source_ptr.add(idx) as *const __m256i);
            let removed_first = _mm256_sub_epi16(
                acc,
                _mm256_loadu_si256(remove_first_ptr.add(idx) as *const __m256i),
            );
            let removed_second = _mm256_sub_epi16(
                removed_first,
                _mm256_loadu_si256(remove_second_ptr.add(idx) as *const __m256i),
            );
            let updated = _mm256_add_epi16(
                removed_second,
                _mm256_loadu_si256(add_ptr.add(idx) as *const __m256i),
            );
            _mm256_storeu_si256(target_ptr.add(idx) as *mut __m256i, updated);
            idx += 16;
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
#[target_feature(enable = "avx2")]
pub(super) unsafe fn screlu_dot_i16_dual(
    left_accumulator: &[i16],
    left_weights: &[i16],
    right_accumulator: &[i16],
    right_weights: &[i16],
    qa: i16,
    narrow_weights: bool,
) -> i64 {
    unsafe {
        if qa <= 255 && narrow_weights {
            return screlu_dot_i16_narrow_dual(
                left_accumulator,
                left_weights,
                right_accumulator,
                right_weights,
                qa,
            );
        }
        if qa <= 255 {
            return screlu_dot_i16_u8(left_accumulator, left_weights, qa)
                + screlu_dot_i16_u8(right_accumulator, right_weights, qa);
        }
        screlu_dot_i16_wide(left_accumulator, left_weights, qa)
            + screlu_dot_i16_wide(right_accumulator, right_weights, qa)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn screlu_dot_i16_narrow_dual(
    left_accumulator: &[i16],
    left_weights: &[i16],
    right_accumulator: &[i16],
    right_weights: &[i16],
    qa: i16,
) -> i64 {
    unsafe {
        let zero = _mm256_setzero_si256();
        let qa_vec = _mm256_set1_epi16(qa);
        let mut sum = _mm256_setzero_si256();
        let mut tail = 0_i64;

        let mut idx = 0_usize;
        while idx + 16 <= left_accumulator.len() {
            let acc = _mm256_loadu_si256(left_accumulator.as_ptr().add(idx) as *const __m256i);
            let clamped = _mm256_min_epi16(_mm256_max_epi16(acc, zero), qa_vec);
            let weights = _mm256_loadu_si256(left_weights.as_ptr().add(idx) as *const __m256i);
            let weighted = _mm256_mullo_epi16(clamped, weights);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(weighted, clamped));
            idx += 16;
        }
        while idx < left_accumulator.len() {
            let value = i64::from(*left_accumulator.get_unchecked(idx)).clamp(0, i64::from(qa));
            tail += value * value * i64::from(*left_weights.get_unchecked(idx));
            idx += 1;
        }

        idx = 0;
        while idx + 16 <= right_accumulator.len() {
            let acc = _mm256_loadu_si256(right_accumulator.as_ptr().add(idx) as *const __m256i);
            let clamped = _mm256_min_epi16(_mm256_max_epi16(acc, zero), qa_vec);
            let weights = _mm256_loadu_si256(right_weights.as_ptr().add(idx) as *const __m256i);
            let weighted = _mm256_mullo_epi16(clamped, weights);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(weighted, clamped));
            idx += 16;
        }
        while idx < right_accumulator.len() {
            let value = i64::from(*right_accumulator.get_unchecked(idx)).clamp(0, i64::from(qa));
            tail += value * value * i64::from(*right_weights.get_unchecked(idx));
            idx += 1;
        }

        let low = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum));
        let high = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum, 1));
        tail + horizontal_sum_i64(_mm256_add_epi64(low, high))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn screlu_dot_i16_u8(accumulator: &[i16], weights: &[i16], qa: i16) -> i64 {
    unsafe {
        let len = accumulator.len();
        let mut idx = 0_usize;
        let acc_ptr = accumulator.as_ptr();
        let weight_ptr = weights.as_ptr();
        let zero = _mm256_setzero_si256();
        let qa_vec = _mm256_set1_epi16(qa);
        let correction_threshold = _mm256_set1_epi16(181);
        let ones = _mm256_set1_epi16(1);
        let mut sum = _mm256_setzero_si256();

        while idx + 16 <= len {
            let acc = _mm256_loadu_si256(acc_ptr.add(idx) as *const __m256i);
            let clamped = _mm256_min_epi16(_mm256_max_epi16(acc, zero), qa_vec);
            let w = _mm256_loadu_si256(weight_ptr.add(idx) as *const __m256i);
            let square = _mm256_mullo_epi16(clamped, clamped);
            let base_pairs = _mm256_madd_epi16(square, w);
            let correction_mask = _mm256_cmpgt_epi16(clamped, correction_threshold);
            let correction_weights = _mm256_and_si256(w, correction_mask);
            let correction_pairs = _mm256_madd_epi16(correction_weights, ones);

            let base_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(base_pairs));
            let base_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(base_pairs, 1));
            let correction_lo = _mm256_slli_epi64::<16>(_mm256_cvtepi32_epi64(
                _mm256_castsi256_si128(correction_pairs),
            ));
            let correction_hi = _mm256_slli_epi64::<16>(_mm256_cvtepi32_epi64(
                _mm256_extracti128_si256(correction_pairs, 1),
            ));
            sum = _mm256_add_epi64(sum, _mm256_add_epi64(base_lo, correction_lo));
            sum = _mm256_add_epi64(sum, _mm256_add_epi64(base_hi, correction_hi));

            idx += 16;
        }

        let mut result = horizontal_sum_i64(sum);
        let qa = i64::from(qa);
        while idx < len {
            let clamped = i64::from(*acc_ptr.add(idx)).clamp(0, qa);
            result += clamped * clamped * i64::from(*weight_ptr.add(idx));
            idx += 1;
        }
        result
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn screlu_dot_i16_wide(accumulator: &[i16], weights: &[i16], qa: i16) -> i64 {
    unsafe {
        let len = accumulator.len();
        let mut idx = 0_usize;
        let acc_ptr = accumulator.as_ptr();
        let weight_ptr = weights.as_ptr();
        let zero = _mm256_setzero_si256();
        let qa_vec = _mm256_set1_epi16(qa);
        let mut sum = _mm256_setzero_si256();

        while idx + 16 <= len {
            let acc = _mm256_loadu_si256(acc_ptr.add(idx) as *const __m256i);
            let clamped = _mm256_min_epi16(_mm256_max_epi16(acc, zero), qa_vec);
            let w = _mm256_loadu_si256(weight_ptr.add(idx) as *const __m256i);

            let v_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(clamped));
            let w_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w));
            let q_lo = _mm256_mullo_epi32(_mm256_mullo_epi32(v_lo, w_lo), v_lo);
            sum = _mm256_add_epi64(sum, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(q_lo)));
            sum = _mm256_add_epi64(sum, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(q_lo, 1)));

            let v_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(clamped, 1));
            let w_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w, 1));
            let q_hi = _mm256_mullo_epi32(_mm256_mullo_epi32(v_hi, w_hi), v_hi);
            sum = _mm256_add_epi64(sum, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(q_hi)));
            sum = _mm256_add_epi64(sum, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(q_hi, 1)));

            idx += 16;
        }

        let mut result = horizontal_sum_i64(sum);
        let qa = i64::from(qa);
        while idx < len {
            let clamped = i64::from(*acc_ptr.add(idx)).clamp(0, qa);
            result += clamped * clamped * i64::from(*weight_ptr.add(idx));
            idx += 1;
        }
        result
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_i64(value: __m256i) -> i64 {
    unsafe {
        let mut lanes = [0_i64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, value);
        lanes.into_iter().sum()
    }
}
