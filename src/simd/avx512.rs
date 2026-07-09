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
pub(super) unsafe fn screlu_dot_i16_dual(
    left_accumulator: &[i16],
    left_weights: &[i16],
    right_accumulator: &[i16],
    right_weights: &[i16],
    qa: i16,
) -> i64 {
    unsafe {
        if qa <= 255 {
            return screlu_dot_i16_u8(left_accumulator, left_weights, qa)
                + screlu_dot_i16_u8(right_accumulator, right_weights, qa);
        }
        screlu_dot_i16_wide(left_accumulator, left_weights, qa)
            + screlu_dot_i16_wide(right_accumulator, right_weights, qa)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx2")]
unsafe fn screlu_dot_i16_u8(accumulator: &[i16], weights: &[i16], qa: i16) -> i64 {
    unsafe {
        let len = accumulator.len();
        let mut idx = 0_usize;
        let acc_ptr = accumulator.as_ptr();
        let weight_ptr = weights.as_ptr();
        let zero = _mm512_setzero_si512();
        let qa_vec = _mm512_set1_epi16(qa);
        let correction_threshold = _mm512_set1_epi16(181);
        let ones = _mm512_set1_epi16(1);
        let mut sum_lo = _mm512_setzero_si512();
        let mut sum_hi = _mm512_setzero_si512();

        while idx + 32 <= len {
            let acc = _mm512_loadu_si512(acc_ptr.add(idx) as *const __m512i);
            let clamped = _mm512_min_epi16(_mm512_max_epi16(acc, zero), qa_vec);
            let w = _mm512_loadu_si512(weight_ptr.add(idx) as *const __m512i);
            let square = _mm512_mullo_epi16(clamped, clamped);
            let base_pairs = _mm512_madd_epi16(square, w);
            let correction_mask = _mm512_cmpgt_epi16_mask(clamped, correction_threshold);
            let correction_weights = _mm512_maskz_mov_epi16(correction_mask, w);
            let correction_pairs = _mm512_madd_epi16(correction_weights, ones);

            let base_pairs_lo = _mm512_castsi512_si256(base_pairs);
            let base_pairs_hi = _mm512_extracti64x4_epi64::<1>(base_pairs);
            let correction_pairs_lo = _mm512_castsi512_si256(correction_pairs);
            let correction_pairs_hi = _mm512_extracti64x4_epi64::<1>(correction_pairs);
            let base_lo = _mm512_cvtepi32_epi64(base_pairs_lo);
            let base_hi = _mm512_cvtepi32_epi64(base_pairs_hi);
            let correction_lo =
                _mm512_slli_epi64::<16>(_mm512_cvtepi32_epi64(correction_pairs_lo));
            let correction_hi =
                _mm512_slli_epi64::<16>(_mm512_cvtepi32_epi64(correction_pairs_hi));
            sum_lo = _mm512_add_epi64(sum_lo, _mm512_add_epi64(base_lo, correction_lo));
            sum_hi = _mm512_add_epi64(sum_hi, _mm512_add_epi64(base_hi, correction_hi));

            idx += 32;
        }

        let mut result = _mm512_reduce_add_epi64(sum_lo) + _mm512_reduce_add_epi64(sum_hi);
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
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx2")]
unsafe fn screlu_dot_i16_wide(accumulator: &[i16], weights: &[i16], qa: i16) -> i64 {
    unsafe {
        let len = accumulator.len();
        let mut idx = 0_usize;
        let acc_ptr = accumulator.as_ptr();
        let weight_ptr = weights.as_ptr();
        let zero = _mm512_setzero_si512();
        let qa_vec = _mm512_set1_epi16(qa);
        let mut sum_lo = _mm512_setzero_si512();
        let mut sum_hi = _mm512_setzero_si512();

        while idx + 32 <= len {
            let acc = _mm512_loadu_si512(acc_ptr.add(idx) as *const __m512i);
            let clamped = _mm512_min_epi16(_mm512_max_epi16(acc, zero), qa_vec);
            let w = _mm512_loadu_si512(weight_ptr.add(idx) as *const __m512i);

            let v0 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(clamped));
            let w0 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(w));
            let q0 = _mm512_mullo_epi32(_mm512_mullo_epi32(v0, w0), v0);
            sum_lo = _mm512_add_epi64(
                sum_lo,
                _mm512_cvtepi32_epi64(_mm512_castsi512_si256(q0)),
            );
            sum_hi = _mm512_add_epi64(
                sum_hi,
                _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(q0)),
            );

            let v1 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(clamped));
            let w1 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(w));
            let q1 = _mm512_mullo_epi32(_mm512_mullo_epi32(v1, w1), v1);
            sum_lo = _mm512_add_epi64(
                sum_lo,
                _mm512_cvtepi32_epi64(_mm512_castsi512_si256(q1)),
            );
            sum_hi = _mm512_add_epi64(
                sum_hi,
                _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(q1)),
            );

            idx += 32;
        }

        let mut result = _mm512_reduce_add_epi64(sum_lo) + _mm512_reduce_add_epi64(sum_hi);
        let qa = i64::from(qa);
        while idx < len {
            let clamped = i64::from(*acc_ptr.add(idx)).clamp(0, qa);
            result += clamped * clamped * i64::from(*weight_ptr.add(idx));
            idx += 1;
        }
        result
    }
}
