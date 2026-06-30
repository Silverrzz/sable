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
pub(super) unsafe fn screlu_dot_i16(accumulator: &[i16], weights: &[i16], qa: i16) -> i64 {
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
pub(super) unsafe fn dot_product_i32(left: &[i32], right: &[i32]) -> i64 {
    unsafe {
        let len = left.len();
        let mut idx = 0_usize;
        let mut sum_even = _mm256_setzero_si256();
        let mut sum_odd = _mm256_setzero_si256();
        let left_ptr = left.as_ptr();
        let right_ptr = right.as_ptr();

        while idx + 8 <= len {
            let a = _mm256_loadu_si256(left_ptr.add(idx) as *const __m256i);
            let b = _mm256_loadu_si256(right_ptr.add(idx) as *const __m256i);
            sum_even = _mm256_add_epi64(sum_even, _mm256_mul_epi32(a, b));
            let a_odd = _mm256_srli_epi64(a, 32);
            let b_odd = _mm256_srli_epi64(b, 32);
            sum_odd = _mm256_add_epi64(sum_odd, _mm256_mul_epi32(a_odd, b_odd));
            idx += 8;
        }

        let mut sum = horizontal_sum_i64(sum_even) + horizontal_sum_i64(sum_odd);
        while idx < len {
            sum += i64::from(*left_ptr.add(idx)) * i64::from(*right_ptr.add(idx));
            idx += 1;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn matrix_vector_i32(
    weights: &[i32],
    input: &[i32],
    rows: usize,
    cols: usize,
    output: &mut [i64],
) {
    unsafe {
        let mut row = 0_usize;
        while row + 4 <= rows {
            let mut sum_even_0 = _mm256_setzero_si256();
            let mut sum_odd_0 = _mm256_setzero_si256();
            let mut sum_even_1 = _mm256_setzero_si256();
            let mut sum_odd_1 = _mm256_setzero_si256();
            let mut sum_even_2 = _mm256_setzero_si256();
            let mut sum_odd_2 = _mm256_setzero_si256();
            let mut sum_even_3 = _mm256_setzero_si256();
            let mut sum_odd_3 = _mm256_setzero_si256();
            let input_ptr = input.as_ptr();
            let w0 = weights.as_ptr().add(row * cols);
            let w1 = weights.as_ptr().add((row + 1) * cols);
            let w2 = weights.as_ptr().add((row + 2) * cols);
            let w3 = weights.as_ptr().add((row + 3) * cols);
            let mut idx = 0_usize;

            while idx + 8 <= cols {
                let x = _mm256_loadu_si256(input_ptr.add(idx) as *const __m256i);
                let x_odd = _mm256_srli_epi64(x, 32);

                let a0 = _mm256_loadu_si256(w0.add(idx) as *const __m256i);
                sum_even_0 = _mm256_add_epi64(sum_even_0, _mm256_mul_epi32(a0, x));
                sum_odd_0 =
                    _mm256_add_epi64(sum_odd_0, _mm256_mul_epi32(_mm256_srli_epi64(a0, 32), x_odd));

                let a1 = _mm256_loadu_si256(w1.add(idx) as *const __m256i);
                sum_even_1 = _mm256_add_epi64(sum_even_1, _mm256_mul_epi32(a1, x));
                sum_odd_1 =
                    _mm256_add_epi64(sum_odd_1, _mm256_mul_epi32(_mm256_srli_epi64(a1, 32), x_odd));

                let a2 = _mm256_loadu_si256(w2.add(idx) as *const __m256i);
                sum_even_2 = _mm256_add_epi64(sum_even_2, _mm256_mul_epi32(a2, x));
                sum_odd_2 =
                    _mm256_add_epi64(sum_odd_2, _mm256_mul_epi32(_mm256_srli_epi64(a2, 32), x_odd));

                let a3 = _mm256_loadu_si256(w3.add(idx) as *const __m256i);
                sum_even_3 = _mm256_add_epi64(sum_even_3, _mm256_mul_epi32(a3, x));
                sum_odd_3 =
                    _mm256_add_epi64(sum_odd_3, _mm256_mul_epi32(_mm256_srli_epi64(a3, 32), x_odd));

                idx += 8;
            }

            let mut out0 = horizontal_sum_i64(sum_even_0) + horizontal_sum_i64(sum_odd_0);
            let mut out1 = horizontal_sum_i64(sum_even_1) + horizontal_sum_i64(sum_odd_1);
            let mut out2 = horizontal_sum_i64(sum_even_2) + horizontal_sum_i64(sum_odd_2);
            let mut out3 = horizontal_sum_i64(sum_even_3) + horizontal_sum_i64(sum_odd_3);
            while idx < cols {
                let x = i64::from(*input_ptr.add(idx));
                out0 += i64::from(*w0.add(idx)) * x;
                out1 += i64::from(*w1.add(idx)) * x;
                out2 += i64::from(*w2.add(idx)) * x;
                out3 += i64::from(*w3.add(idx)) * x;
                idx += 1;
            }

            output[row] = out0;
            output[row + 1] = out1;
            output[row + 2] = out2;
            output[row + 3] = out3;
            row += 4;
        }

        while row < rows {
            let start = row * cols;
            output[row] = dot_product_i32(&weights[start..(start + cols)], input);
            row += 1;
        }
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
