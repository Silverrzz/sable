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
pub(super) unsafe fn dot_product_i32(left: &[i32], right: &[i32]) -> i64 {
    unsafe {
        let len = left.len();
        let mut idx = 0_usize;
        let mut sum_even = _mm512_setzero_si512();
        let mut sum_odd = _mm512_setzero_si512();
        let left_ptr = left.as_ptr();
        let right_ptr = right.as_ptr();

        while idx + 16 <= len {
            let a = _mm512_loadu_si512(left_ptr.add(idx) as *const __m512i);
            let b = _mm512_loadu_si512(right_ptr.add(idx) as *const __m512i);
            sum_even = _mm512_add_epi64(sum_even, _mm512_mul_epi32(a, b));
            sum_odd = _mm512_add_epi64(
                sum_odd,
                _mm512_mul_epi32(_mm512_srli_epi64::<32>(a), _mm512_srli_epi64::<32>(b)),
            );
            idx += 16;
        }

        let mut sum = _mm512_reduce_add_epi64(sum_even) + _mm512_reduce_add_epi64(sum_odd);
        while idx < len {
            sum += i64::from(*left_ptr.add(idx)) * i64::from(*right_ptr.add(idx));
            idx += 1;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx2")]
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
            let mut sum_even_0 = _mm512_setzero_si512();
            let mut sum_odd_0 = _mm512_setzero_si512();
            let mut sum_even_1 = _mm512_setzero_si512();
            let mut sum_odd_1 = _mm512_setzero_si512();
            let mut sum_even_2 = _mm512_setzero_si512();
            let mut sum_odd_2 = _mm512_setzero_si512();
            let mut sum_even_3 = _mm512_setzero_si512();
            let mut sum_odd_3 = _mm512_setzero_si512();
            let input_ptr = input.as_ptr();
            let w0 = weights.as_ptr().add(row * cols);
            let w1 = weights.as_ptr().add((row + 1) * cols);
            let w2 = weights.as_ptr().add((row + 2) * cols);
            let w3 = weights.as_ptr().add((row + 3) * cols);
            let mut idx = 0_usize;

            while idx + 16 <= cols {
                let x = _mm512_loadu_si512(input_ptr.add(idx) as *const __m512i);
                let x_odd = _mm512_srli_epi64::<32>(x);

                let a0 = _mm512_loadu_si512(w0.add(idx) as *const __m512i);
                sum_even_0 = _mm512_add_epi64(sum_even_0, _mm512_mul_epi32(a0, x));
                sum_odd_0 = _mm512_add_epi64(
                    sum_odd_0,
                    _mm512_mul_epi32(_mm512_srli_epi64::<32>(a0), x_odd),
                );

                let a1 = _mm512_loadu_si512(w1.add(idx) as *const __m512i);
                sum_even_1 = _mm512_add_epi64(sum_even_1, _mm512_mul_epi32(a1, x));
                sum_odd_1 = _mm512_add_epi64(
                    sum_odd_1,
                    _mm512_mul_epi32(_mm512_srli_epi64::<32>(a1), x_odd),
                );

                let a2 = _mm512_loadu_si512(w2.add(idx) as *const __m512i);
                sum_even_2 = _mm512_add_epi64(sum_even_2, _mm512_mul_epi32(a2, x));
                sum_odd_2 = _mm512_add_epi64(
                    sum_odd_2,
                    _mm512_mul_epi32(_mm512_srli_epi64::<32>(a2), x_odd),
                );

                let a3 = _mm512_loadu_si512(w3.add(idx) as *const __m512i);
                sum_even_3 = _mm512_add_epi64(sum_even_3, _mm512_mul_epi32(a3, x));
                sum_odd_3 = _mm512_add_epi64(
                    sum_odd_3,
                    _mm512_mul_epi32(_mm512_srli_epi64::<32>(a3), x_odd),
                );

                idx += 16;
            }

            let mut out0 =
                _mm512_reduce_add_epi64(sum_even_0) + _mm512_reduce_add_epi64(sum_odd_0);
            let mut out1 =
                _mm512_reduce_add_epi64(sum_even_1) + _mm512_reduce_add_epi64(sum_odd_1);
            let mut out2 =
                _mm512_reduce_add_epi64(sum_even_2) + _mm512_reduce_add_epi64(sum_odd_2);
            let mut out3 =
                _mm512_reduce_add_epi64(sum_even_3) + _mm512_reduce_add_epi64(sum_odd_3);
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
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx2")]
pub(super) unsafe fn activate_accumulator_i16(
    input: &[i16],
    acc_mul: i32,
    acc_shift: u32,
    use_screlu: bool,
    output: &mut [i32],
) {
    unsafe {
        let len = input.len();
        let mut idx = 0_usize;
        let input_ptr = input.as_ptr();
        let output_ptr = output.as_mut_ptr();
        let acc_mul_scalar = acc_mul;
        let acc_shift_scalar = acc_shift;
        let acc_mul = _mm512_set1_epi64(i64::from(acc_mul_scalar));
        let acc_shift = _mm512_set1_epi64(i64::from(acc_shift_scalar));
        let zero = _mm512_setzero_si512();
        let activation_q = _mm512_set1_epi64(1024);

        while idx + 8 <= len {
            let acc16 = _mm_loadu_si128(input_ptr.add(idx) as *const __m128i);
            let acc64 = _mm512_cvtepi16_epi64(acc16);
            let scaled = _mm512_srav_epi64(_mm512_mullo_epi64(acc64, acc_mul), acc_shift);
            let clamped = _mm512_min_epi64(_mm512_max_epi64(scaled, zero), activation_q);
            let activated = if use_screlu {
                _mm512_srai_epi64::<10>(_mm512_mullo_epi64(clamped, clamped))
            } else {
                clamped
            };
            _mm256_storeu_si256(
                output_ptr.add(idx) as *mut __m256i,
                _mm512_cvtepi64_epi32(activated),
            );
            idx += 8;
        }

        while idx < len {
            let scaled = ((i64::from(*input_ptr.add(idx)) * i64::from(acc_mul_scalar))
                >> acc_shift_scalar) as i32;
            let clamped = scaled.clamp(0, 1024);
            *output_ptr.add(idx) = if use_screlu {
                ((i64::from(clamped) * i64::from(clamped)) / 1024) as i32
            } else {
                clamped
            };
            idx += 1;
        }
    }
}
