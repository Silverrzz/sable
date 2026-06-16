#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn apply_feature_delta(accumulator: &mut [i16], weights: &[i16], sign: i32) {
    if sign == 0 {
        return;
    }
    unsafe {
        let len = accumulator.len();
        let mut idx = 0_usize;
        let acc_ptr = accumulator.as_mut_ptr();
        let weight_ptr = weights.as_ptr();

        while idx + 8 <= len {
            let w = vld1q_s16(weight_ptr.add(idx));
            let acc = vld1q_s16(acc_ptr.add(idx));
            let new_acc = if sign > 0 {
                vaddq_s16(acc, w)
            } else {
                vsubq_s16(acc, w)
            };
            vst1q_s16(acc_ptr.add(idx), new_acc);
            idx += 8;
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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn screlu_dot_i16(accumulator: &[i16], weights: &[i16], qa: i16) -> i64 {
    unsafe {
        let len = accumulator.len();
        let mut idx = 0_usize;
        let acc_ptr = accumulator.as_ptr();
        let weight_ptr = weights.as_ptr();
        let zero = vdupq_n_s16(0);
        let qa_vec = vdupq_n_s16(qa);
        let mut sum = vdupq_n_s64(0);

        while idx + 8 <= len {
            let acc = vld1q_s16(acc_ptr.add(idx));
            let clamped = vminq_s16(vmaxq_s16(acc, zero), qa_vec);
            let w = vld1q_s16(weight_ptr.add(idx));

            let p_lo = vmull_s16(vget_low_s16(clamped), vget_low_s16(w));
            let q_lo = vmulq_s32(p_lo, vmovl_s16(vget_low_s16(clamped)));
            sum = vpadalq_s32(sum, q_lo);

            let p_hi = vmull_high_s16(clamped, w);
            let q_hi = vmulq_s32(p_hi, vmovl_high_s16(clamped));
            sum = vpadalq_s32(sum, q_hi);

            idx += 8;
        }

        let mut result = vaddvq_s64(sum);
        let qa = i64::from(qa);
        while idx < len {
            let clamped = i64::from(*acc_ptr.add(idx)).clamp(0, qa);
            result += clamped * clamped * i64::from(*weight_ptr.add(idx));
            idx += 1;
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn dot_product_i32(left: &[i32], right: &[i32]) -> i64 {
    unsafe {
        let len = left.len();
        let mut idx = 0_usize;
        let left_ptr = left.as_ptr();
        let right_ptr = right.as_ptr();
        let mut sum = vdupq_n_s64(0);

        while idx + 4 <= len {
            let a = vld1q_s32(left_ptr.add(idx));
            let b = vld1q_s32(right_ptr.add(idx));
            sum = vmlal_s32(sum, vget_low_s32(a), vget_low_s32(b));
            sum = vmlal_high_s32(sum, a, b);
            idx += 4;
        }

        let mut result = vaddvq_s64(sum);
        while idx < len {
            result += i64::from(*left_ptr.add(idx)) * i64::from(*right_ptr.add(idx));
            idx += 1;
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
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
            let mut sum0 = vdupq_n_s64(0);
            let mut sum1 = vdupq_n_s64(0);
            let mut sum2 = vdupq_n_s64(0);
            let mut sum3 = vdupq_n_s64(0);
            let input_ptr = input.as_ptr();
            let w0 = weights.as_ptr().add(row * cols);
            let w1 = weights.as_ptr().add((row + 1) * cols);
            let w2 = weights.as_ptr().add((row + 2) * cols);
            let w3 = weights.as_ptr().add((row + 3) * cols);
            let mut idx = 0_usize;

            while idx + 4 <= cols {
                let x = vld1q_s32(input_ptr.add(idx));

                let a0 = vld1q_s32(w0.add(idx));
                sum0 = vmlal_s32(sum0, vget_low_s32(a0), vget_low_s32(x));
                sum0 = vmlal_high_s32(sum0, a0, x);

                let a1 = vld1q_s32(w1.add(idx));
                sum1 = vmlal_s32(sum1, vget_low_s32(a1), vget_low_s32(x));
                sum1 = vmlal_high_s32(sum1, a1, x);

                let a2 = vld1q_s32(w2.add(idx));
                sum2 = vmlal_s32(sum2, vget_low_s32(a2), vget_low_s32(x));
                sum2 = vmlal_high_s32(sum2, a2, x);

                let a3 = vld1q_s32(w3.add(idx));
                sum3 = vmlal_s32(sum3, vget_low_s32(a3), vget_low_s32(x));
                sum3 = vmlal_high_s32(sum3, a3, x);

                idx += 4;
            }

            let mut out0 = vaddvq_s64(sum0);
            let mut out1 = vaddvq_s64(sum1);
            let mut out2 = vaddvq_s64(sum2);
            let mut out3 = vaddvq_s64(sum3);
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
