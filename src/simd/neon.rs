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

