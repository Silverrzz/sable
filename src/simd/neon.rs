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
pub(super) unsafe fn screlu_dot_f32(accumulator: &[i16], weights: &[f32], qa: i16) -> f32 {
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let limit = vdupq_n_f32(f32::from(qa));
        let mut sum = vdupq_n_f32(0.0);
        let mut idx = 0;
        while idx + 4 <= accumulator.len() {
            let values = vcvtq_f32_s32(vmovl_s16(vld1_s16(accumulator.as_ptr().add(idx))));
            let clamped = vminq_f32(vmaxq_f32(values, zero), limit);
            let weight = vld1q_f32(weights.as_ptr().add(idx));
            sum = vaddq_f32(sum, vmulq_f32(vmulq_f32(clamped, clamped), weight));
            idx += 4;
        }
        let mut result = vaddvq_f32(sum);
        while idx < accumulator.len() {
            let clamped = f32::from(accumulator[idx].clamp(0, qa));
            result += clamped * clamped * weights[idx];
            idx += 1;
        }
        result / (f32::from(qa) * f32::from(qa))
    }
}