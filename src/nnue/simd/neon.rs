#![allow(clippy::missing_safety_doc)]

use std::arch::aarch64::*;

pub(super) const BACKEND_NAME: &str = "neon";

pub type I8Vec = int8x16_t;
pub type I16Vec = int16x8_t;
pub type I32Vec = int32x4_t;

#[path = "nnz_table.rs"]
mod nnz_table;

pub mod i8s {
    use super::*;

    pub const LANES: usize = size_of::<I8Vec>() / size_of::<i8>();

    pub use std::arch::aarch64::{vld1q_s8 as load, vst1q_s8 as store};

    #[target_feature(enable = "neon")]
    pub fn dpbusd(acc: I32Vec, l: I8Vec, r: I8Vec) -> I32Vec {
        #[cfg(target_feature = "dotprod")]
        unsafe {
            let mut acc = acc;
            std::arch::asm!(
                "sdot {acc:v}.4s, {l:v}.16b, {r:v}.16b",
                acc = inlateout(vreg) acc,
                l = in(vreg) l,
                r = in(vreg) r,
                options(pure, nostack, nomem, preserves_flags)
            );
            acc
        }
        #[cfg(not(target_feature = "dotprod"))]
        {
            let lo = vmull_s8(vget_low_s8(l), vget_low_s8(r));
            let hi = vmull_high_s8(l, r);
            let p = vpaddq_s16(lo, hi);
            vpadalq_s16(acc, p)
        }
    }

    #[target_feature(enable = "neon")]
    pub fn reinterpret_i32(v: I8Vec) -> I32Vec {
        vreinterpretq_s32_s8(v)
    }
}

pub mod i16s {
    use super::*;

    pub const LANES: usize = size_of::<I16Vec>() / size_of::<i16>();

    pub use std::arch::aarch64::{
        vdupq_n_s16 as splat, vld1q_s16 as load, vmaxq_s16 as max, vminq_s16 as min,
        vst1q_s16 as store,
    };

    #[target_feature(enable = "neon")]
    pub fn add(l: I16Vec, r: I16Vec) -> I16Vec {
        vaddq_s16(l, r)
    }

    #[target_feature(enable = "neon")]
    pub fn mulhi_shl7(l: I16Vec, r: I16Vec) -> I16Vec {
        vqdmulhq_s16(l, vshlq_n_s16(r, 6))
    }

    #[target_feature(enable = "neon")]
    pub fn packus(l: I16Vec, r: I16Vec) -> I8Vec {
        vreinterpretq_s8_u8(vqmovun_high_s16(vqmovun_s16(l), r))
    }
}

pub mod i32s {
    #[target_feature(enable = "neon")]
    pub fn extract_last(v: I32Vec) -> i32 {
        vgetq_lane_s32::<3>(v)
    }

    use super::*;

    pub const LANES: usize = size_of::<I32Vec>() / size_of::<i32>();

    pub use std::arch::aarch64::{
        vaddq_s32 as add, vaddvq_s32 as reduce_add, vdupq_n_s32 as splat, vld1q_s32 as load,
        vmaxq_s32 as max, vminq_s32 as min, vmulq_s32 as mul,
        vreinterpretq_s8_s32 as reinterpret_i8, vshrq_n_s32 as shr_const, vst1q_s32 as store,
    };

    #[target_feature(enable = "neon")]
    pub fn nnz_indices(v: I32Vec) -> (I16Vec, u16) {
        let mask = vtstq_s32(v, v);
        let bitmask = vaddvq_u32(vandq_u32(mask, unsafe { vld1q_u32([1, 2, 4, 8].as_ptr()) }));
        let idxs = unsafe { vld1q_s16(nnz_table::NNZ_TABLE[bitmask as usize].as_ptr()) };
        (idxs, bitmask.count_ones() as u16)
    }
}
