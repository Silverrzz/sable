#![allow(clippy::needless_range_loop)]

use std::mem::MaybeUninit;

use super::{L1, L2, L3, NET, Q, Q0, scale_output};

#[cfg(target_arch = "aarch64")]
#[path = "simd/neon.rs"]
mod simd;
#[cfg(target_arch = "x86_64")]
#[cfg_attr(
    all(target_feature = "avx512bw", target_feature = "avx512vl", target_feature = "avx512vbmi2"),
    path = "simd/avx512.rs"
)]
#[cfg_attr(
    not(all(target_feature = "avx512bw", target_feature = "avx512vl", target_feature = "avx512vbmi2")),
    path = "simd/avx2.rs"
)]
mod simd;

pub(super) const BACKEND_NAME: &str = simd::BACKEND_NAME;

#[inline(always)]
fn activate_ft(
    us: &[i16; L1],
    them: &[i16; L1],
    hmc: &[i16; L1],
) -> [i8; L1] {
    let mut out = MaybeUninit::<[i8; L1]>::uninit();
    let output = out.as_mut_ptr().cast::<i8>();

    const { assert!((L1 / 2).is_multiple_of(2 * simd::i16s::LANES)) };

    unsafe {
        use simd::i16s::{LANES, add, load, max, min, mulhi_shl7, packus, splat};

        let load_values = |values: &[i16; L1], i| {
            add(load(values.as_ptr().add(i)), load(hmc.as_ptr().add(i)))
        };
        for i in (0..L1 / 2).step_by(2 * simd::i16s::LANES) {
            let mut us1 = load_values(us, i);
            let mut us2 = load_values(us, i + L1 / 2);
            let mut us3 = load_values(us, i + LANES);
            let mut us4 = load_values(us, i + L1 / 2 + LANES);

            let mut them1 = load_values(them, i);
            let mut them2 = load_values(them, i + L1 / 2);
            let mut them3 = load_values(them, i + LANES);
            let mut them4 = load_values(them, i + L1 / 2 + LANES);

            us2 = max(us2, splat(0));
            us4 = max(us4, splat(0));

            them2 = max(them2, splat(0));
            them4 = max(them4, splat(0));

            us1 = min(us1, splat(Q0));
            us2 = min(us2, splat(Q0));
            us3 = min(us3, splat(Q0));
            us4 = min(us4, splat(Q0));

            them1 = min(them1, splat(Q0));
            them2 = min(them2, splat(Q0));
            them3 = min(them3, splat(Q0));
            them4 = min(them4, splat(Q0));

            let us_pair1 = mulhi_shl7(us1, us2);
            let us_pair2 = mulhi_shl7(us3, us4);

            let them_pair1 = mulhi_shl7(them1, them2);
            let them_pair2 = mulhi_shl7(them3, them4);

            let p1 = packus(us_pair1, us_pair2);
            let p2 = packus(them_pair1, them_pair2);

            simd::i8s::store(output.add(i), p1);
            simd::i8s::store(output.add(i + L1 / 2), p2);
        }

        out.assume_init()
    }
}

#[inline(always)]
fn propagate_l1(act_ft: &[i8; L1]) -> ([i32; L2 * 2], i32) {
    const UNROLL: usize = 4;
    const { assert!(L2.is_multiple_of(simd::i32s::LANES)) };
    const { assert!((L1 / 4).is_multiple_of(UNROLL)) };

    unsafe {
        let mut nnz_idxs = [const { MaybeUninit::<i16>::uninit() }; L1 / 4 + simd::i16s::LANES];
        let mut nnz_count = 0;
        {
            let mut base = simd::i16s::splat(0);
            for i in (0..L1).step_by(simd::i8s::LANES) {
                let chunk = simd::i8s::reinterpret_i32(simd::i8s::load(act_ft.as_ptr().add(i)));
                let (idxs, cnt) = simd::i32s::nnz_indices(chunk);
                simd::i16s::store(
                    nnz_idxs.as_mut_ptr().add(nnz_count).cast(),
                    simd::i16s::add(base, idxs),
                );
                nnz_count += cnt as usize;
                base = simd::i16s::add(base, simd::i16s::splat(simd::i32s::LANES as i16));
            }
        }

        let mut intermediate = [[simd::i32s::splat(0); UNROLL]; { L2 / simd::i32s::LANES }];
        let ft_32 = act_ft.as_chunks::<4>().0;

        let mut i_outer = 0;

        while i_outer + UNROLL <= nnz_count {
            for j in 0..(L2 / simd::i32s::LANES) {
                for i_inner in 0..UNROLL {
                    let i = nnz_idxs[i_outer + i_inner].assume_init() as usize;
                    let ft_vec = simd::i32s::splat(i32::from_ne_bytes(ft_32[i].map(|byte| byte as u8)));
                    intermediate[j][i_inner] = simd::i8s::dpbusd(
                        intermediate[j][i_inner],
                        simd::i32s::reinterpret_i8(ft_vec),
                        simd::i8s::load(NET.l1w[i].as_ptr().add(j * simd::i8s::LANES)),
                    );
                }
            }

            i_outer += UNROLL;
        }

        while i_outer < nnz_count {
            let i = nnz_idxs[i_outer].assume_init() as usize;
            let ft_vec = simd::i32s::splat(i32::from_ne_bytes(ft_32[i].map(|byte| byte as u8)));
            for j in 0..(L2 / simd::i32s::LANES) {
                intermediate[j][0] = simd::i8s::dpbusd(
                    intermediate[j][0],
                    simd::i32s::reinterpret_i8(ft_vec),
                    simd::i8s::load(NET.l1w[i].as_ptr().add(j * simd::i8s::LANES)),
                )
            }
            i_outer += 1;
        }

        let mut out = [0; L2 * 2];
        let mut skip = 0;
        for i in 0..L2 / simd::i32s::LANES {
            use simd::i32s::*;

            let bias = load(NET.l1b.as_ptr().add(i * LANES));

            let mut sum = splat(0);
            for partial_sum in intermediate[i] {
                sum = add(sum, partial_sum);
            }

            let raw = add(bias, sum);
            if i + 1 == L2 / LANES {
                skip = extract_last(raw);
            }
            let shifted = shr_const::<2>(raw);
            let clipped = min(max(shifted, splat(0)), splat(Q * Q));
            let squared = shr_const::<12>(min(mul(shifted, shifted), splat(Q.pow(4))));
            store(out.as_mut_ptr().add(i * LANES), clipped);
            store(out.as_mut_ptr().add(i * LANES + L2), squared);
        }
        (out, skip)
    }
}

fn propagate_l2(act_l1: &[i32; L2 * 2]) -> [i32; L3] {
    use simd::{I32Vec, i32s::*};
    unsafe {
        let mut sums: [I32Vec; L3 / LANES] =
            std::array::from_fn(|i| load(NET.l2b.as_ptr().add(i * LANES)));

        for (i, &activation) in act_l1
            .as_chunks::<L2>()
            .0
            .iter()
            .flat_map(|chunk| &chunk[..L2 - 1])
            .enumerate()
        {
            let r = splat(activation);
            for j in 0..L3 / LANES {
                let l = load(NET.l2w[i].as_ptr().add(j * LANES));
                sums[j] = add(sums[j], mul(l, r));
            }
        }

        let mut out = [0i32; L3];
        for (i, sum) in sums.iter().enumerate() {
            let clamped = min(max(*sum, splat(0)), splat(Q.pow(3)));
            store(out.as_mut_ptr().add(i * LANES), clamped);
        }

        out
    }
}

fn propagate_l3(act_l2: &[i32; L3]) -> i32 {
    use simd::i32s::*;
    unsafe {
        let mut sum = splat(0);
        for i in (0..L3).step_by(LANES) {
            let l = load(act_l2.as_ptr().add(i));
            let r = load(NET.l3w.as_ptr().add(i));
            sum = add(sum, mul(l, r));
        }

        reduce_add(sum).wrapping_add(NET.l3b)
    }
}

pub fn feed_forward(
    us: &[i16; L1],
    them: &[i16; L1],
    hmc: &[i16; L1],
) -> i32 {
    let act_ft = activate_ft(us, them, hmc);
    let (act_l1, skip) = propagate_l1(&act_ft);
    let act_l2 = propagate_l2(&act_l1);
    scale_output(propagate_l3(&act_l2), skip)
}
