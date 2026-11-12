mod accumulator;
mod feature;
#[cfg_attr(
    not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon"),
    )),
    path = "inference_scalar.rs"
)]
mod inference;
#[allow(clippy::module_inception)]
mod nnue;

pub(crate) use accumulator::Accumulator;
use feature::{FeatureUpdates, PieceFeature};
use inference::feed_forward;
pub(crate) use nnue::Nnue;

use crate::{Square, chess::File};

pub fn backend_name() -> &'static str {
    inference::BACKEND_NAME
}

static NET: Network =
    unsafe { std::mem::transmute(*include_bytes!(concat!(env!("OUT_DIR"), "/goliath.nnue"))) };

const Q0: i16 = 255;
const Q: i32 = 64;
const EVAL_SCALE: i64 = 400;

const INPUT: usize = 779;
const L1: usize = 1536;
const L2: usize = 16;
const L3: usize = 32;

fn scale_output(output: i32, skip: i32) -> i32 {
    let skip = i64::from(skip) * i64::from(Q.pow(3) / 256);
    let scaled = (i64::from(output) + skip) * EVAL_SCALE;
    (scaled / i64::from(Q.pow(4))) as i32
}

#[inline]
pub fn should_mirror(sq: Square) -> bool {
    sq.file() as u8 > File::D as u8
}

#[repr(C, align(64))]
struct Network {
    l0w: [[i16; L1]; INPUT],
    l0b: [i16; L1],
    l1w: [[i8; L2 * 4]; L1 / 4],
    l1b: [i32; L2],
    l2w: [[i32; L3]; (L2 - 1) * 2],
    l2b: [i32; L3],
    l3w: [i32; L3],
    l3b: i32,
}

const _: () = assert!(size_of::<Network>() == 2_424_960);
