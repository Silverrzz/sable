use crate::{Color, Piece, Square};

pub(super) const WIN_SCORE: i32 = 30_000;
pub(crate) const LOSS_SCORE: i32 = -WIN_SCORE;
pub(crate) const DRAW_SCORE: i32 = 0;

pub(super) const PAWN_VALUE: i32 = 100;
pub(super) const KNIGHT_VALUE: i32 = 320;
pub(super) const BISHOP_VALUE: i32 = 330;
pub(super) const ROOK_VALUE: i32 = 500;
pub(super) const QUEEN_VALUE: i32 = 900;

pub(super) const PIECE_SQUARE_FEATURES: usize = 768;
pub(super) const KING_SQUARES: usize = 64;
pub(super) const SHARD_KING_BUCKETS: usize = 16;
pub(super) const SHARD_HIDDEN: usize = 768;
pub const SHARD_OUTPUT_BUCKETS: usize = 8;
pub(super) const SHARD_OUTPUTS_PER_BUCKET: usize = SHARD_HIDDEN * 2;
pub(super) const SHARD_OUTPUT_WEIGHTS: usize = SHARD_OUTPUTS_PER_BUCKET * SHARD_OUTPUT_BUCKETS;
pub(super) const SHARD_INPUT_FEATURES: usize = PIECE_SQUARE_FEATURES * SHARD_KING_BUCKETS;
pub(super) const SHARD_FEATURE_WEIGHTS: usize = SHARD_INPUT_FEATURES * SHARD_HIDDEN;
pub(super) const SHARD_HEADER_BYTES: usize = 180;
pub(super) const SHARD_TENSOR_BYTES: usize = SHARD_HEADER_BYTES
    + (SHARD_FEATURE_WEIGHTS + SHARD_HIDDEN) * 2
    + (SHARD_OUTPUT_WEIGHTS + SHARD_OUTPUT_BUCKETS) * 4;
pub(super) const SHARD_FILE_MAX_BYTES: usize = SHARD_TENSOR_BYTES + 63;
pub(super) const SHARD_QA: i16 = 255;
pub(super) const SHARD_QB: i16 = 64;
pub(super) const SHARD_OUTPUT_SCALE: i32 = 400;
pub(super) const MAX_MOVE_FEATURE_UPDATES: usize = 3;
pub(super) const FINNY_TABLE_ENTRIES: usize = KING_SQUARES * 2;
pub(super) const FINNY_PIECE_BITBOARDS: usize = 12;
pub(super) const SHARD_BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 11, 8, 9, 10,
    11, 12, 13, 14, 15, 12, 13, 14, 15,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NnueArchitectureId {
    Shard,
}

impl NnueArchitectureId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Shard => "shard",
        }
    }
}

#[derive(Debug)]
pub struct NnueModel {
    pub(super) feature_weights: Box<[AlignedFeatureBlock]>,
    pub(super) bias: [i16; SHARD_HIDDEN],
    pub(super) output_weights: AlignedOutputWeights,
    pub(super) narrow_output_weights: bool,
    pub(super) output_bias: [i32; SHARD_OUTPUT_BUCKETS],
}

#[repr(C, align(64))]
#[derive(Debug)]
pub(super) struct AlignedFeatureBlock(pub(super) [i16; SHARD_HIDDEN]);

const _: () = assert!(std::mem::size_of::<AlignedFeatureBlock>() == SHARD_HIDDEN * 2);

#[repr(align(64))]
#[derive(Debug)]
pub(super) struct AlignedOutputWeights(pub(super) [i16; SHARD_OUTPUT_WEIGHTS]);

#[repr(align(64))]
#[derive(Clone, Debug)]
pub struct NnueAccumulators {
    pub(super) white: [i16; SHARD_HIDDEN],
    pub(super) black: [i16; SHARD_HIDDEN],
}

impl NnueAccumulators {
    pub(crate) fn empty_like(source: &Self) -> Self {
        let _ = source;
        Self {
            white: [0; SHARD_HIDDEN],
            black: [0; SHARD_HIDDEN],
        }
    }
}

#[derive(Debug)]
pub(crate) struct NnueFinnyTable {
    entries: Vec<NnueFinnyEntry>,
}

#[derive(Clone, Debug)]
pub(super) struct NnueFinnyEntry {
    pub(super) values: [i16; SHARD_HIDDEN],
    pub(super) pieces: [u64; FINNY_PIECE_BITBOARDS],
    pub(super) valid: bool,
}

impl NnueFinnyTable {
    pub(super) fn new() -> Self {
        Self {
            entries: (0..FINNY_TABLE_ENTRIES)
                .map(|_| NnueFinnyEntry {
                    values: [0; SHARD_HIDDEN],
                    pieces: [0; FINNY_PIECE_BITBOARDS],
                    valid: false,
                })
                .collect(),
        }
    }

    pub(super) fn entry_mut(
        &mut self,
        perspective: Color,
        king_square: usize,
    ) -> Option<&mut NnueFinnyEntry> {
        let index = perspective as usize * KING_SQUARES + king_square;
        self.entries.get_mut(index)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PieceContribution {
    pub square: Square,
    pub piece: Piece,
    pub color: Color,
    /// nnue derived value from white
    pub score_white_cp: i32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FeatureUpdate {
    pub(super) feature: usize,
    pub(super) sign: i32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FeatureUpdateList {
    pub(super) updates: [FeatureUpdate; MAX_MOVE_FEATURE_UPDATES],
    pub(super) len: usize,
}

impl FeatureUpdateList {
    pub(super) fn new() -> Self {
        Self {
            updates: [FeatureUpdate {
                feature: 0,
                sign: 0,
            }; MAX_MOVE_FEATURE_UPDATES],
            len: 0,
        }
    }

    pub(super) fn push(&mut self, update: FeatureUpdate) -> Option<()> {
        if self.len == self.updates.len() {
            return None;
        }
        self.updates[self.len] = update;
        self.len += 1;
        Some(())
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = FeatureUpdate> + '_ {
        self.updates[..self.len].iter().copied()
    }
}
