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
pub(super) const VEX_KING_BUCKETS: usize = 16;
pub(super) const VEX_HIDDEN: usize = 256;
pub(super) const VEX_OUTPUTS: usize = VEX_HIDDEN * 2;
pub(super) const VEX_INPUT_FEATURES: usize = PIECE_SQUARE_FEATURES * VEX_KING_BUCKETS;
pub(super) const VEX_FEATURE_WEIGHTS: usize = VEX_INPUT_FEATURES * VEX_HIDDEN;
pub(super) const VEX_TENSOR_VALUES: usize =
    VEX_FEATURE_WEIGHTS + VEX_HIDDEN + VEX_OUTPUTS + 1;
pub(super) const VEX_TENSOR_BYTES: usize = VEX_TENSOR_VALUES * 2;
pub(super) const VEX_FILE_MAX_BYTES: usize = VEX_TENSOR_BYTES + 63;
pub(super) const VEX_QA: i16 = 255;
pub(super) const VEX_QB: i16 = 64;
pub(super) const VEX_OUTPUT_SCALE: i32 = 400;
pub(super) const MAX_MOVE_FEATURE_UPDATES: usize = 3;
pub(super) const FINNY_TABLE_ENTRIES: usize = KING_SQUARES * 2;
pub(super) const FINNY_PIECE_BITBOARDS: usize = 12;
pub(super) const VEX_BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 11, 8, 9, 10,
    11, 12, 13, 14, 15, 12, 13, 14, 15,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalMode {
    Hce,
    Nnue,
}

impl EvalMode {
    pub fn from_uci(value: &str) -> Option<Self> {
        let mut key = value.to_ascii_lowercase();
        key.retain(|ch| ch != ' ' && ch != '-');
        match key.as_str() {
            "hce" | "handcrafted" | "classical" | "material" => Some(Self::Hce),
            "nnue" => Some(Self::Nnue),
            _ => None,
        }
    }

    pub fn as_uci(self) -> &'static str {
        match self {
            Self::Hce => "hce",
            Self::Nnue => "nnue",
        }
    }
}

impl Default for EvalMode {
    fn default() -> Self {
        Self::Hce
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NnueArchitectureId {
    Vex,
}

impl NnueArchitectureId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vex => "vex",
        }
    }
}

#[derive(Debug)]
pub struct NnueModel {
    pub(super) feature_weights: Box<[i16]>,
    pub(super) bias: [i16; VEX_HIDDEN],
    pub(super) output_weights: [i16; VEX_OUTPUTS],
    pub(super) output_bias: i32,
}

#[derive(Clone, Debug)]
pub struct NnueAccumulators {
    pub(super) white: [i16; VEX_HIDDEN],
    pub(super) black: [i16; VEX_HIDDEN],
}

impl NnueAccumulators {
    pub(crate) fn empty_like(source: &Self) -> Self {
        let _ = source;
        Self {
            white: [0; VEX_HIDDEN],
            black: [0; VEX_HIDDEN],
        }
    }
}

#[derive(Debug)]
pub(crate) struct NnueFinnyTable {
    entries: Vec<NnueFinnyEntry>,
}

#[derive(Clone, Debug)]
pub(super) struct NnueFinnyEntry {
    pub(super) values: [i16; VEX_HIDDEN],
    pub(super) pieces: [u64; FINNY_PIECE_BITBOARDS],
    pub(super) valid: bool,
}

impl NnueFinnyTable {
    pub(super) fn new() -> Self {
        Self {
            entries: (0..FINNY_TABLE_ENTRIES)
                .map(|_| NnueFinnyEntry {
                    values: [0; VEX_HIDDEN],
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
