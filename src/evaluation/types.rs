use crate::{Color, Piece, Square};

pub(super) const WIN_SCORE: i32 = 30_000;
pub(crate) const LOSS_SCORE: i32 = -WIN_SCORE;
pub(crate) const DRAW_SCORE: i32 = 0;

pub(super) const PAWN_VALUE: i32 = 100;
pub(super) const KNIGHT_VALUE: i32 = 320;
pub(super) const BISHOP_VALUE: i32 = 330;
pub(super) const ROOK_VALUE: i32 = 500;
pub(super) const QUEEN_VALUE: i32 = 900;

pub(super) const BULLET_QUANT_MAGIC: &[u8; 8] = b"SBRBLTQ1";
pub(super) const PIECE_SQUARE_FEATURES: usize = 768;
pub(super) const KING_BUCKETS: usize = 64;
pub(super) const KING_BUCKET_FEATURES: usize = PIECE_SQUARE_FEATURES * KING_BUCKETS;
pub(super) const VEX_KING_BUCKETS: usize = 16;
pub(super) const VEX_INPUT_FEATURES: usize = PIECE_SQUARE_FEATURES * VEX_KING_BUCKETS;
pub(super) const SIDE_TO_MOVE_FEATURE: usize = KING_BUCKET_FEATURES;
pub(super) const MAX_MOVE_FEATURE_UPDATES: usize = 6;
pub(super) const FINNY_TABLE_ENTRIES: usize = KING_BUCKETS * 2;
pub(super) const FINNY_PIECE_BITBOARDS: usize = 12;
pub(super) const BULLET_QUANT_HEADER_LEN: usize = 32;
pub(super) const BULLET_FLAG_HAS_SIDE_TO_MOVE: u32 = 1 << 0;
pub(super) const NATIVE_BULLET_QA: i16 = 255;
pub(super) const NATIVE_BULLET_QB: i16 = 64;
pub(super) const NATIVE_BULLET_OUTPUT_SCALE: i32 = 400;
pub(super) const VEX_BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3,
    0, 1, 2, 3,
    4, 5, 6, 7,
    4, 5, 6, 7,
    8, 9, 10, 11,
    8, 9, 10, 11,
    12, 13, 14, 15,
    12, 13, 14, 15,
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
    Nightweave,
    Vex,
}

impl NnueArchitectureId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nightweave => "nightweave",
            Self::Vex => "vex",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NnuePerspectiveMode {
    Single,
    DualConcat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NnueFeatureLayout {
    KingBuckets64,
    MirroredKingBuckets16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NnueArchitecture {
    pub(super) id: NnueArchitectureId,
    pub(super) feature_layout: NnueFeatureLayout,
    pub(super) perspective_mode: NnuePerspectiveMode,
    pub(super) input_features: usize,
}

impl NnueArchitecture {
    pub(super) const fn nightweave() -> Self {
        Self {
            id: NnueArchitectureId::Nightweave,
            feature_layout: NnueFeatureLayout::KingBuckets64,
            perspective_mode: NnuePerspectiveMode::Single,
            input_features: KING_BUCKET_FEATURES,
        }
    }

    pub(super) const fn vex() -> Self {
        Self {
            id: NnueArchitectureId::Vex,
            feature_layout: NnueFeatureLayout::MirroredKingBuckets16,
            perspective_mode: NnuePerspectiveMode::DualConcat,
            input_features: VEX_INPUT_FEATURES,
        }
    }

    pub(super) const fn bucket_count(self) -> usize {
        match self.feature_layout {
            NnueFeatureLayout::KingBuckets64 => KING_BUCKETS,
            NnueFeatureLayout::MirroredKingBuckets16 => VEX_KING_BUCKETS,
        }
    }

    pub(super) const fn output_input_size(self, hidden: usize) -> usize {
        match self.perspective_mode {
            NnuePerspectiveMode::Single => hidden,
            NnuePerspectiveMode::DualConcat => hidden * 2,
        }
    }

    pub(super) const fn side_to_move_feature_index(self) -> usize {
        self.input_features
    }
}

#[derive(Clone, Debug)]
pub struct NnueModel {
    pub(super) architecture: NnueArchitecture,
    pub(super) layers: Vec<QuantizedLayer>,
    pub(super) inference: NnueInference,
    pub(super) output_scale: i32,
    pub(super) has_side_to_move_feature: bool,
    pub(super) side_to_move_relative: bool,
    pub(super) first_layer_feature_weights: Vec<i16>,
}

#[derive(Clone, Debug)]
pub(super) struct QuantizedLayer {
    pub(super) input_size: usize,
    pub(super) weights: Vec<i16>,
    pub(super) bias: Vec<i64>,
    /// combined scale for turning bias back into float
    pub(super) scale: f32,
    /// weight scale straight from export
    pub(super) weight_scale: f32,
}

#[derive(Clone, Debug)]
pub struct NnueAccumulators {
    pub values: Vec<i16>,
    pub(super) black_values: Option<Vec<i16>>,
}

#[derive(Debug)]
pub(crate) struct NnueFinnyTable {
    entries: Vec<NnueFinnyEntry>,
}

#[derive(Clone, Debug)]
pub(super) struct NnueFinnyEntry {
    pub(super) values: Vec<i16>,
    pub(super) pieces: [u64; FINNY_PIECE_BITBOARDS],
    pub(super) side_to_move: Color,
    pub(super) valid: bool,
}

impl NnueFinnyTable {
    pub(super) fn new(hidden: usize) -> Self {
        Self {
            entries: (0..FINNY_TABLE_ENTRIES)
                .map(|_| NnueFinnyEntry {
                    values: vec![0; hidden],
                    pieces: [0; FINNY_PIECE_BITBOARDS],
                    side_to_move: Color::White,
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
        let index = perspective as usize * KING_BUCKETS + king_square;
        self.entries.get_mut(index)
    }
}

#[derive(Clone, Debug)]
pub(super) struct NnueInference {
    pub(super) acc_mul: i32,
    pub(super) acc_shift: u32,
    pub(super) use_screlu: bool,
    pub(super) hidden: Option<IntegerHiddenLayer>,
    pub(super) output: IntegerOutputLayer,
}

#[derive(Clone, Debug)]
pub(super) struct IntegerHiddenLayer {
    pub(super) input_size: usize,
    pub(super) output_size: usize,
    pub(super) weights: Vec<i32>,
    pub(super) bias: Vec<i32>,
}

#[derive(Clone, Debug)]
pub(super) struct IntegerOutputLayer {
    pub(super) weights: Vec<i32>,
    /// i16 copy for the fast screlu path
    pub(super) screlu_weights_i16: Option<Vec<i16>>,
    pub(super) bias: i32,
    pub(super) output_scale: i32,
    pub(super) quantization: IntegerOutputQuantization,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum IntegerOutputQuantization {
    ActivationQ,
    BulletScrelu { qa: i32, qb: i32 },
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

#[derive(Clone, Debug)]
pub struct NnueEvalScratch {
    pub(super) hidden: Vec<i32>,
    pub(super) activations: Vec<i32>,
    pub(super) sums: Vec<i64>,
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
