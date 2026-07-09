use std::{
    path::Path,
    sync::{Arc, OnceLock},
};

use crate::{Board, Color, EngineError, Move, Piece, pieces::ALL_PIECES};

use super::{
    features::{
        apply_feature_delta, apply_feature_deltas, collect_move_feature_updates,
        feature_index_for_perspective, oriented_king_square, validate_i16_accumulator_range,
    },
    types::*,
};

impl NnueModel {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, EngineError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|error| EngineError::InvalidEvalFile {
            path: path.display().to_string(),
            message: error.to_string(),
        })?;
        Self::from_bytes(path, &bytes)
    }

    fn from_bytes(path: &Path, bytes: &[u8]) -> Result<Self, EngineError> {
        if bytes.len() < VEX_TENSOR_BYTES || bytes.len() > VEX_FILE_MAX_BYTES {
            return Err(invalid_eval_file(
                path,
                "expected vex layout (768x16hm->256)x2->1",
            ));
        }

        let mut chunks = bytes[..VEX_TENSOR_BYTES]
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]));
        let mut feature_weights = Vec::with_capacity(VEX_FEATURE_WEIGHTS);
        for _ in 0..VEX_FEATURE_WEIGHTS {
            feature_weights.push(chunks.next().expect("vex feature weights are present"));
        }

        let mut bias = [0; VEX_HIDDEN];
        for value in &mut bias {
            *value = chunks.next().expect("vex accumulator bias is present");
        }

        let mut output_weights = [0; VEX_OUTPUTS];
        for value in &mut output_weights {
            *value = chunks.next().expect("vex output weights are present");
        }
        let output_bias = i32::from(chunks.next().expect("vex output bias is present"));
        debug_assert!(chunks.next().is_none());

        validate_i16_accumulator_range(path, &bias, &feature_weights)?;
        Ok(Self {
            feature_weights: feature_weights.into_boxed_slice(),
            bias,
            output_weights,
            output_bias,
        })
    }

    pub(crate) fn shared_embedded_default() -> Option<Result<Arc<Self>, EngineError>> {
        if option_env!("SABLE_ENGINE_HAS_EMBEDDED_EVAL").unwrap_or("0") != "1" {
            return None;
        }
        static SHARED_EMBEDDED_DEFAULT: OnceLock<Result<Arc<NnueModel>, EngineError>> =
            OnceLock::new();
        Some(
            SHARED_EMBEDDED_DEFAULT
                .get_or_init(|| {
                    let label = NnueModel::embedded_default_label().unwrap_or("<embedded>");
                    let bytes = include_bytes!(env!("SABLE_ENGINE_EMBEDDED_EVAL_PATH"));
                    NnueModel::from_bytes(Path::new(label), bytes).map(Arc::new)
                })
                .clone(),
        )
    }

    pub fn has_embedded_default() -> bool {
        option_env!("SABLE_ENGINE_HAS_EMBEDDED_EVAL").unwrap_or("0") == "1"
    }

    pub fn embedded_default_label() -> Option<&'static str> {
        if !Self::has_embedded_default() {
            return None;
        }
        let label = option_env!("SABLE_ENGINE_EMBEDDED_EVAL_LABEL").unwrap_or("embedded");
        if label == "none" { None } else { Some(label) }
    }

    pub fn embedded_default_hash() -> Option<&'static str> {
        if !Self::has_embedded_default() {
            return None;
        }
        let hash = option_env!("SABLE_ENGINE_EMBEDDED_EVAL_HASH").unwrap_or("none");
        if hash == "none" { None } else { Some(hash) }
    }

    pub fn architecture_id(&self) -> NnueArchitectureId {
        NnueArchitectureId::Vex
    }

    pub fn initial_accumulators(&self, board: &Board) -> Option<NnueAccumulators> {
        let mut accumulators = NnueAccumulators {
            white: [0; VEX_HIDDEN],
            black: [0; VEX_HIDDEN],
        };
        self.refresh_accumulators_into(&mut accumulators, board)
            .then_some(accumulators)
    }

    pub(crate) fn new_finny_table(&self) -> Option<NnueFinnyTable> {
        Some(NnueFinnyTable::new())
    }

    pub(crate) fn seed_finny_table(
        &self,
        table: &mut NnueFinnyTable,
        board: &Board,
        accumulators: &NnueAccumulators,
    ) -> bool {
        self.store_finny_entry(table, board, Color::White, &accumulators.white)
            && self.store_finny_entry(table, board, Color::Black, &accumulators.black)
    }

    pub fn refresh_accumulators_into(
        &self,
        accumulators: &mut NnueAccumulators,
        board: &Board,
    ) -> bool {
        self.refresh_accumulators_into_with_finny(accumulators, board, None)
    }

    pub(crate) fn refresh_accumulators_into_with_finny(
        &self,
        accumulators: &mut NnueAccumulators,
        board: &Board,
        mut finny: Option<&mut NnueFinnyTable>,
    ) -> bool {
        let white = self.refresh_accumulator_values_into(
            &mut accumulators.white,
            board,
            Color::White,
            finny.as_mut().map(|table| &mut **table),
        );
        let black = self.refresh_accumulator_values_into(
            &mut accumulators.black,
            board,
            Color::Black,
            finny.as_mut().map(|table| &mut **table),
        );
        white && black
    }

    fn refresh_accumulator_values_into(
        &self,
        values: &mut [i16; VEX_HIDDEN],
        board: &Board,
        perspective: Color,
        finny: Option<&mut NnueFinnyTable>,
    ) -> bool {
        if let Some(table) = finny {
            return self.refresh_accumulator_values_from_finny(values, board, perspective, table);
        }
        self.refresh_accumulator_values_full_into(values, board, perspective)
    }

    fn refresh_accumulator_values_full_into(
        &self,
        values: &mut [i16; VEX_HIDDEN],
        board: &Board,
        perspective: Color,
    ) -> bool {
        *values = self.bias;
        let Some(king_square) = oriented_king_square(board, perspective) else {
            return false;
        };

        for color in [Color::White, Color::Black] {
            for piece in ALL_PIECES {
                for square in crate::chess::colored_pieces(board, color, piece) {
                    let feature = feature_index_for_perspective(
                        perspective,
                        king_square,
                        color,
                        piece,
                        square as usize,
                    );
                    apply_feature_delta(values, &self.feature_weights, feature, 1);
                }
            }
        }
        true
    }

    fn refresh_accumulator_values_from_finny(
        &self,
        values: &mut [i16; VEX_HIDDEN],
        board: &Board,
        perspective: Color,
        table: &mut NnueFinnyTable,
    ) -> bool {
        let Some(king_square) = oriented_king_square(board, perspective) else {
            return false;
        };
        let current_pieces = board_piece_bitboards(board);
        let Some(entry) = table.entry_mut(perspective, king_square) else {
            return self.refresh_accumulator_values_full_into(values, board, perspective);
        };
        if !entry.valid {
            if !self.refresh_accumulator_values_full_into(values, board, perspective) {
                return false;
            }
            entry.values = *values;
            entry.pieces = current_pieces;
            entry.valid = true;
            return true;
        }

        for color in [Color::White, Color::Black] {
            for piece in ALL_PIECES {
                let index = piece_bitboard_index(color, piece);
                let old = entry.pieces[index];
                let new = current_pieces[index];
                self.apply_piece_bitboard_diff(
                    &mut entry.values,
                    perspective,
                    king_square,
                    color,
                    piece,
                    old & !new,
                    -1,
                );
                self.apply_piece_bitboard_diff(
                    &mut entry.values,
                    perspective,
                    king_square,
                    color,
                    piece,
                    new & !old,
                    1,
                );
            }
        }

        entry.pieces = current_pieces;
        entry.valid = true;
        *values = entry.values;
        true
    }

    fn store_finny_entry(
        &self,
        table: &mut NnueFinnyTable,
        board: &Board,
        perspective: Color,
        values: &[i16; VEX_HIDDEN],
    ) -> bool {
        let Some(king_square) = oriented_king_square(board, perspective) else {
            return false;
        };
        let Some(entry) = table.entry_mut(perspective, king_square) else {
            return false;
        };
        entry.values = *values;
        entry.pieces = board_piece_bitboards(board);
        entry.valid = true;
        true
    }

    fn apply_piece_bitboard_diff(
        &self,
        values: &mut [i16; VEX_HIDDEN],
        perspective: Color,
        king_square: usize,
        color: Color,
        piece: Piece,
        mut bits: u64,
        sign: i32,
    ) {
        while bits != 0 {
            let square = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let feature =
                feature_index_for_perspective(perspective, king_square, color, piece, square);
            apply_feature_delta(values, &self.feature_weights, feature, sign);
        }
    }

    pub fn apply_null_move_delta(
        &self,
        accumulators: &mut NnueAccumulators,
        before: &Board,
    ) -> bool {
        let _ = (accumulators, before);
        true
    }

    pub(crate) fn update_accumulators_after_move(
        &self,
        accumulators: &mut NnueAccumulators,
        before: &Board,
        after: &Board,
        mv: Move,
        mut finny: Option<&mut NnueFinnyTable>,
    ) -> bool {
        let white = self.update_accumulator_after_move_for_perspective(
            &mut accumulators.white,
            before,
            after,
            mv,
            Color::White,
            finny.as_mut().map(|table| &mut **table),
        );
        let black = self.update_accumulator_after_move_for_perspective(
            &mut accumulators.black,
            before,
            after,
            mv,
            Color::Black,
            finny.as_mut().map(|table| &mut **table),
        );
        white && black
    }

    fn update_accumulator_after_move_for_perspective(
        &self,
        values: &mut [i16; VEX_HIDDEN],
        before: &Board,
        after: &Board,
        mv: Move,
        perspective: Color,
        finny: Option<&mut NnueFinnyTable>,
    ) -> bool {
        if self.apply_move_delta_for_perspective(values, before, mv, perspective) {
            return true;
        }
        self.refresh_accumulator_values_into(values, after, perspective, finny)
    }

    fn apply_move_delta_for_perspective(
        &self,
        values: &mut [i16; VEX_HIDDEN],
        before: &Board,
        mv: Move,
        perspective: Color,
    ) -> bool {
        let Some(updates) = collect_move_feature_updates(before, mv, perspective) else {
            return false;
        };
        apply_feature_deltas(values, &self.feature_weights, &updates);
        true
    }

    pub fn evaluate(&self, board: &Board) -> i32 {
        let accumulators = self
            .initial_accumulators(board)
            .expect("valid vex NNUE model should produce accumulators");
        self.evaluate_with_accumulators(board, &accumulators)
    }

    pub fn evaluate_with_accumulators(
        &self,
        board: &Board,
        accumulators: &NnueAccumulators,
    ) -> i32 {
        let (stm, ntm) = match crate::chess::side_to_move(board) {
            Color::White => (&accumulators.white, &accumulators.black),
            Color::Black => (&accumulators.black, &accumulators.white),
        };
        let mut output = crate::simd::screlu_dot_i16_dual(
            stm,
            &self.output_weights[..VEX_HIDDEN],
            ntm,
            &self.output_weights[VEX_HIDDEN..],
            VEX_QA,
        );
        let qa = i64::from(VEX_QA);
        output /= qa;
        output += i64::from(self.output_bias);
        output *= i64::from(VEX_OUTPUT_SCALE);
        output /= qa * i64::from(VEX_QB);
        output.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    }

    pub fn evaluate_for_side_to_move(&self, board: &Board) -> i32 {
        self.evaluate(board)
    }

    pub fn evaluate_for_side_to_move_with_accumulators(
        &self,
        board: &Board,
        accumulators: &NnueAccumulators,
    ) -> i32 {
        self.evaluate_with_accumulators(board, accumulators)
    }

    pub fn piece_contributions_white(&self, board: &Board) -> Vec<PieceContribution> {
        let _ = board;
        Vec::new()
    }
}

fn board_piece_bitboards(board: &Board) -> [u64; FINNY_PIECE_BITBOARDS] {
    let mut pieces = [0; FINNY_PIECE_BITBOARDS];
    for color in [Color::White, Color::Black] {
        for piece in ALL_PIECES {
            pieces[piece_bitboard_index(color, piece)] =
                crate::chess::colored_pieces(board, color, piece).0;
        }
    }
    pieces
}

fn piece_bitboard_index(color: Color, piece: Piece) -> usize {
    color as usize * 6 + piece as usize
}

fn invalid_eval_file(path: &Path, message: &str) -> EngineError {
    EngineError::InvalidEvalFile {
        path: path.display().to_string(),
        message: message.to_owned(),
    }
}
