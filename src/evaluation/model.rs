use std::{
    path::Path,
    sync::{Arc, OnceLock},
};

use crate::{Board, Color, EngineError, Move, Piece, chess::Rank, pieces::ALL_PIECES};

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
        let Some(hidden_size) = shard_hidden_size(bytes) else {
            return Err(invalid_eval_file(
                path,
                "expected a single-hidden-layer shard SABLENET layout",
            ));
        };
        let Some(tensor_bytes) = shard_tensor_bytes(hidden_size) else {
            return Err(invalid_eval_file(path, "shard tensor dimensions overflow"));
        };
        let Some(file_max_bytes) = tensor_bytes.checked_add(SHARD_FILE_PADDING_BYTES) else {
            return Err(invalid_eval_file(path, "shard file dimensions overflow"));
        };
        if bytes.len() < tensor_bytes || bytes.len() > file_max_bytes {
            return Err(invalid_eval_file(
                path,
                "shard file size does not match its declared hidden width",
            ));
        }

        let feature_weight_count = SHARD_INPUT_FEATURES * hidden_size;
        let i16_end = SHARD_HEADER_BYTES + (feature_weight_count + hidden_size) * 2;
        let mut chunks = bytes[SHARD_HEADER_BYTES..i16_end]
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]));
        let mut feature_weights = Vec::with_capacity(feature_weight_count);
        for _ in 0..feature_weight_count {
            feature_weights.push(
                chunks
                    .next()
                    .expect("shard feature weights are present"),
            );
        }

        let mut bias = vec![0; hidden_size];
        for value in &mut bias {
            *value = chunks
                .next()
                .expect("shard accumulator bias is present");
        }
        debug_assert!(chunks.next().is_none());

        let mut chunks = bytes[i16_end..tensor_bytes]
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        let output_weight_count = hidden_size * 2 * SHARD_OUTPUT_BUCKETS;
        let mut output_weights = Vec::with_capacity(output_weight_count);
        for _ in 0..output_weight_count {
            let weight = chunks.next().expect("shard output weights are present");
            output_weights.push(i16::try_from(weight).map_err(|_| {
                invalid_eval_file(path, "shard output weight does not fit the SIMD evaluator")
            })?);
        }
        let mut output_bias = [0; SHARD_OUTPUT_BUCKETS];
        for value in &mut output_bias {
            *value = chunks.next().expect("shard output biases are present");
        }
        debug_assert!(chunks.next().is_none());
        let narrow_output_weights = output_weights.iter().all(|&weight| {
            i32::from(weight).abs() * i32::from(SHARD_QA) <= i32::from(i16::MAX)
        });

        validate_i16_accumulator_range(path, &bias, &feature_weights, hidden_size)?;
        Ok(Self {
            feature_weights: feature_weights.into_boxed_slice(),
            bias: bias.into_boxed_slice(),
            output_weights: output_weights.into_boxed_slice(),
            narrow_output_weights,
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
        NnueArchitectureId::Shard
    }

    pub fn initial_accumulators(&self, board: &Board) -> Option<NnueAccumulators> {
        let hidden_size = self.hidden_size();
        let mut accumulators = NnueAccumulators {
            white: vec![0; hidden_size].into_boxed_slice(),
            black: vec![0; hidden_size].into_boxed_slice(),
        };
        self.refresh_accumulators_into(&mut accumulators, board)
            .then_some(accumulators)
    }

    pub(crate) fn new_finny_table(&self) -> Option<NnueFinnyTable> {
        Some(NnueFinnyTable::new(self.hidden_size()))
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
        values: &mut [i16],
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
        values: &mut [i16],
        board: &Board,
        perspective: Color,
    ) -> bool {
        values.copy_from_slice(&self.bias);
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
                    apply_feature_delta(values, self.feature_weights(), feature, 1);
                }
            }
        }
        true
    }

    fn refresh_accumulator_values_from_finny(
        &self,
        values: &mut [i16],
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
            entry.values.copy_from_slice(values);
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
        values.copy_from_slice(&entry.values);
        true
    }

    fn store_finny_entry(
        &self,
        table: &mut NnueFinnyTable,
        board: &Board,
        perspective: Color,
        values: &[i16],
    ) -> bool {
        let Some(king_square) = oriented_king_square(board, perspective) else {
            return false;
        };
        let Some(entry) = table.entry_mut(perspective, king_square) else {
            return false;
        };
        entry.values.copy_from_slice(values);
        entry.pieces = board_piece_bitboards(board);
        entry.valid = true;
        true
    }

    fn apply_piece_bitboard_diff(
        &self,
        values: &mut [i16],
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
            apply_feature_delta(values, self.feature_weights(), feature, sign);
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
        source: &NnueAccumulators,
        target: &mut NnueAccumulators,
        before: &Board,
        after: &Board,
        mv: Move,
        moving_piece: Piece,
        captured_piece: Option<Piece>,
        mut finny: Option<&mut NnueFinnyTable>,
    ) -> bool {
        let side = crate::chess::side_to_move(before);
        let is_castle = moving_piece == Piece::King && crate::chess::colors(before, side).has(mv.to);
        let captured = captured_piece.map(|piece| {
            let square = if moving_piece == Piece::Pawn
                && !crate::chess::colors(before, !side).has(mv.to)
            {
                crate::Square::new(mv.to.file(), Rank::Fifth.relative_to(side))
            } else {
                mv.to
            };
            (piece, square)
        });
        let white = self.update_accumulator_after_move_for_perspective(
            &source.white,
            &mut target.white,
            before,
            after,
            mv,
            side,
            moving_piece,
            captured,
            is_castle,
            Color::White,
            finny.as_mut().map(|table| &mut **table),
        );
        let black = self.update_accumulator_after_move_for_perspective(
            &source.black,
            &mut target.black,
            before,
            after,
            mv,
            side,
            moving_piece,
            captured,
            is_castle,
            Color::Black,
            finny.as_mut().map(|table| &mut **table),
        );
        white && black
    }

    fn update_accumulator_after_move_for_perspective(
        &self,
        source: &[i16],
        target: &mut [i16],
        before: &Board,
        after: &Board,
        mv: Move,
        side: Color,
        moving_piece: Piece,
        captured: Option<(Piece, crate::Square)>,
        is_castle: bool,
        perspective: Color,
        finny: Option<&mut NnueFinnyTable>,
    ) -> bool {
        if self.apply_move_delta_for_perspective(
            source,
            target,
            before,
            mv,
            side,
            moving_piece,
            captured,
            is_castle,
            perspective,
        ) {
            return true;
        }
        self.refresh_accumulator_values_into(target, after, perspective, finny)
    }

    fn apply_move_delta_for_perspective(
        &self,
        source: &[i16],
        target: &mut [i16],
        before: &Board,
        mv: Move,
        side: Color,
        moving_piece: Piece,
        captured: Option<(Piece, crate::Square)>,
        is_castle: bool,
        perspective: Color,
    ) -> bool {
        let Some(updates) = collect_move_feature_updates(
            before,
            mv,
            side,
            moving_piece,
            captured,
            is_castle,
            perspective,
        ) else {
            return false;
        };
        apply_feature_deltas(source, target, self.feature_weights(), &updates);
        true
    }

    pub fn evaluate(&self, board: &Board) -> i32 {
        let accumulators = self
            .initial_accumulators(board)
            .expect("valid shard NNUE model should produce accumulators");
        self.evaluate_with_accumulators(board, &accumulators)
    }

    pub fn evaluate_with_accumulators(
        &self,
        board: &Board,
        accumulators: &NnueAccumulators,
    ) -> i32 {
        self.evaluate_output_bucket_with_accumulators(
            board,
            accumulators,
            output_bucket(board),
        )
    }

    pub fn output_bucket_values(&self, board: &Board) -> [i32; SHARD_OUTPUT_BUCKETS] {
        let accumulators = self
            .initial_accumulators(board)
            .expect("valid shard NNUE model should produce accumulators");
        std::array::from_fn(|bucket| {
            self.evaluate_output_bucket_with_accumulators(board, &accumulators, bucket)
        })
    }

    fn evaluate_output_bucket_with_accumulators(
        &self,
        board: &Board,
        accumulators: &NnueAccumulators,
        bucket: usize,
    ) -> i32 {
        let (stm, ntm) = match crate::chess::side_to_move(board) {
            Color::White => (&accumulators.white, &accumulators.black),
            Color::Black => (&accumulators.black, &accumulators.white),
        };
        let hidden_size = self.hidden_size();
        let outputs_per_bucket = hidden_size * 2;
        let start = bucket * outputs_per_bucket;
        let weights = &self.output_weights[start..start + outputs_per_bucket];
        let mut output = crate::simd::screlu_dot_i16_dual(
            stm,
            &weights[..hidden_size],
            ntm,
            &weights[hidden_size..],
            SHARD_QA,
            self.narrow_output_weights,
        );
        let qa = i64::from(SHARD_QA);
        output /= qa;
        output += i64::from(self.output_bias[bucket]);
        output *= i64::from(SHARD_OUTPUT_SCALE);
        output /= qa * i64::from(SHARD_QB);
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
        let Some(accumulators) = self.initial_accumulators(board) else {
            return Vec::new();
        };
        let Some(white_king_square) = oriented_king_square(board, Color::White) else {
            return Vec::new();
        };
        let Some(black_king_square) = oriented_king_square(board, Color::Black) else {
            return Vec::new();
        };

        let base_score_white = self.evaluate_white_with_accumulators(board, &accumulators);
        let mut contributions = Vec::new();
        for color in [Color::White, Color::Black] {
            for piece in ALL_PIECES {
                if piece == Piece::King {
                    continue;
                }
                for square in crate::chess::colored_pieces(board, color, piece) {
                    let mut removed = accumulators.clone();
                    self.apply_removed_piece_delta(
                        &mut removed.white,
                        Color::White,
                        white_king_square,
                        color,
                        piece,
                        square as usize,
                    );
                    self.apply_removed_piece_delta(
                        &mut removed.black,
                        Color::Black,
                        black_king_square,
                        color,
                        piece,
                        square as usize,
                    );
                    let removed_score_white = self.evaluate_white_with_accumulators(board, &removed);
                    contributions.push(PieceContribution {
                        square,
                        piece,
                        color,
                        score_white_cp: base_score_white - removed_score_white,
                    });
                }
            }
        }
        contributions
    }

    fn evaluate_white_with_accumulators(
        &self,
        board: &Board,
        accumulators: &NnueAccumulators,
    ) -> i32 {
        let score = self.evaluate_with_accumulators(board, accumulators);
        match crate::chess::side_to_move(board) {
            Color::White => score,
            Color::Black => -score,
        }
    }

    fn apply_removed_piece_delta(
        &self,
        values: &mut [i16],
        perspective: Color,
        king_square: usize,
        color: Color,
        piece: Piece,
        square: usize,
    ) {
        let feature = feature_index_for_perspective(perspective, king_square, color, piece, square);
        apply_feature_delta(values, self.feature_weights(), feature, -1);
    }

    fn feature_weights(&self) -> &[i16] {
        &self.feature_weights
    }

    fn hidden_size(&self) -> usize {
        self.bias.len()
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

fn output_bucket(board: &Board) -> usize {
    let piece_count = crate::chess::colors(board, Color::White).len()
        + crate::chess::colors(board, Color::Black).len();
    ((piece_count.saturating_sub(2) / 4) as usize).min(SHARD_OUTPUT_BUCKETS - 1)
}

fn shard_hidden_size(bytes: &[u8]) -> Option<usize> {
    let hidden_size = usize::try_from(read_u32(bytes, 64)?).ok()?;
    let doubled_hidden = u32::try_from(hidden_size.checked_mul(2)?).ok()?;
    let valid = bytes.get(..8) == Some(&b"SABLENET"[..])
        && read_u32(bytes, 8) == Some(1)
        && read_u32(bytes, 12) == Some(3)
        && read_u32(bytes, 16) == Some(SHARD_HEADER_BYTES as u32)
        && read_u32(bytes, 20) == Some(SHARD_INPUT_FEATURES as u32)
        && read_u32(bytes, 24) == Some(1)
        && read_u32(bytes, 28) == Some(SHARD_OUTPUT_BUCKETS as u32)
        && read_u32(bytes, 32) == Some(SHARD_OUTPUT_SCALE as u32)
        && read_u32(bytes, 36) == Some(SHARD_QA as u32)
        && read_u32(bytes, 40) == Some(SHARD_QB as u32)
        && read_u32(bytes, 44) == Some(3)
        && read_u32(bytes, 48) == Some(1)
        && read_u32(bytes, 52) == Some(4)
        && hidden_size > 0
        && read_u32(bytes, 68) == Some(1)
        && read_u32(bytes, 72) == Some(SHARD_INPUT_FEATURES as u32)
        && read_u32(bytes, 76) == Some(2)
        && read_u32(bytes, 80) == Some(SHARD_QA as u32)
        && read_u32(bytes, 84) == Some(1)
        && read_u32(bytes, 88) == Some(1)
        && read_u32(bytes, 92) == Some(SHARD_INPUT_FEATURES as u32)
        && read_u32(bytes, 96) == Some(hidden_size as u32)
        && read_u32(bytes, 100) == Some(2)
        && read_u32(bytes, 104) == Some(SHARD_QA as u32)
        && read_u32(bytes, 108) == Some(2)
        && read_u32(bytes, 112) == Some(1)
        && read_u32(bytes, 116) == Some(1)
        && read_u32(bytes, 120) == Some(hidden_size as u32)
        && read_u32(bytes, 124) == Some(2)
        && read_u32(bytes, 128) == Some(SHARD_QA as u32)
        && read_u32(bytes, 132) == Some(5)
        && read_u32(bytes, 136) == Some(2)
        && read_u32(bytes, 140) == Some(doubled_hidden)
        && read_u32(bytes, 144) == Some(1)
        && read_u32(bytes, 148) == Some(4)
        && read_u32(bytes, 152) == Some(SHARD_QB as u32)
        && read_u32(bytes, 156) == Some(6)
        && read_u32(bytes, 160) == Some(2)
        && read_u32(bytes, 164) == Some(1)
        && read_u32(bytes, 168) == Some(1)
        && read_u32(bytes, 172) == Some(4)
        && read_u32(bytes, 176) == Some((i32::from(SHARD_QA) * i32::from(SHARD_QB)) as u32);
    valid.then_some(hidden_size)
}

fn shard_tensor_bytes(hidden_size: usize) -> Option<usize> {
    let feature_weights = SHARD_INPUT_FEATURES.checked_mul(hidden_size)?;
    let i16_values = feature_weights.checked_add(hidden_size)?;
    let output_weights = hidden_size
        .checked_mul(2)?
        .checked_mul(SHARD_OUTPUT_BUCKETS)?;
    let i32_values = output_weights.checked_add(SHARD_OUTPUT_BUCKETS)?;
    SHARD_HEADER_BYTES
        .checked_add(i16_values.checked_mul(2)?)?
        .checked_add(i32_values.checked_mul(4)?)
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let value = bytes.get(offset..offset + 4)?;
    Some(u32::from_le_bytes(value.try_into().ok()?))
}

fn invalid_eval_file(path: &Path, message: &str) -> EngineError {
    EngineError::InvalidEvalFile {
        path: path.display().to_string(),
        message: message.to_owned(),
    }
}
