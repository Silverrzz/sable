use crate::{Board, Color, Piece};

use super::super::{
    features::{apply_feature_delta, feature_index_for_perspective, oriented_king_square},
    integer::{integer_forward, integer_forward_dual},
    types::*,
};

impl NnueModel {
    #[allow(dead_code)]
    pub fn architecture_id(&self) -> NnueArchitectureId {
        self.architecture.id
    }

    pub fn evaluate(&self, board: &Board) -> i32 {
        let accumulators = self
            .initial_accumulators(board)
            .expect("valid quantized NNUE model should produce accumulators");
        self.evaluate_with_accumulators(board, &accumulators)
    }

    pub fn evaluate_with_accumulators(
        &self,
        board: &Board,
        accumulators: &NnueAccumulators,
    ) -> i32 {
        let mut scratch = self.eval_scratch();
        self.evaluate_with_accumulators_and_scratch(board, accumulators, &mut scratch)
    }

    pub fn eval_scratch(&self) -> NnueEvalScratch {
        let input_width = self
            .layers
            .first()
            .map(|layer| layer.bias.len())
            .unwrap_or(1);
        let output_input_width = self.architecture.output_input_size(input_width);
        let hidden_width = self
            .inference
            .hidden
            .as_ref()
            .map(|layer| layer.output_size.max(layer.input_size))
            .unwrap_or(output_input_width);
        NnueEvalScratch {
            hidden: vec![0; hidden_width],
            activations: vec![0; output_input_width],
            sums: vec![0; hidden_width],
        }
    }

    pub fn evaluate_with_accumulators_and_scratch(
        &self,
        board: &Board,
        accumulators: &NnueAccumulators,
        scratch: &mut NnueEvalScratch,
    ) -> i32 {
        if self.architecture.perspective_mode == NnuePerspectiveMode::DualConcat {
            return self.evaluate_dual_with_accumulators_and_scratch(board, accumulators, scratch);
        }
        let Some(values) = self.accumulator_values(board, accumulators) else {
            return self.evaluate(board);
        };
        if let Some(first_layer) = self.layers.first()
            && values.len() != first_layer.bias.len()
        {
            return self.evaluate(board);
        }
        let required = self
            .inference
            .hidden
            .as_ref()
            .map(|layer| layer.output_size.max(layer.input_size))
            .unwrap_or(values.len());
        if scratch.hidden.len() < required
            || scratch.activations.len() < values.len()
            || scratch.sums.len() < required
        {
            return self.evaluate_with_accumulators_slow(board, accumulators);
        }
        debug_assert_eq!(self.output_scale, self.inference.output.output_scale);
        integer_forward(values, &self.inference, scratch)
    }

    fn evaluate_dual_with_accumulators_and_scratch(
        &self,
        board: &Board,
        accumulators: &NnueAccumulators,
        scratch: &mut NnueEvalScratch,
    ) -> i32 {
        let Some(first_layer) = self.layers.first() else {
            return self.evaluate(board);
        };
        let hidden = first_layer.bias.len();
        let Some(black_values) = accumulators.black_values.as_deref() else {
            return self.evaluate(board);
        };
        if accumulators.values.len() != hidden || black_values.len() != hidden {
            return self.evaluate(board);
        }
        let (stm_values, ntm_values) = match crate::chess::side_to_move(board) {
            Color::White => (accumulators.values.as_slice(), black_values),
            Color::Black => (black_values, accumulators.values.as_slice()),
        };
        let required_activations = hidden * 2;
        let required_hidden = self
            .inference
            .hidden
            .as_ref()
            .map(|layer| layer.output_size.max(layer.input_size))
            .unwrap_or(required_activations);
        if scratch.hidden.len() < required_hidden
            || scratch.activations.len() < required_activations
            || scratch.sums.len() < required_hidden
        {
            return self.evaluate_with_accumulators_slow(board, accumulators);
        }
        debug_assert_eq!(self.output_scale, self.inference.output.output_scale);
        integer_forward_dual(stm_values, ntm_values, &self.inference, scratch)
    }

    fn accumulator_values<'a>(
        &self,
        board: &Board,
        accumulators: &'a NnueAccumulators,
    ) -> Option<&'a [i16]> {
        if self.side_to_move_relative && crate::chess::side_to_move(board) == Color::Black {
            accumulators.black_values.as_deref()
        } else {
            Some(&accumulators.values)
        }
    }

    fn evaluate_with_accumulators_slow(
        &self,
        board: &Board,
        accumulators: &NnueAccumulators,
    ) -> i32 {
        let mut scratch = self.eval_scratch();
        self.evaluate_with_accumulators_and_scratch(board, accumulators, &mut scratch)
    }

    pub fn piece_contributions_white(&self, board: &Board) -> Vec<PieceContribution> {
        let Some(accumulators) = self.initial_accumulators(board) else {
            return Vec::new();
        };
        let Some(first_layer) = self.layers.first() else {
            return Vec::new();
        };
        let hidden = first_layer.bias.len();
        let Some(white_king_square) = oriented_king_square(board, Color::White) else {
            return Vec::new();
        };
        let Some(black_king_square) = oriented_king_square(board, Color::Black) else {
            return Vec::new();
        };

        let to_white = |stm: i32| match crate::chess::side_to_move(board) {
            Color::White => stm,
            Color::Black => -stm,
        };
        let full_white =
            to_white(self.evaluate_for_side_to_move_with_accumulators(board, &accumulators));

        let mut out = Vec::new();
        for piece in [
            Piece::Pawn,
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
        ] {
            for color in [Color::White, Color::Black] {
                for sq in crate::chess::pieces(board, piece) & crate::chess::colors(board, color) {
                    let sq_idx = sq as usize;
                    let mut modified = accumulators.clone();
                    let feat = feature_index_for_perspective(
                        self.architecture,
                        Color::White,
                        white_king_square,
                        color,
                        piece,
                        sq_idx,
                    );
                    apply_feature_delta(
                        &mut modified.values,
                        hidden,
                        &self.first_layer_feature_weights,
                        feat,
                        -1,
                    );

                    if self.side_to_move_relative
                        && let Some(bv) = modified.black_values.as_mut()
                    {
                        let feat = feature_index_for_perspective(
                            self.architecture,
                            Color::Black,
                            black_king_square,
                            color,
                            piece,
                            sq_idx,
                        );
                        apply_feature_delta(
                            bv,
                            hidden,
                            &self.first_layer_feature_weights,
                            feat,
                            -1,
                        );
                    }

                    let without_white = to_white(
                        self.evaluate_for_side_to_move_with_accumulators(board, &modified),
                    );
                    out.push(PieceContribution {
                        square: sq,
                        piece,
                        color,
                        score_white_cp: full_white - without_white,
                    });
                }
            }
        }
        out
    }

    pub fn evaluate_for_side_to_move(&self, board: &Board) -> i32 {
        if self.side_to_move_relative {
            return self.evaluate(board);
        }
        let white_score = self.evaluate(board);
        match crate::chess::side_to_move(board) {
            Color::White => white_score,
            Color::Black => -white_score,
        }
    }

    pub fn evaluate_for_side_to_move_with_accumulators(
        &self,
        board: &Board,
        accumulators: &NnueAccumulators,
    ) -> i32 {
        if self.side_to_move_relative {
            return self.evaluate_with_accumulators(board, accumulators);
        }
        let white_score = self.evaluate_with_accumulators(board, accumulators);
        match crate::chess::side_to_move(board) {
            Color::White => white_score,
            Color::Black => -white_score,
        }
    }

    pub fn evaluate_for_side_to_move_with_accumulators_and_scratch(
        &self,
        board: &Board,
        accumulators: &NnueAccumulators,
        scratch: &mut NnueEvalScratch,
    ) -> i32 {
        if self.side_to_move_relative {
            return self.evaluate_with_accumulators_and_scratch(board, accumulators, scratch);
        }
        let white_score =
            self.evaluate_with_accumulators_and_scratch(board, accumulators, scratch);
        match crate::chess::side_to_move(board) {
            Color::White => white_score,
            Color::Black => -white_score,
        }
    }
}
