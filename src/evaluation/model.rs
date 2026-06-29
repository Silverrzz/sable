mod evaluate;
mod loading;

use crate::{Board, Color, Move, Piece, pieces::ALL_PIECES};

use super::{
    features::{
        apply_feature_delta, collect_move_feature_updates, feature_index_for_perspective,
        oriented_king_square,
    },
    types::*,
};

impl NnueModel {
    pub fn initial_accumulators(&self, board: &Board) -> Option<NnueAccumulators> {
        let mut accumulators = NnueAccumulators {
            values: Vec::new(),
            black_values: None,
        };
        self.refresh_accumulators_into(&mut accumulators, board)
            .then_some(accumulators)
    }

    pub(crate) fn new_finny_table(&self) -> Option<NnueFinnyTable> {
        Some(NnueFinnyTable::new(self.layers.first()?.bias.len()))
    }

    pub(crate) fn seed_finny_table(
        &self,
        table: &mut NnueFinnyTable,
        board: &Board,
        accumulators: &NnueAccumulators,
    ) -> bool {
        if !self.store_finny_entry(table, board, Color::White, &accumulators.values) {
            return false;
        }
        if self.side_to_move_relative {
            let Some(black_values) = accumulators.black_values.as_deref() else {
                return false;
            };
            if !self.store_finny_entry(table, board, Color::Black, black_values) {
                return false;
            }
        }
        true
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
        let white_ok = if let Some(table) = finny.as_mut() {
            self.refresh_accumulator_values_into(
                &mut accumulators.values,
                board,
                Color::White,
                Some(&mut **table),
            )
        } else {
            self.refresh_accumulator_values_into(&mut accumulators.values, board, Color::White, None)
        };
        if !white_ok {
            return false;
        }
        if self.side_to_move_relative {
            let black_values = accumulators.black_values.get_or_insert_with(Vec::new);
            let black_ok = if let Some(table) = finny.as_mut() {
                self.refresh_accumulator_values_into(
                    black_values,
                    board,
                    Color::Black,
                    Some(&mut **table),
                )
            } else {
                self.refresh_accumulator_values_into(black_values, board, Color::Black, None)
            };
            if !black_ok {
                return false;
            }
        } else {
            accumulators.black_values = None;
        }
        true
    }

    fn refresh_accumulator_values_into(
        &self,
        values: &mut Vec<i16>,
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
        values: &mut Vec<i16>,
        board: &Board,
        perspective: Color,
    ) -> bool {
        let Some(first_layer) = self.layers.first() else {
            return false;
        };
        let hidden = first_layer.bias.len();
        values.clear();
        values.resize(hidden, 0);
        for (slot, bias) in values.iter_mut().zip(first_layer.bias.iter()) {
            let Ok(value) = i16::try_from(*bias) else {
                return false;
            };
            *slot = value;
        }
        let Some(king_square) = oriented_king_square(board, perspective) else {
            return false;
        };

        for color in [Color::White, Color::Black] {
            for piece in ALL_PIECES {
                for square in crate::chess::pieces(board, piece) & crate::chess::colors(board, color) {
                    let feature = feature_index_for_perspective(
                        self.architecture,
                        perspective,
                        king_square,
                        color,
                        piece,
                        square as usize,
                    );
                    apply_feature_delta(
                        values,
                        hidden,
                        &self.first_layer_feature_weights,
                        feature,
                        1,
                    );
                }
            }
        }
        if self.has_side_to_move_feature && crate::chess::side_to_move(board) == Color::Black {
            apply_feature_delta(
                values,
                hidden,
                &self.first_layer_feature_weights,
                self.architecture.side_to_move_feature_index(),
                1,
            );
        }

        true
    }

    fn refresh_accumulator_values_from_finny(
        &self,
        values: &mut Vec<i16>,
        board: &Board,
        perspective: Color,
        table: &mut NnueFinnyTable,
    ) -> bool {
        let Some(first_layer) = self.layers.first() else {
            return false;
        };
        let hidden = first_layer.bias.len();
        let Some(king_square) = oriented_king_square(board, perspective) else {
            return false;
        };
        let current_pieces = board_piece_bitboards(board);
        let current_side = crate::chess::side_to_move(board);
        let Some(entry) = table.entry_mut(perspective, king_square) else {
            return self.refresh_accumulator_values_full_into(values, board, perspective);
        };
        if !entry.valid || entry.values.len() != hidden {
            if !self.refresh_accumulator_values_full_into(values, board, perspective) {
                return false;
            }
            entry.values.clone_from(values);
            entry.pieces = current_pieces;
            entry.side_to_move = current_side;
            entry.valid = true;
            return true;
        }

        // update the cached entry in place
        for color in [Color::White, Color::Black] {
            for piece in ALL_PIECES {
                let index = piece_bitboard_index(color, piece);
                let old = entry.pieces[index];
                let new = current_pieces[index];
                self.apply_piece_bitboard_diff(
                    &mut entry.values,
                    hidden,
                    perspective,
                    king_square,
                    color,
                    piece,
                    old & !new,
                    -1,
                );
                self.apply_piece_bitboard_diff(
                    &mut entry.values,
                    hidden,
                    perspective,
                    king_square,
                    color,
                    piece,
                    new & !old,
                    1,
                );
            }
        }
        if self.has_side_to_move_feature && entry.side_to_move != current_side {
            let sign = if current_side == Color::Black { 1 } else { -1 };
            apply_feature_delta(
                &mut entry.values,
                hidden,
                &self.first_layer_feature_weights,
                self.architecture.side_to_move_feature_index(),
                sign,
            );
        }

        entry.pieces = current_pieces;
        entry.side_to_move = current_side;
        entry.valid = true;
        values.clone_from(&entry.values);
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
        entry.values.clear();
        entry.values.extend_from_slice(values);
        entry.pieces = board_piece_bitboards(board);
        entry.side_to_move = crate::chess::side_to_move(board);
        entry.valid = true;
        true
    }

    fn apply_piece_bitboard_diff(
        &self,
        values: &mut [i16],
        hidden: usize,
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
            let feature = feature_index_for_perspective(
                self.architecture,
                perspective,
                king_square,
                color,
                piece,
                square,
            );
            apply_feature_delta(
                values,
                hidden,
                &self.first_layer_feature_weights,
                feature,
                sign,
            );
        }
    }

    pub fn apply_null_move_delta(
        &self,
        accumulators: &mut NnueAccumulators,
        before: &Board,
    ) -> bool {
        if self.side_to_move_relative {
            return true;
        }
        if !self.has_side_to_move_feature {
            return true;
        }
        let Some(first_layer) = self.layers.first() else {
            return false;
        };
        let hidden = first_layer.bias.len();
        let sign = if crate::chess::side_to_move(before) == Color::White { 1 } else { -1 };
        apply_feature_delta(
            &mut accumulators.values,
            hidden,
            &self.first_layer_feature_weights,
            self.architecture.side_to_move_feature_index(),
            sign,
        );
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
        if self.side_to_move_relative {
            if !self.update_accumulator_after_move_for_perspective(
                &mut accumulators.values,
                before,
                after,
                mv,
                Color::White,
                finny.as_mut().map(|table| &mut **table),
            ) {
                return false;
            }
            let Some(black_values) = accumulators.black_values.as_mut() else {
                return false;
            };
            if !self.update_accumulator_after_move_for_perspective(
                black_values,
                before,
                after,
                mv,
                Color::Black,
                finny.as_mut().map(|table| &mut **table),
            ) {
                return false;
            }
            return true;
        }
        self.update_accumulator_after_move_for_perspective(
            &mut accumulators.values,
            before,
            after,
            mv,
            Color::White,
            finny,
        )
    }

    fn update_accumulator_after_move_for_perspective(
        &self,
        values: &mut Vec<i16>,
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
        values: &mut [i16],
        before: &Board,
        mv: Move,
        perspective: Color,
    ) -> bool {
        let Some(first_layer) = self.layers.first() else {
            return false;
        };
        let Some(updates) = collect_move_feature_updates(
            before,
            mv,
            self.architecture,
            self.has_side_to_move_feature,
            perspective,
        ) else {
            return false;
        };
        let hidden = first_layer.bias.len();
        for update in updates.iter() {
            apply_feature_delta(
                values,
                hidden,
                &self.first_layer_feature_weights,
                update.feature,
                update.sign,
            );
        }
        true
    }

}

fn board_piece_bitboards(board: &Board) -> [u64; FINNY_PIECE_BITBOARDS] {
    let mut pieces = [0; FINNY_PIECE_BITBOARDS];
    for color in [Color::White, Color::Black] {
        for piece in ALL_PIECES {
            pieces[piece_bitboard_index(color, piece)] =
                (crate::chess::pieces(board, piece) & crate::chess::colors(board, color)).0;
        }
    }
    pieces
}

fn piece_bitboard_index(color: Color, piece: Piece) -> usize {
    color as usize * 6 + piece as usize
}
