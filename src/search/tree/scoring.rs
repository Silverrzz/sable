
use crate::{
    Board, Color, GameStatus, Move, Piece,
    evaluation::{
        DRAW_SCORE, LOSS_SCORE,
        is_board_drawn,
    },
};

use super::{
    constants::*,
    move_generation::tactical_move_score_with_history,
    move_ordering::{CandidateMove, MoveOrdering},
};

pub(in crate::search) fn move_score(
    side: Color,
    moving_piece: Piece,
    mv: Move,
    captured_piece: Option<Piece>,
    capture_see: Option<i32>,
    is_capture: bool,
    pv_move: Option<Move>,
    previous_move: Option<Move>,
    ply: u16,
    ordering: &MoveOrdering,
) -> i32 {
    if Some(mv) == pv_move {
        return PV_MOVE_SCORE;
    }

    let promotion_value = mv.promotion.map(piece_value).unwrap_or(0);
    if is_capture {
        let see = capture_see.unwrap_or_else(|| {
            let victim = captured_piece.unwrap_or(Piece::Pawn);
            piece_value(victim) - piece_value(moving_piece)
        });
        return tactical_move_score_with_history(
            ordering,
            side,
            CandidateMove {
                mv,
                moving_piece,
                captured_piece,
                ordinal: 0,
                see: Some(see),
                score: None,
            },
            see,
        );
    }

    if promotion_value > 0 {
        return PROMOTION_SCORE + promotion_value;
    }

    ordering.quiet_score(side, mv, previous_move, ply)
}

pub(in crate::search) fn piece_value(piece: Piece) -> i32 {
    match piece {
        Piece::Pawn => 100,
        Piece::Knight => 320,
        Piece::Bishop => 330,
        Piece::Rook => 500,
        Piece::Queen => 900,
        Piece::King => 20_000,
    }
}

pub(in crate::search) fn terminal_score(board: &Board, repetition: bool, ply: u16) -> Option<i32> {
    if repetition || is_board_drawn(board) {
        return Some(DRAW_SCORE);
    }
    match crate::chess::status(board) {
        GameStatus::Ongoing => None,
        GameStatus::Drawn => Some(DRAW_SCORE),
        GameStatus::Won => Some(LOSS_SCORE.saturating_add(ply as i32)),
    }
}
