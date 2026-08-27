use crate::{
    Board, Color, GameStatus, Move, Piece,
    chess::MoveGenState,
    evaluation::{DRAW_SCORE, LOSS_SCORE, is_board_drawn},
};

use crate::search::{
    constants::*,
    moves::{
        move_generation::tactical_move_score_with_history,
        move_ordering::{CandidateMove, MoveOrdering, compact_see},
    },
    state::move_context::MoveContext,
};

pub(in crate::search) fn move_score(
    board: &Board,
    side: Color,
    moving_piece: Piece,
    mv: Move,
    captured_piece: Option<Piece>,
    capture_see: Option<i32>,
    is_capture: bool,
    pv_move: Option<Move>,
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
                see: compact_see(see),
            },
            see,
        );
    }

    if promotion_value > 0 {
        return PROMOTION_SCORE + promotion_value;
    }

    ordering.quiet_score(board, side, mv, MoveContext::default(), ply)
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

pub(in crate::search) fn terminal_score(
    board: &Board,
    movegen: &MoveGenState,
    repetition: bool,
    ply: u16,
) -> Option<i32> {
    if repetition || is_board_drawn(board) {
        return Some(DRAW_SCORE);
    }
    match crate::chess::status_with_movegen(board, movegen) {
        GameStatus::Ongoing => None,
        GameStatus::Drawn => Some(DRAW_SCORE),
        GameStatus::Won => Some(LOSS_SCORE.saturating_add(ply as i32)),
    }
}

pub(in crate::search) fn immediate_terminal_score(
    board: &Board,
    movegen: &MoveGenState,
    repetition: bool,
    ply: u16,
) -> Option<i32> {
    if repetition || is_board_drawn(board) {
        return Some(DRAW_SCORE);
    }
    if crate::chess::halfmove_clock(board) < 100 {
        return None;
    }
    match crate::chess::status_with_movegen(board, movegen) {
        GameStatus::Won => Some(LOSS_SCORE.saturating_add(ply as i32)),
        GameStatus::Ongoing | GameStatus::Drawn => Some(DRAW_SCORE),
    }
}
