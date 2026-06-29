
use crate::{
    Board, Color, Move, Piece, Square,
    chess::{BitBoard, generate_moves, generate_tactical_moves},
};

use super::{
    board_moves::{captured_piece, en_passant_target, is_en_passant},
    constants::*,
    move_ordering::{CandidateMove, MoveOrdering, MovePicker, ScoredMove, scaled_history_score},
    scoring::{move_score, piece_value},
    see::static_exchange_eval_for_move,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search) enum MoveFilter {
    All,
    Tactical,
}

pub(in crate::search) fn ordered_root_moves(
    board: &Board,
    candidate_moves: &[Move],
    pv_move: Option<Move>,
    ordering: &MoveOrdering,
) -> Vec<ScoredMove> {
    let side = crate::chess::side_to_move(board);
    let ep_target = en_passant_target(board, side);
    let mut moves = Vec::with_capacity(candidate_moves.len());
    for (ordinal, mv) in candidate_moves.iter().enumerate() {
        let moving_piece = crate::chess::piece_on(board, mv.from).unwrap_or(Piece::Pawn);
        let is_capture =
            crate::chess::colors(board, !side).has(mv.to) || is_en_passant(moving_piece, *mv, ep_target);
        let captured_piece = if is_capture {
            captured_piece(board, moving_piece, *mv, ep_target)
        } else {
            None
        };
        let see = is_capture.then(|| {
            static_exchange_eval_for_move(board, *mv, moving_piece, captured_piece)
        });
        moves.push(ScoredMove {
            mv: *mv,
            score: move_score(
                side,
                moving_piece,
                *mv,
                captured_piece,
                see,
                is_capture,
                pv_move,
                None,
                0,
                ordering,
            ),
            ordinal,
            is_quiet: !is_capture && mv.promotion.is_none(),
            moving_piece,
            captured_piece,
            see,
        });
    }
    sort_scored_moves(&mut moves);
    moves
}

pub(in crate::search) fn collect_moves_into(
    board: &Board,
    filter: MoveFilter,
    pv_move: Option<Move>,
    previous_move: Option<Move>,
    ply: u16,
    moves: &mut MovePicker,
) {
    let side = crate::chess::side_to_move(board);
    let enemy_occupancy = crate::chess::colors(board, !side);
    let ep_target = en_passant_target(board, side);
    moves.reset(pv_move, side, previous_move, ply, filter);
    match filter {
        MoveFilter::All => collect_all_moves_into(board, enemy_occupancy, ep_target, moves),
        MoveFilter::Tactical => {
            collect_tactical_moves_into(board, enemy_occupancy, ep_target, moves);
        }
    }
}

fn collect_all_moves_into(
    board: &Board,
    enemy_occupancy: BitBoard,
    ep_target: Option<Square>,
    moves: &mut MovePicker,
) {
    let mut ordinal = 0;
    generate_moves(board, |piece_moves| {
        for mv in piece_moves {
            let captured_piece =
                captured_piece_for_generated_move(board, piece_moves.piece, mv, enemy_occupancy, ep_target);
            let is_tactical = captured_piece.is_some() || mv.promotion.is_some();
            let candidate = CandidateMove {
                mv,
                moving_piece: piece_moves.piece,
                captured_piece,
                ordinal,
                see: None,
                score: None,
            };
            if is_tactical {
                moves.push_tactical(candidate);
            } else {
                moves.push_quiet(candidate);
            }
            ordinal += 1;
        }
        false
    });
}

fn collect_tactical_moves_into(
    board: &Board,
    enemy_occupancy: BitBoard,
    ep_target: Option<Square>,
    moves: &mut MovePicker,
) {
    let mut ordinal = 0;
    generate_tactical_moves(board, |piece_moves| {
        for mv in piece_moves {
            let captured_piece =
                captured_piece_for_generated_move(board, piece_moves.piece, mv, enemy_occupancy, ep_target);
            moves.push_tactical(CandidateMove {
                mv,
                moving_piece: piece_moves.piece,
                captured_piece,
                ordinal,
                see: None,
                score: None,
            });
            ordinal += 1;
        }
        false
    });
}

#[inline]
fn captured_piece_for_generated_move(
    board: &Board,
    moving_piece: Piece,
    mv: Move,
    enemy_occupancy: BitBoard,
    ep_target: Option<Square>,
) -> Option<Piece> {
    if enemy_occupancy.has(mv.to) {
        crate::chess::piece_on(board, mv.to)
    } else if is_en_passant(moving_piece, mv, ep_target) {
        Some(Piece::Pawn)
    } else {
        None
    }
}

pub(in crate::search) fn is_tactical_move(board: &Board, mv: Move) -> bool {
    if mv.promotion.is_some() {
        return true;
    }
    let side = crate::chess::side_to_move(board);
    let moving_piece = crate::chess::piece_on(board, mv.from).unwrap_or(Piece::Pawn);
    crate::chess::colors(board, !side).has(mv.to)
        || is_en_passant(moving_piece, mv, en_passant_target(board, side))
}

pub(in crate::search) fn priority_move_for_node(
    board: &Board,
    pv_move: Option<Move>,
    tt_move: Option<Move>,
    in_check: bool,
) -> Option<Move> {
    let priority = pv_move.or(tt_move);
    if in_check {
        priority.filter(|&mv| is_tactical_move(board, mv))
    } else {
        priority
    }
}

pub(in crate::search) fn pick_better_move(
    current: Option<(usize, i32, u16)>,
    index: usize,
    score: i32,
    ordinal: u16,
) -> Option<(usize, i32, u16)> {
    match current {
        Some((_, best_score, best_ordinal))
            if best_score > score || (best_score == score && best_ordinal < ordinal) =>
        {
            current
        }
        _ => Some((index, score, ordinal)),
    }
}

pub(in crate::search) fn tactical_move_score(candidate: CandidateMove, see: i32) -> i32 {
    let promotion_value = candidate.mv.promotion.map(piece_value).unwrap_or(0);
    if candidate.captured_piece.is_some() {
        let victim = candidate.captured_piece.unwrap_or(Piece::Pawn);
        let see_order = see.clamp(-10_000, 10_000);
        return CAPTURE_SCORE
            + see_order * 1024
            + piece_value(victim) * 32
            - piece_value(candidate.moving_piece)
            + promotion_value;
    }

    PROMOTION_SCORE + promotion_value
}

pub(in crate::search) fn tactical_move_score_with_history(
    ordering: &MoveOrdering,
    side: Color,
    candidate: CandidateMove,
    see: i32,
) -> i32 {
    tactical_move_score(candidate, see).saturating_add(
        scaled_history_score(
            ordering.capture_score(
                side,
                candidate.moving_piece,
                candidate.mv.to,
                candidate.captured_piece,
            ),
            CAPTURE_HISTORY_ORDERING_DIVISOR,
        ),
    )
}

pub(in crate::search) fn sort_scored_moves(moves: &mut [ScoredMove]) {
    moves.sort_unstable_by(|a, b| b.score.cmp(&a.score).then(a.ordinal.cmp(&b.ordinal)));
}
