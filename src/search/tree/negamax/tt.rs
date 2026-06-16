use crate::{
    Board,
    evaluation::DRAW_SCORE,
};

use super::super::{
    constants::MAX_PV_LENGTH,
    context::SearchContext,
    position_key::{is_repetition, position_key},
    root::{PvMove, SearchOutcome, terminal_outcome},
    scoring::terminal_score,
    transposition::{Bound, TranspositionEntry, score_from_tt},
};

pub(super) fn tt_cutoff(
    board: &Board,
    depth: u32,
    alpha: i32,
    beta: i32,
    is_pv_node: bool,
    tt_entry: Option<TranspositionEntry>,
    context: &SearchContext<'_>,
    ply: u16,
) -> Option<SearchOutcome> {
    let entry = tt_entry?;
    if u32::from(entry.depth) < depth {
        return None;
    }

    let entry_score = score_from_tt(entry.score, ply);
    match entry.bound {
        Bound::Exact => exact_tt_cutoff(board, depth, is_pv_node, entry, entry_score, context),
        Bound::Lower if entry_score >= beta => Some(terminal_outcome(entry_score, false)),
        Bound::Upper if entry_score <= alpha => Some(terminal_outcome(entry_score, false)),
        _ => None,
    }
}

fn exact_tt_cutoff(
    board: &Board,
    depth: u32,
    is_pv_node: bool,
    entry: TranspositionEntry,
    entry_score: i32,
    context: &SearchContext<'_>,
) -> Option<SearchOutcome> {
    if is_pv_node {
        let cutoff_pv = tt_cutoff_pv(board, entry.best_move, context, depth);
        return match cutoff_pv.status {
            TtPvStatus::Usable => Some(SearchOutcome {
                score: entry_score,
                repetition_draw: false,
                pv: cutoff_pv.pv,
            }),
            TtPvStatus::RepetitionDraw => Some(SearchOutcome {
                score: DRAW_SCORE,
                repetition_draw: true,
                pv: cutoff_pv.pv,
            }),
            TtPvStatus::IllegalMove => None,
        };
    }

    let pv = entry
        .best_move
        .filter(|&mv| board.is_legal(mv))
        .map(|mv| vec![PvMove::new(board, mv, context.chess960())])
        .unwrap_or_default();
    Some(SearchOutcome {
        score: entry_score,
        repetition_draw: false,
        pv,
    })
}

struct TtCutoffPv {
    pv: Vec<PvMove>,
    status: TtPvStatus,
}

enum TtPvStatus {
    Usable,
    RepetitionDraw,
    IllegalMove,
}

fn tt_cutoff_pv(
    board: &Board,
    first_move: Option<crate::Move>,
    context: &SearchContext<'_>,
    depth: u32,
) -> TtCutoffPv {
    let mut board = board.clone();
    let mut next_move = first_move;
    let mut repetition_keys = context.repetition_keys().to_vec();
    let mut pv = Vec::with_capacity(depth.min(MAX_PV_LENGTH as u32) as usize);
    while pv.len() < depth as usize && pv.len() < MAX_PV_LENGTH {
        let Some(mv) = next_move else {
            break;
        };
        if !board.is_legal(mv) {
            return TtCutoffPv {
                pv,
                status: TtPvStatus::IllegalMove,
            };
        }
        pv.push(PvMove::new(&board, mv, context.chess960()));
        board.play_unchecked(mv);
        let key = position_key(&board);
        let repetition = is_repetition(key, board.halfmove_clock(), &repetition_keys);
        if terminal_score(&board, repetition, pv.len() as u16).is_some() {
            let status = if repetition {
                TtPvStatus::RepetitionDraw
            } else {
                TtPvStatus::Usable
            };
            return TtCutoffPv { pv, status };
        }
        repetition_keys.push(key);
        next_move = context
            .transposition_table()
            .probe(key)
            .filter(|entry| entry.bound == Bound::Exact)
            .and_then(|entry| entry.best_move);
    }
    TtCutoffPv {
        pv,
        status: TtPvStatus::Usable,
    }
}
