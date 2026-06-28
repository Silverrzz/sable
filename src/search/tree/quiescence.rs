
use crate::{
    Board, Move,
    evaluation::LOSS_SCORE,
};

use super::{
    constants::*,
    context::SearchContext,
    correction_history::CorrectionContext,
    move_generation::{MoveFilter, collect_moves_into, priority_move_for_node},
    move_ordering::MovePicker,
    position_key::{PositionKey, position_key},
    pruning::{apply_mate_distance_pruning, should_q_delta_prune_capture},
    root::{PvMove, SearchOutcome, is_better_score, parent_outcome, terminal_outcome},
    scoring::terminal_score,
    see::move_gives_check,
    transposition::{Bound, TranspositionEntry, score_from_tt},
};

pub(in crate::search) fn quiescence(
    board: &Board,
    repetition: bool,
    mut alpha: i32,
    mut beta: i32,
    previous_move: Option<Move>,
    correction_context: CorrectionContext,
    previous_pv: &[PvMove],
    context: &mut SearchContext<'_>,
    ply: u16,
) -> Option<SearchOutcome> {
    context.clear_static_eval_at_ply(ply);
    if context.should_stop().is_some() {
        return None;
    }
    if let Some(score) = terminal_score(board, repetition, ply) {
        return Some(terminal_outcome(score, repetition));
    }
    let in_check = !crate::chess::checkers(board).is_empty();
    if let Some(score) = apply_mate_distance_pruning(&mut alpha, &mut beta, ply) {
        return Some(terminal_outcome(score, false));
    }
    let key = position_key(board);
    let use_tt = !repetition;
    let tt_entry = if use_tt {
        context.transposition_table().probe(key)
    } else {
        None
    };
    if let Some(entry) = tt_entry
        && let Some(outcome) = qsearch_tt_cutoff(board, entry, alpha, beta, ply, context)
    {
        return Some(outcome);
    }
    let alpha_start = alpha;
    if qsearch_at_safety_bound(ply) {
        let (score, raw_static_eval) =
            qsearch_safety_bound_score(board, in_check, correction_context, context, ply);
        if raw_static_eval.is_some() {
            context.record_static_eval_at_ply(ply, score);
        }
        qsearch_store(
            context,
            use_tt,
            key,
            score,
            alpha_start,
            beta,
            None,
            raw_static_eval,
            ply,
        );
        return Some(terminal_outcome(score, false));
    }

    let raw_static_eval = tt_entry.and_then(|entry| entry.static_eval());
    let mut raw_stand_pat = None;
    let stand_pat = if in_check {
        None
    } else {
        let raw_eval = raw_static_eval.unwrap_or_else(|| context.evaluate(board));
        raw_stand_pat = Some(raw_eval);
        let stand_pat = context.corrected_static_eval(board, raw_eval, correction_context);
        context.record_static_eval_at_ply(ply, stand_pat);
        if stand_pat >= beta {
            qsearch_store(
                context,
                use_tt,
                key,
                stand_pat,
                alpha_start,
                beta,
                None,
                Some(raw_eval),
                ply,
            );
            return Some(terminal_outcome(stand_pat, false));
        }
        alpha = alpha.max(stand_pat);
        Some(stand_pat)
    };

    let pv_move = previous_pv.last().map(|pv| pv.mv);
    let tt_move = tt_entry.and_then(|entry| entry.best_move);
    let priority_move = priority_move_for_node(board, pv_move, tt_move, in_check);
    let filter = if in_check {
        MoveFilter::All
    } else {
        MoveFilter::Tactical
    };
    let mut moves = MovePicker::new();
    collect_moves_into(
        board,
        filter,
        priority_move,
        previous_move,
        ply,
        &mut moves,
    );
    let mut best = SearchOutcome {
        score: stand_pat.unwrap_or(i32::MIN),
        repetition_draw: false,
        pv: Vec::new(),
    };
    let mut interrupted = false;
    let mut found_move = false;
    let mut searched_moves = 0_u32;

    while let Some(ordered) = moves.next(board, context.ordering()) {
        if context.should_stop().is_some() {
            interrupted = true;
            break;
        }
        found_move = true;
        if in_check
            && searched_moves >= QSEARCH_MAX_EVASION_MOVES
            && Some(ordered.mv) != pv_move
        {
            continue;
        }
        if !in_check
            && Some(ordered.mv) != pv_move
            && let Some(stand_pat) = stand_pat
            && let Some(captured_piece) = ordered.captured_piece
            && should_q_delta_prune_capture(
                stand_pat,
                alpha,
                captured_piece,
                ordered.mv.promotion,
                ordered.moving_piece,
            )
            && !move_gives_check(board, ordered.mv, ordered.moving_piece, Some(captured_piece))
        {
            continue;
        }

        let mut next = board.clone();
        crate::chess::play_unchecked(&mut next, ordered.mv);
        let next_key = position_key(&next);
        let next_repetition = context.push_position(&next, next_key);
        context.push_eval_state(board, &next, ordered.mv);
        let child_correction_context =
            correction_context.after_move(ordered.mv, ordered.moving_piece);
        let child_pv = if Some(ordered.mv) == pv_move && !previous_pv.is_empty() {
            &previous_pv[..previous_pv.len() - 1]
        } else {
            &[]
        };
        let Some(child) = quiescence(
            &next,
            next_repetition,
            -beta,
            -alpha,
            Some(ordered.mv),
            child_correction_context,
            child_pv,
            context,
            ply + 1,
        ) else {
            context.pop_eval_state(board, ordered.mv);
            context.pop_position(next_key);
            interrupted = true;
            break;
        };
        context.note_searched_move(ply + 1);
        searched_moves += 1;
        context.pop_eval_state(board, ordered.mv);
        context.pop_position(next_key);
        let child_score = -child.score;
        let raised_alpha = is_better_score(child_score, child.repetition_draw, &best);
        if raised_alpha {
            best = parent_outcome(board, ordered.mv, child, context.chess960());
        }
        alpha = alpha.max(child_score);
        let caused_cutoff = alpha >= beta;
        if caused_cutoff {
            break;
        }
    }

    if interrupted {
        None
    } else if found_move {
        if !best.repetition_draw {
            qsearch_store(
                context,
                use_tt,
                key,
                best.score,
                alpha_start,
                beta,
                best.pv.last().map(|pv| pv.mv),
                raw_static_eval.or(raw_stand_pat),
                ply,
            );
        }
        Some(best)
    } else if let Some(stand_pat) = stand_pat {
        qsearch_store(
            context,
            use_tt,
            key,
            stand_pat,
            alpha_start,
            beta,
            None,
            raw_static_eval.or(raw_stand_pat),
            ply,
        );
        Some(terminal_outcome(stand_pat, false))
    } else {
        let score = LOSS_SCORE.saturating_add(ply as i32);
        qsearch_store(
            context,
            use_tt,
            key,
            score,
            alpha_start,
            beta,
            None,
            None,
            ply,
        );
        Some(terminal_outcome(score, false))
    }
}

pub(in crate::search) fn qsearch_tt_cutoff(
    board: &Board,
    entry: TranspositionEntry,
    alpha: i32,
    beta: i32,
    ply: u16,
    context: &mut SearchContext<'_>,
) -> Option<SearchOutcome> {
    let score = score_from_tt(entry.score, ply);
    match entry.bound {
        Bound::Exact => {
            let pv = entry
                .best_move
                .filter(|&mv| crate::chess::is_legal(board, mv))
                .map(|mv| vec![PvMove::new(board, mv, context.chess960())])
                .unwrap_or_default();
            Some(SearchOutcome {
                score,
                repetition_draw: false,
                pv,
            })
        }
        Bound::Lower if score >= beta => {
            Some(terminal_outcome(score, false))
        }
        Bound::Upper if score <= alpha => {
            Some(terminal_outcome(score, false))
        }
        _ => {
            None
        }
    }
}

pub(in crate::search) fn qsearch_store(
    context: &mut SearchContext<'_>,
    use_tt: bool,
    key: PositionKey,
    score: i32,
    alpha_start: i32,
    beta: i32,
    best_move: Option<Move>,
    static_eval: Option<i32>,
    ply: u16,
) {
    if !use_tt {
        return;
    }
    let bound = if score <= alpha_start {
        Bound::Upper
    } else if score >= beta {
        Bound::Lower
    } else {
        Bound::Exact
    };
    context
        .transposition_table()
        .store(key, 0, score, bound, best_move, static_eval, ply);
}

#[inline]
pub(in crate::search) fn qsearch_at_safety_bound(ply: u16) -> bool {
    ply as usize >= MAX_ORDERING_PLY
}

#[inline]
pub(in crate::search) fn qsearch_safety_bound_score(
    board: &Board,
    in_check: bool,
    correction_context: CorrectionContext,
    context: &mut SearchContext<'_>,
    ply: u16,
) -> (i32, Option<i32>) {
    if in_check {
        (LOSS_SCORE.saturating_add(ply as i32), None)
    } else {
        let raw_eval = context.evaluate(board);
        (
            context.corrected_static_eval(board, raw_eval, correction_context),
            Some(raw_eval),
        )
    }
}
