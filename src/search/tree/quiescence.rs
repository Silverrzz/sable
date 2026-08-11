use crate::{Board, Move, chess::MoveGenState, evaluation::LOSS_SCORE};

use super::{
    pruning::{apply_mate_distance_pruning, should_q_delta_prune_capture},
    scoring::terminal_score,
};
use crate::search::{
    constants::*,
    moves::{
        move_generation::{MoveFilter, collect_moves_into, priority_move_for_node},
        move_ordering::MovePicker,
        see::move_gives_check,
    },
    root::outcome::{SearchOutcome, is_better_score, parent_outcome, terminal_outcome},
    state::{
        context::SearchContext,
        correction_history::CorrectionContext,
        position_key::{PositionKey, position_key},
        transposition::{Bound, TranspositionEntry, score_from_tt},
    },
};

pub(in crate::search) fn quiescence(
    board: &Board,
    repetition: bool,
    mut alpha: i32,
    mut beta: i32,
    previous_move: Option<Move>,
    correction_context: CorrectionContext,
    previous_pv: &[Move],
    context: &mut SearchContext<'_>,
    ply: u16,
) -> Option<SearchOutcome> {
    context.enter_node(ply);
    context.clear_static_eval_at_ply(ply);
    if context.should_stop() {
        return None;
    }
    let movegen = MoveGenState::new(board);
    if let Some(score) = terminal_score(board, &movegen, repetition, ply) {
        return Some(terminal_outcome(score, repetition));
    }
    let in_check = movegen.in_check();
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
        && let Some(outcome) = qsearch_tt_cutoff(board, entry, alpha, beta, ply)
    {
        return Some(outcome);
    }
    let alpha_start = alpha;
    if ply as usize >= MAX_ORDERING_PLY {
        let (score, raw_static_eval) = if in_check {
            (LOSS_SCORE.saturating_add(ply as i32), None)
        } else {
            let raw = context.evaluate(board);
            (
                context.corrected_static_eval(board, raw, correction_context),
                Some(raw),
            )
        };
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

    let pv_move = previous_pv.last().copied();
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
        &movegen,
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
    let mut found_move = false;
    let mut searched_moves = 0_u32;

    while let Some(ordered) = moves.next(board, context.ordering()) {
        found_move = true;
        if in_check && searched_moves >= QSEARCH_MAX_EVASION_MOVES() && Some(ordered.mv) != pv_move
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
            && !move_gives_check(
                board,
                ordered.mv,
                ordered.moving_piece,
                Some(captured_piece),
            )
        {
            continue;
        }

        let mut next = board.clone();
        crate::chess::play_generated_move_unchecked(
            &mut next,
            ordered.mv,
            ordered.moving_piece,
            ordered.captured_piece,
        );
        let next_key = position_key(&next);
        context.transposition_table().prefetch(next_key);
        let next_repetition = context.push_position(&next, next_key);
        context.push_eval_state(
            &next,
            ordered.mv,
            ordered.moving_piece,
            ordered.captured_piece,
        );
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
            context.pop_eval_state();
            context.pop_position(next_key);
            return None;
        };
        searched_moves += 1;
        context.pop_eval_state();
        context.pop_position(next_key);
        let child_score = -child.score;
        let raised_alpha = is_better_score(child_score, child.repetition_draw, &best);
        if raised_alpha {
            best = parent_outcome(ordered.mv, child);
        }
        alpha = alpha.max(child_score);
        let caused_cutoff = alpha >= beta;
        if caused_cutoff {
            break;
        }
    }

    if found_move {
        if !best.repetition_draw {
            qsearch_store(
                context,
                use_tt,
                key,
                best.score,
                alpha_start,
                beta,
                best.pv.last().copied(),
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

fn qsearch_tt_cutoff(
    board: &Board,
    entry: TranspositionEntry,
    alpha: i32,
    beta: i32,
    ply: u16,
) -> Option<SearchOutcome> {
    let score = score_from_tt(entry.score, ply);
    match entry.bound {
        Bound::Exact => {
            let pv = entry
                .best_move
                .filter(|&mv| crate::chess::is_legal(board, mv))
                .map(|mv| vec![mv])
                .unwrap_or_default();
            Some(SearchOutcome {
                score,
                repetition_draw: false,
                pv,
            })
        }
        Bound::Lower if score >= beta => Some(terminal_outcome(score, false)),
        Bound::Upper if score <= alpha => Some(terminal_outcome(score, false)),
        _ => None,
    }
}

fn qsearch_store(
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
