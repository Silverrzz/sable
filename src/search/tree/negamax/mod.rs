mod tt;

use crate::{Board, Color, Move, chess::MoveGenState};

use super::{
    pruning::{
        apply_mate_distance_pruning, can_try_see_pruning, can_use_static_eval,
        can_use_static_eval_pruning, internal_iterative_reduction, is_see_prune_candidate,
        is_sparse_pawnless_endgame, late_move_reduction, null_move_reduction,
        requires_full_mate_search, should_futility_prune_quiet, should_prune_late_quiet,
        should_reverse_futility_prune, should_try_null_move, should_try_razoring,
        should_verify_null_move,
    },
    quiescence::quiescence,
    scoring::terminal_score,
};
use crate::search::{
    constants::*,
    moves::{
        move_generation::{MoveFilter, collect_moves_into, priority_move_for_node},
        move_ordering::{MovePicker, ScoredMove},
        see::{
            move_gives_check, static_exchange_eval_for_move, static_exchange_eval_for_quiet_move,
        },
    },
    root::outcome::{SearchOutcome, is_better_score, parent_outcome, terminal_outcome},
    state::{
        context::SearchContext,
        correction_history::{CorrectionContext, should_update_correction_history},
        position_key::position_key,
        transposition::{Bound, is_mate_score, score_from_tt},
    },
};

use tt::tt_cutoff;

#[derive(Clone, Copy)]
pub(super) struct StaticEvalState {
    pub(super) raw: Option<i32>,
    pub(super) corrected: Option<i32>,
    pub(super) can_prune: bool,
    pub(super) improving: bool,
}

pub(in crate::search) fn negamax(
    board: &Board,
    repetition: bool,
    depth: u32,
    mut alpha: i32,
    mut beta: i32,
    previous_pv: &[Move],
    previous_move: Option<Move>,
    correction_context: CorrectionContext,
    context: &mut SearchContext<'_>,
    ply: u16,
    allow_null_move: bool,
    excluded_move: Option<Move>,
) -> Option<SearchOutcome> {
    if depth == 0 {
        return quiescence(
            board,
            repetition,
            alpha,
            beta,
            previous_move,
            correction_context,
            previous_pv,
            context,
            ply,
        );
    }

    context.enter_node(ply);
    context.clear_static_eval_at_ply(ply);
    if context.should_stop() {
        return None;
    }
    let movegen = MoveGenState::new(board);
    if let Some(score) = terminal_score(board, &movegen, repetition, ply) {
        return Some(terminal_outcome(score, repetition));
    }
    let alpha_start = alpha;
    let is_pv_node = beta > alpha.saturating_add(1);
    let key = position_key(board);
    let use_tt = !repetition && excluded_move.is_none();
    let tt_entry = if use_tt {
        context.transposition_table().probe(key)
    } else {
        None
    };
    if let Some(outcome) = tt_cutoff(
        board, depth, alpha, beta, is_pv_node, tt_entry, context, ply,
    ) {
        return Some(outcome);
    }

    let in_check = movegen.in_check();
    let needs_full_mate_search = requires_full_mate_search(alpha, beta);
    let expected_cut_node = !is_pv_node && beta == alpha.saturating_add(1);
    let hash_move = tt_entry.and_then(|entry| entry.best_move);
    let iir = if excluded_move.is_none() {
        internal_iterative_reduction(
            depth,
            repetition,
            is_pv_node,
            expected_cut_node,
            in_check,
            needs_full_mate_search,
            hash_move.is_some(),
        )
    } else {
        0
    };
    let depth = depth.saturating_sub(iir);
    if let Some(score) = apply_mate_distance_pruning(&mut alpha, &mut beta, ply) {
        return Some(terminal_outcome(score, false));
    }
    let can_eval = can_use_static_eval(repetition, in_check, alpha, beta);
    let raw_static_eval = can_eval.then(|| {
        tt_entry
            .and_then(|entry| entry.static_eval())
            .unwrap_or_else(|| context.evaluate(board))
    });
    let corrected_static_eval =
        raw_static_eval.map(|raw| context.corrected_static_eval(board, raw, correction_context));
    let improving =
        corrected_static_eval.is_some_and(|eval| context.is_static_eval_improving(ply, eval));
    if let Some(eval) = corrected_static_eval {
        context.record_static_eval_at_ply(ply, eval);
    }
    let static_eval = StaticEvalState {
        raw: raw_static_eval,
        corrected: corrected_static_eval,
        can_prune: can_use_static_eval_pruning(repetition, is_pv_node, in_check, alpha, beta),
        improving,
    };

    if static_eval.can_prune
        && let Some(eval) = static_eval.corrected
    {
        if let Some(score) = should_reverse_futility_prune(depth, eval, beta) {
            return Some(terminal_outcome(score, false));
        }
        if should_try_razoring(depth, eval, alpha) {
            let razor = quiescence(
                board,
                repetition,
                alpha,
                beta,
                previous_move,
                correction_context,
                &[],
                context,
                ply,
            )?;
            if razor.score <= alpha {
                return Some(razor);
            }
        }
    }

    if !needs_full_mate_search
        && should_try_null_move(board, depth, is_pv_node, in_check, allow_null_move)
        && let Some(null_eval) = static_eval.corrected.filter(|eval| *eval >= beta)
        && let Some(null_board) = crate::chess::null_move(board)
    {
        let sparse_endgame = is_sparse_pawnless_endgame(board);
        let reduction = null_move_reduction(depth, null_eval, beta, sparse_endgame);
        let null_depth = depth.saturating_sub(1 + reduction);
        let null_alpha = beta.saturating_neg();
        let null_beta = null_alpha.saturating_add(1);
        context.push_null_eval_state(&null_board);
        let null_result = negamax(
            &null_board,
            repetition,
            null_depth,
            null_alpha,
            null_beta,
            &[],
            None,
            correction_context.without_move_context(),
            context,
            ply + 1,
            false,
            None,
        );
        context.pop_null_eval_state();
        let null_result = null_result?;
        if -null_result.score >= beta {
            let verified = if should_verify_null_move(depth, sparse_endgame) {
                negamax(
                    board,
                    repetition,
                    depth.saturating_sub(reduction),
                    beta.saturating_sub(1),
                    beta,
                    &[],
                    None,
                    correction_context,
                    context,
                    ply,
                    false,
                    None,
                )?
                .score
                    >= beta
            } else {
                true
            };
            if verified {
                return Some(terminal_outcome(beta, false));
            }
        }
    }

    if depth >= PROBCUT_MIN_DEPTH()
        && !is_pv_node
        && !in_check
        && !needs_full_mate_search
        && !repetition
        && excluded_move.is_none()
        && static_eval.corrected.is_some()
    {
        let probcut_beta = beta.saturating_add(PROBCUT_MARGIN());
        let child_alpha = probcut_beta.saturating_neg();
        let child_beta = child_alpha.saturating_add(1);
        let probcut_depth = depth.saturating_sub(PROBCUT_DEPTH_REDUCTION()).max(1);
        let tt_move = tt_entry.and_then(|entry| entry.best_move);
        let mut moves = MovePicker::new();
        collect_moves_into(
            board,
            &movegen,
            MoveFilter::Tactical,
            tt_move,
            previous_move,
            ply,
            &mut moves,
        );

        while let Some(ordered) = moves.next(board, context.ordering()) {
            let see = ordered.see.unwrap_or_else(|| {
                static_exchange_eval_for_move(
                    board,
                    ordered.mv,
                    ordered.moving_piece,
                    ordered.captured_piece,
                )
            });
            if see < PROBCUT_SEE_THRESHOLD() {
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

            let qsearch = quiescence(
                &next,
                next_repetition,
                child_alpha,
                child_beta,
                Some(ordered.mv),
                child_correction_context,
                &[],
                context,
                ply + 1,
            );
            let Some(qsearch) = qsearch else {
                context.pop_eval_state();
                context.pop_position(next_key);
                return None;
            };
            if -qsearch.score < probcut_beta {
                context.pop_eval_state();
                context.pop_position(next_key);
                continue;
            }

            let reduced = negamax(
                &next,
                next_repetition,
                probcut_depth,
                child_alpha,
                child_beta,
                &[],
                Some(ordered.mv),
                child_correction_context,
                context,
                ply + 1,
                true,
                None,
            );
            let Some(reduced) = reduced else {
                context.pop_eval_state();
                context.pop_position(next_key);
                return None;
            };
            context.pop_eval_state();
            context.pop_position(next_key);

            let score = -reduced.score;
            if score >= probcut_beta {
                return Some(terminal_outcome(score, false));
            }
        }
    }

    let side = crate::chess::side_to_move(board);
    let pv_move = previous_pv.last().copied();
    let tt_move = tt_entry.and_then(|entry| entry.best_move);
    let priority_move = priority_move_for_node(board, pv_move, tt_move, in_check);
    let mut moves = MovePicker::new();
    collect_moves_into(
        board,
        &movegen,
        MoveFilter::All,
        priority_move,
        previous_move,
        ply,
        &mut moves,
    );
    let mut best = SearchOutcome {
        score: i32::MIN,
        repetition_draw: false,
        pv: Vec::new(),
    };
    let mut searched_moves = 0_u32;
    let mut captures_tried = 0_u32;
    let child_depth = depth - 1;
    while let Some(ordered) = moves.next(board, context.ordering()) {
        if Some(ordered.mv) == excluded_move {
            continue;
        }
        let capture_prune = see_capture_prune(
            board,
            ordered,
            depth,
            is_pv_node,
            in_check,
            needs_full_mate_search,
            pv_move,
            captures_tried,
        );
        if capture_prune.pruned {
            captures_tried += 1;
            continue;
        }
        let mut next = board.clone();
        crate::chess::play_generated_move_unchecked(
            &mut next,
            ordered.mv,
            ordered.moving_piece,
            ordered.captured_piece,
        );
        let gives_check = capture_prune
            .gives_check
            .unwrap_or_else(|| !crate::chess::checkers(&next).is_empty());
        if should_static_prune_quiet(
            static_eval,
            depth,
            alpha,
            ordered,
            searched_moves,
            gives_check,
        ) || see_quiet_prune(
            board,
            ordered,
            depth,
            is_pv_node,
            in_check,
            needs_full_mate_search,
            pv_move,
            searched_moves,
            gives_check,
        ) {
            continue;
        }

        let extension = if excluded_move.is_some()
            || in_check
            || needs_full_mate_search
            || depth < SINGULAR_EXTENSION_MIN_DEPTH()
            || Some(ordered.mv) != tt_move
        {
            0
        } else if let Some(entry) = tt_entry
            && matches!(entry.bound, Bound::Lower | Bound::Exact)
            && u32::from(entry.depth).saturating_add(SINGULAR_EXTENSION_TT_DEPTH_MARGIN()) >= depth
            && !is_mate_score(score_from_tt(entry.score, ply))
        {
            let tt_score = score_from_tt(entry.score, ply);
            let singular_beta = tt_score.saturating_sub(SINGULAR_EXTENSION_BASE_MARGIN());
            let excluded = negamax(
                board,
                repetition,
                depth.saturating_sub(1) / 2,
                singular_beta.saturating_sub(1),
                singular_beta,
                &[],
                previous_move,
                correction_context,
                context,
                ply,
                false,
                Some(ordered.mv),
            )?;
            if excluded.score < singular_beta {
                let double_beta = tt_score.saturating_sub(DOUBLE_SINGULAR_EXTENSION_BASE_MARGIN());
                if !is_pv_node && excluded.score < double_beta {
                    2
                } else {
                    1
                }
            } else if !is_pv_node && singular_beta >= beta {
                return Some(terminal_outcome(singular_beta, false));
            } else {
                0
            }
        } else {
            0
        };

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
        let full_depth = child_depth.saturating_add(extension);
        let reduction = if needs_full_mate_search {
            0
        } else {
            late_move_reduction(
                depth,
                searched_moves,
                is_pv_node,
                ordered.is_quiet,
                in_check,
                gives_check,
                static_eval.improving,
                ordered.score,
            )
        };
        let scout_beta = alpha.saturating_neg();
        let scout_alpha = scout_beta.saturating_sub(1);
        let child = if reduction > 0 {
            match negamax(
                &next,
                next_repetition,
                full_depth.saturating_sub(reduction),
                scout_alpha,
                scout_beta,
                &[],
                Some(ordered.mv),
                child_correction_context,
                context,
                ply + 1,
                true,
                None,
            ) {
                Some(reduced) if -reduced.score <= alpha => Some(reduced),
                Some(_) => negamax(
                    &next,
                    next_repetition,
                    full_depth,
                    beta.saturating_neg(),
                    alpha.saturating_neg(),
                    child_pv,
                    Some(ordered.mv),
                    child_correction_context,
                    context,
                    ply + 1,
                    true,
                    None,
                ),
                None => None,
            }
        } else if is_pv_node && searched_moves > 0 && beta > alpha.saturating_add(1) {
            match negamax(
                &next,
                next_repetition,
                full_depth,
                scout_alpha,
                scout_beta,
                &[],
                Some(ordered.mv),
                child_correction_context,
                context,
                ply + 1,
                true,
                None,
            ) {
                Some(scout) if -scout.score > alpha && -scout.score < beta => negamax(
                    &next,
                    next_repetition,
                    full_depth,
                    beta.saturating_neg(),
                    alpha.saturating_neg(),
                    child_pv,
                    Some(ordered.mv),
                    child_correction_context,
                    context,
                    ply + 1,
                    true,
                    None,
                ),
                scout => scout,
            }
        } else {
            negamax(
                &next,
                next_repetition,
                full_depth,
                beta.saturating_neg(),
                alpha.saturating_neg(),
                child_pv,
                Some(ordered.mv),
                child_correction_context,
                context,
                ply + 1,
                true,
                None,
            )
        };
        let Some(child) = child else {
            context.pop_eval_state();
            context.pop_position(next_key);
            return None;
        };
        context.pop_eval_state();
        context.pop_position(next_key);
        if ordered.captured_piece.is_some() {
            captures_tried += 1;
        }
        searched_moves += 1;
        let child_score = -child.score;
        if is_better_score(child_score, child.repetition_draw, &best) {
            best = parent_outcome(ordered.mv, child);
        }
        alpha = alpha.max(best.score);
        if alpha >= beta {
            record_cutoff_and_failures(&moves, ordered, side, previous_move, depth, ply, context);
            break;
        }
    }

    if excluded_move.is_some() && best.score == i32::MIN {
        best = terminal_outcome(alpha, false);
    }

    let bound = if best.score <= alpha_start {
        Bound::Upper
    } else if best.score >= beta {
        Bound::Lower
    } else {
        Bound::Exact
    };
    let best_move = best.pv.last().copied();
    if use_tt
        && !best.repetition_draw
        && let Some(raw_eval) = static_eval.raw
        && let Some(corrected_eval) = static_eval.corrected
        && should_update_correction_history(board, best_move, bound, corrected_eval, best.score)
    {
        context.update_correction_history(board, correction_context, raw_eval, best.score, depth);
    }
    if use_tt && !best.repetition_draw {
        context.transposition_table().store(
            key,
            depth,
            best.score,
            bound,
            best_move,
            static_eval.raw,
            ply,
        );
    }
    Some(best)
}

struct SeeCapturePruneResult {
    gives_check: Option<bool>,
    pruned: bool,
}

fn see_capture_prune(
    board: &Board,
    ordered: ScoredMove,
    depth: u32,
    is_pv_node: bool,
    in_check: bool,
    needs_full_mate_search: bool,
    pv_move: Option<Move>,
    captures_tried: u32,
) -> SeeCapturePruneResult {
    if in_check || needs_full_mate_search || Some(ordered.mv) == pv_move {
        return SeeCapturePruneResult {
            gives_check: None,
            pruned: false,
        };
    }
    let Some(captured_piece) = ordered.captured_piece else {
        return SeeCapturePruneResult {
            gives_check: None,
            pruned: false,
        };
    };
    if !can_try_see_pruning(depth, is_pv_node, captures_tried) {
        return SeeCapturePruneResult {
            gives_check: None,
            pruned: false,
        };
    }
    let see = ordered.see.unwrap_or_else(|| {
        static_exchange_eval_for_move(
            board,
            ordered.mv,
            ordered.moving_piece,
            ordered.captured_piece,
        )
    });
    if !is_see_prune_candidate(depth, is_pv_node, captures_tried, see) {
        return SeeCapturePruneResult {
            gives_check: None,
            pruned: false,
        };
    }
    let gives_check = move_gives_check(
        board,
        ordered.mv,
        ordered.moving_piece,
        Some(captured_piece),
    );
    SeeCapturePruneResult {
        gives_check: Some(gives_check),
        pruned: !gives_check,
    }
}

fn see_quiet_prune(
    board: &Board,
    ordered: ScoredMove,
    depth: u32,
    is_pv_node: bool,
    in_check: bool,
    needs_full_mate_search: bool,
    pv_move: Option<Move>,
    searched_moves: u32,
    gives_check: bool,
) -> bool {
    if in_check
        || needs_full_mate_search
        || !ordered.is_quiet
        || gives_check
        || Some(ordered.mv) == pv_move
        || !can_try_see_pruning(depth, is_pv_node, searched_moves)
    {
        return false;
    }
    let see = static_exchange_eval_for_quiet_move(board, ordered.mv, ordered.moving_piece);
    is_see_prune_candidate(depth, is_pv_node, searched_moves, see)
}

fn should_static_prune_quiet(
    static_eval: StaticEvalState,
    depth: u32,
    alpha: i32,
    ordered: ScoredMove,
    searched_moves: u32,
    gives_check: bool,
) -> bool {
    if !static_eval.can_prune || !ordered.is_quiet || searched_moves == 0 || gives_check {
        return false;
    }
    let Some(eval) = static_eval.corrected else {
        return false;
    };
    should_futility_prune_quiet(depth, eval, alpha, ordered.score, static_eval.improving)
        || should_prune_late_quiet(depth, searched_moves, ordered.score, static_eval.improving)
}

fn record_cutoff_and_failures(
    moves: &MovePicker,
    ordered: ScoredMove,
    side: Color,
    previous_move: Option<Move>,
    depth: u32,
    ply: u16,
    context: &mut SearchContext<'_>,
) {
    if ordered.is_quiet {
        context
            .ordering_mut()
            .record_quiet_cutoff(side, ordered.mv, previous_move, depth, ply);
    } else if let Some(captured_piece) = ordered.captured_piece {
        context.ordering_mut().record_capture_cutoff(
            side,
            ordered.moving_piece,
            ordered.mv,
            Some(captured_piece),
            depth,
        );
    }

    for candidate in moves.searched_candidates() {
        if candidate.mv == ordered.mv {
            break;
        }
        if candidate.is_quiet() {
            context
                .ordering_mut()
                .record_quiet_failure(side, previous_move, candidate.mv, depth);
        } else if let Some(captured_piece) = candidate.captured_piece {
            context.ordering_mut().record_capture_failure(
                side,
                candidate.mv,
                candidate.moving_piece,
                captured_piece,
                depth,
            );
        }
    }
}
