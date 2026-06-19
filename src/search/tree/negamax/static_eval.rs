use crate::{Board, Move};

use super::{
    PruneResult,
    super::{
        context::SearchContext,
        correction_history::CorrectionContext,
        pruning::{
            can_use_static_eval, can_use_static_eval_pruning, should_reverse_futility_prune,
            should_try_razoring,
        },
        quiescence::quiescence,
        root::terminal_outcome,
        transposition::TranspositionEntry,
    },
};

#[derive(Clone, Copy)]
pub(super) struct StaticEvalState {
    pub(super) raw: Option<i32>,
    pub(super) corrected: Option<i32>,
    pub(super) can_prune: bool,
    pub(super) improving: bool,
}

pub(super) struct StaticEvalParams<'a> {
    pub(super) board: &'a Board,
    pub(super) repetition: bool,
    pub(super) in_check: bool,
    pub(super) is_pv_node: bool,
    pub(super) alpha: i32,
    pub(super) beta: i32,
    pub(super) correction_context: CorrectionContext,
    pub(super) tt_entry: Option<TranspositionEntry>,
    pub(super) ply: u16,
}

pub(super) fn prepare_static_eval(
    params: StaticEvalParams<'_>,
    context: &mut SearchContext<'_>,
) -> StaticEvalState {
    let StaticEvalParams {
        board,
        repetition,
        in_check,
        is_pv_node,
        alpha,
        beta,
        correction_context,
        tt_entry,
        ply,
    } = params;

    let can_eval = can_use_static_eval(repetition, in_check, alpha, beta);
    let raw = if can_eval {
        Some(
            tt_entry
                .and_then(|entry| entry.static_eval())
                .unwrap_or_else(|| context.evaluate(board)),
        )
    } else {
        None
    };
    let corrected = raw.map(|raw_eval| {
        context.corrected_static_eval(board, raw_eval, correction_context)
    });
    let improving = corrected
        .map(|eval| context.is_static_eval_improving(ply, eval))
        .unwrap_or(false);
    if let Some(eval) = corrected {
        context.record_static_eval_at_ply(ply, eval);
    }
    StaticEvalState {
        raw,
        corrected,
        can_prune: can_use_static_eval_pruning(repetition, is_pv_node, in_check, alpha, beta),
        improving,
    }
}

pub(super) struct StaticPruningParams<'a> {
    pub(super) board: &'a Board,
    pub(super) repetition: bool,
    pub(super) depth: u32,
    pub(super) alpha: i32,
    pub(super) beta: i32,
    pub(super) previous_move: Option<Move>,
    pub(super) correction_context: CorrectionContext,
    pub(super) ply: u16,
}

pub(super) fn try_static_eval_pruning(
    params: StaticPruningParams<'_>,
    static_eval: StaticEvalState,
    context: &mut SearchContext<'_>,
) -> PruneResult {
    if !static_eval.can_prune {
        return PruneResult::Continue;
    }
    let Some(eval) = static_eval.corrected else {
        return PruneResult::Continue;
    };

    if let Some(score) = should_reverse_futility_prune(params.depth, eval, params.beta) {
        return PruneResult::Done(terminal_outcome(score, false));
    }

    if should_try_razoring(params.depth, eval, params.alpha) {
        let Some(razor) = quiescence(
            params.board,
            params.repetition,
            params.alpha,
            params.beta,
            params.previous_move,
            params.correction_context,
            &[],
            context,
            params.ply,
        ) else {
            return PruneResult::Interrupted;
        };
        if razor.score <= params.alpha {
            return PruneResult::Done(razor);
        }
    }

    PruneResult::Continue
}
