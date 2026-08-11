use crate::{Board, Move, evaluation::DRAW_SCORE};

use super::super::constants::{
    DRAW_PREFERENCE_MAX_SCORE, MAX_PV_LENGTH, ROOT_REPETITION_DEFER_MIN_SCORE,
};

#[cfg(debug_assertions)]
use crate::GameStatus;

#[derive(Clone, Debug, Default)]
pub(in crate::search) struct SearchOutcome {
    pub(in crate::search) score: i32,
    pub(in crate::search) repetition_draw: bool,
    pub(in crate::search) pv: Vec<Move>,
}

#[inline]
pub(in crate::search) fn is_better_score(
    score: i32,
    repetition_draw: bool,
    best: &SearchOutcome,
) -> bool {
    if score != best.score {
        return score > best.score;
    }
    if score > DRAW_SCORE {
        return !repetition_draw && best.repetition_draw;
    }
    if score == DRAW_SCORE {
        return repetition_draw && !best.repetition_draw;
    }
    false
}

#[inline]
pub(in crate::search) fn is_better_root_score(
    score: i32,
    repetition_draw: bool,
    best: &SearchOutcome,
    root_repetitions: u8,
) -> bool {
    if is_better_score(score, repetition_draw, best) {
        return true;
    }
    let draw_pressure = root_repetitions == 2;
    if !draw_pressure {
        return false;
    }
    repetition_draw
        && score == DRAW_SCORE
        && best.score > DRAW_SCORE
        && best.score < DRAW_PREFERENCE_MAX_SCORE()
}

#[inline]
pub(in crate::search) fn is_better_root_outcome(
    candidate: &SearchOutcome,
    best: &SearchOutcome,
    root_repetitions: u8,
) -> bool {
    is_better_root_score(
        candidate.score,
        candidate.repetition_draw,
        best,
        root_repetitions,
    )
}

#[inline]
pub(in crate::search) fn should_defer_repetition_root_switch(
    completed_depth: u32,
    previous_move: Option<Move>,
    previous_score: i32,
    candidate_move: Move,
    candidate: &SearchOutcome,
) -> bool {
    let switched_move = matches!(previous_move, Some(best) if best != candidate_move);
    completed_depth > 0
        && switched_move
        && previous_score > ROOT_REPETITION_DEFER_MIN_SCORE()
        && candidate.score == DRAW_SCORE
        && candidate.repetition_draw
}

pub(in crate::search) fn terminal_outcome(score: i32, repetition_draw: bool) -> SearchOutcome {
    SearchOutcome {
        score,
        repetition_draw,
        pv: Vec::new(),
    }
}

pub(in crate::search) fn parent_outcome(mv: Move, child: SearchOutcome) -> SearchOutcome {
    let mut pv = child.pv;
    if pv.len() >= MAX_PV_LENGTH {
        pv.remove(0);
    }
    pv.push(mv);
    SearchOutcome {
        score: -child.score,
        repetition_draw: child.repetition_draw,
        pv,
    }
}

#[cfg(debug_assertions)]
pub(in crate::search) fn debug_validate_pv(board: &Board, pv: &[Move], tag: &str) {
    let mut b = board.clone();
    for (i, &mv) in pv.iter().rev().enumerate() {
        if crate::chess::status(&b) != GameStatus::Ongoing {
            let seq: Vec<String> = pv.iter().rev().map(ToString::to_string).collect();
            eprintln!(
                "PVBUG[{tag}] move #{i} {} continues from terminal {} | pv: {}",
                mv,
                b,
                seq.join(" ")
            );
            return;
        }
        if !crate::chess::is_legal(&b, mv) {
            let seq: Vec<String> = pv.iter().rev().map(ToString::to_string).collect();
            eprintln!(
                "PVBUG[{tag}] illegal move #{i} {} from {} | pv: {}",
                mv,
                b,
                seq.join(" ")
            );
            return;
        }
        crate::chess::play_unchecked(&mut b, mv);
    }
}

#[cfg(not(debug_assertions))]
#[inline]
pub(in crate::search) fn debug_validate_pv(_board: &Board, _pv: &[Move], _tag: &str) {}
