use crate::{Board, Move, evaluation::DRAW_SCORE, protocol::uci::format_uci_move_for_board};

use super::super::constants::{
    DRAW_PREFERENCE_MAX_SCORE, MAX_PV_LENGTH, ROOT_REPETITION_DEFER_MIN_SCORE,
};

#[derive(Clone, Debug, Default)]
pub(in crate::search) struct SearchOutcome {
    pub(in crate::search) score: i32,
    pub(in crate::search) repetition_draw: bool,
    pub(in crate::search) pv: Vec<PvMove>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search) struct PvMove {
    pub(in crate::search) mv: Move,
    pub(in crate::search) uci: Option<[u8; 5]>,
    pub(in crate::search) uci_len: usize,
}

impl PvMove {
    pub(in crate::search) fn new(board: &Board, mv: Move, chess960: bool) -> Self {
        let uci = format_uci_move_for_board(board, mv, chess960);
        let mut bytes = [0; 5];
        let uci_len = uci.len().min(bytes.len());
        bytes[..uci_len].copy_from_slice(&uci.as_bytes()[..uci_len]);
        Self {
            mv,
            uci: Some(bytes),
            uci_len,
        }
    }

    pub(in crate::search) fn to_uci(self) -> String {
        self.uci
            .map(|bytes| String::from_utf8_lossy(&bytes[..self.uci_len]).into_owned())
            .unwrap_or_else(|| self.mv.to_string())
    }
}

#[inline]
pub(in crate::search) fn is_better_score(score: i32, repetition_draw: bool, best: &SearchOutcome) -> bool {
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
        && best.score < DRAW_PREFERENCE_MAX_SCORE
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
        && previous_score > ROOT_REPETITION_DEFER_MIN_SCORE
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

pub(in crate::search) fn parent_outcome(
    board: &Board,
    mv: Move,
    child: SearchOutcome,
    chess960: bool,
) -> SearchOutcome {
    let mut pv = Vec::with_capacity(1 + child.pv.len().min(MAX_PV_LENGTH - 1));
    pv.push(PvMove::new(board, mv, chess960));
    if !child.repetition_draw {
        pv.extend(child.pv.into_iter().take(MAX_PV_LENGTH - 1));
    }
    SearchOutcome {
        score: -child.score,
        repetition_draw: child.repetition_draw,
        pv,
    }
}

pub(in crate::search) fn debug_validate_pv(board: &Board, pv: &[PvMove], tag: &str) {
    let mut b = board.clone();
    for (i, pm) in pv.iter().enumerate() {
        if !b.is_legal(pm.mv) {
            let seq: Vec<String> = pv.iter().map(|p| p.mv.to_string()).collect();
            eprintln!(
                "PVBUG[{tag}] illegal move #{i} {} from {} | pv: {}",
                pm.mv,
                b.to_string(),
                seq.join(" ")
            );
            return;
        }
        b.play_unchecked(pm.mv);
    }
}
