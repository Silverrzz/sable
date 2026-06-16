use std::mem::MaybeUninit;

use crate::{
    Board, Color, Move, Piece, Square,
};

use super::{
    constants::*,
    move_generation::{MoveFilter, pick_better_move, tactical_move_score_with_history},
    see::static_exchange_eval,
};

#[derive(Clone, Debug)]
pub(in crate::search) struct MoveOrdering {
    killers: [[Option<Move>; KILLER_SLOTS]; MAX_ORDERING_PLY],
    history: Vec<i32>,
    continuation_history: Vec<i32>,
    capture_history: Vec<i32>,
    counter_moves: Vec<Option<Move>>,
}

#[inline]
pub(in crate::search) fn ordering_ply(ply: u16) -> usize {
    (ply as usize).min(MAX_ORDERING_PLY - 1)
}

impl Default for MoveOrdering {
    fn default() -> Self {
        Self {
            killers: [[None; KILLER_SLOTS]; MAX_ORDERING_PLY],
            history: vec![0; 2 * 64 * 64],
            continuation_history: vec![0; 2 * 64 * 64 * 64],
            capture_history: vec![0; 2 * 6 * 64 * 6],
            counter_moves: vec![None; 2 * 64 * 64],
        }
    }
}

impl MoveOrdering {
    pub(in crate::search) fn clear_search_local(&mut self) {
        self.killers = [[None; KILLER_SLOTS]; MAX_ORDERING_PLY];
    }

    pub(in crate::search) fn decay_persistent(&mut self) {
        decay_history_table(&mut self.history);
        decay_history_table(&mut self.continuation_history);
        decay_history_table(&mut self.capture_history);
    }

    pub(in crate::search) fn history_index(side: usize, from: usize, to: usize) -> usize {
        ((side * 64) + from) * 64 + to
    }

    pub(in crate::search) fn continuation_index(side: usize, previous_to: usize, from: usize, to: usize) -> usize {
        (((side * 64) + previous_to) * 64 + from) * 64 + to
    }

    pub(in crate::search) fn capture_index(side: usize, moving_piece: usize, to: usize, captured_piece: usize) -> usize {
        (((side * 6) + moving_piece) * 64 + to) * 6 + captured_piece
    }

    pub(in crate::search) fn counter_move_index(side: usize, from: usize, to: usize) -> usize {
        ((side * 64) + from) * 64 + to
    }

    pub(in crate::search) fn quiet_score(&self, side: Color, mv: Move, previous_move: Option<Move>, ply: u16) -> i32 {
        let ply = ordering_ply(ply);
        let side = side as usize;
        let from = mv.from as usize;
        let to = mv.to as usize;
        if self.killers[ply][0] == Some(mv) {
            return FIRST_KILLER_SCORE;
        }
        if self.killers[ply][1] == Some(mv) {
            return SECOND_KILLER_SCORE;
        }
        if let Some(previous_move) = previous_move
            && self.counter_moves[Self::counter_move_index(
                side,
                previous_move.from as usize,
                previous_move.to as usize,
            )] == Some(mv)
        {
            return COUNTER_MOVE_SCORE;
        }
        let mut score = self.history[Self::history_index(side, from, to)];
        if let Some(previous_move) = previous_move {
            let continuation_score = scaled_history_score(
                self.continuation_history[Self::continuation_index(
                    side,
                    previous_move.to as usize,
                    from,
                    to,
                )],
                CONTINUATION_HISTORY_ORDERING_DIVISOR,
            );
            score = score.saturating_add(continuation_score);
        }
        score.clamp(-MAX_HISTORY_SCORE, MAX_HISTORY_SCORE)
    }

    pub(in crate::search) fn capture_score(
        &self,
        side: Color,
        moving_piece: Piece,
        to: Square,
        captured_piece: Option<Piece>,
    ) -> i32 {
        let Some(captured_piece) = captured_piece else {
            return 0;
        };
        self.capture_history[Self::capture_index(
            side as usize,
            moving_piece as usize,
            to as usize,
            captured_piece as usize,
        )]
        .clamp(-MAX_HISTORY_SCORE, MAX_HISTORY_SCORE)
    }

    pub(in crate::search) fn record_quiet_cutoff(
        &mut self,
        side: Color,
        mv: Move,
        previous_move: Option<Move>,
        depth: u32,
        ply: u16,
    ) {
        let side = side as usize;
        self.record_killer(mv, ply);
        if let Some(previous_move) = previous_move {
            let index = Self::counter_move_index(
                side,
                previous_move.from as usize,
                previous_move.to as usize,
            );
            self.counter_moves[index] = Some(mv);
        }
        let bonus = history_bonus(depth);
        let history_index = Self::history_index(side, mv.from as usize, mv.to as usize);
        update_history_value(&mut self.history[history_index], bonus);
        if let Some(previous_move) = previous_move {
            let continuation_index = Self::continuation_index(
                side,
                previous_move.to as usize,
                mv.from as usize,
                mv.to as usize,
            );
            update_history_value(
                &mut self.continuation_history[continuation_index],
                bonus,
            );
        }
    }

    pub(in crate::search) fn record_quiet_failure(
        &mut self,
        side: Color,
        previous_move: Option<Move>,
        mv: Move,
        depth: u32,
    ) {
        let malus = -history_malus(depth);
        let side = side as usize;
        let history_index = Self::history_index(side, mv.from as usize, mv.to as usize);
        update_history_value(&mut self.history[history_index], malus);
        if let Some(previous_move) = previous_move {
            let continuation_index = Self::continuation_index(
                side,
                previous_move.to as usize,
                mv.from as usize,
                mv.to as usize,
            );
            update_history_value(
                &mut self.continuation_history[continuation_index],
                malus,
            );
        }
    }

    pub(in crate::search) fn record_capture_cutoff(
        &mut self,
        side: Color,
        moving_piece: Piece,
        mv: Move,
        captured_piece: Option<Piece>,
        depth: u32,
    ) {
        let Some(captured_piece) = captured_piece else {
            return;
        };
        let bonus = history_bonus(depth);
        let capture_index = Self::capture_index(
            side as usize,
            moving_piece as usize,
            mv.to as usize,
            captured_piece as usize,
        );
        update_history_value(
            &mut self.capture_history[capture_index],
            bonus,
        );
    }

    pub(in crate::search) fn record_capture_failure(
        &mut self,
        side: Color,
        mv: Move,
        moving_piece: Piece,
        captured_piece: Piece,
        depth: u32,
    ) {
        let malus = -history_malus(depth);
        let side = side as usize;
        let capture_index = Self::capture_index(
            side,
            moving_piece as usize,
            mv.to as usize,
            captured_piece as usize,
        );
        update_history_value(&mut self.capture_history[capture_index], malus);
    }

    pub(in crate::search) fn record_killer(&mut self, mv: Move, ply: u16) {
        let ply = ordering_ply(ply);
        if self.killers[ply][0] == Some(mv) {
            return;
        }
        self.killers[ply][1] = self.killers[ply][0];
        self.killers[ply][0] = Some(mv);
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::search) struct ScoredMove {
    pub(in crate::search) mv: Move,
    pub(in crate::search) score: i32,
    pub(in crate::search) ordinal: usize,
    pub(in crate::search) is_quiet: bool,
    pub(in crate::search) moving_piece: Piece,
    pub(in crate::search) captured_piece: Option<Piece>,
    pub(in crate::search) see: Option<i32>,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::search) struct CandidateMove {
    pub(in crate::search) mv: Move,
    pub(in crate::search) moving_piece: Piece,
    pub(in crate::search) captured_piece: Option<Piece>,
    pub(in crate::search) is_capture: bool,
    pub(in crate::search) is_promotion: bool,
    pub(in crate::search) ordinal: usize,
    pub(in crate::search) see: Option<i32>,
    pub(in crate::search) score: Option<i32>,
    pub(in crate::search) tried: bool,
}

impl CandidateMove {
    pub(in crate::search) fn is_quiet(self) -> bool {
        !self.is_capture && !self.is_promotion
    }

    pub(in crate::search) fn is_tactical(self) -> bool {
        self.is_capture || self.is_promotion
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search) enum MovePickerStage {
    Priority,
    Tactical,
    Quiet,
    Done,
}

pub(in crate::search) struct MovePicker {
    moves: [MaybeUninit<CandidateMove>; MAX_CANDIDATE_MOVES],
    pub(in crate::search) len: usize,
    stage: MovePickerStage,
    priority_move: Option<Move>,
    side: Color,
    previous_move: Option<Move>,
    ply: u16,
    filter: MoveFilter,
}

impl MovePicker {
    pub(in crate::search) fn new(
        priority_move: Option<Move>,
        side: Color,
        previous_move: Option<Move>,
        ply: u16,
        filter: MoveFilter,
    ) -> Self {
        Self {
            moves: [const { MaybeUninit::uninit() }; MAX_CANDIDATE_MOVES],
            len: 0,
            stage: MovePickerStage::Priority,
            priority_move,
            side,
            previous_move,
            ply,
            filter,
        }
    }

    pub(in crate::search) fn push(&mut self, candidate: CandidateMove) {
        assert!(self.len < MAX_CANDIDATE_MOVES, "move picker capacity exceeded");
        self.moves[self.len].write(candidate);
        self.len += 1;
    }

    pub(in crate::search) fn get(&self, index: usize) -> CandidateMove {
        debug_assert!(index < self.len);
        unsafe { self.moves[index].assume_init() }
    }

    pub(in crate::search) fn get_mut(&mut self, index: usize) -> &mut CandidateMove {
        debug_assert!(index < self.len);
        unsafe { self.moves[index].assume_init_mut() }
    }

    pub(in crate::search) fn next(
        &mut self,
        board: &Board,
        ordering: &MoveOrdering,
    ) -> Option<ScoredMove> {
        loop {
            match self.stage {
                MovePickerStage::Priority => {
                    self.stage = MovePickerStage::Tactical;
                    if let Some(mv) = self.priority_move {
                        for index in 0..self.len {
                            let candidate = self.get(index);
                            if !candidate.tried && candidate.mv == mv {
                                return Some(self.take_scored(index, PV_MOVE_SCORE));
                            }
                        }
                    }
                }
                MovePickerStage::Tactical => {
                    if let Some((index, score)) = self.best_tactical(board, ordering) {
                        return Some(self.take_scored(index, score));
                    }
                    self.stage = MovePickerStage::Quiet;
                }
                MovePickerStage::Quiet => {
                    if self.filter == MoveFilter::All
                        && let Some((index, score)) = self.best_quiet(ordering)
                    {
                        return Some(self.take_scored(index, score));
                    }
                    self.stage = MovePickerStage::Done;
                }
                MovePickerStage::Done => return None,
            }
        }
    }

    pub(in crate::search) fn take_scored(&mut self, index: usize, score: i32) -> ScoredMove {
        let candidate = self.get_mut(index);
        candidate.tried = true;
        ScoredMove {
            mv: candidate.mv,
            score,
            ordinal: candidate.ordinal,
            is_quiet: candidate.is_quiet(),
            moving_piece: candidate.moving_piece,
            captured_piece: candidate.captured_piece,
            see: candidate.see,
        }
    }

    pub(in crate::search) fn best_quiet(&mut self, ordering: &MoveOrdering) -> Option<(usize, i32)> {
        let mut best = None;
        for index in 0..self.len {
            let candidate = self.get(index);
            if candidate.tried || !candidate.is_quiet() {
                continue;
            }
            let score = if let Some(score) = candidate.score {
                score
            } else {
                let score = ordering.quiet_score(
                    self.side,
                    candidate.mv,
                    self.previous_move,
                    self.ply,
                );
                self.get_mut(index).score = Some(score);
                score
            };
            best = pick_better_move(best, index, score, candidate.ordinal);
        }
        best.map(|(index, score, _)| (index, score))
    }

    pub(in crate::search) fn best_tactical(
        &mut self,
        board: &Board,
        ordering: &MoveOrdering,
    ) -> Option<(usize, i32)> {
        let mut best = None;
        for index in 0..self.len {
            let candidate = self.get(index);
            if candidate.tried || !candidate.is_tactical() {
                continue;
            }
            let score = self.tactical_score(board, index, ordering);
            if self.filter == MoveFilter::Tactical
                && self.get(index).see.unwrap_or(0) < 0
            {
                continue;
            }
            best = pick_better_move(
                best,
                index,
                score,
                candidate.ordinal,
            );
        }
        best.map(|(index, score, _)| (index, score))
    }

    pub(in crate::search) fn tactical_see(&mut self, board: &Board, index: usize) -> i32 {
        if let Some(see) = self.get(index).see {
            return see;
        }
        let see = static_exchange_eval(board, self.get(index).mv);
        self.get_mut(index).see = Some(see);
        see
    }

    pub(in crate::search) fn tactical_score(&mut self, board: &Board, index: usize, ordering: &MoveOrdering) -> i32 {
        if let Some(score) = self.get(index).score {
            return score;
        }
        let see = self.tactical_see(board, index);
        let score = tactical_move_score_with_history(ordering, self.side, self.get(index), see);
        self.get_mut(index).score = Some(score);
        score
    }
}

pub(in crate::search) fn history_bonus(depth: u32) -> i32 {
    let depth = depth.min(64);
    depth.saturating_mul(depth).saturating_mul(16).max(16)
        .min(MAX_HISTORY_SCORE as u32) as i32
}

pub(in crate::search) fn history_malus(depth: u32) -> i32 {
    (history_bonus(depth) / 2).max(16)
}

pub(in crate::search) fn update_history_value(value: &mut i32, delta: i32) {
    let delta = delta.clamp(-MAX_HISTORY_SCORE, MAX_HISTORY_SCORE);
    let gravity = (*value as i64 * delta.abs() as i64) / MAX_HISTORY_SCORE as i64;
    *value = (*value as i64 + delta as i64 - gravity)
        .clamp(-MAX_HISTORY_SCORE as i64, MAX_HISTORY_SCORE as i64) as i32;
}

pub(in crate::search) fn decay_history_table(values: &mut [i32]) {
    for value in values {
        *value /= 2;
    }
}

pub(in crate::search) fn scaled_history_score(score: i32, divisor: i32) -> i32 {
    if divisor <= 0 {
        0
    } else {
        score / divisor
    }
}
