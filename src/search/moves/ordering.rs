use arrayvec::ArrayVec;
use crate::{
    Board, Color, Move, Piece, Square,
};

use super::{
    constants::*,
    move_generation::{MoveFilter, pick_better_move, tactical_move_score_with_history},
    see::static_exchange_eval_for_move,
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
    pub(in crate::search) ordinal: u16,
    pub(in crate::search) see: Option<i32>,
    pub(in crate::search) score: Option<i32>,
}

#[derive(Clone, Copy, Debug)]
struct QuietScoreContext {
    first_killer: Option<Move>,
    second_killer: Option<Move>,
    counter_move: Option<Move>,
    history_base: usize,
    continuation_base: Option<usize>,
}

impl CandidateMove {
    pub(in crate::search) fn is_quiet(self) -> bool {
        !self.is_tactical()
    }

    pub(in crate::search) fn is_tactical(self) -> bool {
        self.captured_piece.is_some() || self.mv.promotion.is_some()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search) enum MovePickerStage {
    Priority,
    GoodTactical,
    Quiet,
    BadTactical,
    Done,
}

pub(in crate::search) struct MovePicker {
    moves: ArrayVec<CandidateMove, MAX_CANDIDATE_MOVES>,
    tactical_indices: ArrayVec<u16, MAX_CANDIDATE_MOVES>,
    quiet_indices: ArrayVec<u16, MAX_CANDIDATE_MOVES>,
    bad_tactical_indices: ArrayVec<u16, MAX_CANDIDATE_MOVES>,
    searched_indices: ArrayVec<u16, MAX_CANDIDATE_MOVES>,
    priority_index: Option<u16>,
    stage: MovePickerStage,
    priority_move: Option<Move>,
    side: Color,
    previous_move: Option<Move>,
    ply: u16,
    filter: MoveFilter,
    quiets_sorted: bool,
}

impl MovePicker {
    pub(in crate::search) fn new() -> Self {
        Self {
            moves: ArrayVec::new(),
            tactical_indices: ArrayVec::new(),
            quiet_indices: ArrayVec::new(),
            bad_tactical_indices: ArrayVec::new(),
            searched_indices: ArrayVec::new(),
            priority_index: None,
            stage: MovePickerStage::Done,
            priority_move: None,
            side: Color::White,
            previous_move: None,
            ply: 0,
            filter: MoveFilter::All,
            quiets_sorted: false,
        }
    }

    pub(in crate::search) fn reset(
        &mut self,
        priority_move: Option<Move>,
        side: Color,
        previous_move: Option<Move>,
        ply: u16,
        filter: MoveFilter,
    ) {
        self.moves.clear();
        self.tactical_indices.clear();
        self.quiet_indices.clear();
        self.bad_tactical_indices.clear();
        self.searched_indices.clear();
        self.priority_index = None;
        self.stage = MovePickerStage::Priority;
        self.priority_move = priority_move;
        self.side = side;
        self.previous_move = previous_move;
        self.ply = ply;
        self.filter = filter;
        self.quiets_sorted = false;
    }

    #[inline]
    pub(in crate::search) fn push_tactical(&mut self, candidate: CandidateMove) {
        self.push_classified(candidate, true);
    }

    #[inline]
    pub(in crate::search) fn push_quiet(&mut self, candidate: CandidateMove) {
        self.push_classified(candidate, false);
    }

    #[inline]
    fn push_classified(&mut self, candidate: CandidateMove, is_tactical: bool) {
        assert!(self.moves.len() < MAX_CANDIDATE_MOVES, "move picker capacity exceeded");
        let index = self.moves.len();
        self.moves.push(candidate);
        let index = index as u16;
        if Some(candidate.mv) == self.priority_move && self.priority_index.is_none() {
            self.priority_index = Some(index);
        } else if is_tactical {
            self.tactical_indices.push(index);
        } else {
            self.quiet_indices.push(index);
        }
    }

    pub(in crate::search) fn get(&self, index: usize) -> CandidateMove {
        self.moves[index]
    }

    pub(in crate::search) fn get_mut(&mut self, index: usize) -> &mut CandidateMove {
        &mut self.moves[index]
    }

    pub(in crate::search) fn searched_candidates(&self) -> impl Iterator<Item = CandidateMove> + '_ {
        self.searched_indices
            .iter()
            .map(|&index| self.get(index as usize))
    }

    pub(in crate::search) fn next(
        &mut self,
        board: &Board,
        ordering: &MoveOrdering,
    ) -> Option<ScoredMove> {
        loop {
            match self.stage {
                MovePickerStage::Priority => {
                    self.stage = MovePickerStage::GoodTactical;
                    if let Some(index) = self.priority_index {
                        return Some(self.take_scored(index as usize, PV_MOVE_SCORE));
                    }
                }
                MovePickerStage::GoodTactical => {
                    if let Some((index, score)) = self.best_tactical(board, ordering, false) {
                        return Some(self.take_scored(index, score));
                    }
                    self.stage = MovePickerStage::Quiet;
                }
                MovePickerStage::Quiet => {
                    if self.filter == MoveFilter::All
                        && let Some((index, score)) = self.next_quiet(ordering)
                    {
                        return Some(self.take_scored(index, score));
                    }
                    self.stage = MovePickerStage::BadTactical;
                }
                MovePickerStage::BadTactical => {
                    if self.filter == MoveFilter::All
                        && let Some((index, score)) = self.best_tactical(board, ordering, true)
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
        self.searched_indices.push(index as u16);
        let candidate = self.get(index);
        ScoredMove {
            mv: candidate.mv,
            score,
            ordinal: candidate.ordinal as usize,
            is_quiet: candidate.is_quiet(),
            moving_piece: candidate.moving_piece,
            captured_piece: candidate.captured_piece,
            see: candidate.see,
        }
    }

    pub(in crate::search) fn next_quiet(&mut self, ordering: &MoveOrdering) -> Option<(usize, i32)> {
        self.sort_quiets_once(ordering);
        let index = self.quiet_indices.pop()? as usize;
        let score = self.get(index).score.unwrap_or(0);
        Some((index, score))
    }

    fn sort_quiets_once(&mut self, ordering: &MoveOrdering) {
        if self.quiets_sorted {
            return;
        }
        if self.quiet_indices.len() <= 1 {
            if let Some(&index) = self.quiet_indices.first() {
                let index = index as usize;
                let candidate = self.get(index);
                if candidate.score.is_none() {
                    let context = self.quiet_score_context(ordering);
                    let score = Self::quiet_score_for_candidate(ordering, context, candidate);
                    self.get_mut(index).score = Some(score);
                }
            }
            self.quiets_sorted = true;
            return;
        }
        let context = self.quiet_score_context(ordering);
        let mut scored = ArrayVec::<(u16, i32, u16), MAX_CANDIDATE_MOVES>::new();
        for position in 0..self.quiet_indices.len() {
            let index = self.quiet_indices[position];
            let index_usize = index as usize;
            let candidate = self.get(index_usize);
            let score = if let Some(score) = candidate.score {
                score
            } else {
                let score = Self::quiet_score_for_candidate(ordering, context, candidate);
                self.get_mut(index_usize).score = Some(score);
                score
            };
            scored.push((index, score, candidate.ordinal));
        }
        scored.sort_unstable_by(|(_, left_score, left_ordinal), (_, right_score, right_ordinal)| {
            left_score
                .cmp(right_score)
                .then_with(|| right_ordinal.cmp(left_ordinal))
        });
        self.quiet_indices.clear();
        for (index, _, _) in scored {
            self.quiet_indices.push(index);
        }
        self.quiets_sorted = true;
    }

    #[inline]
    fn quiet_score_context(&self, ordering: &MoveOrdering) -> QuietScoreContext {
        let ply = ordering_ply(self.ply);
        let side = self.side as usize;
        let previous_move = self.previous_move;
        let counter_move = previous_move.and_then(|previous_move| {
            ordering.counter_moves.get(MoveOrdering::counter_move_index(
                side,
                previous_move.from as usize,
                previous_move.to as usize,
            )).copied().flatten()
        });
        QuietScoreContext {
            first_killer: ordering.killers[ply][0],
            second_killer: ordering.killers[ply][1],
            counter_move,
            history_base: side * 64 * 64,
            continuation_base: previous_move.map(|previous_move| {
                ((side * 64) + previous_move.to as usize) * 64 * 64
            }),
        }
    }

    #[inline]
    fn quiet_score_for_candidate(
        ordering: &MoveOrdering,
        context: QuietScoreContext,
        candidate: CandidateMove,
    ) -> i32 {
        let mv = candidate.mv;
        if context.first_killer == Some(mv) {
            return FIRST_KILLER_SCORE;
        }
        if context.second_killer == Some(mv) {
            return SECOND_KILLER_SCORE;
        }
        if context.counter_move == Some(mv) {
            return COUNTER_MOVE_SCORE;
        }
        let move_offset = (mv.from as usize) * 64 + mv.to as usize;
        let mut score = ordering.history[context.history_base + move_offset];
        if let Some(continuation_base) = context.continuation_base {
            score = score.saturating_add(
                ordering.continuation_history[continuation_base + move_offset]
                    / CONTINUATION_HISTORY_ORDERING_DIVISOR,
            );
        }
        score.clamp(-MAX_HISTORY_SCORE, MAX_HISTORY_SCORE)
    }

    pub(in crate::search) fn best_tactical(
        &mut self,
        board: &Board,
        ordering: &MoveOrdering,
        bad_tactical: bool,
    ) -> Option<(usize, i32)> {
        if bad_tactical {
            return self.best_bad_tactical(board, ordering);
        }
        let mut best = None;
        let mut position = 0;
        while position < self.tactical_indices.len() {
            let index = self.tactical_indices[position] as usize;
            let candidate = self.get(index);
            let see = self.tactical_see(board, index);
            if see < 0 {
                let index = self.tactical_indices.swap_remove(position);
                self.bad_tactical_indices.push(index);
                continue;
            }
            let score = self.tactical_score(board, index, ordering);
            best = pick_better_move(
                best,
                position,
                score,
                candidate.ordinal,
            );
            position += 1;
        }
        best.map(|(position, score, _)| (self.tactical_indices.swap_remove(position) as usize, score))
    }

    pub(in crate::search) fn best_bad_tactical(
        &mut self,
        board: &Board,
        ordering: &MoveOrdering,
    ) -> Option<(usize, i32)> {
        let mut best = None;
        for position in 0..self.bad_tactical_indices.len() {
            let index = self.bad_tactical_indices[position] as usize;
            let candidate = self.get(index);
            let score = self.tactical_score(board, index, ordering);
            best = pick_better_move(
                best,
                position,
                score,
                candidate.ordinal,
            );
        }
        best.map(|(position, score, _)| (self.bad_tactical_indices.swap_remove(position) as usize, score))
    }

    pub(in crate::search) fn tactical_see(&mut self, board: &Board, index: usize) -> i32 {
        let candidate = self.get(index);
        if let Some(see) = candidate.see {
            return see;
        }
        let see = static_exchange_eval_for_move(
            board,
            candidate.mv,
            candidate.moving_piece,
            candidate.captured_piece,
        );
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
