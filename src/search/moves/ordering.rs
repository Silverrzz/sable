use crate::{
    Board, Color, Move, Piece, Square,
    chess::{BitBoard, BoardParts},
};
use arrayvec::ArrayVec;

use super::{
    move_generation::{MoveFilter, tactical_move_score_with_history},
    see::static_exchange_eval_for_move,
};
use crate::search::constants::*;
use crate::search::state::move_context::{ContextMove, MoveContext};

const CONTINUATION_HISTORY_PLY_COUNT: usize = 2;
const CONTINUATION_HISTORY_SIZE: usize = CONTINUATION_HISTORY_PLY_COUNT * 6 * 64 * 6 * 64;

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
            history: vec![0; 2 * 2 * 2 * 64 * 64],
            continuation_history: vec![0; CONTINUATION_HISTORY_SIZE],
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

    pub(in crate::search) fn history_index(
        side: usize,
        from_threatened: usize,
        to_threatened: usize,
        from: usize,
        to: usize,
    ) -> usize {
        (((((side * 2) + from_threatened) * 2 + to_threatened) * 64) + from) * 64 + to
    }

    fn quiet_history_index(board: &Board, side: Color, mv: Move) -> usize {
        let parts = BoardParts::from_board(board);
        let enemy = !side;
        Self::history_index(
            side as usize,
            parts.is_square_attacked(mv.from, enemy) as usize,
            parts.is_square_attacked(mv.to, enemy) as usize,
            mv.from as usize,
            mv.to as usize,
        )
    }

    pub(in crate::search) fn continuation_index(
        distance_index: usize,
        previous_piece: usize,
        previous_to: usize,
        moving_piece: usize,
        move_to: usize,
    ) -> usize {
        (((((distance_index * 6) + previous_piece) * 64 + previous_to) * 6 + moving_piece) * 64)
            + move_to
    }

    #[inline]
    fn continuation_base(distance_index: usize, previous: ContextMove) -> usize {
        Self::continuation_index(
            distance_index,
            previous.piece as usize,
            previous.mv.to as usize,
            0,
            0,
        )
    }

    pub(in crate::search) fn capture_index(
        side: usize,
        moving_piece: usize,
        to: usize,
        captured_piece: usize,
    ) -> usize {
        (((side * 6) + moving_piece) * 64 + to) * 6 + captured_piece
    }

    pub(in crate::search) fn counter_move_index(side: usize, from: usize, to: usize) -> usize {
        ((side * 64) + from) * 64 + to
    }

    pub(in crate::search) fn quiet_score(
        &self,
        board: &Board,
        side: Color,
        mv: Move,
        move_context: MoveContext,
        ply: u16,
    ) -> i32 {
        let ply = ordering_ply(ply);
        let side_index = side as usize;
        if self.killers[ply][0] == Some(mv) {
            return FIRST_KILLER_SCORE;
        }
        if self.killers[ply][1] == Some(mv) {
            return SECOND_KILLER_SCORE;
        }
        if let Some(previous_move) = move_context.previous_move()
            && self.counter_moves[Self::counter_move_index(
                side_index,
                previous_move.from as usize,
                previous_move.to as usize,
            )] == Some(mv)
        {
            return COUNTER_MOVE_SCORE;
        }
        let moving_piece = crate::chess::piece_on(board, mv.from).unwrap_or(Piece::Pawn);
        let mut score = self.history[Self::quiet_history_index(board, side, mv)];
        score = score.saturating_add(self.continuation_score(move_context, moving_piece, mv));
        score.clamp(-MAX_HISTORY_SCORE, MAX_HISTORY_SCORE)
    }

    fn continuation_score(&self, move_context: MoveContext, moving_piece: Piece, mv: Move) -> i32 {
        let move_offset = moving_piece as usize * 64 + mv.to as usize;
        let previous = move_context.previous.map(|previous| {
            scaled_history_score(
                self.continuation_history[Self::continuation_base(0, previous) + move_offset],
                CONTINUATION_HISTORY_ORDERING_DIVISOR(),
            )
        });
        let previous_same_side = move_context.previous_same_side.map(|previous| {
            scaled_history_score(
                self.continuation_history[Self::continuation_base(1, previous) + move_offset],
                CONTINUATION_HISTORY_SAME_SIDE_ORDERING_DIVISOR(),
            )
        });
        previous
            .unwrap_or(0)
            .saturating_add(previous_same_side.unwrap_or(0))
    }

    fn update_continuation_history(
        &mut self,
        move_context: MoveContext,
        moving_piece: Piece,
        mv: Move,
        delta: i32,
    ) {
        let move_offset = moving_piece as usize * 64 + mv.to as usize;
        if let Some(previous) = move_context.previous {
            let index = Self::continuation_base(0, previous) + move_offset;
            update_history_value(&mut self.continuation_history[index], delta);
        }
        if let Some(previous) = move_context.previous_same_side {
            let index = Self::continuation_base(1, previous) + move_offset;
            update_history_value(&mut self.continuation_history[index], delta);
        }
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
        board: &Board,
        side: Color,
        mv: Move,
        move_context: MoveContext,
        depth: u32,
        ply: u16,
    ) {
        let side_index = side as usize;
        self.record_killer(mv, ply);
        if let Some(previous_move) = move_context.previous_move() {
            let index = Self::counter_move_index(
                side_index,
                previous_move.from as usize,
                previous_move.to as usize,
            );
            self.counter_moves[index] = Some(mv);
        }
        let bonus = history_bonus(depth);
        let history_index = Self::quiet_history_index(board, side, mv);
        update_history_value(&mut self.history[history_index], bonus);
        let moving_piece = crate::chess::piece_on(board, mv.from).unwrap_or(Piece::Pawn);
        self.update_continuation_history(move_context, moving_piece, mv, bonus);
    }

    pub(in crate::search) fn record_quiet_failure(
        &mut self,
        board: &Board,
        side: Color,
        move_context: MoveContext,
        mv: Move,
        moving_piece: Piece,
        depth: u32,
    ) {
        let malus = -history_malus(depth);
        let history_index = Self::quiet_history_index(board, side, mv);
        update_history_value(&mut self.history[history_index], malus);
        self.update_continuation_history(move_context, moving_piece, mv, malus);
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
        update_history_value(&mut self.capture_history[capture_index], bonus);
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
    pub(in crate::search) see: i16,
}

pub(in crate::search) const UNCACHED_SEE: i16 = i16::MIN;

#[inline]
pub(in crate::search) fn compact_see(see: i32) -> i16 {
    see.clamp(i32::from(i16::MIN) + 1, i32::from(i16::MAX)) as i16
}

const _: () = assert!(MAX_CANDIDATE_MOVES <= u8::MAX as usize + 1);
const _: () = assert!(std::mem::size_of::<CandidateMove>() == 8);

#[derive(Clone, Copy, Debug)]
struct QuietScoreContext {
    first_killer: Option<Move>,
    second_killer: Option<Move>,
    counter_move: Option<Move>,
    enemy_attacks: BitBoard,
    side: Color,
    continuation_bases: [Option<usize>; CONTINUATION_HISTORY_PLY_COUNT],
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
    score_heap: ArrayVec<u64, MAX_CANDIDATE_MOVES>,
    tactical_indices: ArrayVec<u8, MAX_CANDIDATE_MOVES>,
    quiet_indices: ArrayVec<u8, MAX_CANDIDATE_MOVES>,
    bad_tactical_indices: ArrayVec<u8, MAX_CANDIDATE_MOVES>,
    searched_indices: ArrayVec<u8, MAX_CANDIDATE_MOVES>,
    priority_index: Option<u8>,
    stage: MovePickerStage,
    priority_move: Option<Move>,
    side: Color,
    move_context: MoveContext,
    ply: u16,
    filter: MoveFilter,
    good_tacticals_heapified: bool,
    quiets_heapified: bool,
    bad_tacticals_heapified: bool,
}

impl MovePicker {
    pub(in crate::search) fn new() -> Self {
        Self {
            moves: ArrayVec::new(),
            score_heap: ArrayVec::new(),
            tactical_indices: ArrayVec::new(),
            quiet_indices: ArrayVec::new(),
            bad_tactical_indices: ArrayVec::new(),
            searched_indices: ArrayVec::new(),
            priority_index: None,
            stage: MovePickerStage::Done,
            priority_move: None,
            side: Color::White,
            move_context: MoveContext::default(),
            ply: 0,
            filter: MoveFilter::All,
            good_tacticals_heapified: false,
            quiets_heapified: false,
            bad_tacticals_heapified: false,
        }
    }

    pub(in crate::search) fn reset(
        &mut self,
        priority_move: Option<Move>,
        side: Color,
        move_context: MoveContext,
        ply: u16,
        filter: MoveFilter,
    ) {
        self.moves.clear();
        self.score_heap.clear();
        self.tactical_indices.clear();
        self.quiet_indices.clear();
        self.bad_tactical_indices.clear();
        self.searched_indices.clear();
        self.priority_index = None;
        self.stage = MovePickerStage::Priority;
        self.priority_move = priority_move;
        self.side = side;
        self.move_context = move_context;
        self.ply = ply;
        self.filter = filter;
        self.good_tacticals_heapified = false;
        self.quiets_heapified = false;
        self.bad_tacticals_heapified = false;
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
        assert!(
            self.moves.len() < MAX_CANDIDATE_MOVES,
            "move picker capacity exceeded"
        );
        let index = self.moves.len();
        self.moves.push(candidate);
        let index = index as u8;
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

    pub(in crate::search) fn searched_candidates(
        &self,
    ) -> impl Iterator<Item = CandidateMove> + '_ {
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
                        && let Some((index, score)) = self.next_quiet(board, ordering)
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
        self.searched_indices.push(index as u8);
        let candidate = self.get(index);
        ScoredMove {
            mv: candidate.mv,
            score,
            ordinal: index,
            is_quiet: candidate.is_quiet(),
            moving_piece: candidate.moving_piece,
            captured_piece: candidate.captured_piece,
            see: (candidate.see != UNCACHED_SEE).then_some(i32::from(candidate.see)),
        }
    }

    pub(in crate::search) fn next_quiet(
        &mut self,
        board: &Board,
        ordering: &MoveOrdering,
    ) -> Option<(usize, i32)> {
        if !self.quiets_heapified {
            self.score_heap.clear();
            let context = self.quiet_score_context(board, ordering);
            for position in 0..self.quiet_indices.len() {
                let index = self.quiet_indices[position] as usize;
                let candidate = self.get(index);
                let score = Self::quiet_score_for_candidate(ordering, context, candidate);
                self.score_heap.push(score_heap_entry(index as u8, score));
            }
            self.quiet_indices.clear();
            heapify_score_entries(&mut self.score_heap);
            self.quiets_heapified = true;
        }
        pop_score_entry(&mut self.score_heap)
    }

    #[inline]
    fn quiet_score_context(&self, board: &Board, ordering: &MoveOrdering) -> QuietScoreContext {
        let ply = ordering_ply(self.ply);
        let side = self.side as usize;
        let previous_move = self.move_context.previous_move();
        let counter_move = previous_move.and_then(|previous_move| {
            ordering
                .counter_moves
                .get(MoveOrdering::counter_move_index(
                    side,
                    previous_move.from as usize,
                    previous_move.to as usize,
                ))
                .copied()
                .flatten()
        });
        QuietScoreContext {
            first_killer: ordering.killers[ply][0],
            second_killer: ordering.killers[ply][1],
            counter_move,
            enemy_attacks: BoardParts::from_board(board).attacked_squares(!self.side),
            side: self.side,
            continuation_bases: [
                self.move_context
                    .previous
                    .map(|previous| MoveOrdering::continuation_base(0, previous)),
                self.move_context
                    .previous_same_side
                    .map(|previous| MoveOrdering::continuation_base(1, previous)),
            ],
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
        let move_offset = candidate.moving_piece as usize * 64 + mv.to as usize;
        let history_index = MoveOrdering::history_index(
            context.side as usize,
            context.enemy_attacks.has(mv.from) as usize,
            context.enemy_attacks.has(mv.to) as usize,
            mv.from as usize,
            mv.to as usize,
        );
        let mut score = ordering.history[history_index];
        if let Some(continuation_base) = context.continuation_bases[0] {
            score = score.saturating_add(scaled_history_score(
                ordering.continuation_history[continuation_base + move_offset],
                CONTINUATION_HISTORY_ORDERING_DIVISOR(),
            ));
        }
        if let Some(continuation_base) = context.continuation_bases[1] {
            score = score.saturating_add(scaled_history_score(
                ordering.continuation_history[continuation_base + move_offset],
                CONTINUATION_HISTORY_SAME_SIDE_ORDERING_DIVISOR(),
            ));
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
        if !self.good_tacticals_heapified {
            self.score_heap.clear();
            let mut position = 0;
            while position < self.tactical_indices.len() {
                let index = self.tactical_indices[position] as usize;
                let see = self.tactical_see(board, index);
                if see < 0 {
                    let index = self.tactical_indices.swap_remove(position);
                    self.bad_tactical_indices.push(index);
                    continue;
                }
                let score =
                    tactical_move_score_with_history(ordering, self.side, self.get(index), see);
                self.score_heap.push(score_heap_entry(index as u8, score));
                position += 1;
            }
            self.tactical_indices.clear();
            heapify_score_entries(&mut self.score_heap);
            self.good_tacticals_heapified = true;
        }
        pop_score_entry(&mut self.score_heap)
    }

    pub(in crate::search) fn best_bad_tactical(
        &mut self,
        board: &Board,
        ordering: &MoveOrdering,
    ) -> Option<(usize, i32)> {
        if !self.bad_tacticals_heapified {
            self.score_heap.clear();
            for position in 0..self.bad_tactical_indices.len() {
                let index = self.bad_tactical_indices[position] as usize;
                let see = self.tactical_see(board, index);
                let score =
                    tactical_move_score_with_history(ordering, self.side, self.get(index), see);
                self.score_heap.push(score_heap_entry(index as u8, score));
            }
            self.bad_tactical_indices.clear();
            heapify_score_entries(&mut self.score_heap);
            self.bad_tacticals_heapified = true;
        }
        pop_score_entry(&mut self.score_heap)
    }

    pub(in crate::search) fn tactical_see(&mut self, board: &Board, index: usize) -> i32 {
        let candidate = self.get(index);
        if candidate.see != UNCACHED_SEE {
            return i32::from(candidate.see);
        }
        let see = static_exchange_eval_for_move(
            board,
            candidate.mv,
            candidate.moving_piece,
            candidate.captured_piece,
        );
        self.get_mut(index).see = compact_see(see);
        see
    }
}

fn heapify_score_entries(entries: &mut [u64]) {
    for root in (0..entries.len() / 2).rev() {
        sift_score_entries_down(entries, root);
    }
}

fn pop_score_entry(entries: &mut ArrayVec<u64, MAX_CANDIDATE_MOVES>) -> Option<(usize, i32)> {
    let entry = *entries.first()?;
    let replacement = entries.pop().expect("non-empty move heap must pop");
    if !entries.is_empty() {
        entries[0] = replacement;
        sift_score_entries_down(entries, 0);
    }
    Some((score_heap_index(entry) as usize, score_heap_score(entry)))
}

fn sift_score_entries_down(entries: &mut [u64], mut root: usize) {
    let entry = entries[root];
    loop {
        let left = root * 2 + 1;
        if left >= entries.len() {
            break;
        }
        let right = left + 1;
        let better_child = if right < entries.len() && entries[right] > entries[left] {
            right
        } else {
            left
        };
        if entries[better_child] <= entry {
            break;
        }
        entries[root] = entries[better_child];
        root = better_child;
    }
    entries[root] = entry;
}

#[inline]
fn score_heap_entry(index: u8, score: i32) -> u64 {
    let ordered_score = (score as u32) ^ 0x8000_0000;
    (u64::from(ordered_score) << 8) | u64::from(u8::MAX - index)
}

#[inline]
fn score_heap_index(entry: u64) -> u8 {
    u8::MAX - entry as u8
}

#[inline]
fn score_heap_score(entry: u64) -> i32 {
    (((entry >> 8) as u32) ^ 0x8000_0000) as i32
}

pub(in crate::search) fn history_bonus(depth: u32) -> i32 {
    let depth = depth.min(64);
    depth
        .saturating_mul(depth)
        .saturating_mul(16)
        .max(16)
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
    if divisor <= 0 { 0 } else { score / divisor }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet_move(from: Square, to: Square) -> Move {
        Move {
            from,
            to,
            promotion: None,
        }
    }

    #[test]
    fn continuation_history_combines_previous_and_same_side_contexts() {
        let previous = ContextMove::new(quiet_move(Square::E7, Square::E5), Piece::Pawn);
        let previous_same_side =
            ContextMove::new(quiet_move(Square::G1, Square::F3), Piece::Knight);
        let move_context = MoveContext {
            previous: Some(previous),
            previous_same_side: Some(previous_same_side),
        };
        let mv = quiet_move(Square::F1, Square::B5);
        let mut ordering = MoveOrdering::default();
        let move_offset = Piece::Bishop as usize * 64 + mv.to as usize;
        let previous_index = MoveOrdering::continuation_base(0, previous) + move_offset;
        let same_side_index = MoveOrdering::continuation_base(1, previous_same_side) + move_offset;
        ordering.continuation_history[previous_index] = 120;
        ordering.continuation_history[same_side_index] = 80;

        let expected = scaled_history_score(120, CONTINUATION_HISTORY_ORDERING_DIVISOR())
            + scaled_history_score(80, CONTINUATION_HISTORY_SAME_SIDE_ORDERING_DIVISOR());
        assert_eq!(
            ordering.continuation_score(move_context, Piece::Bishop, mv),
            expected
        );
    }

    #[test]
    fn continuation_history_separates_distance_and_piece_context() {
        let one_ply = MoveOrdering::continuation_index(
            0,
            Piece::Pawn as usize,
            Square::E5 as usize,
            Piece::Bishop as usize,
            Square::B5 as usize,
        );
        let two_ply = MoveOrdering::continuation_index(
            1,
            Piece::Pawn as usize,
            Square::E5 as usize,
            Piece::Bishop as usize,
            Square::B5 as usize,
        );
        let different_piece = MoveOrdering::continuation_index(
            0,
            Piece::Knight as usize,
            Square::E5 as usize,
            Piece::Bishop as usize,
            Square::B5 as usize,
        );

        assert_ne!(one_ply, two_ply);
        assert_ne!(one_ply, different_piece);
        assert!(two_ply < CONTINUATION_HISTORY_SIZE);
    }
}
