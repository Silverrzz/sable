use std::sync::{
    Arc,
    atomic::{AtomicU8, AtomicU64, Ordering},
};

use crate::{
    Move, Piece, Square,
    evaluation::LOSS_SCORE,
};

use super::{constants::*, position_key::PositionKey};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::search) enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::search) struct TranspositionEntry {
    pub(in crate::search) depth: u8,
    pub(in crate::search) score: i32,
    pub(in crate::search) bound: Bound,
    pub(in crate::search) best_move: Option<Move>,
    pub(in crate::search) static_eval: i32,
    pub(in crate::search) age: u8,
}

impl TranspositionEntry {
    pub(in crate::search) fn static_eval(self) -> Option<i32> {
        (self.static_eval != NO_STATIC_EVAL).then_some(self.static_eval)
    }
}

#[derive(Debug)]
pub(in crate::search) struct AtomicTtEntry {
    key_xor_data: AtomicU64,
    data: AtomicU64,
}

impl AtomicTtEntry {
    const fn empty() -> Self {
        Self {
            key_xor_data: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }

    pub(in crate::search) fn load(&self) -> Option<(PositionKey, u64)> {
        let data = self.data.load(Ordering::Relaxed);
        if data == 0 {
            return None;
        }
        let key_xor_data = self.key_xor_data.load(Ordering::Relaxed);
        Some((key_xor_data ^ data, data))
    }

    pub(in crate::search) fn load_for_key(&self, key: PositionKey) -> Option<TranspositionEntry> {
        // writers publish the two words separately, so torn reads can happen
        // that's fine: the rebuilt key misses and the reader shrugs it off
        let (entry_key, data) = self.load()?;
        if entry_key != key {
            return None;
        }
        decode_tt_data(data)
    }

    pub(in crate::search) fn store(&self, key: PositionKey, data: u64) {
        debug_assert_ne!(data, 0);
        self.data.store(data, Ordering::Relaxed);
        self.key_xor_data.store(key ^ data, Ordering::Relaxed);
    }
}

#[repr(align(64))]
#[derive(Debug)]
pub(in crate::search) struct Cluster {
    entries: [AtomicTtEntry; TT_CLUSTER_SIZE],
}

impl Cluster {
    const fn empty() -> Self {
        Self {
            entries: [
                AtomicTtEntry::empty(),
                AtomicTtEntry::empty(),
                AtomicTtEntry::empty(),
                AtomicTtEntry::empty(),
            ],
        }
    }
}

#[derive(Debug)]
pub(in crate::search) struct TranspositionTableInner {
    clusters: Vec<Cluster>,
    age: AtomicU8,
}

#[derive(Clone, Debug)]
pub(crate) struct TranspositionTable {
    inner: Arc<TranspositionTableInner>,
}

impl TranspositionTable {
    pub(crate) fn new(hash_mb: u32) -> Self {
        debug_assert_eq!(std::mem::size_of::<Cluster>(), TT_CLUSTER_BYTES);
        let bytes = (hash_mb as usize).saturating_mul(1024 * 1024);
        let len = (bytes / TT_CLUSTER_BYTES).max(1);
        let clusters = (0..len).map(|_| Cluster::empty()).collect();
        Self {
            inner: Arc::new(TranspositionTableInner {
                clusters,
                age: AtomicU8::new(0),
            }),
        }
    }

    pub(in crate::search) fn next_age(&self) {
        let next = self.current_age().wrapping_add(1) & TT_AGE_MASK;
        self.inner.age.store(next, Ordering::Relaxed);
    }

    pub(in crate::search) fn probe(&self, key: PositionKey) -> Option<TranspositionEntry> {
        let cluster = &self.inner.clusters[self.cluster_index(key)];
        for entry in &cluster.entries {
            let Some(packed) = entry.load_for_key(key) else {
                continue;
            };
            return Some(packed);
        }
        None
    }

    pub(in crate::search) fn store(
        &self,
        key: PositionKey,
        depth: u32,
        score: i32,
        bound: Bound,
        best_move: Option<Move>,
        static_eval: Option<i32>,
        ply: u16,
    ) {
        let age = self.current_age();
        let depth = depth.min(u32::from(u8::MAX)) as u8;
        let data = encode_tt_data(
            depth,
            score_to_tt(score, ply),
            bound,
            best_move,
            static_eval,
            age,
        );
        let cluster = &self.inner.clusters[self.cluster_index(key)];
        let mut victim_index = 0;
        let mut victim_priority = i32::MAX;

        for (index, entry) in cluster.entries.iter().enumerate() {
            let Some((entry_key, entry_data)) = entry.load() else {
                entry.store(key, data);
                return;
            };
            let Some(existing) = decode_tt_data(entry_data) else {
                entry.store(key, data);
                return;
            };
            if entry_key == key {
                if should_overwrite_same_key(existing, depth, bound, age) {
                    entry.store(key, data);
                }
                return;
            }

            let priority = replacement_priority(existing.depth, relative_age(age, existing.age));
            if priority < victim_priority {
                victim_index = index;
                victim_priority = priority;
            }
        }

        cluster.entries[victim_index].store(key, data);
    }

    pub(in crate::search) fn hashfull(&self) -> u16 {
        let sampled_clusters = self.inner.clusters.len().min(1000);
        if sampled_clusters == 0 {
            return 0;
        }
        let age = self.current_age();
        let mut used = 0_usize;
        for cluster in self.inner.clusters.iter().take(sampled_clusters) {
            for entry in &cluster.entries {
                let data = entry.data.load(Ordering::Relaxed);
                if data != 0 && packed_age(data) == age {
                    used += 1;
                }
            }
        }
        let sampled_entries = sampled_clusters * TT_CLUSTER_SIZE;
        ((used.saturating_mul(1000)) / sampled_entries).min(1000) as u16
    }

    pub(in crate::search) fn cluster_index(&self, key: PositionKey) -> usize {
        ((u128::from(key) * self.inner.clusters.len() as u128) >> 64) as usize
    }

    pub(in crate::search) fn current_age(&self) -> u8 {
        self.inner.age.load(Ordering::Relaxed) & TT_AGE_MASK
    }
}

pub(in crate::search) const TT_CLUSTER_SIZE: usize = 4;
pub(in crate::search) const TT_CLUSTER_BYTES: usize = 64;
pub(in crate::search) const TT_AGE_MASK: u8 = 0x3f;
pub(in crate::search) const TT_MOVE_MASK: u64 = 0xffff;
pub(in crate::search) const TT_STATIC_EVAL_NONE: i16 = i16::MIN;

pub(in crate::search) fn should_overwrite_same_key(
    existing: TranspositionEntry,
    depth: u8,
    bound: Bound,
    _age: u8,
) -> bool {
    (bound == Bound::Exact && u16::from(depth) + 2 >= u16::from(existing.depth))
        || u16::from(depth) + 4 >= u16::from(existing.depth)
}

pub(in crate::search) fn replacement_priority(depth: u8, age_distance: u8) -> i32 {
    i32::from(depth) - 8 * i32::from(age_distance)
}

pub(in crate::search) fn relative_age(current: u8, entry_age: u8) -> u8 {
    current.wrapping_sub(entry_age) & TT_AGE_MASK
}

pub(in crate::search) fn encode_tt_data(
    depth: u8,
    score: i32,
    bound: Bound,
    best_move: Option<Move>,
    static_eval: Option<i32>,
    age: u8,
) -> u64 {
    let mut data = u64::from(encode_optional_move(best_move))
        | (u64::from(pack_tt_score(score) as u16) << 16)
        | (u64::from(pack_static_eval(static_eval) as u16) << 32)
        | (u64::from(depth) << 48)
        | ((bound as u64) << 56)
        | (u64::from(age & TT_AGE_MASK) << 58);

    // zero means empty, so if the packed payload is somehow zero, drop only static eval
    // slot stays non-empty and everyone moves on
    if data == 0 {
        data = u64::from(TT_STATIC_EVAL_NONE as u16) << 32;
    }
    data
}

pub(in crate::search) fn decode_tt_data(data: u64) -> Option<TranspositionEntry> {
    if data == 0 {
        return None;
    }
    let bound = match (data >> 56) & 0x3 {
        0 => Bound::Exact,
        1 => Bound::Lower,
        2 => Bound::Upper,
        _ => return None,
    };
    let best_move = decode_optional_move((data & TT_MOVE_MASK) as u16)?;
    Some(TranspositionEntry {
        depth: ((data >> 48) & 0xff) as u8,
        score: i32::from(((data >> 16) as u16) as i16),
        bound,
        best_move,
        static_eval: unpack_static_eval(((data >> 32) as u16) as i16),
        age: packed_age(data),
    })
}

pub(in crate::search) fn packed_age(data: u64) -> u8 {
    ((data >> 58) as u8) & TT_AGE_MASK
}

pub(in crate::search) fn pack_tt_score(score: i32) -> i16 {
    debug_assert!(
        score >= i32::from(i16::MIN) && score <= i32::from(i16::MAX),
        "TT score out of i16 range: {}",
        score
    );
    score as i16
}

pub(in crate::search) fn pack_static_eval(static_eval: Option<i32>) -> i16 {
    static_eval
        .map(|eval| eval.clamp(i32::from(i16::MIN) + 1, i32::from(i16::MAX)) as i16)
        .unwrap_or(TT_STATIC_EVAL_NONE)
}

pub(in crate::search) fn unpack_static_eval(static_eval: i16) -> i32 {
    if static_eval == TT_STATIC_EVAL_NONE {
        NO_STATIC_EVAL
    } else {
        i32::from(static_eval)
    }
}

pub(in crate::search) fn is_mate_score(score: i32) -> bool {
    let mate_score_ply_window = MAX_ORDERING_PLY as i32;
    score >= -LOSS_SCORE - mate_score_ply_window || score <= LOSS_SCORE + mate_score_ply_window
}

pub(in crate::search) fn score_to_tt(score: i32, ply: u16) -> i32 {
    if !is_mate_score(score) {
        return score;
    }
    if score > 0 {
        score.saturating_add(ply as i32)
    } else {
        score.saturating_sub(ply as i32)
    }
}

pub(in crate::search) fn score_from_tt(score: i32, ply: u16) -> i32 {
    if !is_mate_score(score) {
        return score;
    }
    if score > 0 {
        score.saturating_sub(ply as i32)
    } else {
        score.saturating_add(ply as i32)
    }
}

pub(in crate::search) fn encode_optional_move(mv: Option<Move>) -> u16 {
    let Some(mv) = mv else {
        return 0;
    };
    let promotion = mv.promotion.map(|piece| piece as u16 + 1).unwrap_or(0);
    1 + ((mv.from as u16) | ((mv.to as u16) << 6) | (promotion << 12))
}

pub(in crate::search) fn decode_optional_move(encoded: u16) -> Option<Option<Move>> {
    if encoded == 0 {
        return Some(None);
    }
    let raw = encoded - 1;
    let from = Square::try_index((raw & 0x3f) as usize)?;
    let to = Square::try_index(((raw >> 6) & 0x3f) as usize)?;
    let promotion = match (raw >> 12) & 0x7 {
        0 => None,
        value => Some(Piece::try_index(value.saturating_sub(1) as usize)?),
    };
    Some(Some(Move {
        from,
        to,
        promotion,
    }))
}
