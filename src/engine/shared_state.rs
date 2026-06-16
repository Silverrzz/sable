use std::sync::{
    Mutex, MutexGuard,
    atomic::{AtomicU64, Ordering},
};

use crate::search::PersistentSearchState;

#[derive(Debug)]
pub(super) struct SharedSearchState {
    generation: AtomicU64,
    state: Mutex<PersistentSearchState>,
}

impl Default for SharedSearchState {
    fn default() -> Self {
        Self {
            generation: AtomicU64::new(0),
            state: Mutex::new(PersistentSearchState::default()),
        }
    }
}

impl SharedSearchState {
    pub(super) fn snapshot(&self) -> (u64, PersistentSearchState) {
        let mut state = self.lock_state();
        (
            self.generation.load(Ordering::Acquire),
            std::mem::take(&mut *state),
        )
    }

    pub(super) fn store_if_current(&self, generation: u64, state: PersistentSearchState) {
        if self.generation.load(Ordering::Acquire) != generation {
            return;
        }
        let mut current = self.lock_state();
        if self.generation.load(Ordering::Acquire) == generation {
            *current = state;
        }
    }

    pub(super) fn reset(&self) {
        let mut state = self.lock_state();
        *state = PersistentSearchState::default();
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    fn lock_state(&self) -> MutexGuard<'_, PersistentSearchState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
