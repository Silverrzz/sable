use crate::{Move, Piece};

#[derive(Clone, Copy, Debug)]
pub(in crate::search) struct ContextMove {
    pub(in crate::search) mv: Move,
    pub(in crate::search) piece: Piece,
}

impl ContextMove {
    pub(in crate::search) fn new(mv: Move, piece: Piece) -> Self {
        Self { mv, piece }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::search) struct MoveContext {
    pub(in crate::search) previous: Option<ContextMove>,
    pub(in crate::search) previous_same_side: Option<ContextMove>,
}

impl MoveContext {
    pub(in crate::search) fn previous_move(self) -> Option<Move> {
        self.previous.map(|entry| entry.mv)
    }

    pub(in crate::search) fn after_move(self, mv: Move, piece: Piece) -> Self {
        Self {
            previous: Some(ContextMove::new(mv, piece)),
            previous_same_side: self.previous,
        }
    }

    pub(in crate::search) fn without_moves(self) -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Square;

    #[test]
    fn move_context_tracks_previous_two_plies() {
        let first = Move {
            from: Square::E2,
            to: Square::E4,
            promotion: None,
        };
        let second = Move {
            from: Square::E7,
            to: Square::E5,
            promotion: None,
        };
        let context = MoveContext::default()
            .after_move(first, Piece::Pawn)
            .after_move(second, Piece::Pawn);

        assert_eq!(context.previous.map(|entry| entry.mv), Some(second));
        assert_eq!(
            context.previous_same_side.map(|entry| entry.mv),
            Some(first)
        );
    }
}
