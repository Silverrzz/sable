use crate::{Color, Piece, Square};

use super::should_mirror;

#[derive(Debug, Copy, Clone)]
pub struct PieceFeature {
    pub piece: Piece,
    pub color: Color,
    pub sq: Square,
}

impl PieceFeature {
    #[inline]
    pub fn new(piece: Piece, color: Color, sq: Square) -> Self {
        Self { piece, color, sq }
    }

    #[inline]
    pub fn to_index(self, king: Square, perspective: Color) -> usize {
        let mut sq = self.sq as usize;
        if perspective == Color::Black {
            sq ^= 56;
        }
        if should_mirror(king) {
            sq ^= 7;
        }

        usize::from(self.color != perspective) * 384 + self.piece as usize * 64 + sq
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct FeatureUpdates {
    pub add: Option<PieceFeature>,
    pub add2: Option<PieceFeature>,
    pub sub: Option<PieceFeature>,
    pub sub2: Option<PieceFeature>,
}
