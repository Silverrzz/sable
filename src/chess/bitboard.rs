use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Sub, SubAssign,
};

use super::Square;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct BitBoard(pub u64);

impl BitBoard {
    pub const EMPTY: Self = Self(0);
    pub const FULL: Self = Self(u64::MAX);

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn is_disjoint(self, other: Self) -> bool {
        (self.0 & other.0) == 0
    }

    #[inline]
    pub const fn has(self, square: Square) -> bool {
        (self.0 & square.bitboard().0) != 0
    }

    #[inline]
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    pub const fn next_square(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Square::try_index(self.0.trailing_zeros() as usize)
        }
    }

    #[inline]
    pub(crate) const fn from_cozy(bitboard: cozy_chess::BitBoard) -> Self {
        Self(bitboard.0)
    }

    #[inline]
    pub(crate) const fn to_cozy(self) -> cozy_chess::BitBoard {
        cozy_chess::BitBoard(self.0)
    }
}

impl IntoIterator for BitBoard {
    type Item = Square;
    type IntoIter = BitBoardIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        BitBoardIter(self)
    }
}

pub struct BitBoardIter(BitBoard);

impl Iterator for BitBoardIter {
    type Item = Square;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let square = self.0.next_square()?;
        self.0 ^= square.bitboard();
        Some(square)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.len() as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for BitBoardIter {}

impl BitAnd for BitBoard {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for BitBoard {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOr for BitBoard {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for BitBoard {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitXor for BitBoard {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for BitBoard {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Sub for BitBoard {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 & !rhs.0)
    }
}

impl SubAssign for BitBoard {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 &= !rhs.0;
    }
}

impl Not for BitBoard {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}
