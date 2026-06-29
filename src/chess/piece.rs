use std::{fmt, ops::Not};

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Not for Color {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl Piece {
    #[inline]
    pub const fn try_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Pawn),
            1 => Some(Self::Knight),
            2 => Some(Self::Bishop),
            3 => Some(Self::Rook),
            4 => Some(Self::Queen),
            5 => Some(Self::King),
            _ => None,
        }
    }

}

impl std::str::FromStr for Piece {
    type Err = PieceParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "p" | "P" => Ok(Self::Pawn),
            "n" | "N" => Ok(Self::Knight),
            "b" | "B" => Ok(Self::Bishop),
            "r" | "R" => Ok(Self::Rook),
            "q" | "Q" => Ok(Self::Queen),
            "k" | "K" => Ok(Self::King),
            _ => Err(PieceParseError),
        }
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let letter = match self {
            Self::Pawn => 'p',
            Self::Knight => 'n',
            Self::Bishop => 'b',
            Self::Rook => 'r',
            Self::Queen => 'q',
            Self::King => 'k',
        };
        f.write_str(&letter.to_string())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PieceParseError;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GameStatus {
    Ongoing,
    Won,
    Drawn,
}
