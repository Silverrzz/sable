use std::{fmt, str::FromStr};

use super::{BitBoard, Color};

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum File {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
    F = 5,
    G = 6,
    H = 7,
}

impl File {
    #[inline]
    pub const fn try_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::A),
            1 => Some(Self::B),
            2 => Some(Self::C),
            3 => Some(Self::D),
            4 => Some(Self::E),
            5 => Some(Self::F),
            6 => Some(Self::G),
            7 => Some(Self::H),
            _ => None,
        }
    }

    #[inline]
    pub(crate) const fn index(index: usize) -> Self {
        match index {
            0 => Self::A,
            1 => Self::B,
            2 => Self::C,
            3 => Self::D,
            4 => Self::E,
            5 => Self::F,
            6 => Self::G,
            7 => Self::H,
            _ => panic!("file index out of range"),
        }
    }

}

impl fmt::Display for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = (b'a' + *self as u8) as char;
        f.write_str(&file.to_string())
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Rank {
    First = 0,
    Second = 1,
    Third = 2,
    Fourth = 3,
    Fifth = 4,
    Sixth = 5,
    Seventh = 6,
    Eighth = 7,
}

impl Rank {
    #[inline]
    pub const fn try_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::First),
            1 => Some(Self::Second),
            2 => Some(Self::Third),
            3 => Some(Self::Fourth),
            4 => Some(Self::Fifth),
            5 => Some(Self::Sixth),
            6 => Some(Self::Seventh),
            7 => Some(Self::Eighth),
            _ => None,
        }
    }

    #[inline]
    pub(crate) const fn index(index: usize) -> Self {
        match index {
            0 => Self::First,
            1 => Self::Second,
            2 => Self::Third,
            3 => Self::Fourth,
            4 => Self::Fifth,
            5 => Self::Sixth,
            6 => Self::Seventh,
            7 => Self::Eighth,
            _ => panic!("rank index out of range"),
        }
    }

    #[inline]
    pub const fn relative_to(self, color: Color) -> Self {
        match color {
            Color::White => self,
            Color::Black => Self::index(7 - self as usize),
        }
    }

    #[inline]
    pub const fn bitboard(self) -> BitBoard {
        BitBoard(0xffu64 << ((self as u8) * 8))
    }

}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rank = (b'1' + *self as u8) as char;
        f.write_str(&rank.to_string())
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum Square {
    A1 = 0, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
}

impl Square {
    #[inline]
    pub const fn new(file: File, rank: Rank) -> Self {
        Self::index_const(((rank as usize) << 3) | file as usize)
    }

    #[inline]
    pub const fn try_index(index: usize) -> Option<Self> {
        if index < 64 {
            Some(Self::index_const(index))
        } else {
            None
        }
    }

    #[inline]
    pub(crate) const fn index(index: usize) -> Self {
        Self::index_const(index)
    }

    #[inline]
    const fn index_const(index: usize) -> Self {
        match index {
            0 => Self::A1, 1 => Self::B1, 2 => Self::C1, 3 => Self::D1,
            4 => Self::E1, 5 => Self::F1, 6 => Self::G1, 7 => Self::H1,
            8 => Self::A2, 9 => Self::B2, 10 => Self::C2, 11 => Self::D2,
            12 => Self::E2, 13 => Self::F2, 14 => Self::G2, 15 => Self::H2,
            16 => Self::A3, 17 => Self::B3, 18 => Self::C3, 19 => Self::D3,
            20 => Self::E3, 21 => Self::F3, 22 => Self::G3, 23 => Self::H3,
            24 => Self::A4, 25 => Self::B4, 26 => Self::C4, 27 => Self::D4,
            28 => Self::E4, 29 => Self::F4, 30 => Self::G4, 31 => Self::H4,
            32 => Self::A5, 33 => Self::B5, 34 => Self::C5, 35 => Self::D5,
            36 => Self::E5, 37 => Self::F5, 38 => Self::G5, 39 => Self::H5,
            40 => Self::A6, 41 => Self::B6, 42 => Self::C6, 43 => Self::D6,
            44 => Self::E6, 45 => Self::F6, 46 => Self::G6, 47 => Self::H6,
            48 => Self::A7, 49 => Self::B7, 50 => Self::C7, 51 => Self::D7,
            52 => Self::E7, 53 => Self::F7, 54 => Self::G7, 55 => Self::H7,
            56 => Self::A8, 57 => Self::B8, 58 => Self::C8, 59 => Self::D8,
            60 => Self::E8, 61 => Self::F8, 62 => Self::G8, 63 => Self::H8,
            _ => panic!("square index out of range"),
        }
    }

    #[inline]
    pub const fn file(self) -> File {
        File::index(self as usize & 7)
    }

    #[inline]
    pub const fn rank(self) -> Rank {
        Rank::index(self as usize >> 3)
    }

    #[inline]
    pub const fn bitboard(self) -> BitBoard {
        BitBoard(1u64 << self as u8)
    }

}

impl FromStr for Square {
    type Err = SquareParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let bytes = value.as_bytes();
        if bytes.len() != 2 {
            return Err(SquareParseError);
        }
        let file = match bytes[0] {
            b'a' | b'A' => File::A,
            b'b' | b'B' => File::B,
            b'c' | b'C' => File::C,
            b'd' | b'D' => File::D,
            b'e' | b'E' => File::E,
            b'f' | b'F' => File::F,
            b'g' | b'G' => File::G,
            b'h' | b'H' => File::H,
            _ => return Err(SquareParseError),
        };
        let rank = match bytes[1] {
            b'1' => Rank::First,
            b'2' => Rank::Second,
            b'3' => Rank::Third,
            b'4' => Rank::Fourth,
            b'5' => Rank::Fifth,
            b'6' => Rank::Sixth,
            b'7' => Rank::Seventh,
            b'8' => Rank::Eighth,
            _ => return Err(SquareParseError),
        };
        Ok(Self::new(file, rank))
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.file(), self.rank())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SquareParseError;
