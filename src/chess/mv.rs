use std::{fmt, str::FromStr};

use super::{Piece, Square};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<Piece>,
}

impl FromStr for Move {
    type Err = MoveParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let from = value.get(0..2).ok_or(MoveParseError)?.parse().map_err(|_| MoveParseError)?;
        let to = value.get(2..4).ok_or(MoveParseError)?.parse().map_err(|_| MoveParseError)?;
        let promotion = if let Some(raw) = value.get(4..5) {
            let piece = raw.parse().map_err(|_| MoveParseError)?;
            if matches!(piece, Piece::Pawn | Piece::King) {
                None
            } else {
                Some(piece)
            }
        } else {
            None
        };
        if value.len() > 5 {
            return Err(MoveParseError);
        }
        Ok(Self {
            from,
            to,
            promotion,
        })
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.from, self.to)?;
        if let Some(promotion) = self.promotion {
            write!(f, "{promotion}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoveParseError;
