use anyhow::Result;
use sable_engine::Engine;

use super::command::{PositionBase, PositionCommand};

pub(super) fn apply_position(position: PositionCommand, engine: &mut Engine) -> Result<()> {
    match position.base {
        PositionBase::StartPos => engine.set_startpos_with_moves(&position.moves)?,
        PositionBase::Fen(fen) => engine.set_fen_with_moves(&fen, &position.moves)?,
    }
    Ok(())
}
