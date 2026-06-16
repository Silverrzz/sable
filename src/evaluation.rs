mod api;
mod bullet;
mod evaluator;
mod features;
mod hce;
mod integer;
mod io;
mod material;
mod model;
mod types;

pub use api::{embedded_eval_label, has_embedded_eval};
pub use types::{EvalMode, NnueAccumulators, NnueEvalScratch, NnueModel, PieceContribution};

pub(crate) use evaluator::Evaluator;
pub(crate) use integer::evaluate_position;
pub(crate) use material::{is_board_drawn, material_score_for_white};
pub(crate) use types::{DRAW_SCORE, LOSS_SCORE, NnueFinnyTable};
