use std::sync::Arc;


use super::types::NnueModel;

pub(crate) type SharedNnueModel = Arc<NnueModel>;

pub fn has_embedded_eval() -> bool {
    NnueModel::has_embedded_default()
}

pub fn embedded_eval_label() -> Option<&'static str> {
    NnueModel::embedded_default_label()
}
