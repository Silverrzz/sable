use std::sync::atomic::{AtomicI64, Ordering};

use crate::EngineError;

pub(super) const DEFAULT_MAX_DEPTH: u32 = 128;
pub(super) const MAX_ORDERING_PLY: usize = 128;
pub(super) const MAX_CANDIDATE_MOVES: usize = 256;
pub(super) const MAX_PV_LENGTH: usize = MAX_ORDERING_PLY;
pub(super) const STOP_CHECK_NODE_INTERVAL: u64 = 1024;
pub(super) const KILLER_SLOTS: usize = 2;
pub(super) const PV_MOVE_SCORE: i32 = 2_000_000_000;
pub(super) const CAPTURE_SCORE: i32 = 1_000_000_000;
pub(super) const PROMOTION_SCORE: i32 = 900_000_000;
pub(super) const FIRST_KILLER_SCORE: i32 = 800_000_000;
pub(super) const SECOND_KILLER_SCORE: i32 = 799_000_000;
pub(super) const COUNTER_MOVE_SCORE: i32 = 790_000_000;
pub(super) const MAX_HISTORY_SCORE: i32 = 8_000_000;
pub(super) const LMR_SCALE: i32 = 1024;
pub(super) const CORRECTION_HISTORY_UPDATE_DIVISOR: i32 = 466;
pub(super) const CORRECTION_HISTORY_PAWN_WEIGHT: i32 = 262;
pub(super) const CORRECTION_HISTORY_BUCKETS: usize = 16_384;
pub(super) const ASPIRATION_MAX_WINDOW: i32 = 4096;
pub(super) const SPARSE_ENDGAME_MAX_NON_KING_PIECES: u32 = 4;
pub(super) const MATE_PRUNING_GUARD: i32 = 512;
pub(super) const NO_STATIC_EVAL: i32 = i32::MIN;

#[derive(Clone, Copy, Debug)]
pub struct SpsaParameter {
    pub name: &'static str,
    pub value: i64,
    pub default: i64,
    pub min: i64,
    pub max: i64,
    pub c_end: f64,
    pub r_end: f64,
}

struct TunableParameter {
    name: &'static str,
    value: AtomicI64,
    default: i64,
    min: i64,
    max: i64,
    c_end: f64,
}

impl TunableParameter {
    const fn new(
        name: &'static str,
        default: i64,
        min: i64,
        max: i64,
        c_end: f64,
    ) -> Self {
        Self {
            name,
            value: AtomicI64::new(default),
            default,
            min,
            max,
            c_end,
        }
    }

    fn descriptor(&self) -> SpsaParameter {
        SpsaParameter {
            name: self.name,
            value: self.value.load(Ordering::Relaxed),
            default: self.default,
            min: self.min,
            max: self.max,
            c_end: self.c_end,
            r_end: if self.name.contains("_UNCERTAINTY_") {
                ((self.max - self.min) as f64 / (750.0 * self.c_end) * 10_000.0).round()
                    / 10_000.0
            } else {
                0.002
            },
        }
    }
}

macro_rules! define_tunable_parameters {
    ($(($name:ident, $accessor:ident, $ty:ty, $default:expr, $min:expr, $max:expr, $c_end:expr)),+ $(,)?) => {
        $(
            fn $accessor() -> &'static TunableParameter {
                static PARAMETER: TunableParameter = TunableParameter::new(
                    stringify!($name),
                    $default,
                    $min,
                    $max,
                    $c_end,
                );
                &PARAMETER
            }

            #[allow(non_snake_case)]
            #[inline(always)]
            pub(crate) fn $name() -> $ty {
                $accessor().value.load(Ordering::Relaxed) as $ty
            }
        )+


    };
}

define_tunable_parameters!(
    (RFP_UNCERTAINTY_SCALE, rfp_uncertainty_scale_parameter, i32, 52, 0, 192, 8.0),
    (RFP_UNCERTAINTY_MAX_MARGIN, rfp_uncertainty_max_margin_parameter, i32, 80, 0, 256, 8.0),
    (RFP_UNCERTAINTY_MIN_DEPTH, rfp_uncertainty_min_depth_parameter, u32, 2, 1, 3, 0.5),
    (RFP_UNCERTAINTY_MAX_DEPTH, rfp_uncertainty_max_depth_parameter, u32, 4, 3, 5, 0.5),
    (RFP_UNCERTAINTY_CONFIDENT_WEIGHT, rfp_uncertainty_confident_weight_parameter, i32, 168, 0, 384, 16.0),
    (RFP_UNCERTAINTY_UNCERTAIN_WEIGHT, rfp_uncertainty_uncertain_weight_parameter, i32, 135, 0, 384, 16.0),
    (RFP_UNCERTAINTY_DEADBAND, rfp_uncertainty_deadband_parameter, i32, 2, 0, 64, 2.0),
    (MAX_CORRECTION_HISTORY_SCORE, max_correction_history_score_parameter, i32, 521, 402, 670, 54.0),
    (CORRECTION_HISTORY_MINOR_WEIGHT, correction_history_minor_weight_parameter, i32, 224, 157, 261, 21.0),
    (CORRECTION_HISTORY_NON_PAWN_WEIGHT, correction_history_non_pawn_weight_parameter, i32, 363, 258, 430, 34.0),
    (CORRECTION_HISTORY_PREVIOUS_WEIGHT, correction_history_previous_weight_parameter, i32, 169, 117, 195, 16.0),
    (CORRECTION_HISTORY_SAME_SIDE_WEIGHT, correction_history_same_side_weight_parameter, i32, 44, 35, 59, 5.0),
    (CORRECTION_HISTORY_PAWN_UPDATE_SCALE, correction_history_pawn_update_scale_parameter, i32, 152, 119, 199, 16.0),
    (CORRECTION_HISTORY_MINOR_UPDATE_SCALE, correction_history_minor_update_scale_parameter, i32, 152, 113, 188, 15.0),
    (CORRECTION_HISTORY_NON_PAWN_UPDATE_SCALE, correction_history_non_pawn_update_scale_parameter, i32, 207, 161, 269, 22.0),
    (CORRECTION_HISTORY_PREVIOUS_UPDATE_SCALE, correction_history_previous_update_scale_parameter, i32, 136, 109, 181, 15.0),
    (CORRECTION_HISTORY_SAME_SIDE_UPDATE_SCALE, correction_history_same_side_update_scale_parameter, i32, 72, 55, 91, 7.0),
    (CONTINUATION_HISTORY_ORDERING_DIVISOR, continuation_history_ordering_divisor_parameter, i32, 2, 1, 16, 0.5),
    (CAPTURE_HISTORY_ORDERING_DIVISOR, capture_history_ordering_divisor_parameter, i32, 2, 1, 16, 0.5),
    (ASPIRATION_MIN_DEPTH, aspiration_min_depth_parameter, u32, 3, 1, 12, 0.5),
    (ASPIRATION_INITIAL_WINDOW, aspiration_initial_window_parameter, i32, 20, 14, 24, 2.0),
    (INTERNAL_ITERATIVE_REDUCTION_MIN_DEPTH, internal_iterative_reduction_min_depth_parameter, u32, 7, 2, 16, 1.0),
    (INTERNAL_ITERATIVE_REDUCTION, internal_iterative_reduction_parameter, u32, 1, 1, 4, 0.5),
    (SINGULAR_EXTENSION_MIN_DEPTH, singular_extension_min_depth_parameter, u32, 10, 2, 16, 1.0),
    (SINGULAR_EXTENSION_TT_DEPTH_MARGIN, singular_extension_tt_depth_margin_parameter, u32, 1, 1, 8, 0.5),
    (SINGULAR_EXTENSION_BASE_MARGIN, singular_extension_base_margin_parameter, i32, 49, 39, 65, 5.0),
    (DOUBLE_SINGULAR_EXTENSION_BASE_MARGIN, double_singular_extension_base_margin_parameter, i32, 0, 0, 256, 16.0),
    (TRIPLE_SINGULAR_EXTENSION_BASE_MARGIN, triple_singular_extension_base_margin_parameter, i32, 118, 90, 150, 12.0),
    (LMR_MIN_DEPTH, lmr_min_depth_parameter, u32, 3, 2, 8, 0.5),
    (LMR_UNCERTAINTY_REFERENCE, lmr_uncertainty_reference_parameter, i32, 939, 200, 1600, 64.0),
    (LMR_UNCERTAINTY_WEIGHT, lmr_uncertainty_weight_parameter, i32, 269, 0, 768, 32.0),
    (LMR_UNCERTAINTY_CONFIDENT_WEIGHT, lmr_uncertainty_confident_weight_parameter, i32, 269, 0, 768, 32.0),
    (LMR_UNCERTAINTY_DEADBAND, lmr_uncertainty_deadband_parameter, i32, 70, 0, 256, 8.0),
    (LMR_UNCERTAINTY_MIN_DEPTH, lmr_uncertainty_min_depth_parameter, u32, 3, 2, 6, 0.5),
    (LMR_UNCERTAINTY_MAX_DEPTH, lmr_uncertainty_max_depth_parameter, u32, 9, 6, 16, 1.0),
    (LMR_UNCERTAINTY_MIN_MOVE, lmr_uncertainty_min_move_parameter, u32, 3, 2, 12, 1.0),
    (LMR_BASE, lmr_base_parameter, i32, 1024, 0, 2048, 64.0),
    (LMR_DEPTH_MOVE_WEIGHT, lmr_depth_move_weight_parameter, i32, 256, 0, 512, 16.0),
    (LMR_HISTORY_WEIGHT, lmr_history_weight_parameter, i32, 1024, 0, 2048, 64.0),
    (LMR_CONTINUATION_HISTORY_WEIGHT, lmr_continuation_history_weight_parameter, i32, 512, 0, 2048, 64.0),
    (LMR_HISTORY_DIVISOR, lmr_history_divisor_parameter, i32, 2048, 512, 8192, 128.0),
    (LMR_HISTORY_MAX_ADJUSTMENT, lmr_history_max_adjustment_parameter, i32, 2048, 0, 4096, 128.0),
    (LMR_KILLER_PROTECTION, lmr_killer_protection_parameter, i32, 512, 0, 2048, 64.0),
    (LMR_COUNTER_MOVE_PROTECTION, lmr_counter_move_protection_parameter, i32, 256, 0, 2048, 64.0),
    (SPARSE_ENDGAME_QUIET_CHECK_LMR_PROTECTION, sparse_endgame_quiet_check_lmr_protection_parameter, u32, 0, 0, 4, 0.5),
    (PROBCUT_MIN_DEPTH, probcut_min_depth_parameter, u32, 8, 2, 12, 1.0),
    (PROBCUT_MARGIN, probcut_margin_parameter, i32, 233, 0, 500, 20.0),
    (PROBCUT_SEE_THRESHOLD, probcut_see_threshold_parameter, i32, 63, -300, 500, 20.0),
    (PROBCUT_DEPTH_REDUCTION, probcut_depth_reduction_parameter, u32, 7, 1, 8, 0.5),
    (NULL_MOVE_MIN_DEPTH, null_move_min_depth_parameter, u32, 3, 2, 8, 0.5),
    (NULL_MOVE_BASE_REDUCTION, null_move_base_reduction_parameter, u32, 4, 1, 8, 0.5),
    (NULL_MOVE_DEPTH_REDUCTION_DIVISOR, null_move_depth_reduction_divisor_parameter, u32, 3, 1, 16, 1.0),
    (NULL_MOVE_EVAL_MARGIN_PER_REDUCTION, null_move_eval_margin_per_reduction_parameter, i32, 272, 1, 600, 25.0),
    (NULL_MOVE_MAX_EVAL_REDUCTION, null_move_max_eval_reduction_parameter, u32, 2, 0, 8, 0.5),
    (NULL_MOVE_SPARSE_ENDGAME_REDUCTION_PROTECTION, null_move_sparse_endgame_reduction_protection_parameter, u32, 3, 0, 4, 0.5),
    (NULL_MOVE_VERIFICATION_MIN_DEPTH, null_move_verification_min_depth_parameter, u32, 13, 4, 24, 1.0),
    (REVERSE_FUTILITY_MAX_DEPTH, reverse_futility_max_depth_parameter, u32, 5, 1, 12, 1.0),
    (REVERSE_FUTILITY_BASE_MARGIN, reverse_futility_base_margin_parameter, i32, 29, -100, 400, 20.0),
    (REVERSE_FUTILITY_MARGIN_PER_DEPTH, reverse_futility_margin_per_depth_parameter, i32, 64, 0, 300, 15.0),
    (RAZOR_MAX_DEPTH, razor_max_depth_parameter, u32, 2, 1, 6, 0.5),
    (RAZOR_BASE_MARGIN, razor_base_margin_parameter, i32, 256, 0, 500, 25.0),
    (RAZOR_MARGIN_PER_DEPTH, razor_margin_per_depth_parameter, i32, 72, 0, 300, 15.0),
    (FUTILITY_MAX_DEPTH, futility_max_depth_parameter, u32, 8, 1, 20, 1.0),
    (FUTILITY_UNCERTAINTY_REFERENCE, futility_uncertainty_reference_parameter, i32, 266, 100, 1000, 32.0),
    (FUTILITY_UNCERTAINTY_WEIGHT, futility_uncertainty_weight_parameter, i32, 38, 0, 96, 8.0),
    (FUTILITY_UNCERTAINTY_CONFIDENT_WEIGHT, futility_uncertainty_confident_weight_parameter, i32, 18, 0, 96, 8.0),
    (FUTILITY_UNCERTAINTY_DEADBAND, futility_uncertainty_deadband_parameter, i32, 35, 0, 128, 4.0),
    (FUTILITY_UNCERTAINTY_MIN_DEPTH, futility_uncertainty_min_depth_parameter, u32, 2, 1, 4, 0.5),
    (FUTILITY_UNCERTAINTY_MAX_DEPTH, futility_uncertainty_max_depth_parameter, u32, 5, 4, 8, 0.5),
    (FUTILITY_BASE_MARGIN, futility_base_margin_parameter, i32, 10, 0, 400, 20.0),
    (FUTILITY_MARGIN_PER_DEPTH, futility_margin_per_depth_parameter, i32, 114, 0, 300, 15.0),
    (FUTILITY_IMPROVING_MARGIN, futility_improving_margin_parameter, i32, 109, 0, 300, 15.0),
    (SEE_PRUNING_MAX_DEPTH, see_pruning_max_depth_parameter, u32, 9, 1, 16, 1.0),
    (SEE_PRUNING_BASE_MARGIN, see_pruning_base_margin_parameter, i32, 8, 0, 300, 15.0),
    (SEE_PRUNING_MARGIN_PER_DEPTH, see_pruning_margin_per_depth_parameter, i32, 21, 0, 150, 10.0),
    (Q_DELTA_PRUNING_MARGIN, q_delta_pruning_margin_parameter, i32, 495, 327, 545, 44.0),
    (QSEARCH_MAX_EVASION_MOVES, qsearch_max_evasion_moves_parameter, u32, 2, 1, 12, 0.5),
    (LATE_QUIET_PRUNING_MAX_DEPTH, late_quiet_pruning_max_depth_parameter, u32, 10, 1, 16, 1.0),
    (LATE_QUIET_PRUNING_BASE_THRESHOLD, late_quiet_pruning_base_threshold_parameter, u32, 4, 1, 16, 1.0),
    (DRAW_PREFERENCE_MAX_SCORE, draw_preference_max_score_parameter, i32, 43, 33, 55, 4.0),
    (ROOT_REPETITION_DEFER_MIN_SCORE, root_repetition_defer_min_score_parameter, i32, 129, 94, 156, 13.0),
    (DEFAULT_TIME_ALLOCATION_DIVISOR, default_time_allocation_divisor_parameter, u64, 10, 8, 32, 1.0),
    (INCREMENT_TIME_PERMILLE, increment_time_permille_parameter, u64, 903, 500, 1100, 25.0),
    (HARD_TIME_SOFT_MULTIPLIER_PERMILLE, hard_time_soft_multiplier_permille_parameter, u64, 4458, 2000, 8000, 250.0),
    (HARD_TIME_CLOCK_PERMILLE, hard_time_clock_permille_parameter, u64, 869, 600, 950, 25.0),
    (TIME_MANAGER_MIN_PREDICTION_DEPTH, time_manager_min_prediction_depth_parameter, u32, 1, 1, 5, 0.5),
    (TIME_MANAGER_DEFAULT_NODE_GROWTH_PERMILLE, time_manager_default_node_growth_permille_parameter, u64, 3522, 2000, 5000, 100.0),
    (TIME_MANAGER_MIN_NODE_GROWTH_PERMILLE, time_manager_min_node_growth_permille_parameter, u64, 1708, 1000, 2000, 50.0),
    (TIME_MANAGER_MAX_NODE_GROWTH_PERMILLE, time_manager_max_node_growth_permille_parameter, u64, 5531, 3000, 8000, 100.0),
    (TIME_MANAGER_STABLE_SCORE_CP, time_manager_stable_score_cp_parameter, i32, 26, 18, 40, 2.0),
    (TIME_MANAGER_VERY_STABLE_SCORE_CP, time_manager_very_stable_score_cp_parameter, i32, 14, 4, 16, 1.0),
    (TIME_MANAGER_FAIL_LOW_SMALL_DROP_CP, time_manager_fail_low_small_drop_cp_parameter, i32, 34, 30, 90, 5.0),
    (TIME_MANAGER_FAIL_LOW_MEDIUM_DROP_CP, time_manager_fail_low_medium_drop_cp_parameter, i32, 140, 100, 180, 5.0),
    (TIME_MANAGER_FAIL_LOW_BIG_DROP_CP, time_manager_fail_low_big_drop_cp_parameter, i32, 217, 200, 400, 10.0),
    (TIME_MANAGER_FAIL_LOW_SMALL_MULTIPLIER, time_manager_fail_low_small_multiplier_parameter, u64, 1179, 1050, 1400, 25.0),
    (TIME_MANAGER_FAIL_LOW_MEDIUM_MULTIPLIER, time_manager_fail_low_medium_multiplier_parameter, u64, 1584, 1400, 1800, 25.0),
    (TIME_MANAGER_FAIL_LOW_BIG_MULTIPLIER, time_manager_fail_low_big_multiplier_parameter, u64, 1875, 1800, 3000, 50.0),
    (TIME_MANAGER_MIN_SOFT_MULTIPLIER, time_manager_min_soft_multiplier_parameter, u64, 833, 700, 950, 10.0),
    (TIME_MANAGER_MAX_SOFT_MULTIPLIER, time_manager_max_soft_multiplier_parameter, u64, 3881, 3000, 6000, 100.0),
    (TIME_MANAGER_MOVE_UNSTABLE_MULTIPLIER, time_manager_move_unstable_multiplier_parameter, u64, 1304, 1100, 1600, 25.0),
    (TIME_MANAGER_MOVE_STABILITY_1_MULTIPLIER, time_manager_move_stability_1_multiplier_parameter, u64, 1122, 1050, 1300, 10.0),
    (TIME_MANAGER_MOVE_STABILITY_2_MULTIPLIER, time_manager_move_stability_2_multiplier_parameter, u64, 1034, 1000, 1150, 10.0),
    (TIME_MANAGER_MOVE_STABILITY_3_MULTIPLIER, time_manager_move_stability_3_multiplier_parameter, u64, 1010, 950, 1050, 5.0),
    (TIME_MANAGER_MOVE_STABLE_MULTIPLIER, time_manager_move_stable_multiplier_parameter, u64, 887, 800, 1000, 10.0),
    (TIME_MANAGER_SCORE_BIG_DELTA_MULTIPLIER, time_manager_score_big_delta_multiplier_parameter, u64, 1263, 1150, 1600, 25.0),
    (TIME_MANAGER_SCORE_MEDIUM_DELTA_MULTIPLIER, time_manager_score_medium_delta_multiplier_parameter, u64, 1205, 1050, 1400, 25.0),
    (TIME_MANAGER_SCORE_SMALL_DELTA_MULTIPLIER, time_manager_score_small_delta_multiplier_parameter, u64, 1127, 1000, 1250, 10.0),
    (TIME_MANAGER_SCORE_VERY_STABLE_MULTIPLIER, time_manager_score_very_stable_multiplier_parameter, u64, 935, 800, 990, 10.0),
    (TIME_MANAGER_SCORE_STABLE_MULTIPLIER, time_manager_score_stable_multiplier_parameter, u64, 924, 850, 1100, 10.0),
    (TIME_MANAGER_VERY_STABLE_ITERATIONS, time_manager_very_stable_iterations_parameter, u32, 3, 2, 5, 0.5),
    (TIME_MANAGER_STABLE_ITERATIONS, time_manager_stable_iterations_parameter, u32, 2, 1, 3, 0.5),
);

static SPSA_PARAMETERS: &[fn() -> &'static TunableParameter] = &[
    rfp_uncertainty_scale_parameter,
    rfp_uncertainty_max_margin_parameter,
    rfp_uncertainty_min_depth_parameter,
    rfp_uncertainty_max_depth_parameter,
    rfp_uncertainty_confident_weight_parameter,
    rfp_uncertainty_uncertain_weight_parameter,
    rfp_uncertainty_deadband_parameter,
    lmr_uncertainty_reference_parameter,
    lmr_uncertainty_weight_parameter,
    lmr_uncertainty_confident_weight_parameter,
    lmr_uncertainty_deadband_parameter,
    lmr_uncertainty_min_depth_parameter,
    lmr_uncertainty_max_depth_parameter,
    lmr_uncertainty_min_move_parameter,
    futility_uncertainty_reference_parameter,
    futility_uncertainty_weight_parameter,
    futility_uncertainty_confident_weight_parameter,
    futility_uncertainty_deadband_parameter,
    futility_uncertainty_min_depth_parameter,
    futility_uncertainty_max_depth_parameter,
];

fn find_spsa_parameter(name: &str) -> Option<&'static TunableParameter> {
    let name = name.trim();
    SPSA_PARAMETERS
        .iter()
        .map(|accessor| accessor())
        .find(|parameter| parameter.name.eq_ignore_ascii_case(name))
}

pub fn spsa_parameters() -> Vec<SpsaParameter> {
    if !crate::SPSA_UCI_OPTIONS_ENABLED {
        return Vec::new();
    }
    SPSA_PARAMETERS
        .iter()
        .map(|accessor| accessor().descriptor())
        .collect()
}

pub(crate) fn set_spsa_parameter(
    name: &str,
    value: Option<&str>,
) -> Result<bool, EngineError> {
    let Some(parameter) = find_spsa_parameter(name) else {
        return Ok(false);
    };
    let raw = value.ok_or_else(|| EngineError::InvalidOptionValue {
        option: name.to_owned(),
        value: "<missing>".to_owned(),
    })?;
    let parsed = raw
        .parse::<i64>()
        .map_err(|_| EngineError::InvalidOptionValue {
            option: name.to_owned(),
            value: raw.to_owned(),
        })?;
    if parsed < parameter.min || parsed > parameter.max {
        return Err(EngineError::InvalidOptionValue {
            option: name.to_owned(),
            value: raw.to_owned(),
        });
    }
    parameter.value.store(parsed, Ordering::Relaxed);
    Ok(true)
}
