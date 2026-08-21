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
            r_end: 0.002,
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

        static SPSA_PARAMETERS: &[fn() -> &'static TunableParameter] = &[
            $($accessor),+
        ];
    };
}

define_tunable_parameters!(
    (MAX_CORRECTION_HISTORY_SCORE, max_correction_history_score_parameter, i32, 654, 32, 1024, 32.0),
    (CORRECTION_HISTORY_MINOR_WEIGHT, correction_history_minor_weight_parameter, i32, 200, 0, 512, 16.0),
    (CORRECTION_HISTORY_NON_PAWN_WEIGHT, correction_history_non_pawn_weight_parameter, i32, 482, 0, 512, 16.0),
    (CORRECTION_HISTORY_PREVIOUS_WEIGHT, correction_history_previous_weight_parameter, i32, 206, 0, 512, 16.0),
    (CORRECTION_HISTORY_SAME_SIDE_WEIGHT, correction_history_same_side_weight_parameter, i32, 75, 0, 512, 16.0),
    (CORRECTION_HISTORY_PAWN_UPDATE_SCALE, correction_history_pawn_update_scale_parameter, i32, 161, 0, 384, 16.0),
    (CORRECTION_HISTORY_MINOR_UPDATE_SCALE, correction_history_minor_update_scale_parameter, i32, 167, 0, 384, 16.0),
    (CORRECTION_HISTORY_NON_PAWN_UPDATE_SCALE, correction_history_non_pawn_update_scale_parameter, i32, 156, 0, 384, 16.0),
    (CORRECTION_HISTORY_PREVIOUS_UPDATE_SCALE, correction_history_previous_update_scale_parameter, i32, 184, 0, 384, 16.0),
    (CORRECTION_HISTORY_SAME_SIDE_UPDATE_SCALE, correction_history_same_side_update_scale_parameter, i32, 6, 0, 384, 16.0),
    (CONTINUATION_HISTORY_ORDERING_DIVISOR, continuation_history_ordering_divisor_parameter, i32, 5, 1, 16, 0.5),
    (CAPTURE_HISTORY_ORDERING_DIVISOR, capture_history_ordering_divisor_parameter, i32, 7, 1, 16, 0.5),
    (ASPIRATION_MIN_DEPTH, aspiration_min_depth_parameter, u32, 5, 1, 12, 0.5),
    (ASPIRATION_INITIAL_WINDOW, aspiration_initial_window_parameter, i32, 10, 8, 256, 8.0),
    (INTERNAL_ITERATIVE_REDUCTION_MIN_DEPTH, internal_iterative_reduction_min_depth_parameter, u32, 9, 2, 16, 1.0),
    (INTERNAL_ITERATIVE_REDUCTION, internal_iterative_reduction_parameter, u32, 4, 1, 4, 0.5),
    (SINGULAR_EXTENSION_MIN_DEPTH, singular_extension_min_depth_parameter, u32, 10, 2, 16, 1.0),
    (SINGULAR_EXTENSION_TT_DEPTH_MARGIN, singular_extension_tt_depth_margin_parameter, u32, 1, 1, 8, 0.5),
    (SINGULAR_EXTENSION_BASE_MARGIN, singular_extension_base_margin_parameter, i32, 10, 0, 128, 8.0),
    (DOUBLE_SINGULAR_EXTENSION_BASE_MARGIN, double_singular_extension_base_margin_parameter, i32, 3, 0, 256, 16.0),
    (TRIPLE_SINGULAR_EXTENSION_BASE_MARGIN, triple_singular_extension_base_margin_parameter, i32, 24, 0, 384, 16.0),
    (LMR_MIN_DEPTH, lmr_min_depth_parameter, u32, 2, 2, 8, 0.5),
    (SPARSE_ENDGAME_QUIET_CHECK_LMR_PROTECTION, sparse_endgame_quiet_check_lmr_protection_parameter, u32, 0, 0, 4, 0.5),
    (PROBCUT_MIN_DEPTH, probcut_min_depth_parameter, u32, 8, 2, 12, 1.0),
    (PROBCUT_MARGIN, probcut_margin_parameter, i32, 122, 0, 500, 20.0),
    (PROBCUT_SEE_THRESHOLD, probcut_see_threshold_parameter, i32, 108, -300, 500, 20.0),
    (PROBCUT_DEPTH_REDUCTION, probcut_depth_reduction_parameter, u32, 4, 1, 8, 0.5),
    (NULL_MOVE_MIN_DEPTH, null_move_min_depth_parameter, u32, 2, 2, 8, 0.5),
    (NULL_MOVE_BASE_REDUCTION, null_move_base_reduction_parameter, u32, 3, 1, 8, 0.5),
    (NULL_MOVE_DEPTH_REDUCTION_DIVISOR, null_move_depth_reduction_divisor_parameter, u32, 1, 1, 16, 1.0),
    (NULL_MOVE_EVAL_MARGIN_PER_REDUCTION, null_move_eval_margin_per_reduction_parameter, i32, 293, 1, 600, 25.0),
    (NULL_MOVE_MAX_EVAL_REDUCTION, null_move_max_eval_reduction_parameter, u32, 1, 0, 8, 0.5),
    (NULL_MOVE_SPARSE_ENDGAME_REDUCTION_PROTECTION, null_move_sparse_endgame_reduction_protection_parameter, u32, 2, 0, 4, 0.5),
    (NULL_MOVE_VERIFICATION_MIN_DEPTH, null_move_verification_min_depth_parameter, u32, 9, 4, 24, 1.0),
    (REVERSE_FUTILITY_MAX_DEPTH, reverse_futility_max_depth_parameter, u32, 6, 1, 12, 1.0),
    (REVERSE_FUTILITY_BASE_MARGIN, reverse_futility_base_margin_parameter, i32, -34, -100, 400, 20.0),
    (REVERSE_FUTILITY_MARGIN_PER_DEPTH, reverse_futility_margin_per_depth_parameter, i32, 70, 0, 300, 15.0),
    (RAZOR_MAX_DEPTH, razor_max_depth_parameter, u32, 3, 1, 6, 0.5),
    (RAZOR_BASE_MARGIN, razor_base_margin_parameter, i32, 208, 0, 500, 25.0),
    (RAZOR_MARGIN_PER_DEPTH, razor_margin_per_depth_parameter, i32, 152, 0, 300, 15.0),
    (FUTILITY_MAX_DEPTH, futility_max_depth_parameter, u32, 6, 1, 20, 1.0),
    (FUTILITY_BASE_MARGIN, futility_base_margin_parameter, i32, 0, 0, 400, 20.0),
    (FUTILITY_MARGIN_PER_DEPTH, futility_margin_per_depth_parameter, i32, 31, 0, 300, 15.0),
    (FUTILITY_IMPROVING_MARGIN, futility_improving_margin_parameter, i32, 196, 0, 300, 15.0),
    (SEE_PRUNING_MAX_DEPTH, see_pruning_max_depth_parameter, u32, 15, 1, 16, 1.0),
    (SEE_PRUNING_BASE_MARGIN, see_pruning_base_margin_parameter, i32, 5, 0, 300, 15.0),
    (SEE_PRUNING_MARGIN_PER_DEPTH, see_pruning_margin_per_depth_parameter, i32, 71, 0, 150, 10.0),
    (Q_DELTA_PRUNING_MARGIN, q_delta_pruning_margin_parameter, i32, 344, 0, 600, 25.0),
    (QSEARCH_MAX_EVASION_MOVES, qsearch_max_evasion_moves_parameter, u32, 3, 1, 12, 0.5),
    (LATE_QUIET_PRUNING_MAX_DEPTH, late_quiet_pruning_max_depth_parameter, u32, 13, 1, 16, 1.0),
    (LATE_QUIET_PRUNING_BASE_THRESHOLD, late_quiet_pruning_base_threshold_parameter, u32, 1, 1, 16, 1.0),
    (DRAW_PREFERENCE_MAX_SCORE, draw_preference_max_score_parameter, i32, 68, 0, 200, 10.0),
    (ROOT_REPETITION_DEFER_MIN_SCORE, root_repetition_defer_min_score_parameter, i32, 115, 0, 600, 25.0),
    (DEFAULT_TIME_ALLOCATION_DIVISOR, default_time_allocation_divisor_parameter, u64, 9, 8, 32, 1.0),
    (INCREMENT_TIME_PERMILLE, increment_time_permille_parameter, u64, 907, 500, 1100, 25.0),
    (HARD_TIME_SOFT_MULTIPLIER_PERMILLE, hard_time_soft_multiplier_permille_parameter, u64, 5553, 2000, 8000, 250.0),
    (HARD_TIME_CLOCK_PERMILLE, hard_time_clock_permille_parameter, u64, 743, 600, 950, 25.0),
    (TIME_MANAGER_MIN_PREDICTION_DEPTH, time_manager_min_prediction_depth_parameter, u32, 3, 1, 5, 0.5),
    (TIME_MANAGER_DEFAULT_NODE_GROWTH_PERMILLE, time_manager_default_node_growth_permille_parameter, u64, 3207, 2000, 5000, 100.0),
    (TIME_MANAGER_MIN_NODE_GROWTH_PERMILLE, time_manager_min_node_growth_permille_parameter, u64, 1994, 1000, 2000, 50.0),
    (TIME_MANAGER_MAX_NODE_GROWTH_PERMILLE, time_manager_max_node_growth_permille_parameter, u64, 5010, 3000, 8000, 100.0),
    (TIME_MANAGER_STABLE_SCORE_CP, time_manager_stable_score_cp_parameter, i32, 26, 18, 40, 2.0),
    (TIME_MANAGER_VERY_STABLE_SCORE_CP, time_manager_very_stable_score_cp_parameter, i32, 12, 4, 16, 1.0),
    (TIME_MANAGER_FAIL_LOW_SMALL_DROP_CP, time_manager_fail_low_small_drop_cp_parameter, i32, 32, 30, 90, 5.0),
    (TIME_MANAGER_FAIL_LOW_MEDIUM_DROP_CP, time_manager_fail_low_medium_drop_cp_parameter, i32, 168, 100, 180, 5.0),
    (TIME_MANAGER_FAIL_LOW_BIG_DROP_CP, time_manager_fail_low_big_drop_cp_parameter, i32, 277, 200, 400, 10.0),
    (TIME_MANAGER_FAIL_LOW_SMALL_MULTIPLIER, time_manager_fail_low_small_multiplier_parameter, u64, 1118, 1050, 1400, 25.0),
    (TIME_MANAGER_FAIL_LOW_MEDIUM_MULTIPLIER, time_manager_fail_low_medium_multiplier_parameter, u64, 1458, 1400, 1800, 25.0),
    (TIME_MANAGER_FAIL_LOW_BIG_MULTIPLIER, time_manager_fail_low_big_multiplier_parameter, u64, 1806, 1800, 3000, 50.0),
    (TIME_MANAGER_MIN_SOFT_MULTIPLIER, time_manager_min_soft_multiplier_parameter, u64, 775, 700, 950, 10.0),
    (TIME_MANAGER_MAX_SOFT_MULTIPLIER, time_manager_max_soft_multiplier_parameter, u64, 4319, 3000, 6000, 100.0),
    (TIME_MANAGER_MOVE_UNSTABLE_MULTIPLIER, time_manager_move_unstable_multiplier_parameter, u64, 1212, 1100, 1600, 25.0),
    (TIME_MANAGER_MOVE_STABILITY_1_MULTIPLIER, time_manager_move_stability_1_multiplier_parameter, u64, 1092, 1050, 1300, 10.0),
    (TIME_MANAGER_MOVE_STABILITY_2_MULTIPLIER, time_manager_move_stability_2_multiplier_parameter, u64, 1057, 1000, 1150, 10.0),
    (TIME_MANAGER_MOVE_STABILITY_3_MULTIPLIER, time_manager_move_stability_3_multiplier_parameter, u64, 1015, 950, 1050, 5.0),
    (TIME_MANAGER_MOVE_STABLE_MULTIPLIER, time_manager_move_stable_multiplier_parameter, u64, 827, 800, 1000, 10.0),
    (TIME_MANAGER_SCORE_BIG_DELTA_MULTIPLIER, time_manager_score_big_delta_multiplier_parameter, u64, 1393, 1150, 1600, 25.0),
    (TIME_MANAGER_SCORE_MEDIUM_DELTA_MULTIPLIER, time_manager_score_medium_delta_multiplier_parameter, u64, 1195, 1050, 1400, 25.0),
    (TIME_MANAGER_SCORE_SMALL_DELTA_MULTIPLIER, time_manager_score_small_delta_multiplier_parameter, u64, 1095, 1000, 1250, 10.0),
    (TIME_MANAGER_SCORE_VERY_STABLE_MULTIPLIER, time_manager_score_very_stable_multiplier_parameter, u64, 988, 800, 990, 10.0),
    (TIME_MANAGER_SCORE_STABLE_MULTIPLIER, time_manager_score_stable_multiplier_parameter, u64, 868, 850, 1100, 10.0),
    (TIME_MANAGER_VERY_STABLE_ITERATIONS, time_manager_very_stable_iterations_parameter, u32, 2, 2, 5, 0.5),
    (TIME_MANAGER_STABLE_ITERATIONS, time_manager_stable_iterations_parameter, u32, 3, 1, 3, 0.5),
);

fn find_spsa_parameter(name: &str) -> Option<&'static TunableParameter> {
    let name = name.trim();
    SPSA_PARAMETERS
        .iter()
        .map(|accessor| accessor())
        .find(|parameter| parameter.name.eq_ignore_ascii_case(name))
}

pub fn spsa_parameters() -> Vec<SpsaParameter> {
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
