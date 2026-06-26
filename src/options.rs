use crate::{EngineError, EvalMode};

#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub hash_mb: u32,
    pub threads: u32,
    pub ponder: bool,
    pub multi_pv: u32,
    pub use_soft_nodes: bool,
    pub uci_chess960: bool,
    pub uci_show_wdl: bool,
    pub move_overhead_ms: u64,
    pub eval_mode: EvalMode,
    pub eval_file: Option<String>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            hash_mb: 16,
            threads: 1,
            ponder: false,
            multi_pv: 1,
            use_soft_nodes: false,
            uci_chess960: false,
            uci_show_wdl: false,
            move_overhead_ms: 100,
            eval_mode: compiled_default_eval_mode(),
            eval_file: None,
        }
    }
}

fn compiled_default_eval_mode() -> EvalMode {
    option_env!("SABLE_ENGINE_DEFAULT_EVAL_MODE")
        .and_then(EvalMode::from_uci)
        .unwrap_or_default()
}

pub(crate) fn apply_engine_option(
    options: &mut EngineOptions,
    name: &str,
    value: Option<&str>,
) -> Result<(), EngineError> {
    match EngineOption::from_name(name)? {
        EngineOption::Hash => options.hash_mb = parse_u32_option(name, value, 1, 32768)?,
        EngineOption::Threads => options.threads = parse_u32_option(name, value, 1, 256)?,
        EngineOption::Ponder => options.ponder = parse_bool_option(name, value)?,
        EngineOption::MultiPv => options.multi_pv = parse_u32_option(name, value, 1, 256)?,
        EngineOption::UseSoftNodes => options.use_soft_nodes = parse_bool_option(name, value)?,
        EngineOption::UciChess960 => options.uci_chess960 = parse_bool_option(name, value)?,
        EngineOption::UciShowWdl => options.uci_show_wdl = parse_bool_option(name, value)?,
        EngineOption::MoveOverhead => {
            options.move_overhead_ms = u64::from(parse_u32_option(name, value, 0, 10_000)?);
        }
        EngineOption::EvalMode => options.eval_mode = parse_eval_mode_option(name, value)?,
        EngineOption::EvalFile => {
            options.eval_file = Some(required_option_value(name, value)?.to_owned());
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EngineOption {
    Hash,
    Threads,
    Ponder,
    MultiPv,
    UseSoftNodes,
    UciChess960,
    UciShowWdl,
    MoveOverhead,
    EvalMode,
    EvalFile,
}

impl EngineOption {
    fn from_name(name: &str) -> Result<Self, EngineError> {
        match normalize_option_name(name).as_str() {
            "hash" => Ok(Self::Hash),
            "threads" => Ok(Self::Threads),
            "ponder" => Ok(Self::Ponder),
            "multipv" => Ok(Self::MultiPv),
            "usesoftnodes" => Ok(Self::UseSoftNodes),
            "uci_chess960" => Ok(Self::UciChess960),
            "uci_showwdl" => Ok(Self::UciShowWdl),
            "moveoverhead" => Ok(Self::MoveOverhead),
            "eval" | "evaluation" => Ok(Self::EvalMode),
            "evalfile" => Ok(Self::EvalFile),
            _ => Err(EngineError::InvalidOption(name.to_owned())),
        }
    }
}

fn parse_eval_mode_option(name: &str, value: Option<&str>) -> Result<EvalMode, EngineError> {
    let raw = required_option_value(name, value)?;
    EvalMode::from_uci(raw).ok_or_else(|| EngineError::InvalidOptionValue {
        option: name.to_owned(),
        value: raw.to_owned(),
    })
}

fn normalize_option_name(name: &str) -> String {
    name.to_ascii_lowercase().replace(' ', "")
}

fn parse_u32_option(
    name: &str,
    value: Option<&str>,
    min: u32,
    max: u32,
) -> Result<u32, EngineError> {
    let raw = required_option_value(name, value)?;
    let parsed = raw
        .parse::<u32>()
        .map_err(|_| EngineError::InvalidOptionValue {
            option: name.to_owned(),
            value: raw.to_owned(),
        })?;
    if parsed < min || parsed > max {
        return Err(EngineError::InvalidOptionValue {
            option: name.to_owned(),
            value: raw.to_owned(),
        });
    }
    Ok(parsed)
}

fn parse_bool_option(name: &str, value: Option<&str>) -> Result<bool, EngineError> {
    let raw = required_option_value(name, value)?;
    match raw.to_ascii_lowercase().as_str() {
        "true" | "on" | "1" => Ok(true),
        "false" | "off" | "0" => Ok(false),
        _ => Err(EngineError::InvalidOptionValue {
            option: name.to_owned(),
            value: raw.to_owned(),
        }),
    }
}

fn required_option_value<'a>(name: &str, value: Option<&'a str>) -> Result<&'a str, EngineError> {
    value.ok_or_else(|| EngineError::InvalidOptionValue {
        option: name.to_owned(),
        value: "<missing>".to_owned(),
    })
}
