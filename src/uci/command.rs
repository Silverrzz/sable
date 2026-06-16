use sable_engine::{SearchLimits, SearchRequest, TimeControl};

#[derive(Debug)]
pub(super) enum UciCommand {
    Uci,
    IsReady,
    UciNewGame,
    Position(PositionCommand),
    Go(SearchRequest),
    Stop,
    PonderHit,
    SetOption {
        name: String,
        value: Option<String>,
    },
    Register {
        name: Option<String>,
        code: Option<String>,
        later: bool,
    },
    Debug(bool),
    Eval,
    VerboseEval,
    Quit,
    Unknown,
    ParseError(String),
}

#[derive(Debug)]
pub(super) enum PositionBase {
    StartPos,
    Fen(String),
}

#[derive(Debug)]
pub(super) struct PositionCommand {
    pub(super) base: PositionBase,
    pub(super) moves: Vec<String>,
}

pub(super) fn parse_uci_command(input: &str) -> UciCommand {
    let (command, rest) = input.split_once(' ').unwrap_or((input, ""));
    match command {
        "uci" if rest.is_empty() => UciCommand::Uci,
        "isready" if rest.is_empty() => UciCommand::IsReady,
        "ucinewgame" if rest.is_empty() => UciCommand::UciNewGame,
        "stop" if rest.is_empty() => UciCommand::Stop,
        "ponderhit" if rest.is_empty() => UciCommand::PonderHit,
        "quit" if rest.is_empty() => UciCommand::Quit,
        "eval" if rest.is_empty() => UciCommand::Eval,
        "veval" if rest.is_empty() => UciCommand::VerboseEval,
        "debug" if input.contains(' ') => UciCommand::Debug(rest.trim() == "on"),
        "setoption" if input.contains(' ') => parse_setoption(rest),
        "register" if input.contains(' ') => parse_register(rest),
        "position" if input.contains(' ') => {
            parse_position_command(rest).map_or(UciCommand::Unknown, UciCommand::Position)
        }
        "go" if rest.is_empty() => UciCommand::Go(SearchRequest::default()),
        "go" => match parse_go(rest) {
            Ok(request) => UciCommand::Go(request),
            Err(err) => UciCommand::ParseError(format!("go parse error: {err}")),
        },
        _ => UciCommand::Unknown,
    }
}

fn parse_setoption(rest: &str) -> UciCommand {
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.is_empty() {
        return UciCommand::Unknown;
    }

    let Some(name) = named_field(&tokens, "name", &["value"]) else {
        return UciCommand::Unknown;
    };
    let value = named_field(&tokens, "value", &[]);

    UciCommand::SetOption {
        name,
        value: value.filter(|v| !v.is_empty()),
    }
}

fn parse_register(rest: &str) -> UciCommand {
    let trimmed = rest.trim();
    if trimmed == "later" {
        return UciCommand::Register {
            name: None,
            code: None,
            later: true,
        };
    }

    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    let name = named_field(&tokens, "name", &["code"]);
    let code = named_field(&tokens, "code", &[]);

    UciCommand::Register {
        name: name.filter(|v| !v.is_empty()),
        code: code.filter(|v| !v.is_empty()),
        later: false,
    }
}

fn named_field(tokens: &[&str], marker: &str, stop_markers: &[&str]) -> Option<String> {
    let start = tokens.iter().position(|token| *token == marker)? + 1;
    let end = tokens[start..]
        .iter()
        .position(|token| stop_markers.contains(token))
        .map(|offset| start + offset)
        .unwrap_or(tokens.len());
    (start < end).then(|| tokens[start..end].join(" "))
}

fn parse_position_command(rest: &str) -> Option<PositionCommand> {
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.is_empty() {
        return None;
    }

    let moves_index = tokens.iter().position(|token| *token == "moves");
    let moves = moves_index
        .map(|idx| {
            tokens[idx + 1..]
                .iter()
                .map(|mv| (*mv).to_owned())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    match tokens[0] {
        "startpos" => Some(PositionCommand {
            base: PositionBase::StartPos,
            moves,
        }),
        "fen" => {
            let fen_end = moves_index.unwrap_or(tokens.len());
            let fen = tokens[1..fen_end].join(" ");
            Some(PositionCommand {
                base: PositionBase::Fen(fen),
                moves,
            })
        }
        _ => None,
    }
}

fn parse_go(rest: &str) -> Result<SearchRequest, String> {
    let mut parser = GoParser::new(rest);
    let mut request = SearchRequest::default();
    let mut time_control = TimeControl::default();
    let mut limits = SearchLimits::default();

    while let Some(token) = parser.current() {
        match token {
            "ponder" => {
                request.ponder = true;
                parser.advance_flag();
            }
            "wtime" => {
                time_control.white_time_ms = Some(parser.u64_arg("wtime")?);
            }
            "btime" => {
                time_control.black_time_ms = Some(parser.u64_arg("btime")?);
            }
            "winc" => {
                time_control.white_increment_ms = Some(parser.u64_arg("winc")?);
            }
            "binc" => {
                time_control.black_increment_ms = Some(parser.u64_arg("binc")?);
            }
            "movestogo" => {
                time_control.moves_to_go = Some(parser.nonzero_u32_arg("movestogo")?);
            }
            "depth" => {
                limits.depth = Some(parser.nonzero_u32_arg("depth")?);
            }
            "nodes" => {
                limits.nodes = Some(parser.nonzero_u64_arg("nodes")?);
            }
            "softnodes" => {
                limits.soft_nodes = Some(parser.nonzero_u64_arg("softnodes")?);
            }
            "hardnodes" => {
                limits.hard_nodes = Some(parser.nonzero_u64_arg("hardnodes")?);
            }
            "mate" => {
                limits.mate = Some(parser.nonzero_u32_arg("mate")?);
            }
            "movetime" => {
                limits.move_time_ms = Some(parser.u64_arg("movetime")?);
            }
            "infinite" => {
                limits.infinite = true;
                parser.advance_flag();
            }
            "searchmoves" => {
                request.search_moves.extend(parser.search_moves());
            }
            other => return Err(format!("unknown go token: {other}")),
        }
    }

    if time_control.white_time_ms.is_some()
        || time_control.black_time_ms.is_some()
        || time_control.white_increment_ms.is_some()
        || time_control.black_increment_ms.is_some()
        || time_control.moves_to_go.is_some()
    {
        request.time_control = Some(time_control);
    }
    request.limits = limits;
    Ok(request)
}

struct GoParser<'a> {
    tokens: Vec<&'a str>,
    idx: usize,
}

impl<'a> GoParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            tokens: input.split_whitespace().collect(),
            idx: 0,
        }
    }

    fn current(&self) -> Option<&'a str> {
        self.tokens.get(self.idx).copied()
    }

    fn advance_flag(&mut self) {
        self.idx += 1;
    }

    fn arg(&mut self, name: &str) -> Result<&'a str, String> {
        let value = self
            .tokens
            .get(self.idx + 1)
            .copied()
            .ok_or_else(|| format!("missing value for {name}"))?;
        self.idx += 2;
        Ok(value)
    }

    fn u64_arg(&mut self, name: &str) -> Result<u64, String> {
        let raw = self.arg(name)?;
        raw
            .parse::<u64>()
            .map_err(|_| format!("invalid integer for {name}: {raw}"))
    }

    fn u32_arg(&mut self, name: &str) -> Result<u32, String> {
        let raw = self.arg(name)?;
        raw
            .parse::<u32>()
            .map_err(|_| format!("invalid integer for {name}: {raw}"))
    }

    fn nonzero_u64_arg(&mut self, name: &str) -> Result<u64, String> {
        let value = self.u64_arg(name)?;
        if value == 0 {
            Err(format!("{name} must be >= 1"))
        } else {
            Ok(value)
        }
    }

    fn nonzero_u32_arg(&mut self, name: &str) -> Result<u32, String> {
        let value = self.u32_arg(name)?;
        if value == 0 {
            Err(format!("{name} must be >= 1"))
        } else {
            Ok(value)
        }
    }

    fn search_moves(&mut self) -> impl Iterator<Item = String> + '_ {
        self.idx += 1;
        std::iter::from_fn(move || {
            let token = self.current()?;
            if is_go_keyword(token) {
                None
            } else {
                self.idx += 1;
                Some(token.to_owned())
            }
        })
    }
}

fn is_go_keyword(token: &str) -> bool {
    matches!(
        token,
        "ponder"
            | "wtime"
            | "btime"
            | "winc"
            | "binc"
            | "movestogo"
            | "depth"
            | "nodes"
            | "softnodes"
            | "hardnodes"
            | "mate"
            | "movetime"
            | "infinite"
            | "searchmoves"
    )
}
