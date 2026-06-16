use anyhow::Result;
use sable_engine::{Engine, SearchRequest};
use std::{
    io::{self, Write},
    sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use super::protocol::format_uci_info;

pub(super) enum WorkerCommand {
    Start {
        engine: Box<Engine>,
        request: Box<SearchRequest>,
        search_id: u64,
    },
    Stop,
    PonderHit,
    Quit,
}

pub(super) enum WorkerEvent {
    Info {
        search_id: u64,
        line: String,
    },
    BestMove {
        search_id: u64,
        best: String,
        ponder: Option<String>,
    },
    Error {
        search_id: u64,
        err: String,
    },
}

struct ActiveSearch {
    stop_flag: Arc<AtomicBool>,
    pondering: Arc<AtomicBool>,
    handle: thread::JoinHandle<()>,
}

pub(super) fn spawn_search_worker() -> (Sender<WorkerCommand>, Receiver<WorkerEvent>) {
    let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCommand>();
    let (evt_tx, evt_rx) = mpsc::channel::<WorkerEvent>();
    thread::spawn(move || worker_loop(cmd_rx, evt_tx));
    (cmd_tx, evt_rx)
}

pub(super) fn drain_worker_events(
    worker_rx: &Receiver<WorkerEvent>,
    stdout: &mut io::Stdout,
    search_in_progress: &mut bool,
    active_search_id: &mut Option<u64>,
) -> Result<()> {
    loop {
        match worker_rx.try_recv() {
            Ok(WorkerEvent::Info { search_id, line }) => {
                if Some(search_id) == *active_search_id {
                    writeln!(stdout, "{line}")?;
                    stdout.flush()?;
                }
            }
            Ok(WorkerEvent::BestMove {
                search_id,
                best,
                ponder,
            }) => {
                if Some(search_id) == *active_search_id {
                    write_best_move(stdout, &best, ponder.as_deref())?;
                    *search_in_progress = false;
                    *active_search_id = None;
                }
            }
            Ok(WorkerEvent::Error { search_id, err }) => {
                if Some(search_id) == *active_search_id {
                    writeln!(stdout, "info string search error: {err}")?;
                    writeln!(stdout, "bestmove 0000")?;
                    stdout.flush()?;
                    *search_in_progress = false;
                    *active_search_id = None;
                }
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => break,
        }
    }
    Ok(())
}

pub(super) fn cancel_active_search(
    worker_tx: &Sender<WorkerCommand>,
    search_in_progress: &mut bool,
    active_search_id: &mut Option<u64>,
) {
    if active_search_id.is_some() {
        let _ = worker_tx.send(WorkerCommand::Stop);
    }
    *search_in_progress = false;
    *active_search_id = None;
}

fn write_best_move(stdout: &mut io::Stdout, best: &str, ponder: Option<&str>) -> Result<()> {
    if let Some(ponder) = ponder {
        writeln!(stdout, "bestmove {best} ponder {ponder}")?;
    } else {
        writeln!(stdout, "bestmove {best}")?;
    }
    stdout.flush()?;
    Ok(())
}

fn worker_loop(cmd_rx: Receiver<WorkerCommand>, evt_tx: Sender<WorkerEvent>) {
    let mut active: Option<ActiveSearch> = None;
    loop {
        match cmd_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(WorkerCommand::Start {
                engine,
                request,
                search_id,
            }) => {
                stop_active_search(&mut active);
                active = Some(start_search(*engine, *request, search_id, evt_tx.clone()));
            }
            Ok(WorkerCommand::Stop) => {
                if let Some(active_search) = &active {
                    active_search.stop_flag.store(true, Ordering::Relaxed);
                }
            }
            Ok(WorkerCommand::PonderHit) => {
                if let Some(active_search) = &active {
                    active_search.pondering.store(false, Ordering::Relaxed);
                }
            }
            Ok(WorkerCommand::Quit) => {
                stop_active_search(&mut active);
                break;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                stop_active_search(&mut active);
                break;
            }
        }

        join_completed_search(&mut active);
    }
}

fn start_search(
    engine: Engine,
    request: SearchRequest,
    search_id: u64,
    evt_tx: Sender<WorkerEvent>,
) -> ActiveSearch {
    let stop_flag = Arc::new(AtomicBool::new(false));
    let pondering = Arc::new(AtomicBool::new(request.ponder));
    let stop_flag_clone = Arc::clone(&stop_flag);
    let pondering_clone = Arc::clone(&pondering);
    let handle = thread::spawn(move || {
        run_search(
            engine,
            request,
            search_id,
            stop_flag_clone,
            pondering_clone,
            evt_tx,
        )
    });
    ActiveSearch {
        stop_flag,
        pondering,
        handle,
    }
}

fn join_completed_search(active: &mut Option<ActiveSearch>) {
    if active
        .as_ref()
        .is_some_and(|search| search.handle.is_finished())
        && let Some(search) = active.take()
    {
        let _ = search.handle.join();
    }
}

fn run_search(
    engine: Engine,
    request: SearchRequest,
    search_id: u64,
    stop_flag: Arc<AtomicBool>,
    pondering: Arc<AtomicBool>,
    evt_tx: Sender<WorkerEvent>,
) {
    let result = match engine.search_with_controls(
        &request,
        Some(stop_flag.as_ref()),
        Some(pondering.as_ref()),
        |info| {
            let _ = evt_tx.send(WorkerEvent::Info {
                search_id,
                line: format_uci_info(info, engine.show_wdl_option_value()),
            });
        },
    ) {
        Ok(result) => result,
        Err(err) => {
            let _ = evt_tx.send(WorkerEvent::Error {
                search_id,
                err: err.to_string(),
            });
            return;
        }
    };

    wait_for_ponderhit_or_stop(&request, &pondering, &stop_flag);
    send_final_info(&engine, &result, search_id, &evt_tx);
    send_best_move(engine, request, result, search_id, pondering, evt_tx);
}

fn wait_for_ponderhit_or_stop(
    request: &SearchRequest,
    pondering: &AtomicBool,
    stop_flag: &AtomicBool,
) {
    while request.ponder && pondering.load(Ordering::Relaxed) && !stop_flag.load(Ordering::Relaxed)
    {
        thread::sleep(Duration::from_millis(5));
    }
}

fn send_best_move(
    engine: Engine,
    request: SearchRequest,
    result: sable_engine::SearchResult,
    search_id: u64,
    pondering: Arc<AtomicBool>,
    evt_tx: Sender<WorkerEvent>,
) {
    let best = result
        .best_move
        .map(|mv| engine.format_uci_move(mv))
        .unwrap_or_else(|| "0000".to_owned());
    let ponder = if request.ponder && pondering.load(Ordering::Relaxed) {
        None
    } else {
        result.ponder_move.map(|mv| engine.format_uci_move(mv))
    };
    let _ = evt_tx.send(WorkerEvent::BestMove {
        search_id,
        best,
        ponder,
    });
}

fn send_final_info(
    engine: &Engine,
    result: &sable_engine::SearchResult,
    search_id: u64,
    evt_tx: &Sender<WorkerEvent>,
) {
    let _ = evt_tx.send(WorkerEvent::Info {
        search_id,
        line: format_uci_info(&result.info, engine.show_wdl_option_value()),
    });
}

fn stop_active_search(active: &mut Option<ActiveSearch>) {
    if let Some(search) = active.take() {
        search.stop_flag.store(true, Ordering::Relaxed);
        let _ = search.handle.join();
    }
}
