use anyhow::Result;
use sable_engine::{Engine, SearchRequest};
use std::{
    io::{self, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender, TryRecvError},
    },
    thread,
    time::Duration,
};

use super::protocol::format_uci_info;

enum WorkerEvent {
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
    stop: Arc<AtomicBool>,
    pondering: Arc<AtomicBool>,
    handle: thread::JoinHandle<()>,
}

pub(super) struct SearchWorker {
    event_tx: Sender<WorkerEvent>,
    event_rx: Receiver<WorkerEvent>,
    active: Option<ActiveSearch>,
}

impl SearchWorker {
    pub(super) fn new() -> Self {
        let (event_tx, event_rx) = mpsc::channel();
        Self {
            event_tx,
            event_rx,
            active: None,
        }
    }

    pub(super) fn start(&mut self, engine: Engine, request: SearchRequest, search_id: u64) {
        self.shutdown();
        let stop = Arc::new(AtomicBool::new(false));
        let pondering = Arc::new(AtomicBool::new(request.ponder));
        let search_stop = Arc::clone(&stop);
        let search_pondering = Arc::clone(&pondering);
        let event_tx = self.event_tx.clone();
        let handle = thread::spawn(move || {
            run_search_task(
                engine,
                request,
                search_id,
                search_stop,
                search_pondering,
                event_tx,
            );
        });
        self.active = Some(ActiveSearch {
            stop,
            pondering,
            handle,
        });
    }

    pub(super) fn stop(&self) {
        if let Some(search) = &self.active {
            search.stop.store(true, Ordering::Relaxed);
        }
    }

    pub(super) fn ponder_hit(&self) {
        if let Some(search) = &self.active {
            search.pondering.store(false, Ordering::Relaxed);
        }
    }

    pub(super) fn shutdown(&mut self) {
        if let Some(search) = self.active.take() {
            search.stop.store(true, Ordering::Relaxed);
            let _ = search.handle.join();
        }
    }

    pub(super) fn drain_events(
        &mut self,
        stdout: &mut io::Stdout,
        active_search_id: &mut Option<u64>,
    ) -> Result<()> {
        loop {
            match self.event_rx.try_recv() {
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
                        if let Some(ponder) = ponder {
                            writeln!(stdout, "bestmove {best} ponder {ponder}")?;
                        } else {
                            writeln!(stdout, "bestmove {best}")?;
                        }
                        stdout.flush()?;
                        *active_search_id = None;
                    }
                }
                Ok(WorkerEvent::Error { search_id, err }) => {
                    if Some(search_id) == *active_search_id {
                        writeln!(stdout, "info string search error: {err}")?;
                        writeln!(stdout, "bestmove 0000")?;
                        stdout.flush()?;
                        *active_search_id = None;
                    }
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }

        if self
            .active
            .as_ref()
            .is_some_and(|search| search.handle.is_finished())
            && let Some(search) = self.active.take()
        {
            let _ = search.handle.join();
        }
        Ok(())
    }
}

fn run_search_task(
    engine: Engine,
    request: SearchRequest,
    search_id: u64,
    stop: Arc<AtomicBool>,
    pondering: Arc<AtomicBool>,
    event_tx: Sender<WorkerEvent>,
) {
    let result = match engine.search_with_controls(
        &request,
        Some(stop.as_ref()),
        Some(pondering.as_ref()),
        |info| {
            let _ = event_tx.send(WorkerEvent::Info {
                search_id,
                line: format_uci_info(&engine, info, engine.show_wdl_option_value()),
            });
        },
    ) {
        Ok(result) => result,
        Err(err) => {
            let _ = event_tx.send(WorkerEvent::Error {
                search_id,
                err: err.to_string(),
            });
            return;
        }
    };

    while request.ponder && pondering.load(Ordering::Relaxed) && !stop.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(5));
    }

    let _ = event_tx.send(WorkerEvent::Info {
        search_id,
        line: format_uci_info(&engine, &result.info, engine.show_wdl_option_value()),
    });
    let best = result
        .best_move
        .map(|mv| engine.format_uci_move(mv))
        .unwrap_or_else(|| "0000".to_owned());
    let ponder = (!request.ponder || !pondering.load(Ordering::Relaxed))
        .then(|| result.ponder_move.map(|mv| engine.format_uci_move(mv)))
        .flatten();
    let _ = event_tx.send(WorkerEvent::BestMove {
        search_id,
        best,
        ponder,
    });
}
