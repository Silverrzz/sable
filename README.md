# Sable

An in-development personal project written in Rust designed to play Chess better than i can.

## Included Features
- Sable uses a principal variation search plus quiescence search with iterative deepening
    - transposition table
    - move ordering
        - transposition table move ordering
        - previous PV move ordering
        - promotion ordering
        - SEE move ordering
    - null move pruning
    - late move reductions
    - history
        - quiet history
        - capture history
        - continuation history
        - counter moves
    - killers (2 per ply)
    - repetition handling
    - mate distance pruning
    - check extensions
    - futility pruning
    - reverse futility pruning
    - late quiet pruning
    - SEE pruning
    - razoring
    - aspiration windows
    - correction history
        - pawn
        - minor
        - non-pawn
        - continuation
- A hand crafted evaluation based on PeSTO's evaluation function
- An efficiently updatable neural network
    - Nightweave (Currently used model)
        - (768x64)>64>1 architecture
        - Trained on 3 iterations of training, with ~70 million positions of selfplay data per iteration
        - Trained using [Bullet](https://github.com/jw1912/bullet)
- A poor attempt at Lazy SMP for efficient multi-thread usage
- UCI protocol

## Future Features
- Sable's 1.0 net 'Nightweave' is a weak proof-of-concept to show myself I could do it, future Sable releases will hopefully come with a considerably stronger net
- Loads of search improvements, such as various types of extensions and corrhist features that I am missing
- A better packaged testing suite

## UCI Options
|Name|Type|Default|Min/Max or Vars|Description|
|-|-|-|-|-|
|Hash|spin|16|1 / 32768|Transposition table size in MiB.|
|Threads|spin|1|1 / 128|Number of search threads.|
|Ponder|check|false||Standard UCI ponder option. `go ponder` is held until `ponderhit` or `stop`.|
|MultiPV|spin|1|1 / 256|Number of principal variations to search and report.|
|UCI_Chess960|check|false||Enables Chess960 FEN parsing and castling move notation.|
|UCI_ShowWDL|check|false||Adds WDL values to UCI info lines.|
|Move Overhead|spin|100|0 / 10000|Milliseconds reserved from time controls to avoid flagging.|
|Clear Hash|button|||Clears the transposition table.|
|Eval|combo|build default|hce / nnue|Chooses between hand crafted evaluation and NNUE.|
|Eval File|string|embedded if compiled in, otherwise blank||Loads an NNUE file from disk, or `embedded` for the compiled-in net.|

## Strength
This section will be updated when more testing is done, from my own low-effort
evaluations and tests I estimate Sable to be around the level of 2900 elo in STC.

## Project Details
Sable is releasing with 1.0 in a blank repository. This is because the git repo used from 0.1.0 through to 0.61.3
was filled with private files, large blobs of data that shouldnt have been commited, and it was an all-round mess.

I've been working on Sable for about 2 months for sometimes over 12 hours a day,
primarily trying to teach myself how everything worked and using various wikis and other engines.

With only a moderate proficiency in Rust, in-line completions were frequently used to assist with writing Rust syntax.
A coding agent never directly touched my codebase, though while developing the code for NNUE, Mr GPT was consulted for explanations and snippets.

You can create your own Sable build with cargo build --release.

## Thanks :D
- Many members in the Stockfish discord server for their help with my questions (no particular order)
    - Chef
    - Dr Extension
    - Matt
    - Ciekce
    - DarkNeutrino
    - jb1729
    - Many others..!!
- Close friends who helped with datagen for my net
    - wnnb3dgy
    - HipHop
    - Tosiakowa
    - Bedthyme