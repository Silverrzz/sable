# Sable

An in-development personal project written in Rust designed to play Chess better than i can!

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
    - singular extensions
    - internal iterative reductions
    - futility pruning
    - reverse futility pruning
    - late quiet pruning
    - capture SEE pruning
    - quiet SEE pruning
    - razoring
    - aspiration windows
    - correction history
        - pawn
        - minor
        - non-pawn
        - continuation
- A hand crafted evaluation based on PeSTO's evaluation function
- An efficiently updatable neural network
    - Nightweave
        - (768x64)>64>1 arch
        - Trained on 3 iterations of selfplay, with ~70 million positions of data per iteration
        - Trained using [Bullet](https://github.com/jw1912/bullet)
    - Vex (Currently used net)
        - (768x16hm>256)x2->1 arch
        - Trained on 3 iterations of selfplay, with ~800 million positions of data per iteration
        - Trained using [Bullet](https://github.com/jw1912/bullet)
- A rewritten movegen/board layer, because the old cozy middle layer had to go eventually
- A rebuilt movepicker, faster PV building, and a pile of small search speedups
- Lazy SMP for efficient multi-thread usage
- UCI protocol

## Future Features
- A net with more complex input features and multi hidden layers
- A better packaged testing suite

## UCI Options
|Name|Type|Default|Min/Max or Vars|Description|
|-|-|-|-|-|
|Hash|spin|16|1 / 32768|Transposition table size in MiB.|s
|Threads|spin|1|1 / 256|Number of search threads.|
|Ponder|check|false||`go ponder` is held until `ponderhit` or `stop`.|
|MultiPV|spin|1|1 / 256|Number of principal variations to search and report.|
|UseSoftNodes|check|false||Treats go nodes as a soft node limit for datagen.|
|UCI_Chess960|check|false||Enables Chess960 FEN parsing and castling move notation.|
|UCI_ShowWDL|check|false||Adds WDL values to UCI info lines.|
|Move Overhead|spin|100|0 / 10000|Milliseconds reserved from time controls to avoid flagging.|
|Clear Hash|button|||Clears the transposition table.|
|Eval|combo|build default|hce / nnue|Chooses between hand crafted evaluation and NNUE.|
|Eval File|string|embedded if compiled in, otherwise blank||Loads an NNUE file from disk, or `embedded` for the compiled-in net.|

## Strength
|Version|My Estimate|
|-|-|
|2.0|3300|
|1.1|2900|
|1.0|2800|

## Project Details
My primary goal with Sable is to learn more about low-level programming and also training cool networks

With only a moderate proficiency in Rust, in-line completions were frequently used to assist with writing Rust syntax.
A coding agent never directly touched my codebase, though while developing the code for NNUE, Mr GPT was consulted for explanations and snippets.

You can create your own Sable build with cargo build --release.

## Release Builds
The embedded NNUE is read from data/quantised.bin by default.
the source and identity with environment variables:

```text
SABLE_RELEASE_ID=2.0.0
SABLE_EVAL_LABEL=vex-1b
SABLE_DEFAULT_EVAL=nnue
SABLE_EVAL_FILE=<path to quantised net>
```

## Thanks :D
- Many members in the Stockfish discord server for their help with my questions (no particular order)
    - Chef
    - Dr Extension
    - Matt
    - Ciekce
    - DarkNeutrino
    - jb1729
    - Dan
    - Many others..!!
- Close friends who helped with datagen for my net
    - wnnb3dgy
    - HipHop
    - Tosiakowa
    - Bedthyme
- Members of MattBench for compute stuff
