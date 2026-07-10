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
- An efficiently updatable neural network - Shard 1.4
    - (768x16hm>512)x2->8 arch
    - Trained on a single iteration of selfplay, with ~1 billion positions of data
    - Trained using [Bullet](https://github.com/jw1912/bullet)
- Lazy SMP for efficient multi-thread usage
- UCI protocol

## Future Features
- A net with more complex input features and multi hidden layers
- Multilayer
- More layers after the first layer in my net
- A more complicated net with things like multiple layers and output buckets
- more elo

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
|Eval File|string|embedded if compiled in, otherwise blank||Loads a native Shard NNUE file from disk, or `embedded` for the compiled-in net.|

## Strength
|Version|My Estimate|CCRL 40/15|CCRL FRC 40/2|
|-|-|-|-|
|3.0.0|3450|-|-|
|2.0.0|3300|3258|3323|
|1.1.0|2900|2915|-|
|1.0.0|2800|-|-|

## Project Details
My primary goal with Sable is to learn more about low-level programming and also training cool networks

With only a moderate proficiency in Rust, in-line completions were frequently used to assist with writing Rust syntax.
While developing the code for NNUE, Mr GPT was consulted for explanations and snippets.
LLM agents have modified the codebase for faster mass deletions or param changes.

You can create your own Sable build with cargo build --release.

## Release Builds
The embedded Shard NNUE is read from data/quantised.bin when that file exists.
Builds without an embedded net still compile, but search requires `Eval File` to be loaded before use.

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
