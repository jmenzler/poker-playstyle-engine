# Architecture

## Data flow

1. A `GameState` describes the decision point: cards, position, action sequence, pot and stack information.
2. The canonicalizer and feature extractors produce preflop or postflop embeddings and structured retrieval filters. Runtime encoding expects caller-supplied equity data; normalization manifests are separate inputs.
3. The decision engine queries Milvus, blends retrieved action distributions and constrains the result to the available action vocabulary. A preflop chart path is also present; no production chart is included.
4. The simulation layer adapts game state and actions to its RLCard-based environment. Evaluation modules compare strategies and record analysis results.
5. Study workflows expose decisions, neighboring observations, patch history, solver verification, hand replay and evaluation through FastAPI. The React browser client calls those endpoints and renders analysis and study controls.

## Persistence and optional solver

PostgreSQL/TimescaleDB stores observations, metrics, patches and workflow records. Milvus supplies vector retrieval. SQL schema migrations are retained for inspection; running them changes a database and is not part of the credential-free checks.

The Python solver wrapper communicates with an external executable through JSON and subprocess calls. Solver execution requires a separately supplied compatible binary. The solver-linked Rust crates and all other Rust/native-shell sources are excluded from this snapshot. Python wrapper and mocked protocol tests remain; there is no replacement solver implementation or silent demo backend.

Several tools prepare data, fit normalization inputs or populate retrieval collections. They require user-owned input data and configured services. Their presence is not evidence that the snapshot contains a prepared corpus or that those integrations have been verified here.

## Source layout

- `src/protocols`, `src/canonicalizer`: data contracts and encoding.
- `src/decision_engine`, `tools/feature_extractors`: retrieval decisions and feature computation.
- `src/solver`, `src/autoloop`, `src/patch_engine.py`: optional solver workflows and patch application.
- `src/sim`, `src/eval`, `src/metrics`: simulation and measurement code.
- `src/study`, `src/api`, `src/cli`: study operations, API and CLI entry points.
- `migrations`, `runner`, `scripts`: schema and operational tooling; not an automatic deployment recipe.
- `frontend`: browser application, frontend tests and package-manager lockfile.
- `tests`: unit, property and service-backed tests. Fabricated test inputs do not form a runtime dataset.

## Limits

The application uses local listener defaults and a local-origin CORS allowlist, but has no application authentication system. External access control is the operator's responsibility. Some older tests disagree with current contracts, and strict typing is incomplete. The native desktop shell, real datasets, solver outputs and deployment configuration are deliberately absent.
