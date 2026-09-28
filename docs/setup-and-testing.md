# Setup and testing

## Install from a source checkout

Use Python 3.11, [uv](https://docs.astral.sh/uv/) and a Node.js version satisfying the frontend lockfile's engine requirements. The recorded local checks used Python 3.11.14 and Node.js 25.9.0 on macOS. Linux and native desktop builds have not been verified for this snapshot.

Run from the repository root:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
(cd frontend && npm ci)
```

Installation may download packages. Python dependencies include both pinned and ranged requirements; there is no Python lockfile. The frontend uses `package-lock.json`.

Keep Python execution in the checkout root. Editable source-checkout use is the supported development procedure here; wheel installation and execution outside the checkout are unverified. The package imports top-level `tools` modules and uses checkout-relative resources.

## Isolate checks from real services

Do not run tests with production credentials or access to a real hand-history database. Private-deployment SSH/compose checks are excluded; optional API/CLI checks require explicit service configuration. Some older service checks rely on environment or executable availability rather than markers alone.

The following macOS wrapper matches the verification approach: a clean environment, isolated home directory and denied network access. On other platforms, use an equivalent network-isolated disposable environment; that environment has not been validated here.

```sh
TEST_HOME="$(mktemp -d)"
offline_check() {
  sandbox-exec -p '(version 1)(allow default)(deny network*)' \
    env -i PATH="$PATH" HOME="$TEST_HOME" CI=true \
    PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$@"
}
```

After dependency installation, run the focused backend checks:

```sh
offline_check .venv/bin/python -m pytest \
  -p pytest_asyncio.plugin -p _hypothesis_pytestplugin \
  tests/unit/test_export_boundaries.py \
  tests/phase6/test_api_smoke.py tests/phase6/test_study_config.py \
  tests/phase9/test_vendor_treys.py tests/phase14/test_knn_obs_id.py -q
```

The pytest-only input fixture creates small fabricated normalization, equity, chart and palette files for legacy checkout-relative readers, then removes them at teardown. It refuses to overwrite existing files. These are test inputs, not poker recommendations, fitted production artifacts or a demo corpus. If a test process is forcibly terminated, inspect any leftover generated files before another run.

Frontend checks and build:

```sh
(cd frontend && offline_check npm test -- --run \
  src/lib/__tests__/publicConfig.test.ts \
  src/lib/__tests__/rangeClient.test.ts \
  src/components/__tests__/ownedCards.test.tsx \
  src/components/__tests__/inlineReplayer.test.tsx)
(cd frontend && offline_check npm run build)
```

## Verification status

Recorded local checkpoint, 25 September 2026:

| Check | Result |
|---|---|
| Editable Python development installation; `npm ci` | Passed |
| Focused backend checks above | 28 passed |
| Focused frontend checks above | 16 passed |
| TypeScript/Vite production build | Passed |
| Full backend selection below | 1,395 passed; 45 failed; 21 skipped; 84 deselected |
| Full frontend tests | 105 passed; 9 failed |
| Strict Ruff | 4 findings |
| Strict mypy | 402 errors across 75 files |

Known failures include legacy collaborator mocks and response/schema contracts, simulation mock serialization, dashboard/parser expectations, frontend routing mocks, validation and SSE expectations. Typing debt includes missing generic parameters and dependency typing metadata. These checks are not suppressed or represented as green CI. Passing builds and focused tests do not establish runtime correctness, solver quality or production readiness. Browser visual verification has not been performed.

To reproduce the broader checks without changing their scope:

```sh
offline_check .venv/bin/python -m pytest \
  -p pytest_asyncio.plugin -p _hypothesis_pytestplugin \
  -m 'not integration and not e2e' -q
offline_check .venv/bin/python -m ruff check .
offline_check .venv/bin/python -m mypy src
(cd frontend && offline_check npm test -- --run)
```

## Optional runtime

The browser development server binds to `127.0.0.1:1420`. Its API URL defaults to `http://127.0.0.1:8765`; set `VITE_API_BASE_URL` before starting or building the frontend to override it. Variables with a `VITE_` prefix are browser-visible: never place credentials in them.

```sh
.venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8765 --workers 1
# In another terminal, from the checkout:
(cd frontend && npm run dev)
```

These commands do not provision services or data. Most analysis routes require caller-supplied PostgreSQL/TimescaleDB and Milvus instances, schemas, normalization inputs, equity data and retrieval records. Configure `TSDB_HOST`, `TSDB_PORT`, `TSDB_DB`, `TSDB_USER`, `TSDB_PASSWORD`, `MILVUS_HOST`, `MILVUS_PORT` and, where applicable, `MILVUS_TOKEN` through the environment. A `.env` file is not an automatic setup mechanism.

Solver operations require a separately supplied compatible executable; relevant wrappers accept explicit binary paths and some CLI paths read `POSTFLOP_CLI_BIN` or `EQUITY_DECILE_BIN`. Hand replay may require a user-owned database selected through `HM3_PATH`. No such binary or database is included. The native-shell files are excluded, so `npm run tauri` is not a supported build path for this snapshot.

The API can serve an existing `frontend/dist` build when started from the checkout. `/api/health` reports process/job-registry status, not database or corpus readiness. Review [security and data boundaries](security-and-data-boundaries.md) before running it. Migration, ingestion and solver scripts can modify databases or create substantial outputs; inspect and configure them deliberately rather than running them as installation steps.
