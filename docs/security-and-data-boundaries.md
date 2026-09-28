# Security and data boundaries

## Trusted local use only

The API has no application authentication or authorization layer. Listener defaults and the browser API URL are loopback-only, and CORS permits local development origins. CORS is not authentication and does not protect against non-browser clients.

Do not expose this development application directly to an untrusted network. Any remote deployment needs an independently designed access-control and transport-security boundary. The static-file fallback and filesystem/database operations have not been hardened for hostile users. There is no claim of production readiness or a completed application-security audit.

Use one API worker: the job registry and related coordination live in process memory. Multiple workers are not a verified deployment mode.

## Data and credentials

The snapshot excludes private histories, hand-history databases, player datasets, fitted normalization data, production equity tables, retrieved corpora, solver/range outputs, deployment secrets and operational topology. Tests use fabricated inputs where data is required; these do not supply a valid runtime corpus.

Supply only data you own or are authorized to process. Review the terms of any hand-history source, solver, model or external service separately. The repository's inclusion of an importer or wrapper grants no rights to third-party data or binaries.

Set service credentials through a local environment or secret-management mechanism, not committed files. Frontend environment variables are compiled into browser code and are not a secret store. Logs, generated bundles, caches, test output and database dumps may expose data from your own environment; keep them out of source releases.

## Optional integrations and mutation

The Python external-binary solver interface is retained. Solver-linked Rust crates and solver binaries are absent. Missing runtime dependencies or data can produce explicit errors; no simulated runtime service is substituted.

Database migrations, ingestion, patch application, rollback, evaluation and solver-queue operations can write to configured services. Do not point experiments or tests at a production database. A passing health endpoint or browser build does not verify those integrations.

The native desktop shell is excluded. Browser helpers may retain optional desktop detection, but native packaging is not supported by this snapshot.

## Review boundaries

Recorded verification covers source-checkout installation, focused mocked/configuration checks, the browser build and the explicitly disclosed wider check results. It does not establish poker profitability, numerical solver validity, a current populated corpus, operational uptime or safe autonomous execution.

Third-party licenses are preserved in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Release updates remain manual and must review source, tests, configuration, generated material and redistribution rights again.
