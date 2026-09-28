"""Phase 6 Study Tool subsystem.

Pure-function utilities consumed by:
- src/api/* (FastAPI study-tool endpoints)
- src/cli/* (CLI-01..07 surface)
- src/autoloop/* (Stage A leak ranking)

This package is import-side-effect-free. Submodules may import each other freely
but MUST NOT import database/network code at module load time.
"""
