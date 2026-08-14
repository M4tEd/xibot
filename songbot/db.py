"""SQLite persistence layer: schema, versioned migrations, connection helper.

Sync sqlite3 (check_same_thread=False, WAL, foreign_keys ON). The engine owns
all SQL through this module — no SQL elsewhere.
"""
