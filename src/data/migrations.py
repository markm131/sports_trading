# src/data/migrations.py
"""Database migrations for adding new columns."""

from src.data.db_writer import get_db_connection


def add_xg_columns():
    """Add xG columns to matches table."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if columns already exist
    cursor.execute("PRAGMA table_info(matches)")
    existing_cols = [row[1] for row in cursor.fetchall()]

    migrations = []

    if "home_xg" not in existing_cols:
        migrations.append("ALTER TABLE matches ADD COLUMN home_xg REAL")

    if "away_xg" not in existing_cols:
        migrations.append("ALTER TABLE matches ADD COLUMN away_xg REAL")

    if "xg_source" not in existing_cols:
        migrations.append("ALTER TABLE matches ADD COLUMN xg_source TEXT")

    if not migrations:
        print("xG columns already exist")
        return

    for sql in migrations:
        print(f"Running: {sql}")
        cursor.execute(sql)

    conn.commit()
    conn.close()
    print(f"Added {len(migrations)} columns")


def verify_schema():
    """Show current schema."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(matches)")
    cols = cursor.fetchall()

    print("\nCurrent schema:")
    for col in cols:
        print(f"  {col[1]}: {col[2]}")

    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Database migrations")
    parser.add_argument("--add-xg", action="store_true", help="Add xG columns")
    parser.add_argument("--verify", action="store_true", help="Show schema")

    args = parser.parse_args()

    if args.add_xg:
        add_xg_columns()
    elif args.verify:
        verify_schema()
    else:
        parser.print_help()
