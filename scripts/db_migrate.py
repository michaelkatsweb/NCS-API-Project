#!/usr/bin/env python3
"""
NeuroCluster Streamer API - Database Migration Management Script

This script manages database schema migrations, data seeding, and maintenance tasks:
- Initialize database schema from migrations
- Apply incremental schema updates
- Rollback migrations to previous versions
- Seed database with initial data
- Backup and restore database content
- Validate database integrity and performance
- Environment-specific migration handling

Usage:
    python scripts/db_migrate.py [command] [options]

Commands:
    init              Initialize database schema
    migrate           Apply pending migrations
    rollback          Rollback to previous migration
    seed              Seed database with initial data
    backup            Create database backup
    restore           Restore from backup
    status            Show migration status
    validate          Validate database integrity
    reset             Reset database (dangerous!)

Options:
    --environment ENV     Target environment (development, staging, production)
    --migration ID        Specific migration ID
    --backup-file FILE    Backup file path
    --dry-run            Show what would be done without executing
    --force              Force execution without confirmation
    --verbose            Enable verbose output
    --help               Show this help message

Examples:
    python scripts/db_migrate.py init --environment development
    python scripts/db_migrate.py migrate --dry-run
    python scripts/db_migrate.py rollback --migration 002
    python scripts/db_migrate.py backup --backup-file backup.sql
    python scripts/db_migrate.py seed --environment staging
"""

import os
import sys
import argparse
import logging
import psycopg2
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse
import tempfile
import shutil

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    from sqlalchemy import create_engine, text, MetaData, Table, Column, String, DateTime, Integer
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.exc import SQLAlchemyError
    from config import settings
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running from the project root with dependencies installed")
    sys.exit(1)

# Color constants
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    NC = '\033[0m'

def log(level: str, message: str, color: bool = True):
    """Log message with optional color."""
    colors = {
        'INFO': Colors.GREEN,
        'WARN': Colors.YELLOW,
        'ERROR': Colors.RED,
        'DEBUG': Colors.BLUE,
        'MIGRATE': Colors.PURPLE
    }
    
    if color and sys.stdout.isatty():
        color_code = colors.get(level, Colors.NC)
        print(f"{color_code}[{level}]{Colors.NC} {message}")
    else:
        print(f"[{level}] {message}")

def error_exit(message: str):
    """Print error and exit."""
    log("ERROR", message)
    sys.exit(1)

class MigrationTracker:
    """Track migration status in database."""
    
    def __init__(self, engine):
        self.engine = engine
        self.metadata = MetaData()
        
        # Define migrations table
        self.migrations_table = Table(
            'migration_history',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('migration_id', String(255), unique=True, nullable=False),
            Column('filename', String(255), nullable=False),
            Column('checksum', String(64), nullable=False),
            Column('applied_at', DateTime(timezone=True), nullable=False),
            Column('applied_by', String(255), nullable=False),
            Column('execution_time_ms', Integer),
            Column('success', String(10), nullable=False)
        )
        
    def ensure_migration_table(self):
        """Create migration tracking table if it doesn't exist."""
        try:
            self.metadata.create_all(self.engine, tables=[self.migrations_table])
            log("DEBUG", "Migration tracking table ensured")
        except Exception as e:
            error_exit(f"Failed to create migration table: {e}")
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration IDs."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT migration_id FROM migration_history WHERE success = 'true' ORDER BY applied_at")
                )
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            log("WARN", f"Could not fetch applied migrations: {e}")
            return []
    
    def record_migration(self, migration_id: str, filename: str, checksum: str, 
                        execution_time_ms: int, success: bool, applied_by: str = "db_migrate.py"):
        """Record migration execution."""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO migration_history 
                        (migration_id, filename, checksum, applied_at, applied_by, execution_time_ms, success)
                        VALUES (:migration_id, :filename, :checksum, :applied_at, :applied_by, :execution_time_ms, :success)
                    """),
                    {
                        'migration_id': migration_id,
                        'filename': filename,
                        'checksum': checksum,
                        'applied_at': datetime.now(timezone.utc),
                        'applied_by': applied_by,
                        'execution_time_ms': execution_time_ms,
                        'success': str(success).lower()
                    }
                )
                conn.commit()
        except Exception as e:
            log("ERROR", f"Failed to record migration: {e}")
    
    def remove_migration_record(self, migration_id: str):
        """Remove migration record (for rollback)."""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("DELETE FROM migration_history WHERE migration_id = :migration_id"),
                    {'migration_id': migration_id}
                )
                conn.commit()
                log("DEBUG", f"Removed migration record: {migration_id}")
        except Exception as e:
            log("ERROR", f"Failed to remove migration record: {e}")

class DatabaseMigrator:
    """Main database migration manager."""
    
    def __init__(self, environment: str = "development", dry_run: bool = False):
        self.environment = environment
        self.dry_run = dry_run
        self.migrations_dir = project_root / "database" / "migrations"
        self.seeds_dir = project_root / "database" / "seeds"
        
        # Get database URL for environment
        self.database_url = self._get_database_url()
        
        # Create engine
        try:
            self.engine = create_engine(self.database_url, echo=False)
            self.tracker = MigrationTracker(self.engine)
        except Exception as e:
            error_exit(f"Failed to connect to database: {e}")
    
    def _get_database_url(self) -> str:
        """Get database URL for environment."""
        # Try environment variable first
        env_var = f"DATABASE_URL_{self.environment.upper()}"
        if env_var in os.environ:
            return os.environ[env_var]
        
        # Fall back to default DATABASE_URL
        if "DATABASE_URL" in os.environ:
            return os.environ["DATABASE_URL"]
        
        # Fall back to environment-specific defaults
        defaults = {
            "development": "postgresql://ncs_dev:ncs_dev_password@localhost:5432/ncs_dev",
            "testing": "postgresql://ncs_test:ncs_test_password@localhost:5432/ncs_test",
            "staging": "postgresql://ncs_staging:ncs_staging_password@localhost:5432/ncs_staging",
            "production": "postgresql://ncs_prod:ncs_prod_password@localhost:5432/ncs_prod"
        }
        
        return defaults.get(self.environment, defaults["development"])
    
    def _calculate_file_checksum(self, filepath: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _get_migration_files(self) -> List[Tuple[str, Path]]:
        """Get sorted list of migration files."""
        if not self.migrations_dir.exists():
            log("WARN", f"Migrations directory not found: {self.migrations_dir}")
            return []
        
        migrations = []
        for file_path in self.migrations_dir.glob("*.sql"):
            # Extract migration ID from filename (e.g., "001_init.sql" -> "001")
            migration_id = file_path.stem.split('_')[0]
            migrations.append((migration_id, file_path))
        
        # Sort by migration ID
        migrations.sort(key=lambda x: x[0])
        return migrations
    
    def _execute_sql_file(self, filepath: Path) -> Tuple[bool, int]:
        """Execute SQL file and return success status and execution time."""
        log("INFO", f"Executing SQL file: {filepath.name}")
        
        if self.dry_run:
            log("INFO", f"[DRY RUN] Would execute: {filepath}")
            return True, 0
        
        start_time = datetime.now()
        
        try:
            with open(filepath, 'r') as f:
                sql_content = f.read()
            
            # Split into individual statements
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            
            with self.engine.connect() as conn:
                for statement in statements:
                    if statement:
                        conn.execute(text(statement))
                conn.commit()
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            log("INFO", f"Successfully executed {filepath.name} in {execution_time}ms")
            return True, execution_time
            
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            log("ERROR", f"Failed to execute {filepath.name}: {e}")
            return False, execution_time
    
    def init_database(self):
        """Initialize database schema."""
        log("MIGRATE", f"Initializing database for {self.environment} environment")
        
        # Ensure migration tracking table exists
        if not self.dry_run:
            self.tracker.ensure_migration_table()
        
        # Check if database is already initialized
        applied_migrations = self.tracker.get_applied_migrations()
        if applied_migrations:
            log("WARN", f"Database already initialized with {len(applied_migrations)} migrations")
            return
        
        # Look for init.sql file
        init_file = self.migrations_dir / "init.sql"
        if not init_file.exists():
            error_exit(f"init.sql not found in {self.migrations_dir}")
        
        log("INFO", "Running initial database setup...")
        
        # Execute init.sql
        success, execution_time = self._execute_sql_file(init_file)
        
        if success and not self.dry_run:
            # Record the initialization
            checksum = self._calculate_file_checksum(init_file)
            self.tracker.record_migration("000", "init.sql", checksum, execution_time, True)
            log("INFO", "Database initialization completed successfully")
        elif success:
            log("INFO", "[DRY RUN] Database initialization would complete successfully")
        else:
            error_exit("Database initialization failed")
    
    def migrate(self, target_migration: Optional[str] = None):
        """Apply pending migrations."""
        log("MIGRATE", f"Applying migrations for {self.environment} environment")
        
        # Ensure migration tracking table exists
        if not self.dry_run:
            self.tracker.ensure_migration_table()
        
        # Get migration files and applied migrations
        migration_files = self._get_migration_files()
        applied_migrations = self.tracker.get_applied_migrations()
        
        if not migration_files:
            log("INFO", "No migration files found")
            return
        
        # Filter to pending migrations
        pending_migrations = []
        for migration_id, filepath in migration_files:
            if migration_id not in applied_migrations:
                pending_migrations.append((migration_id, filepath))
                if target_migration and migration_id == target_migration:
                    break
        
        if not pending_migrations:
            log("INFO", "No pending migrations")
            return
        
        log("INFO", f"Found {len(pending_migrations)} pending migrations")
        
        # Apply each pending migration
        for migration_id, filepath in pending_migrations:
            log("MIGRATE", f"Applying migration {migration_id}: {filepath.name}")
            
            success, execution_time = self._execute_sql_file(filepath)
            
            if success and not self.dry_run:
                checksum = self._calculate_file_checksum(filepath)
                self.tracker.record_migration(migration_id, filepath.name, checksum, execution_time, True)
                log("INFO", f"Migration {migration_id} applied successfully")
            elif success:
                log("INFO", f"[DRY RUN] Migration {migration_id} would apply successfully")
            else:
                if not self.dry_run:
                    checksum = self._calculate_file_checksum(filepath)
                    self.tracker.record_migration(migration_id, filepath.name, checksum, execution_time, False)
                error_exit(f"Migration {migration_id} failed")
        
        if not self.dry_run:
            log("INFO", "All migrations applied successfully")
        else:
            log("INFO", "[DRY RUN] All migrations would apply successfully")
    
    def rollback(self, target_migration: Optional[str] = None):
        """Rollback to previous migration."""
        log("MIGRATE", f"Rolling back migrations for {self.environment} environment")
        
        applied_migrations = self.tracker.get_applied_migrations()
        
        if not applied_migrations:
            log("INFO", "No migrations to rollback")
            return
        
        if target_migration:
            if target_migration not in applied_migrations:
                error_exit(f"Migration {target_migration} is not applied")
            
            # Find migrations to rollback
            rollback_index = applied_migrations.index(target_migration)
            migrations_to_rollback = applied_migrations[rollback_index + 1:]
        else:
            # Rollback last migration only
            migrations_to_rollback = [applied_migrations[-1]]
        
        if not migrations_to_rollback:
            log("INFO", "No migrations to rollback")
            return
        
        log("WARN", f"Rolling back {len(migrations_to_rollback)} migrations: {migrations_to_rollback}")
        
        if not self.dry_run and self.environment == "production":
            confirmation = input("This is PRODUCTION. Type 'ROLLBACK' to confirm: ")
            if confirmation != "ROLLBACK":
                log("INFO", "Rollback cancelled")
                return
        
        # Look for rollback scripts
        for migration_id in reversed(migrations_to_rollback):
            rollback_file = self.migrations_dir / f"{migration_id}_rollback.sql"
            
            if rollback_file.exists():
                log("MIGRATE", f"Rolling back migration {migration_id}")
                success, _ = self._execute_sql_file(rollback_file)
                
                if success and not self.dry_run:
                    self.tracker.remove_migration_record(migration_id)
                    log("INFO", f"Migration {migration_id} rolled back successfully")
                elif success:
                    log("INFO", f"[DRY RUN] Migration {migration_id} would rollback successfully")
                else:
                    error_exit(f"Rollback of migration {migration_id} failed")
            else:
                log("WARN", f"No rollback script found for migration {migration_id}")
                if not self.dry_run:
                    self.tracker.remove_migration_record(migration_id)
    
    def seed_database(self):
        """Seed database with initial data."""
        log("MIGRATE", f"Seeding database for {self.environment} environment")
        
        if not self.seeds_dir.exists():
            log("WARN", f"Seeds directory not found: {self.seeds_dir}")
            return
        
        # Look for environment-specific seed file
        seed_files = [
            self.seeds_dir / f"{self.environment}.sql",
            self.seeds_dir / "common.sql",
            self.seeds_dir / "seed.sql"
        ]
        
        executed_files = 0
        for seed_file in seed_files:
            if seed_file.exists():
                log("INFO", f"Executing seed file: {seed_file.name}")
                success, _ = self._execute_sql_file(seed_file)
                
                if success:
                    log("INFO", f"Seed file {seed_file.name} executed successfully")
                    executed_files += 1
                else:
                    log("ERROR", f"Seed file {seed_file.name} failed")
        
        if executed_files == 0:
            log("WARN", "No seed files found or executed")
        else:
            log("INFO", f"Database seeding completed ({executed_files} files)")
    
    def backup_database(self, backup_file: str):
        """Create database backup."""
        log("MIGRATE", f"Creating database backup: {backup_file}")
        
        # Parse database URL
        parsed_url = urlparse(self.database_url)
        
        # Build pg_dump command
        cmd = [
            "pg_dump",
            "--host", parsed_url.hostname or "localhost",
            "--port", str(parsed_url.port or 5432),
            "--username", parsed_url.username or "postgres",
            "--dbname", parsed_url.path.lstrip('/'),
            "--verbose",
            "--clean",
            "--if-exists",
            "--file", backup_file
        ]
        
        # Set password environment variable
        env = os.environ.copy()
        if parsed_url.password:
            env["PGPASSWORD"] = parsed_url.password
        
        if self.dry_run:
            log("INFO", f"[DRY RUN] Would run: {' '.join(cmd)}")
            return
        
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                log("INFO", f"Database backup created successfully: {backup_file}")
                
                # Add metadata to backup
                metadata = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "environment": self.environment,
                    "database_url": self.database_url.replace(parsed_url.password or "", "***"),
                    "applied_migrations": self.tracker.get_applied_migrations()
                }
                
                metadata_file = f"{backup_file}.metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                log("INFO", f"Backup metadata saved: {metadata_file}")
            else:
                error_exit(f"Backup failed: {result.stderr}")
                
        except FileNotFoundError:
            error_exit("pg_dump command not found. Please install PostgreSQL client tools.")
        except Exception as e:
            error_exit(f"Backup failed: {e}")
    
    def restore_database(self, backup_file: str, force: bool = False):
        """Restore database from backup."""
        log("MIGRATE", f"Restoring database from backup: {backup_file}")
        
        if not os.path.exists(backup_file):
            error_exit(f"Backup file not found: {backup_file}")
        
        # Check metadata if available
        metadata_file = f"{backup_file}.metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            log("INFO", f"Backup created: {metadata['created_at']}")
            log("INFO", f"Source environment: {metadata['environment']}")
            log("INFO", f"Applied migrations: {len(metadata['applied_migrations'])}")
        
        # Confirmation for production
        if not force and self.environment == "production":
            confirmation = input("This will DESTROY the production database. Type 'RESTORE' to confirm: ")
            if confirmation != "RESTORE":
                log("INFO", "Restore cancelled")
                return
        
        # Parse database URL
        parsed_url = urlparse(self.database_url)
        
        # Build psql command
        cmd = [
            "psql",
            "--host", parsed_url.hostname or "localhost",
            "--port", str(parsed_url.port or 5432),
            "--username", parsed_url.username or "postgres",
            "--dbname", parsed_url.path.lstrip('/'),
            "--file", backup_file
        ]
        
        # Set password environment variable
        env = os.environ.copy()
        if parsed_url.password:
            env["PGPASSWORD"] = parsed_url.password
        
        if self.dry_run:
            log("INFO", f"[DRY RUN] Would run: {' '.join(cmd)}")
            return
        
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                log("INFO", "Database restored successfully")
            else:
                error_exit(f"Restore failed: {result.stderr}")
                
        except FileNotFoundError:
            error_exit("psql command not found. Please install PostgreSQL client tools.")
        except Exception as e:
            error_exit(f"Restore failed: {e}")
    
    def show_status(self):
        """Show migration status."""
        log("INFO", f"Migration status for {self.environment} environment")
        
        try:
            # Get applied migrations
            applied_migrations = self.tracker.get_applied_migrations()
            
            # Get all migration files
            migration_files = self._get_migration_files()
            
            print(f"\nDatabase URL: {self.database_url}")
            print(f"Environment: {self.environment}")
            print(f"Applied migrations: {len(applied_migrations)}")
            print(f"Available migrations: {len(migration_files)}")
            
            if migration_files:
                print("\nMigration Status:")
                print("ID    | Status  | Filename")
                print("------|---------|------------------")
                
                for migration_id, filepath in migration_files:
                    status = "Applied" if migration_id in applied_migrations else "Pending"
                    print(f"{migration_id:5} | {status:7} | {filepath.name}")
            
            # Show detailed history
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(
                        text("""
                            SELECT migration_id, filename, applied_at, execution_time_ms, success
                            FROM migration_history 
                            ORDER BY applied_at DESC 
                            LIMIT 10
                        """)
                    )
                    
                    history = result.fetchall()
                    if history:
                        print("\nRecent Migration History:")
                        print("ID    | Filename         | Applied At          | Time(ms) | Success")
                        print("------|------------------|---------------------|----------|--------")
                        
                        for row in history:
                            applied_at = row[2].strftime("%Y-%m-%d %H:%M:%S") if row[2] else "Unknown"
                            print(f"{row[0]:5} | {row[1]:16} | {applied_at} | {row[3]:8} | {row[4]}")
            
            except Exception as e:
                log("WARN", f"Could not fetch migration history: {e}")
                
        except Exception as e:
            error_exit(f"Failed to get migration status: {e}")
    
    def validate_database(self):
        """Validate database integrity."""
        log("MIGRATE", f"Validating database for {self.environment} environment")
        
        checks = []
        
        try:
            with self.engine.connect() as conn:
                # Check if all expected tables exist
                tables_query = text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                """)
                
                result = conn.execute(tables_query)
                existing_tables = {row[0] for row in result.fetchall()}
                
                expected_tables = {
                    'migration_history', 'processing_sessions', 'clusters', 
                    'data_points', 'performance_metrics', 'audit_logs',
                    'user_activities', 'system_configurations'
                }
                
                missing_tables = expected_tables - existing_tables
                extra_tables = existing_tables - expected_tables
                
                if missing_tables:
                    checks.append(f"❌ Missing tables: {', '.join(missing_tables)}")
                else:
                    checks.append("✅ All expected tables present")
                
                if extra_tables:
                    checks.append(f"⚠️  Extra tables: {', '.join(extra_tables)}")
                
                # Check indexes
                indexes_query = text("""
                    SELECT schemaname, tablename, indexname
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                """)
                
                result = conn.execute(indexes_query)
                index_count = len(result.fetchall())
                checks.append(f"✅ Found {index_count} indexes")
                
                # Check constraints
                constraints_query = text("""
                    SELECT COUNT(*) 
                    FROM information_schema.table_constraints 
                    WHERE table_schema = 'public'
                """)
                
                result = conn.execute(constraints_query)
                constraint_count = result.scalar()
                checks.append(f"✅ Found {constraint_count} constraints")
                
                # Basic connectivity test
                conn.execute(text("SELECT 1"))
                checks.append("✅ Database connectivity test passed")
        
        except Exception as e:
            checks.append(f"❌ Database validation failed: {e}")
        
        print(f"\nDatabase Validation Results ({self.environment}):")
        for check in checks:
            print(f"  {check}")
        
        # Check migration consistency
        try:
            applied_migrations = self.tracker.get_applied_migrations()
            migration_files = self._get_migration_files()
            
            file_ids = {migration_id for migration_id, _ in migration_files}
            applied_ids = set(applied_migrations)
            
            if applied_ids <= file_ids:
                print("  ✅ Migration consistency check passed")
            else:
                missing_files = applied_ids - file_ids
                print(f"  ❌ Applied migrations missing files: {', '.join(missing_files)}")
        
        except Exception as e:
            print(f"  ❌ Migration consistency check failed: {e}")
    
    def reset_database(self, force: bool = False):
        """Reset database (dangerous operation)."""
        log("MIGRATE", f"RESETTING database for {self.environment} environment")
        
        if not force:
            if self.environment == "production":
                error_exit("Cannot reset production database without --force flag")
            
            confirmation = input(f"This will DESTROY all data in {self.environment}. Type 'RESET' to confirm: ")
            if confirmation != "RESET":
                log("INFO", "Reset cancelled")
                return
        
        if self.dry_run:
            log("INFO", "[DRY RUN] Would reset database")
            return
        
        try:
            with self.engine.connect() as conn:
                # Drop all tables
                conn.execute(text("DROP SCHEMA public CASCADE"))
                conn.execute(text("CREATE SCHEMA public"))
                conn.execute(text("GRANT ALL ON SCHEMA public TO public"))
                conn.commit()
            
            log("INFO", "Database reset completed")
            
            # Reinitialize
            self.init_database()
            
        except Exception as e:
            error_exit(f"Database reset failed: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Database migration management for NCS API",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        '--environment',
        choices=['development', 'staging', 'production', 'testing'],
        default='development',
        help='Target environment'
    )
    common_parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    common_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Init command
    subparsers.add_parser('init', parents=[common_parser], help='Initialize database schema')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', parents=[common_parser], help='Apply migrations')
    migrate_parser.add_argument('--migration', help='Target specific migration ID')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', parents=[common_parser], help='Rollback migrations')
    rollback_parser.add_argument('--migration', help='Rollback to specific migration ID')
    
    # Seed command
    subparsers.add_parser('seed', parents=[common_parser], help='Seed database with data')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', parents=[common_parser], help='Create database backup')
    backup_parser.add_argument('--backup-file', required=True, help='Backup file path')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', parents=[common_parser], help='Restore from backup')
    restore_parser.add_argument('--backup-file', required=True, help='Backup file path')
    restore_parser.add_argument('--force', action='store_true', help='Force restore without confirmation')
    
    # Status command
    subparsers.add_parser('status', parents=[common_parser], help='Show migration status')
    
    # Validate command
    subparsers.add_parser('validate', parents=[common_parser], help='Validate database integrity')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', parents=[common_parser], help='Reset database (dangerous!)')
    reset_parser.add_argument('--force', action='store_true', help='Force reset without confirmation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize migrator
    try:
        migrator = DatabaseMigrator(args.environment, args.dry_run)
    except Exception as e:
        error_exit(f"Failed to initialize migrator: {e}")
    
    # Execute command
    try:
        if args.command == 'init':
            migrator.init_database()
        elif args.command == 'migrate':
            migrator.migrate(getattr(args, 'migration', None))
        elif args.command == 'rollback':
            migrator.rollback(getattr(args, 'migration', None))
        elif args.command == 'seed':
            migrator.seed_database()
        elif args.command == 'backup':
            migrator.backup_database(args.backup_file)
        elif args.command == 'restore':
            migrator.restore_database(args.backup_file, getattr(args, 'force', False))
        elif args.command == 'status':
            migrator.show_status()
        elif args.command == 'validate':
            migrator.validate_database()
        elif args.command == 'reset':
            migrator.reset_database(getattr(args, 'force', False))
        else:
            error_exit(f"Unknown command: {args.command}")
    
    except KeyboardInterrupt:
        log("ERROR", "Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        error_exit(f"Command failed: {e}")

if __name__ == "__main__":
    main()