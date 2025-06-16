#!/usr/bin/env python3
"""
NeuroCluster Streamer API - Secret Generation Script

This script generates cryptographically secure secrets for the NCS API including:
- JWT secret keys with appropriate entropy
- API keys for external access
- Database passwords with complexity requirements
- Encryption keys for data protection
- Session secrets and CSRF tokens
- TLS/SSL certificates (self-signed for development)

Usage:
    python scripts/generate_secrets.py [options]

Options:
    --format FORMAT         Output format: json, env, yaml (default: json)
    --output FILE          Output file (default: stdout)
    --environment ENV      Environment: development, staging, production
    --length LENGTH        Secret length for keys (default: 64)
    --api-keys COUNT       Number of API keys to generate (default: 3)
    --no-passwords         Skip password generation
    --include-certs        Generate self-signed certificates
    --force                Overwrite existing output file
    --help                 Show this help message

Examples:
    python scripts/generate_secrets.py --format env --output .env.secrets
    python scripts/generate_secrets.py --environment production --length 128
    python scripts/generate_secrets.py --include-certs --format yaml
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import string
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)
from cryptography.x509.oid import NameOID


# Color constants for output
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    NC = "\033[0m"  # No Color


def log(level: str, message: str, color: bool = True):
    """Log message with optional color."""
    colors = {
        "INFO": Colors.GREEN,
        "WARN": Colors.YELLOW,
        "ERROR": Colors.RED,
        "DEBUG": Colors.BLUE,
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


class SecretGenerator:
    """Main secret generator class."""

    def __init__(self, environment: str = "development", length: int = 64):
        self.environment = environment
        self.length = length
        self.secrets = {}

        # Character sets for different types of secrets
        self.charset_alphanumeric = string.ascii_letters + string.digits
        self.charset_base64 = string.ascii_letters + string.digits + "+/"
        self.charset_hex = string.hexdigits.lower()
        self.charset_password = string.ascii_letters + string.digits + "!@#$%^&*"

    def generate_jwt_secret(self) -> str:
        """Generate a JWT secret key with high entropy."""
        log("INFO", f"Generating JWT secret key ({self.length} chars)")

        # Use URL-safe base64 encoding for JWT secrets
        random_bytes = secrets.token_bytes(self.length)
        jwt_secret = base64.urlsafe_b64encode(random_bytes).decode("utf-8")

        # Ensure it meets minimum length requirement
        if len(jwt_secret) < self.length:
            jwt_secret += secrets.token_urlsafe(self.length - len(jwt_secret))

        return jwt_secret[: self.length]

    def generate_api_keys(self, count: int = 3) -> List[str]:
        """Generate API keys for external access."""
        log("INFO", f"Generating {count} API keys")

        api_keys = []
        for i in range(count):
            # Create API key with prefix for identification
            key_id = secrets.token_hex(4)  # 8 character identifier
            key_secret = secrets.token_urlsafe(32)  # 43 character secret
            api_key = f"ncs_api_{self.environment}_{key_id}_{key_secret}"
            api_keys.append(api_key)

        return api_keys

    def generate_database_password(self, min_length: int = 16) -> str:
        """Generate a secure database password."""
        log("INFO", f"Generating database password ({min_length}+ chars)")

        # Ensure password meets complexity requirements
        password_length = max(min_length, 16)

        # Start with required character types
        password_chars = [
            secrets.choice(string.ascii_lowercase),
            secrets.choice(string.ascii_uppercase),
            secrets.choice(string.digits),
            secrets.choice("!@#$%^&*"),
        ]

        # Fill rest with random characters from full charset
        remaining_length = password_length - len(password_chars)
        for _ in range(remaining_length):
            password_chars.append(secrets.choice(self.charset_password))

        # Shuffle the characters
        secrets.SystemRandom().shuffle(password_chars)

        return "".join(password_chars)

    def generate_encryption_key(self, key_size: int = 32) -> str:
        """Generate encryption key for data protection."""
        log("INFO", f"Generating encryption key ({key_size} bytes)")

        # Generate random bytes and encode as base64
        key_bytes = secrets.token_bytes(key_size)
        return base64.b64encode(key_bytes).decode("utf-8")

    def generate_session_secret(self) -> str:
        """Generate session secret for web frameworks."""
        log("INFO", "Generating session secret")

        return secrets.token_urlsafe(64)

    def generate_csrf_token(self) -> str:
        """Generate CSRF protection token."""
        log("INFO", "Generating CSRF token")

        return secrets.token_hex(32)

    def generate_webhook_secret(self) -> str:
        """Generate webhook signing secret."""
        log("INFO", "Generating webhook secret")

        return secrets.token_hex(32)

    def generate_monitoring_token(self) -> str:
        """Generate token for monitoring endpoints."""
        log("INFO", "Generating monitoring token")

        return secrets.token_urlsafe(48)

    def generate_backup_encryption_key(self) -> str:
        """Generate key for backup encryption."""
        log("INFO", "Generating backup encryption key")

        return base64.b64encode(secrets.token_bytes(32)).decode("utf-8")

    def generate_redis_password(self) -> str:
        """Generate Redis password."""
        log("INFO", "Generating Redis password")

        return self.generate_database_password(20)

    def generate_salt(self, length: int = 32) -> str:
        """Generate cryptographic salt."""
        return secrets.token_hex(length)

    def generate_self_signed_cert(self) -> Dict[str, str]:
        """Generate self-signed SSL certificate for development."""
        log("INFO", "Generating self-signed SSL certificate")

        # Generate private key
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Create certificate
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Development"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "NCS API Development"),
                x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName("localhost"),
                        x509.DNSName("127.0.0.1"),
                        x509.IPAddress("127.0.0.1".encode()),
                    ]
                ),
                critical=False,
            )
            .sign(private_key, hashes.SHA256())
        )

        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption(),
        ).decode("utf-8")

        # Serialize certificate
        cert_pem = cert.public_bytes(Encoding.PEM).decode("utf-8")

        return {"private_key": private_pem, "certificate": cert_pem}

    def generate_all_secrets(
        self,
        api_key_count: int = 3,
        include_passwords: bool = True,
        include_certs: bool = False,
    ) -> Dict[str, Any]:
        """Generate all required secrets."""
        log("INFO", f"Generating secrets for {self.environment} environment")

        secrets_dict = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "environment": self.environment,
                "version": "1.0.0",
                "generator": "NCS API Secret Generator",
            },
            "jwt": {
                "secret_key": self.generate_jwt_secret(),
                "algorithm": "HS256",
                "expiry_hours": 24 if self.environment == "development" else 8,
            },
            "api_keys": self.generate_api_keys(api_key_count),
            "encryption": {
                "data_key": self.generate_encryption_key(32),
                "backup_key": self.generate_backup_encryption_key(),
                "salt": self.generate_salt(),
            },
            "session": {
                "secret": self.generate_session_secret(),
                "csrf_token": self.generate_csrf_token(),
            },
            "monitoring": {
                "token": self.generate_monitoring_token(),
                "webhook_secret": self.generate_webhook_secret(),
            },
        }

        # Add passwords if requested
        if include_passwords:
            secrets_dict["passwords"] = {
                "database": {
                    "postgres_password": self.generate_database_password(),
                    "admin_password": self.generate_database_password(20),
                },
                "redis_password": self.generate_redis_password(),
                "admin_user_password": self.generate_database_password(16),
            }

        # Add certificates if requested
        if include_certs:
            cert_data = self.generate_self_signed_cert()
            secrets_dict["certificates"] = {
                "ssl": {
                    "private_key": cert_data["private_key"],
                    "certificate": cert_data["certificate"],
                    "validity_days": 365,
                    "created_at": datetime.utcnow().isoformat(),
                }
            }

        # Environment-specific additions
        if self.environment == "production":
            secrets_dict["security"] = {
                "rate_limit_secret": self.generate_salt(16),
                "audit_signing_key": self.generate_encryption_key(32),
                "backup_verification_key": self.generate_salt(24),
            }

        return secrets_dict


class SecretFormatter:
    """Format secrets for different output types."""

    @staticmethod
    def format_json(secrets: Dict[str, Any], indent: int = 2) -> str:
        """Format secrets as JSON."""
        return json.dumps(secrets, indent=indent, sort_keys=True)

    @staticmethod
    def format_yaml(secrets: Dict[str, Any]) -> str:
        """Format secrets as YAML."""
        return yaml.dump(secrets, default_flow_style=False, sort_keys=True)

    @staticmethod
    def format_env(secrets: Dict[str, Any]) -> str:
        """Format secrets as environment variables."""
        lines = [
            "# NeuroCluster Streamer API Secrets",
            f"# Generated: {secrets['metadata']['generated_at']}",
            f"# Environment: {secrets['metadata']['environment']}",
            "",
        ]

        # JWT secrets
        lines.extend(
            [
                "# JWT Configuration",
                f"JWT_SECRET_KEY={secrets['jwt']['secret_key']}",
                f"JWT_ALGORITHM={secrets['jwt']['algorithm']}",
                f"JWT_EXPIRY_HOURS={secrets['jwt']['expiry_hours']}",
                "",
            ]
        )

        # API Keys
        api_keys_str = ",".join(secrets["api_keys"])
        lines.extend(["# API Keys", f"API_KEYS={api_keys_str}", ""])

        # Encryption keys
        lines.extend(
            [
                "# Encryption Configuration",
                f"DATA_ENCRYPTION_KEY={secrets['encryption']['data_key']}",
                f"BACKUP_ENCRYPTION_KEY={secrets['encryption']['backup_key']}",
                f"CRYPTO_SALT={secrets['encryption']['salt']}",
                "",
            ]
        )

        # Session secrets
        lines.extend(
            [
                "# Session Configuration",
                f"SESSION_SECRET={secrets['session']['secret']}",
                f"CSRF_TOKEN={secrets['session']['csrf_token']}",
                "",
            ]
        )

        # Monitoring
        lines.extend(
            [
                "# Monitoring Configuration",
                f"MONITORING_TOKEN={secrets['monitoring']['token']}",
                f"WEBHOOK_SECRET={secrets['monitoring']['webhook_secret']}",
                "",
            ]
        )

        # Passwords (if present)
        if "passwords" in secrets:
            lines.extend(
                [
                    "# Database Passwords",
                    f"DB_PASSWORD={secrets['passwords']['database']['postgres_password']}",
                    f"DB_ADMIN_PASSWORD={secrets['passwords']['database']['admin_password']}",
                    f"REDIS_PASSWORD={secrets['passwords']['redis_password']}",
                    f"ADMIN_USER_PASSWORD={secrets['passwords']['admin_user_password']}",
                    "",
                ]
            )

        # Security (if present)
        if "security" in secrets:
            lines.extend(
                [
                    "# Security Configuration",
                    f"RATE_LIMIT_SECRET={secrets['security']['rate_limit_secret']}",
                    f"AUDIT_SIGNING_KEY={secrets['security']['audit_signing_key']}",
                    f"BACKUP_VERIFICATION_KEY={secrets['security']['backup_verification_key']}",
                    "",
                ]
            )

        return "\n".join(lines)


def validate_environment(env: str) -> str:
    """Validate and return environment."""
    valid_environments = ["development", "staging", "production", "testing"]
    if env not in valid_environments:
        error_exit(
            f"Invalid environment: {env}. Must be one of: {', '.join(valid_environments)}"
        )
    return env


def validate_format(fmt: str) -> str:
    """Validate and return format."""
    valid_formats = ["json", "yaml", "env"]
    if fmt not in valid_formats:
        error_exit(f"Invalid format: {fmt}. Must be one of: {', '.join(valid_formats)}")
    return fmt


def check_output_file(output_file: str, force: bool) -> None:
    """Check if output file can be written."""
    if os.path.exists(output_file) and not force:
        error_exit(f"Output file exists: {output_file}. Use --force to overwrite.")

    # Check if directory is writable
    output_dir = os.path.dirname(output_file) or "."
    if not os.access(output_dir, os.W_OK):
        error_exit(f"Cannot write to directory: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate cryptographically secure secrets for NCS API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --format env --output .env.secrets
  %(prog)s --environment production --length 128
  %(prog)s --include-certs --format yaml
  %(prog)s --api-keys 5 --no-passwords
        """,
    )

    parser.add_argument(
        "--format",
        choices=["json", "yaml", "env"],
        default="json",
        help="Output format (default: json)",
    )

    parser.add_argument("--output", help="Output file (default: stdout)")

    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production", "testing"],
        default="development",
        help="Target environment (default: development)",
    )

    parser.add_argument(
        "--length", type=int, default=64, help="Secret length for keys (default: 64)"
    )

    parser.add_argument(
        "--api-keys",
        type=int,
        default=3,
        help="Number of API keys to generate (default: 3)",
    )

    parser.add_argument(
        "--no-passwords", action="store_true", help="Skip password generation"
    )

    parser.add_argument(
        "--include-certs", action="store_true", help="Generate self-signed certificates"
    )

    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output file"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Validate inputs
    environment = validate_environment(args.environment)
    output_format = validate_format(args.format)

    if args.output:
        check_output_file(args.output, args.force)

    if args.length < 32:
        log("WARN", "Secret length less than 32 characters may not be secure")

    # Generate secrets
    try:
        log("INFO", "Starting secret generation...")

        generator = SecretGenerator(environment, args.length)
        secrets_data = generator.generate_all_secrets(
            api_key_count=args.api_keys,
            include_passwords=not args.no_passwords,
            include_certs=args.include_certs,
        )

        # Format output
        if output_format == "json":
            output = SecretFormatter.format_json(secrets_data)
        elif output_format == "yaml":
            output = SecretFormatter.format_yaml(secrets_data)
        elif output_format == "env":
            output = SecretFormatter.format_env(secrets_data)

        # Write output
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)

            # Set secure permissions on output file
            os.chmod(args.output, 0o600)

            log("INFO", f"Secrets written to: {args.output}")
            log("INFO", f"File permissions set to 600 (owner read/write only)")

            if args.verbose:
                log("INFO", f"Generated {len(secrets_data['api_keys'])} API keys")
                if "passwords" in secrets_data:
                    log("INFO", "Generated database and admin passwords")
                if "certificates" in secrets_data:
                    log("INFO", "Generated self-signed SSL certificate")
        else:
            print(output)

        log("INFO", "Secret generation completed successfully")

        # Security reminder for production
        if environment == "production":
            log("WARN", "PRODUCTION SECRETS GENERATED!")
            log(
                "WARN",
                "Store these secrets securely and never commit to version control",
            )
            log("WARN", "Consider using a secrets management system for production")

    except KeyboardInterrupt:
        log("ERROR", "Secret generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        error_exit(f"Secret generation failed: {str(e)}")


if __name__ == "__main__":
    main()
