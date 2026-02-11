"""Tests for regex patterns in examples/example_config.toml."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from looselips.cli.config import build_regex_patterns, load_config

EXAMPLE_CONFIG = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "example_config.toml"
)

Patterns = dict[str, re.Pattern[str]]


@pytest.fixture(scope="module")
def patterns() -> Patterns:
    """Load regex patterns from the example config."""
    config = load_config(EXAMPLE_CONFIG)
    return dict(build_regex_patterns(config))


def _find(pattern: re.Pattern[str], text: str) -> str | None:
    m = pattern.search(text)
    return m.group() if m else None


def test_config_loads_without_error() -> None:
    config = load_config(EXAMPLE_CONFIG)
    assert len(config.matchers) >= 3


SSN = "US SSN / ITIN"
CC = "Credit Card Number"
CRED = "Credential / Secret"


# -- SSN / ITIN --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("my ssn is 219-45-6789", "219-45-6789"),
        ("219 45 6789", "219 45 6789"),
        ("219456789", "219456789"),
        ("912-70-1234", "912-70-1234"),  # ITIN (9xx prefix)
    ],
    ids=["dashed", "spaced", "no-sep", "itin-9xx"],
)
def test_ssn_matches(
    patterns: Patterns, text: str, expected: str
) -> None:
    assert _find(patterns[SSN], text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "000-12-3456",  # invalid 000 prefix
        "666-12-3456",  # invalid 666 prefix
        "12-34-5678",  # too few digits in first group
        "12345678901",  # embedded in longer number
        "(212) 867-5309",  # phone number
        "2025-01-15",  # date
        "123-45-6789",  # well-known example SSN
        "078-05-1120",  # Woolworth wallet example
        "987-65-4320",  # IRS advertising range
        "219-00-1234",  # invalid group 00
        "219-45-0000",  # invalid serial 0000
    ],
    ids=[
        "invalid-000",
        "invalid-666",
        "short-prefix",
        "longer-number",
        "phone",
        "date",
        "example-123",
        "example-woolworth",
        "example-irs-ad",
        "group-00",
        "serial-0000",
    ],
)
def test_ssn_rejects(patterns: Patterns, text: str) -> None:
    assert _find(patterns[SSN], text) is None


# -- Credit Card -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4111111111111111", "4111111111111111"),
        ("4111-1111-1111-1111", "4111-1111-1111-1111"),
        ("4111 1111 1111 1111", "4111 1111 1111 1111"),
        ("5105105105105100", "5105105105105100"),
        ("378282246310005", "378282246310005"),  # Amex 15-digit
        ("6011111111111117", "6011111111111117"),
    ],
    ids=[
        "visa",
        "visa-dashed",
        "visa-spaced",
        "mastercard",
        "amex",
        "discover",
    ],
)
def test_cc_matches(
    patterns: Patterns, text: str, expected: str
) -> None:
    assert _find(patterns[CC], text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "1234567890123456",  # wrong prefix
        "4111-1111",  # too short
        "9111111111111111",  # 9xxx not a card prefix
        "5600000000000000",  # MC 56xx out of 51-55 range
    ],
    ids=["wrong-prefix", "too-short", "9xxx", "mc-out-of-range"],
)
def test_cc_rejects(patterns: Patterns, text: str) -> None:
    assert _find(patterns[CC], text) is None


# -- Credential / Secret -----------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # API key labels
        'api_key = "sk_abcdefghijklmnopqrst"',
        "bearer: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abcdef",
        'access_token="abcdefghijklmnopqrstuvwxyz"',
        # AWS
        "AKIAIOSFODNN7EXAMPLE",
        # GitHub (all 5 prefixes)
        "ghp_" + "a" * 40,
        "gho_" + "a" * 40,
        "ghu_" + "a" * 40,
        "ghs_" + "a" * 40,
        "ghr_" + "a" * 40,
        # GitLab
        "glpat-" + "x" * 20,
        # Slack
        "xoxb-123456-abcdefghij",
        "xoxp-999-abcdefghij",
        # Stripe
        "sk_live_abcdefghij",
        "pk_test_abcdefghij",
        "rk_live_abcdefghij",
        # Google
        "AIzaSy" + "A" * 33,
        # npm
        "npm_" + "A" * 36,
        # Passwords
        "my password is hunter2",
        "passwd: s3cret",
        "pwd=abc123",
        # PEM
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----BEGIN EC PRIVATE KEY-----",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        # Database URLs with creds
        "postgresql://admin:pass@db.example.com:5432/mydb",
        "mysql://root:password@localhost/db",
        "mongodb+srv://user:pass@cluster.mongodb.net/db",
        "redis://default:pass@redis.example.com:6379",
    ],
    ids=[
        "api-key-assignment",
        "bearer-token",
        "access-token",
        "aws-key",
        "github-ghp",
        "github-gho",
        "github-ghu",
        "github-ghs",
        "github-ghr",
        "gitlab-pat",
        "slack-bot",
        "slack-user",
        "stripe-secret",
        "stripe-publishable",
        "stripe-restricted",
        "google-api-key",
        "npm-token",
        "password-is",
        "passwd-colon",
        "pwd-equals",
        "pem-rsa",
        "pem-ec",
        "pem-openssh",
        "postgres-url",
        "mysql-url",
        "mongodb-url",
        "redis-url",
    ],
)
def test_cred_matches(patterns: Patterns, text: str) -> None:
    assert _find(patterns[CRED], text) is not None


@pytest.mark.parametrize(
    "text",
    [
        # Value too short for api_key label pattern
        'api_key = "sk-short"',
        # Wrong AWS prefix (AKIB not AKIA)
        "AKIB" + "A" * 16,
        # GitHub token too short
        "ghp_1234567890",
        # Stripe wrong first letter
        "xk_live_abcdefghij",
        # DB URL without user:pass@
        "postgres://localhost:5432/mydb",
        # "password" without a value assignment
        "change your password tomorrow",
        # Public key, not private
        "-----BEGIN PUBLIC KEY-----",
        # Almost a Google key (AIzaXy not AIzaSy)
        "AIzaXy" + "A" * 33,
    ],
    ids=[
        "api-key-too-short",
        "aws-wrong-prefix",
        "github-truncated",
        "stripe-wrong-prefix",
        "db-url-no-creds",
        "password-no-value",
        "pem-public-key",
        "google-wrong-prefix",
    ],
)
def test_cred_rejects(patterns: Patterns, text: str) -> None:
    assert _find(patterns[CRED], text) is None
