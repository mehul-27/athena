"""Symmetric encryption for provider API keys stored on disk.

Adapted from Odysseus `src/secret_storage.py` — same *pattern*, deliberately
shrunk: a single Fernet key file under `data/`, and an `enc:` prefix that makes
encryption idempotent and lets plaintext (legacy) values pass through unchanged.

Threat model (same as Odysseus): protects against the config file being
exfiltrated (stolen backup, copied data dir) — **not** against a process
compromise. Anyone who can read both `providers.json` and the key file has the
plaintext. That is the right trade-off for a local, single-user app.

`cryptography` is imported lazily so Athena keeps booting on a machine where the
optional dependency is missing: in that case the box degrades to a clearly
logged pass-through rather than crashing the application.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

logger = logging.getLogger(__name__)

_PREFIX = "enc:"
_warned = False


def is_encrypted(value: str) -> bool:
    return bool(value) and value.startswith(_PREFIX)


def _restrict(path: Path) -> None:
    """Best-effort 0o600 for freshly created key files (no-op on Windows)."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


class SecretBox:
    """Encrypts/decrypts short secrets with a key stored beside the data dir."""

    def __init__(self, key_path: Path) -> None:
        self.key_path = Path(key_path)
        self._fernet = None
        self._unavailable = False
        self._missing_reported = False

    # ------------------------------------------------------------------
    @property
    def encryption_available(self) -> bool:
        """True when `cryptography` is importable (a key is made on first encrypt)."""
        try:
            from cryptography.fernet import Fernet  # noqa: F401
        except ImportError:  # pragma: no cover - environment dependent
            return False
        return True

    @property
    def key_available(self) -> bool:
        """True when the key file exists (i.e. stored secrets can be decrypted)."""
        return self.key_path.exists()

    def _read_key(self) -> bytes | None:
        if not self.key_path.exists():
            return None
        try:
            return self.key_path.read_bytes()
        except OSError as exc:  # pragma: no cover - defensive
            logger.error("Could not read %s: %s", self.key_path, exc)
            return None

    def _create_key(self, fernet_cls) -> bytes:
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        key = fernet_cls.generate_key()
        self.key_path.write_bytes(key)
        _restrict(self.key_path)
        logger.info("Generated provider secret key at %s", self.key_path)
        return key

    def _fernet_for(self, *, create: bool):
        """Return a Fernet, or None.

        `create=True` (encryption) will generate the key file if absent.
        `create=False` (decryption) will **never** create one: if the key is gone
        the stored ciphertext is undecryptable, and silently generating a new key
        would make every saved API key disappear without explanation.
        """
        if self._fernet is not None:
            return self._fernet
        if self._unavailable:
            return None
        global _warned
        try:
            from cryptography.fernet import Fernet
        except ImportError:  # pragma: no cover - environment dependent
            self._unavailable = True
            if not _warned:
                logger.warning(
                    "cryptography is not installed — provider API keys will be "
                    "stored WITHOUT encryption. Install it to protect keys at rest."
                )
                _warned = True
            return None

        raw = self._read_key()
        if raw is None:
            if not create:
                if not self._missing_reported:
                    logger.error(
                        "Provider key file %s is missing, so stored API keys cannot be "
                        "decrypted. Re-enter the affected keys in Settings.",
                        self.key_path,
                    )
                    self._missing_reported = True
                return None
            raw = self._create_key(Fernet)

        try:
            self._fernet = Fernet(raw)
        except Exception as exc:
            logger.error("Provider key file %s is not a usable Fernet key: %s", self.key_path, exc)
            return None
        return self._fernet

    # ------------------------------------------------------------------
    def encrypt(self, plaintext: str) -> str:
        """Encrypt `plaintext`. Empty/None and already-encrypted values pass through."""
        if not plaintext:
            return plaintext or ""
        if is_encrypted(plaintext):
            return plaintext
        fernet = self._fernet_for(create=True)
        if fernet is None:
            return plaintext
        token = fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")
        return _PREFIX + token

    def decrypt(self, value: str) -> str:
        """Decrypt an `enc:` value. Plaintext passes through; failures return ""."""
        if not value:
            return value or ""
        if not is_encrypted(value):
            return value
        fernet = self._fernet_for(create=False)
        if fernet is None:
            return ""
        try:
            from cryptography.fernet import InvalidToken
        except ImportError:  # pragma: no cover - defensive
            return ""
        try:
            return fernet.decrypt(value[len(_PREFIX):].encode("ascii")).decode("utf-8")
        except InvalidToken:
            logger.error("Could not decrypt a stored provider key — wrong key or corrupt token")
            return ""
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Provider key decryption failed: %s", exc)
            return ""
