"""Encryption of secrets at rest (AI keys, OAuth tokens).

The master key comes from the environment (MASTER_KEY, a Fernet key) and is never
stored in the database or in backups.
"""

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken


class SecretBoxError(Exception):
    pass


class SecretBox:
    def __init__(self, master_key: str) -> None:
        try:
            self._fernet = Fernet(
                master_key.encode() if isinstance(master_key, str) else master_key
            )
        except (ValueError, TypeError) as exc:  # pragma: no cover - config error
            raise SecretBoxError("MASTER_KEY must be a urlsafe base64 32-byte Fernet key") from exc

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, token: str) -> str:
        try:
            return self._fernet.decrypt(token.encode()).decode()
        except InvalidToken as exc:
            raise SecretBoxError(
                "cannot decrypt secret: wrong MASTER_KEY or corrupted value"
            ) from exc

    @staticmethod
    def generate_key() -> str:
        return Fernet.generate_key().decode()


def last4(secret: str) -> str:
    return secret[-4:] if len(secret) >= 4 else "*" * len(secret)
