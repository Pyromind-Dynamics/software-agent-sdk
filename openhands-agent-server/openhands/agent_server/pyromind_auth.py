import os
from datetime import datetime
from functools import lru_cache

import jwt
from fastapi import Cookie, HTTPException, WebSocketException, status
from pydantic import BaseModel

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

ALGORITHM = "HS256"
PYROMIND_AUTH_COOKIE_NAME = "auth_token"
LOGIN_REQUIRED_DETAIL = (
    "Sorry, you need to log in first—or your session ended. Re-login to access this."
)
_DEV_SECRET_KEY = "Kij823420JITRE21i21248cbsxhuexvS"


class AvatarInfo(BaseModel):
    filename: str | None = None
    object_path: str | None = None
    size_mb: float | None = None
    etag: str | None = None
    version_id: str | None = None
    url: str | None = None


class CurrentLoginUser(BaseModel):
    username: str
    email: str
    user_id: int
    group_id: int | None = None
    full_phone_num: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    uid: int | None = None
    avatar_url: str | None = None
    avatar_info: AvatarInfo | None = None


def get_env_value() -> str:
    app_env: str = os.getenv("APP_ENV", "dev")
    return app_env


def is_dev() -> bool:
    mock = get_env_value()
    return mock == "dev"


def find_secret_key() -> str:
    if is_dev():
        return _DEV_SECRET_KEY
    return find_secret_key_v2()


@lru_cache
def find_secret_key_v2() -> str:
    return read_secret("web_secret_key")


def read_secret(secret_name: str, secret_path: str = "/etc/secrets") -> str:
    """Read a mounted secret file."""
    try:
        with open(f"{secret_path}/{secret_name}") as f:
            return f.read().strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Secret file not found: {secret_name}") from exc
    except PermissionError as exc:
        raise RuntimeError(f"Permission denied reading secret: {secret_name}") from exc


def verify_jwt_token(token: str) -> CurrentLoginUser | None:
    try:
        payload = jwt.decode(token, find_secret_key(), algorithms=[ALGORITHM])
        if payload.get("type") == "password_reset":
            logger.warning("Password reset token used for authentication")
            return None
        required_fields = ["sub", "email", "user_id"]
        for field in required_fields:
            if field not in payload:
                logger.warning("Missing required field '%s' in token payload", field)
                return None
        group_id = payload.get("group_id", 0)
        full_phone_num = payload.get("full_phone_num", "")
        return CurrentLoginUser(
            username=payload["sub"],
            email=payload["email"],
            user_id=payload["user_id"],
            group_id=group_id,
            full_phone_num=full_phone_num,
        )
    except jwt.ExpiredSignatureError as e:
        logger.warning("Token expired: %s", e)
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid token: %s", e)
    except KeyError as e:
        logger.warning("Missing required field in token: %s", e)
    except Exception as e:
        logger.error("Unexpected error verifying token: %s", e, exc_info=True)
    return None


def v_jwt_token(token: str) -> CurrentLoginUser | None:
    return verify_jwt_token(token)


def get_current_login_user_from_token(
    auth_token: str | None,
) -> CurrentLoginUser | None:
    if not auth_token:
        return None
    return v_jwt_token(auth_token)


def require_login_from_cookie(
    auth_token: str | None = Cookie(None),
) -> CurrentLoginUser:
    current_user = get_current_login_user_from_token(auth_token)
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=LOGIN_REQUIRED_DETAIL,
        )
    return current_user


def require_login_from_websocket_cookie(
    auth_token: str | None = Cookie(None),
) -> CurrentLoginUser:
    current_user = get_current_login_user_from_token(auth_token)
    if current_user is None:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason=LOGIN_REQUIRED_DETAIL,
        )
    return current_user
