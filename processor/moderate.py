import asyncio
import datetime
import logging
import os

from atproto import AsyncClient
from atproto_client.exceptions import BadRequestError, UnauthorizedError
from dotenv import load_dotenv

from constants import THRESHOLD

load_dotenv()

logger = logging.getLogger("xblock.moderate")

# Prevents concurrent re-auth attempts from stampeding the login endpoint.
_auth_lock = asyncio.Lock()

SESSION_FILE = os.path.join(os.path.dirname(__file__), "session.txt")

_AUTH_ERROR_CODES = {"ExpiredToken", "AuthenticationRequired", "InvalidToken"}


def _make_client() -> AsyncClient:
    c = AsyncClient()
    c.configure_proxy_header("atproto_labeler", "did:plc:newitj5jo3uel7o4mnf3vj2o")
    return c


def _is_auth_error(exc: Exception) -> bool:
    if isinstance(exc, UnauthorizedError):
        return True
    if isinstance(exc, BadRequestError):
        try:
            return exc.response.content.error in _AUTH_ERROR_CODES
        except AttributeError:
            pass
    # Fallback for any wrapped or unexpected exception shape.
    msg = str(exc).lower()
    return any(k in msg for k in ("expiredtoken", "authenticationrequired", "invalidtoken", "401"))


client = _make_client()


async def _fresh_login() -> None:
    global client
    logger.info("Logging in with credentials...")
    # The atproto client caches the expired JWT in _access_jwt/_refresh_jwt and
    # tries to refresh them before any outgoing call, including the credential
    # login itself. Creating a new instance is the only reliable way to start
    # from a clean slate.
    fresh = _make_client()
    await fresh.login(os.environ["BSKY_USERNAME"], os.environ["BSKY_PASSWORD"])
    client = fresh
    with open(SESSION_FILE, "w") as f:
        f.write(client.export_session_string())
    logger.info("Login successful, session saved.")


async def auth_client() -> None:
    """Authenticate on startup. Tries cached session first, falls back to credentials."""
    global client
    async with _auth_lock:
        try:
            with open(SESSION_FILE) as f:
                session_string = f.read().strip()
            if not session_string:
                raise ValueError("empty session file")
            # Always use a fresh client so there are no stale JWTs that would
            # cause the library to attempt a refresh before the session load.
            fresh = _make_client()
            await fresh.login(session_string=session_string)
            await fresh.get_profile(actor=fresh._session.did)
            client = fresh
            logger.info("Resumed session for %s", client._session.did)
        except Exception as e:
            logger.warning("Session load failed (%s), re-authenticating...", e)
            await _fresh_login()


async def _reauth_if_needed(exc: Exception) -> bool:
    """Re-authenticate if the exception looks like an auth failure.
    Returns True if re-auth was attempted (caller should retry), False otherwise."""
    if not _is_auth_error(exc):
        return False
    async with _auth_lock:
        try:
            await _fresh_login()
            return True
        except Exception as reauth_exc:
            logger.error("Re-authentication failed: %s", reauth_exc)
            return False


async def create_label(result: dict) -> None:
    try:
        await _emit_label(result)
    except Exception as e:
        if await _reauth_if_needed(e):
            try:
                await _emit_label(result)
            except Exception as retry_exc:
                logger.error("create_label retry failed: %s", retry_exc)
        else:
            logger.error("create_label failed: %s", e)


async def _emit_label(result: dict) -> None:
    commit = result["commit"]
    uri = f"at://{result['did']}/{commit['collection']}/{commit['rkey']}"
    cid = commit["cid"]

    blob_cids = []
    add_labels = []
    for image in result["image_results"]:
        for label, score in image.get("labels", {}).items():
            if float(score) >= THRESHOLD:
                if label in ("news", "newsmedia"):
                    pass
                elif label != "negative":
                    add_labels.append(f"{label}-screenshot")
                    blob_cids.append(image["blob_cid"])

    if not add_labels:
        return

    data = {
        "event": {
            "$type": "tools.ozone.moderation.defs#modEventLabel",
            "createLabelVals": add_labels,
            "negateLabelVals": [],
            "comment": "model:howdyaendra/swin_s3_base_224-xblockm-timm",
        },
        "subject": {
            "$type": "com.atproto.repo.strongRef",
            "uri": uri,
            "cid": cid,
        },
        "createdBy": client._session.did,
        "createdAt": datetime.datetime.now().isoformat(),
        "subjectBlobCids": blob_cids,
    }

    logger.debug("Emitting labels %s for %s", add_labels, uri)
    await client.tools.ozone.moderation.emit_event(data=data)
