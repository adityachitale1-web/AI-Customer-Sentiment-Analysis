"""SentiLens authentication.

Accounts live in a `users` table in the project database. Passwords are hashed
with PBKDF2-HMAC-SHA256 (200k iterations, per-user random salt) — never stored
in plain text.

"Continue with Google" uses Streamlit's native OIDC login (`st.login`) when the
deployment has an [auth] section in .streamlit/secrets.toml (client_id/secret
from Google Cloud Console). Without those secrets — e.g. the local demo — it
falls back to a clearly-labelled demo SSO that only asks for the Google email.
"""

import hashlib
import re
import secrets as pysecrets
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "4_Database"))
import db  # noqa: E402

from branding import APP_NAME, wordmark  # noqa: E402

PBKDF2_ITERATIONS = 200_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ------------------------------------------------------------------ storage

def init_users() -> None:
    with db.get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL,
                email         TEXT NOT NULL UNIQUE,
                password_hash TEXT,
                salt          TEXT,
                provider      TEXT NOT NULL DEFAULT 'local',
                created_at    TEXT NOT NULL
            )
            """
        )


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt),
                               PBKDF2_ITERATIONS).hex()


def create_user(name: str, email: str, password: str = "", provider: str = "local"):
    """Returns (user_dict, None) on success or (None, error_message)."""
    email = email.strip().lower()
    name = name.strip()
    if not name:
        return None, "Please enter your name."
    if not EMAIL_RE.match(email):
        return None, "Please enter a valid email address."
    if provider == "local" and len(password) < 8:
        return None, "Password must be at least 8 characters."

    salt = pysecrets.token_hex(16) if provider == "local" else None
    pw_hash = _hash_password(password, salt) if provider == "local" else None
    try:
        with db.get_connection() as conn:
            conn.execute(
                "INSERT INTO users (name, email, password_hash, salt, provider, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, email, pw_hash, salt, provider,
                 datetime.now().isoformat(timespec="seconds")),
            )
    except Exception:
        return None, "An account with this email already exists. Try signing in."
    return {"name": name, "email": email, "provider": provider}, None


def verify_user(email: str, password: str):
    """Returns (user_dict, None) on success or (None, error_message)."""
    email = email.strip().lower()
    with db.get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if row is None:
        return None, "No account found with this email. Create one below."
    if row["provider"] != "local":
        return None, f"This account signs in with {row['provider'].title()}."
    if not pysecrets.compare_digest(row["password_hash"],
                                    _hash_password(password, row["salt"])):
        return None, "Incorrect password. Please try again."
    return {"name": row["name"], "email": row["email"], "provider": "local"}, None


def get_or_create_google_user(email: str, name: str = ""):
    email = email.strip().lower()
    with db.get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if row:
        return {"name": row["name"], "email": row["email"], "provider": row["provider"]}, None
    return create_user(name or email.split("@")[0].replace(".", " ").title(),
                       email, provider="google")


# ------------------------------------------------------------------ UI

def _native_oidc_available() -> bool:
    try:
        return hasattr(st, "login") and "auth" in st.secrets
    except Exception:
        return False


def _sign_in(user: dict) -> None:
    st.session_state["user"] = user
    st.rerun()


@st.dialog(f"Welcome to {APP_NAME}", width="small")
def login_dialog():
    st.markdown(wordmark(30), unsafe_allow_html=True)
    st.caption("Sign in to open your sentiment dashboard.")

    # --- Google ---
    if _native_oidc_available():
        if st.button("🔵  Continue with Google", use_container_width=True):
            st.login("google")
    else:
        if st.session_state.get("google_flow"):
            g_email = st.text_input("Google account email", key="g_email",
                                    placeholder="you@gmail.com")
            if st.button("Continue", type="primary", use_container_width=True):
                user, err = get_or_create_google_user(g_email)
                if err:
                    st.error(err)
                else:
                    _sign_in(user)
        else:
            if st.button("🔵  Continue with Google", use_container_width=True):
                st.session_state["google_flow"] = True
                st.rerun(scope="fragment")
            st.caption("Demo SSO — connects a Google account by email. Full OAuth "
                       "activates automatically when OIDC secrets are configured.")

    st.markdown('<div class="sl-divider">or use email</div>', unsafe_allow_html=True)

    tab_in, tab_up = st.tabs(["Sign in", "Create account"])

    with tab_in:
        email = st.text_input("Email", key="si_email", placeholder="you@company.com")
        password = st.text_input("Password", key="si_pw", type="password")
        if st.button("Sign in", type="primary", use_container_width=True, key="si_btn"):
            user, err = verify_user(email, password)
            if err:
                st.error(err)
            else:
                _sign_in(user)

    with tab_up:
        name = st.text_input("Full name", key="su_name", placeholder="Aditya Chitale")
        email_u = st.text_input("Email", key="su_email", placeholder="you@company.com")
        pw1 = st.text_input("Password (min 8 characters)", key="su_pw1", type="password")
        pw2 = st.text_input("Confirm password", key="su_pw2", type="password")
        if st.button("Create account", type="primary", use_container_width=True, key="su_btn"):
            if pw1 != pw2:
                st.error("Passwords do not match.")
            else:
                user, err = create_user(name, email_u, pw1)
                if err:
                    st.error(err)
                else:
                    _sign_in(user)


def current_user():
    return st.session_state.get("user")


def sign_out() -> None:
    st.session_state.pop("user", None)
    st.session_state.pop("google_flow", None)
