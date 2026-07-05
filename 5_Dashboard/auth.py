"""CXSentinel authentication.

Accounts live in a `users` table in the project database. Passwords are hashed
with PBKDF2-HMAC-SHA256 (200k iterations, per-user random salt).

Email verification: new email/password accounts must confirm a 6-digit code.
The code is emailed via SMTP when mail credentials are configured (an [smtp]
section in .streamlit/secrets.toml or SMTP_* environment variables); without
them the code is shown on screen, clearly labelled as the demo fallback.

Roles: the first account is seeded as the owner (role='admin') from
MASTER_EMAIL / MASTER_PASSWORD (env or secrets), giving access to the
Admin Settings tab.

"Continue with Google" uses Streamlit's native OIDC login (st.login) when the
deployment has an [auth] section in secrets; otherwise a labelled demo SSO.
"""

import hashlib
import os
import re
import secrets as pysecrets
import smtplib
import sys
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "4_Database"))
import db  # noqa: E402

from branding import APP_NAME, wordmark  # noqa: E402

PBKDF2_ITERATIONS = 200_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
CODE_TTL_MINUTES = 15

DEFAULT_MASTER_EMAIL = "adityachitale1@gmail.com"
DEFAULT_MASTER_PASSWORD = "CXS-Master#2026"


# ------------------------------------------------------------------ storage

def _secret(section: str, key: str, env_name: str, default: str = "") -> str:
    try:
        if section in st.secrets and key in st.secrets[section]:
            return str(st.secrets[section][key])
    except Exception:
        pass
    return os.getenv(env_name, default)


def init_users() -> None:
    with db.get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                name           TEXT NOT NULL,
                email          TEXT NOT NULL UNIQUE,
                password_hash  TEXT,
                salt           TEXT,
                provider       TEXT NOT NULL DEFAULT 'local',
                role           TEXT NOT NULL DEFAULT 'user',
                verified       INTEGER NOT NULL DEFAULT 0,
                verify_code    TEXT,
                verify_expires TEXT,
                created_at     TEXT NOT NULL
            )
            """
        )
        # Migrate pre-existing tables that lack the newer columns
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
        for col, ddl in [("role", "TEXT NOT NULL DEFAULT 'user'"),
                         ("verified", "INTEGER NOT NULL DEFAULT 0"),
                         ("verify_code", "TEXT"),
                         ("verify_expires", "TEXT")]:
            if col not in existing:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col} {ddl}")


def seed_master() -> None:
    """Ensure the owner account exists (idempotent)."""
    email = _secret("master", "email", "MASTER_EMAIL", DEFAULT_MASTER_EMAIL).lower()
    password = _secret("master", "password", "MASTER_PASSWORD", DEFAULT_MASTER_PASSWORD)
    with db.get_connection() as conn:
        row = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if row:
            return
        salt = pysecrets.token_hex(16)
        conn.execute(
            "INSERT INTO users (name, email, password_hash, salt, provider, role, "
            "verified, created_at) VALUES (?, ?, ?, ?, 'local', 'admin', 1, ?)",
            ("Owner", email, _hash_password(password, salt), salt,
             datetime.now().isoformat(timespec="seconds")),
        )


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt),
                               PBKDF2_ITERATIONS).hex()


def _row_to_user(row) -> dict:
    return {"name": row["name"], "email": row["email"],
            "provider": row["provider"], "role": row["role"]}


def _get_user_row(email: str):
    with db.get_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE email = ?",
                            (email.strip().lower(),)).fetchone()


def create_user(name: str, email: str, password: str = "", provider: str = "local"):
    """Returns (user_dict, None) on success or (None, error_message).
    Local accounts start unverified; call issue_verification next."""
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
    verified = 0 if provider == "local" else 1
    try:
        with db.get_connection() as conn:
            conn.execute(
                "INSERT INTO users (name, email, password_hash, salt, provider, "
                "verified, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, email, pw_hash, salt, provider, verified,
                 datetime.now().isoformat(timespec="seconds")),
            )
    except Exception:
        return None, "An account with this email already exists. Try signing in."
    return {"name": name, "email": email, "provider": provider, "role": "user"}, None


def verify_user(email: str, password: str):
    """Returns (user_dict, None), or (None, 'UNVERIFIED'), or (None, error)."""
    row = _get_user_row(email)
    if row is None:
        return None, "No account found with this email. Create one below."
    if row["provider"] != "local":
        return None, f"This account signs in with {row['provider'].title()}."
    if not pysecrets.compare_digest(row["password_hash"],
                                    _hash_password(password, row["salt"])):
        return None, "Incorrect password. Please try again."
    if not row["verified"]:
        return None, "UNVERIFIED"
    return _row_to_user(row), None


def change_password(email: str, current: str, new: str):
    """Returns (True, None) or (False, error_message)."""
    row = _get_user_row(email)
    if row is None:
        return False, "Account not found."
    if not pysecrets.compare_digest(row["password_hash"],
                                    _hash_password(current, row["salt"])):
        return False, "Current password is incorrect."
    if len(new) < 8:
        return False, "New password must be at least 8 characters."
    salt = pysecrets.token_hex(16)
    with db.get_connection() as conn:
        conn.execute("UPDATE users SET password_hash = ?, salt = ? WHERE email = ?",
                     (_hash_password(new, salt), salt, email.strip().lower()))
    return True, None


def get_or_create_google_user(email: str, name: str = ""):
    row = _get_user_row(email)
    if row:
        return _row_to_user(row), None
    return create_user(name or email.split("@")[0].replace(".", " ").title(),
                       email, provider="google")


def list_users() -> list:
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT name, email, provider, role, verified, created_at "
            "FROM users ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------ verification

def _send_email(to_addr: str, subject: str, body: str) -> bool:
    host = _secret("smtp", "host", "SMTP_HOST")
    if not host:
        return False
    port = int(_secret("smtp", "port", "SMTP_PORT", "587"))
    user = _secret("smtp", "user", "SMTP_USER")
    password = _secret("smtp", "password", "SMTP_PASSWORD")
    sender = _secret("smtp", "from", "SMTP_FROM", user)
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to_addr
        msg.set_content(body)
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            if user:
                server.login(user, password)
            server.send_message(msg)
        return True
    except Exception:
        return False


def issue_verification(email: str):
    """Generate + store a 6-digit code and try to email it.
    Returns (code, emailed: bool)."""
    code = f"{pysecrets.randbelow(10**6):06d}"
    expires = (datetime.now() + timedelta(minutes=CODE_TTL_MINUTES)).isoformat(timespec="seconds")
    with db.get_connection() as conn:
        conn.execute("UPDATE users SET verify_code = ?, verify_expires = ? WHERE email = ?",
                     (code, expires, email.strip().lower()))
    emailed = _send_email(
        email,
        f"{APP_NAME} — Verify Your Email",
        f"Welcome to {APP_NAME}!\n\nYour verification code is: {code}\n\n"
        f"It expires in {CODE_TTL_MINUTES} minutes. If you didn't create an "
        f"account, you can ignore this email.",
    )
    return code, emailed


def verify_email_code(email: str, code: str):
    """Returns (user_dict, None) on success or (None, error_message)."""
    row = _get_user_row(email)
    if row is None:
        return None, "Account not found."
    if not row["verify_code"] or not code.strip():
        return None, "Please enter the 6-digit code."
    if row["verify_expires"] and row["verify_expires"] < datetime.now().isoformat(timespec="seconds"):
        return None, "This code has expired. Use Resend Code to get a new one."
    if not pysecrets.compare_digest(row["verify_code"], code.strip()):
        return None, "Incorrect code. Please check your email and try again."
    with db.get_connection() as conn:
        conn.execute("UPDATE users SET verified = 1, verify_code = NULL, "
                     "verify_expires = NULL WHERE email = ?", (row["email"],))
    return _row_to_user(row), None


# ------------------------------------------------------------ supabase

@st.cache_resource(show_spinner=False)
def _supabase():
    """Supabase client when [supabase] url/anon_key secrets (or SUPABASE_URL /
    SUPABASE_ANON_KEY env vars) are configured; None otherwise."""
    url = _secret("supabase", "url", "SUPABASE_URL")
    key = _secret("supabase", "anon_key", "SUPABASE_ANON_KEY")
    if not (url and key):
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None


def supabase_enabled() -> bool:
    return _supabase() is not None


def supabase_sign_up(first: str, last: str, email: str, password: str):
    """Register through Supabase Auth (it emails the confirmation link).
    Returns (True, None) or (False, error_message)."""
    if not first.strip() or not last.strip():
        return False, "Please enter your first and last name."
    if not EMAIL_RE.match(email.strip().lower()):
        return False, "Please enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    try:
        _supabase().auth.sign_up({
            "email": email.strip().lower(),
            "password": password,
            "options": {"data": {"first_name": first.strip(),
                                 "last_name": last.strip()}},
        })
    except Exception as exc:
        return False, f"Sign up failed: {exc}"
    # Mirror the account locally so the admin panel and roles work
    create_user(f"{first.strip()} {last.strip()}", email, provider="supabase")
    return True, None


def supabase_sign_in(email: str, password: str):
    """Authenticate against Supabase Auth. Returns (user_dict, None) or (None, error)."""
    try:
        res = _supabase().auth.sign_in_with_password(
            {"email": email.strip().lower(), "password": password})
    except Exception as exc:
        msg = str(exc)
        if "confirm" in msg.lower():
            return None, ("Please verify your email first — Supabase sent you a "
                          "confirmation link when you signed up.")
        return None, "Incorrect email or password."
    row = _get_user_row(email)
    if row:
        return _row_to_user(row), None
    meta = (res.user.user_metadata or {}) if res and res.user else {}
    name = (f"{meta.get('first_name', '')} {meta.get('last_name', '')}".strip()
            or email.split("@")[0].replace(".", " ").title())
    return create_user(name, email, provider="supabase")


# ------------------------------------------------------------------ UI

def _native_oidc_available() -> bool:
    try:
        return hasattr(st, "login") and "auth" in st.secrets
    except Exception:
        return False


def _sign_in(user: dict) -> None:
    st.session_state["user"] = user
    for key in ("google_flow", "pending_verify", "demo_code"):
        st.session_state.pop(key, None)
    st.rerun()


def _start_verification(email: str) -> None:
    code, emailed = issue_verification(email)
    st.session_state["pending_verify"] = email
    st.session_state["demo_code"] = None if emailed else code
    st.rerun(scope="fragment")


@st.dialog(f"Welcome to {APP_NAME}", width="small")
def login_dialog():
    st.markdown(wordmark(30), unsafe_allow_html=True)

    # ---- Verification step (after sign-up, or unverified sign-in) ----
    pending = st.session_state.get("pending_verify")
    if pending:
        st.caption(f"Verify your email to activate your account.")
        if st.session_state.get("demo_code"):
            st.warning(f"Email delivery isn't configured on this deployment, so "
                       f"here is your code: **{st.session_state['demo_code']}**  \n"
                       f"(With SMTP credentials in secrets.toml this is emailed "
                       f"to {pending} instead.)")
        else:
            st.info(f"We've emailed a 6-digit code to **{pending}**. "
                    f"It expires in {CODE_TTL_MINUTES} minutes.")
        code = st.text_input("Verification Code", key="vf_code", max_chars=6,
                             placeholder="123456")
        c1, c2 = st.columns(2)
        if c1.button("Verify Email", type="primary", use_container_width=True):
            user, err = verify_email_code(pending, code)
            if err:
                st.error(err)
            else:
                st.balloons()
                _sign_in(user)
        if c2.button("Resend Code", use_container_width=True):
            _start_verification(pending)
        if st.button("Back To Sign In", use_container_width=True):
            for key in ("pending_verify", "demo_code"):
                st.session_state.pop(key, None)
            st.rerun(scope="fragment")
        return

    st.caption("Sign in to open your sentiment dashboard.")

    # ---- Google (the "G" logo is injected via CSS on the google_btn key) ----
    if _native_oidc_available():
        if st.button("Continue With Google", use_container_width=True, key="google_btn"):
            st.login("google")
    else:
        if st.session_state.get("google_flow"):
            g_email = st.text_input("Google Account Email", key="g_email",
                                    placeholder="you@gmail.com")
            if st.button("Continue", type="primary", use_container_width=True,
                         key="g_continue"):
                user, err = get_or_create_google_user(g_email)
                if err:
                    st.error(err)
                else:
                    _sign_in(user)
        else:
            if st.button("Continue With Google", use_container_width=True,
                         key="google_btn"):
                st.session_state["google_flow"] = True
                st.rerun(scope="fragment")
            st.caption("Demo SSO — connects a Google account by email. Full OAuth "
                       "activates automatically when OIDC secrets are configured.")

    st.markdown('<div class="sl-divider">or use email</div>', unsafe_allow_html=True)

    tab_in, tab_up = st.tabs(["Sign In", "Create Account"])

    with tab_in:
        email = st.text_input("Email", key="si_email", placeholder="you@company.com")
        password = st.text_input("Password", key="si_pw", type="password")
        if st.button("Sign In", type="primary", use_container_width=True, key="si_btn"):
            if not EMAIL_RE.match(email.strip().lower()):
                st.error("Please enter a valid email address (e.g. you@company.com).")
            elif not password:
                st.error("Please enter your password.")
            elif supabase_enabled():
                user, err = supabase_sign_in(email, password)
                if err:
                    st.error(err)
                else:
                    _sign_in(user)
            else:
                user, err = verify_user(email, password)
                if err == "UNVERIFIED":
                    _start_verification(email)
                elif err:
                    st.error(err)
                else:
                    _sign_in(user)

    with tab_up:
        n1, n2 = st.columns(2)
        first = n1.text_input("First Name", key="su_first", placeholder="Aditya")
        last = n2.text_input("Last Name", key="su_last", placeholder="Chitale")
        email_u = st.text_input("Email", key="su_email", placeholder="you@company.com")
        pw1 = st.text_input("Password (Min 8 Characters)", key="su_pw1", type="password")
        pw2 = st.text_input("Confirm Password", key="su_pw2", type="password")
        if supabase_enabled():
            st.caption("🔐 Verification Powered By Supabase Auth — a confirmation "
                       "link is emailed to you.")
        if st.button("Create Account", type="primary", use_container_width=True, key="su_btn"):
            if pw1 != pw2:
                st.error("Passwords do not match.")
            elif not first.strip() or not last.strip():
                st.error("Please enter your first and last name.")
            elif supabase_enabled():
                ok, err = supabase_sign_up(first, last, email_u, pw1)
                if err:
                    st.error(err)
                else:
                    st.success(f"Account created! Supabase has emailed a confirmation "
                               f"link to **{email_u}** — click it, then sign in here.")
            else:
                user, err = create_user(f"{first.strip()} {last.strip()}", email_u, pw1)
                if err:
                    st.error(err)
                else:
                    _start_verification(user["email"])


def current_user():
    return st.session_state.get("user")


def is_admin(user) -> bool:
    return bool(user) and user.get("role") == "admin"


def sign_out() -> None:
    for key in ("user", "google_flow", "pending_verify", "demo_code"):
        st.session_state.pop(key, None)
