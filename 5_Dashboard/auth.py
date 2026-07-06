"""CXSentinel authentication.

Accounts live in a `users` table in the project database. Passwords are hashed
with PBKDF2-HMAC-SHA256 (200k iterations, per-user random salt).

Sign-up is instant: the email address is validated (format, common-domain typo
detection, and a DNS check that the domain can receive mail) but there is no
OTP / email-verification step.

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

# Email deliverability: typo detection + DNS mail-server check
COMMON_DOMAINS = ("gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
                  "icloud.com", "aol.com", "protonmail.com", "proton.me",
                  "live.com", "msn.com", "googlemail.com")
REAL_LOOKALIKE_DOMAINS = {"mail.com", "ymail.com"}  # genuine providers, no typo warning


def _edit_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein: adjacent-swap typos (hotmial) count as one edit."""
    rows = [list(range(len(b) + 1))]
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = min(rows[-1][j] + 1, cur[j - 1] + 1, rows[-1][j - 1] + (ca != cb))
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cost = min(cost, rows[-2][j - 2] + 1)
            cur.append(cost)
        rows.append(cur)
        if len(rows) > 3:
            rows.pop(0)
    return rows[-1][-1]


def check_email_deliverability(email: str):
    """Returns None if the address looks deliverable, else an error message.

    Catches domain typos (gnail.com -> gmail.com) and domains with no mail
    server (via DNS MX lookup). Mailbox typos on a real domain can only be
    caught by the verification email itself.
    """
    domain = email.strip().lower().rsplit("@", 1)[-1]

    if domain not in COMMON_DOMAINS and domain not in REAL_LOOKALIKE_DOMAINS:
        for known in COMMON_DOMAINS:
            if _edit_distance(domain, known) == 1:
                return (f"'{domain}' looks like a typo — did you mean "
                        f"**@{known}**? Please check and try again.")

    try:
        import dns.resolver
        resolver = dns.resolver.Resolver()
        resolver.timeout = resolver.lifetime = 3.0
        try:
            resolver.resolve(domain, "MX")
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
            try:
                resolver.resolve(domain, "A")  # rare: mail via A record
            except Exception:
                return (f"The domain '{domain}' does not exist or cannot "
                        f"receive email. Please check the address.")
    except ImportError:
        pass
    except Exception:
        pass  # DNS unavailable/offline — never block sign-up on our own outage
    return None

DEFAULT_MASTER_EMAIL = "adityachitale1@gmail.com"
DEFAULT_MASTER_PASSWORD = "CXS-Master#2026"
DEFAULT_MASTER_NAME = "Aditya Chitale"


def initials(name: str) -> str:
    """Up to two initials: first + last name (e.g. 'Aditya Chitale' -> 'AC')."""
    parts = [p for p in str(name).strip().split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][0].upper()
    return (parts[0][0] + parts[-1][0]).upper()


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
    name = _secret("master", "name", "MASTER_NAME", DEFAULT_MASTER_NAME)
    with db.get_connection() as conn:
        row = conn.execute("SELECT id, name FROM users WHERE email = ?",
                           (email,)).fetchone()
        if row:
            # Replace placeholder names (generic "Owner", or one auto-derived
            # from the email like "Adityachitale1") with the configured name.
            email_derived = email.split("@")[0].replace(".", " ").title()
            if row["name"] in ("Owner", email_derived):
                conn.execute("UPDATE users SET name = ? WHERE id = ?",
                             (name, row["id"]))
            return
        salt = pysecrets.token_hex(16)
        conn.execute(
            "INSERT INTO users (name, email, password_hash, salt, provider, role, "
            "verified, created_at) VALUES (?, ?, ?, ?, 'local', 'admin', 1, ?)",
            (name, email, _hash_password(password, salt), salt,
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
    Accounts are active immediately — no email-verification step."""
    email = email.strip().lower()
    name = name.strip()
    if not name:
        return None, "Please enter your name."
    if not EMAIL_RE.match(email):
        return None, "Please enter a valid email address."
    deliverability_error = check_email_deliverability(email)
    if deliverability_error:
        return None, deliverability_error
    if provider == "local" and len(password) < 8:
        return None, "Password must be at least 8 characters."

    salt = pysecrets.token_hex(16) if provider == "local" else None
    pw_hash = _hash_password(password, salt) if provider == "local" else None
    try:
        with db.get_connection() as conn:
            conn.execute(
                "INSERT INTO users (name, email, password_hash, salt, provider, "
                "verified, created_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
                (name, email, pw_hash, salt, provider,
                 datetime.now().isoformat(timespec="seconds")),
            )
    except Exception:
        return None, "An account with this email already exists. Try signing in."
    return {"name": name, "email": email, "provider": provider, "role": "user"}, None


def verify_user(email: str, password: str):
    """Returns (user_dict, None) or (None, error_message)."""
    row = _get_user_row(email)
    if row is None:
        return None, "No account found with this email. Create one below."
    if row["provider"] != "local":
        return None, f"This account signs in with {row['provider'].title()}."
    if not pysecrets.compare_digest(row["password_hash"],
                                    _hash_password(password, row["salt"])):
        return None, "Incorrect password. Please try again."
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


def get_profile(email: str) -> dict:
    """Full account details for the profile view."""
    row = _get_user_row(email)
    if row is None:
        return {}
    return {"name": row["name"], "email": row["email"],
            "provider": row["provider"], "role": row["role"],
            "created_at": row["created_at"]}


def list_users() -> list:
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT name, email, provider, role, verified, created_at "
            "FROM users ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


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
    deliverability_error = check_email_deliverability(email)
    if deliverability_error:
        return False, deliverability_error
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
    st.session_state.pop("google_flow", None)
    st.rerun()


def open_login(mode: str = "signin") -> None:
    """Open the auth dialog on a specific view: 'signin' or 'signup'."""
    st.session_state["auth_mode"] = mode
    login_dialog()


@st.dialog(f"Welcome to {APP_NAME}", width="small")
def login_dialog():
    st.markdown(wordmark(30), unsafe_allow_html=True)

    if st.session_state.get("auth_mode", "signin") == "signup":
        st.caption("Get Started — Create Your Account In Under A Minute.")
    else:
        st.caption("Sign In To Open Your Sentiment Dashboard.")

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
                deliverability_error = (check_email_deliverability(g_email)
                                        if EMAIL_RE.match(g_email.strip().lower())
                                        else None)
                if not EMAIL_RE.match(g_email.strip().lower()):
                    st.error("Please enter a valid email address.")
                elif deliverability_error:
                    st.error(deliverability_error)
                else:
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

    mode = st.session_state.get("auth_mode", "signin")

    if mode == "signin":
        email = st.text_input("Email", key="si_email", placeholder="you@company.com")
        password = st.text_input("Password", key="si_pw", type="password")
        if st.button("Sign In", type="primary", use_container_width=True, key="si_btn"):
            deliverability_error = (check_email_deliverability(email)
                                    if EMAIL_RE.match(email.strip().lower()) else None)
            if not EMAIL_RE.match(email.strip().lower()):
                st.error("Please enter a valid email address (e.g. you@company.com).")
            elif deliverability_error:
                st.error(deliverability_error)
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
                if err:
                    st.error(err)
                else:
                    _sign_in(user)
        if st.button("New To CXSentinel?  Create An Account →",
                     use_container_width=True, key="switch_signup"):
            st.session_state["auth_mode"] = "signup"
            st.rerun(scope="fragment")

    else:  # ---- sign-up view (Get Started) ----
        st.caption("Create Your Free Account")
        n1, n2 = st.columns(2)
        first = n1.text_input("First Name", key="su_first", placeholder="Aditya")
        last = n2.text_input("Last Name", key="su_last", placeholder="Chitale")
        email_u = st.text_input("Email", key="su_email", placeholder="you@company.com")
        pw1 = st.text_input("Password (Min 8 Characters)", key="su_pw1", type="password")
        pw2 = st.text_input("Confirm Password", key="su_pw2", type="password")
        if st.button("Create Account", type="primary", use_container_width=True, key="su_btn"):
            if pw1 != pw2:
                st.error("Passwords do not match.")
            elif not first.strip() or not last.strip():
                st.error("Please enter your first and last name.")
            else:
                user, err = create_user(f"{first.strip()} {last.strip()}", email_u, pw1)
                if err:
                    st.error(err)
                else:
                    _sign_in(user)
        if st.button("Already Have An Account?  Sign In →",
                     use_container_width=True, key="switch_signin"):
            st.session_state["auth_mode"] = "signin"
            st.rerun(scope="fragment")


def current_user():
    return st.session_state.get("user")


def is_admin(user) -> bool:
    return bool(user) and user.get("role") == "admin"


def sign_out() -> None:
    for key in ("user", "google_flow"):
        st.session_state.pop(key, None)
    # End the native OIDC (Google) session too, if there is one
    try:
        if getattr(st, "user", None) is not None and st.user.is_logged_in:
            st.logout()
    except Exception:
        pass


PROVIDER_LABEL = {"google": "Google", "supabase": "Supabase Auth",
                  "local": "Email & Password"}


@st.dialog("My Profile", width="small")
def profile_dialog(user: dict):
    profile = get_profile(user["email"]) or user
    initial = initials(profile.get("name", "?"))
    is_owner = profile.get("role") == "admin"

    # ---- Avatar + name header ----
    st.markdown(
        '<div style="display:flex;align-items:center;gap:1rem;margin:.2rem 0 1rem;">'
        f'<div style="width:64px;height:64px;border-radius:50%;flex:none;'
        'display:flex;align-items:center;justify-content:center;font-size:1.9rem;'
        'font-weight:800;color:#fff;'
        'background:linear-gradient(135deg,#5227FF,#B970E8);">'
        f'{initial}</div>'
        f'<div><div style="font-size:1.25rem;font-weight:800;color:#F2EFFD;">'
        f'{profile.get("name","")}</div>'
        f'<div style="color:#B4A9DC;font-size:.9rem;">{profile.get("email","")}</div>'
        '</div></div>',
        unsafe_allow_html=True)

    # ---- Detail rows ----
    role_txt = "Owner · Administrator" if is_owner else "Member"
    created = (profile.get("created_at") or "")[:10] or "—"
    c1, c2 = st.columns(2)
    c1.metric("Role", "Owner" if is_owner else "Member")
    c2.metric("Signed In Via", PROVIDER_LABEL.get(profile.get("provider"), "Email"))
    st.markdown(f"**Account Role:** {role_txt}  \n"
                f"**Member Since:** {created}  \n"
                f"**Account Email:** {profile.get('email','')}")

    if is_owner:
        st.info("You Have Owner Access — The Admin Settings Tab Is Available.")

    # ---- Change password (email/password accounts only) ----
    if profile.get("provider") == "local":
        with st.expander("🔒 Change Password"):
            cur = st.text_input("Current Password", type="password", key="pf_cur")
            new1 = st.text_input("New Password (Min 8 Characters)", type="password",
                                 key="pf_new1")
            new2 = st.text_input("Confirm New Password", type="password", key="pf_new2")
            if st.button("Update Password", type="primary", key="pf_update"):
                if new1 != new2:
                    st.error("New passwords do not match.")
                else:
                    ok, err = change_password(profile["email"], cur, new1)
                    st.success("Password updated.") if ok else st.error(err)
    else:
        st.caption(f"Password Is Managed By "
                   f"{PROVIDER_LABEL.get(profile.get('provider'), 'your provider')}.")

    st.divider()
    if st.button("Sign Out", use_container_width=True, key="pf_signout"):
        sign_out()
        st.rerun()
