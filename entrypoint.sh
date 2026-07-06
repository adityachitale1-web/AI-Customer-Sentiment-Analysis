#!/bin/bash
# Container startup: write the OAuth secrets from Space secrets (env vars)
# BEFORE launching Streamlit, so st.login('google') is configured at boot.
set -e

mkdir -p /app/.streamlit

if [ -n "$GOOGLE_CLIENT_ID" ] && [ -n "$GOOGLE_CLIENT_SECRET" ]; then
  if [ -n "$OAUTH_REDIRECT_URI" ]; then
    REDIRECT="$OAUTH_REDIRECT_URI"
  elif [ -n "$SPACE_HOST" ]; then
    REDIRECT="https://${SPACE_HOST}/oauth2callback"
  else
    REDIRECT="https://aicreator1010101-cxsentinel.hf.space/oauth2callback"
  fi
  COOKIE="${OAUTH_COOKIE_SECRET:-cxsentinel-docker-cookie-set-OAUTH_COOKIE_SECRET}"
  cat > /app/.streamlit/secrets.toml <<EOF
[auth]
redirect_uri = "${REDIRECT}"
cookie_secret = "${COOKIE}"

[auth.google]
client_id = "${GOOGLE_CLIENT_ID}"
client_secret = "${GOOGLE_CLIENT_SECRET}"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
EOF
  echo "OIDC secrets written (redirect_uri=${REDIRECT})"
else
  echo "No Google OAuth env vars — running in demo-SSO mode."
fi

exec streamlit run 5_Dashboard/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true
