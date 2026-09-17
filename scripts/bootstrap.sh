#!/usr/bin/env bash
# Set up JARVIS on a fresh machine (macOS or Linux).
#
#   cd jarvis && ./scripts/bootstrap.sh
#
# Installs nothing system-wide except a virtualenv inside this directory, and
# never touches a credential. Safe to run more than once.

set -euo pipefail

cd "$(dirname "$0")/.."
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()  { printf '  \033[32mok\033[0m   %s\n' "$*"; }
bad() { printf '  \033[31mmiss\033[0m %s\n' "$*"; }

say "Checking what is already here"

MISSING=0

if command -v python3 >/dev/null 2>&1; then
  VERSION=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  MAJOR=${VERSION%%.*}
  MINOR=${VERSION##*.}
  if [ "$MAJOR" -gt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ]; }; then
    ok "python $VERSION"
  else
    bad "python $VERSION is too old; 3.11 or newer is required"
    MISSING=1
  fi
else
  bad "python3 is not installed"
  MISSING=1
fi

command -v git >/dev/null 2>&1 && ok "git $(git --version | awk '{print $3}')" || { bad "git"; MISSING=1; }
command -v docker >/dev/null 2>&1 && ok "docker $(docker --version | awk '{print $3}' | tr -d ,)" \
  || bad "docker (optional now, needed to run agents in containers later)"

if [ "$MISSING" -ne 0 ]; then
  cat <<'HINT'

Install what is missing, then run this again.

  macOS:  brew install python@3.11 git && brew install --cask docker
  Ubuntu: sudo apt install -y python3.11 python3.11-venv git

HINT
  exit 1
fi

say "Creating the virtual environment"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ok "created .venv"
else
  ok ".venv already exists"
fi

say "Installing dependencies"
# setuptools below 83 carries a known advisory (PYSEC-2026-3447).
./.venv/bin/pip install --quiet --upgrade pip "setuptools>=83.0.0"
./.venv/bin/pip install --quiet -e ".[dev]"
ok "installed"

say "Configuration"
if [ ! -f .env ]; then
  cp .env.example .env
  ok "created .env from the template"
else
  ok ".env already exists"
fi

say "Verifying the install"
./.venv/bin/pytest -q
./.venv/bin/python examples/demo_offline.py >/dev/null
ok "tests pass and the pipeline runs end to end"

cat <<'NEXT'

Done.

Next:
  1. Put your Claude key in jarvis/.env  (ANTHROPIC_API_KEY=...)
  2. source .venv/bin/activate
  3. jarvis doctor
  4. jarvis ask "research the AI agent framework market"

Day to day, `make help` lists everything.

NEXT
