#!/bin/bash
# EdgeFinder: one-shot bootstrap for the Mac.
#   1. puts Homebrew on PATH and installs Node + the GitHub CLI
#   2. logs in to GitHub (browser window)
#   3. pushes the restored NBA app (~/Projects/edgefinder-nba-pro) to a private repo
#   4. downloads the deployed Cloudflare worker (browser login) and pushes it to a private repo
# Safe to re-run: every step skips what is already done. No secrets are read or written.
set -u

log()  { printf '\n==> %s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
GIT_NAME="${GIT_NAME:-khalid}"
GIT_EMAIL="${GIT_EMAIL:-khalidinter87@gmail.com}"

# 1. Homebrew, Node, gh
[ -x /opt/homebrew/bin/brew ] || fail "Homebrew is not at /opt/homebrew. Install it from https://brew.sh and re-run."
eval "$(/opt/homebrew/bin/brew shellenv)"
grep -q 'brew shellenv' ~/.zprofile 2>/dev/null || printf '\neval "$(/opt/homebrew/bin/brew shellenv zsh)"\n' >> ~/.zprofile
log "Installing node and gh (already-installed ones are skipped)"
brew list node >/dev/null 2>&1 || brew install node
brew list gh   >/dev/null 2>&1 || brew install gh
node -v && npm -v && gh --version | head -1

# 2. GitHub login
if ! gh auth status >/dev/null 2>&1; then
  log "GitHub login: a browser window will open"
  gh auth login -h github.com -p https -w || fail "GitHub login did not complete"
fi
GH_USER="$(gh api user -q .login)"

push_repo() {  # push_repo <dir> <repo-name> <commit message>
  local dir="$1" name="$2" msg="$3"
  cd "$dir" || fail "missing folder $dir"
  [ -d .git ] || git init -q
  git add -A
  if git status --porcelain | grep -qE '(^|/)\.env|\.dev\.vars|\.wrangler/'; then
    fail "a secrets file is staged in $dir; add it to .gitignore and re-run"
  fi
  git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" commit -q -m "$msg" 2>/dev/null || true
  if gh repo view "$GH_USER/$name" >/dev/null 2>&1; then
    git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$GH_USER/$name.git"
    git push -u origin HEAD
  else
    gh repo create "$name" --private --source=. --remote=origin --push
  fi
  log "$GH_USER/$name is on GitHub"
}

# 3. NBA app
APP="$HOME/Projects/edgefinder-nba-pro"
[ -f "$APP/src/api/adapter.js" ] || fail "NBA app source not found at $APP (restore it first)"
printf '.env\n.env.*\nnode_modules/\ndist/\n.vercel/\n.DS_Store\n' > "$APP/.gitignore"
push_repo "$APP" edgefinder-nba-pro "Import NBA app source restored from the Vercel deployment"

# 4. Worker
WORKER="$HOME/Projects/edgefinder-server"
if [ ! -f "$WORKER/wrangler.toml" ] && [ ! -f "$WORKER/wrangler.jsonc" ] && [ ! -f "$WORKER/wrangler.json" ]; then
  log "Downloading the deployed worker from Cloudflare: a browser login window will open"
  cd "$HOME/Projects"
  npm create cloudflare@latest edgefinder-server -- --type=pre-existing --existing-script=edgefinder-server --lang=js --git=false --deploy=false
fi
[ -d "$WORKER" ] || fail "worker download did not create $WORKER"
printf '.env\n.env.*\n.dev.vars\n.wrangler/\nnode_modules/\ndist/\n.DS_Store\n' > "$WORKER/.gitignore"
push_repo "$WORKER" edgefinder-server "Import worker source pulled from Cloudflare"

log "All done. Tell Claude both repos exist."
