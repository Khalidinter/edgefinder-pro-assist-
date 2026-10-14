#!/usr/bin/env bash
# One-shot: pull the live edgefinder-pro source out of Vercel, strip the old keys,
# and push it to a new private GitHub repo so the code can be audited and fixed.
#
#   bash ops/restore_live_site.sh
#
# It asks for a Vercel token (create one at https://vercel.com/account/settings/tokens,
# any name, scope "Full Account"). The token is read without echo and never stored.
set -euo pipefail

DEPLOYMENT="dpl_Ahcq9PTSE4S2Kn5vGMc1GhQYswuR"
TEAM="team_Xr2jyDONLp0U1ASqG8vWmuZI"
DEST="$HOME/Projects/edgefinder-pro-live"
REPO="Khalidinter/edgefinder-pro-live"
PUB_KEY="sb_publishable_yxIfmyO3pj6611XOxBivIw_1cvoEXak"
NEW_TOKEN="a120e72eeff4639dd8141105a595cb0bcc2879e62c7777a865502faa15f05908"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${VERCEL_TOKEN:-}" ]; then
  printf 'Paste your Vercel token (input hidden) and press Enter: '
  read -r -s VERCEL_TOKEN
  echo
fi
export VERCEL_TOKEN

if [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
  echo "$DEST already exists and is not empty. Move it away and run again." >&2
  exit 1
fi

echo "1/4 restoring source from Vercel deployment $DEPLOYMENT ..."
python3 "$HERE/restore_vercel_deployment.py" "$DEPLOYMENT" "$TEAM" "$DEST"

echo "2/4 replacing the retired keys with the current ones ..."
cd "$DEST"
grep -rl --exclude-dir=node_modules --exclude-dir=.git -e 'eyJhbGci' -e '6641b73a' . | while read -r f; do
  sed -i '' -e "s/eyJhbGci[A-Za-z0-9._-]*/$PUB_KEY/g" -e "s/6641b73a[0-9a-f]*/$NEW_TOKEN/g" "$f"
  echo "   patched $f"
done
if grep -rq --exclude-dir=node_modules --exclude-dir=.git 'eyJhbGci' .; then
  echo "A legacy JWT is still present; refusing to push." >&2
  exit 1
fi
printf 'node_modules/\ndist/\n.vercel/\n.env\n.env.*\n.DS_Store\n' > .gitignore

echo "3/4 committing ..."
git init -q -b main
git add -A
git -c user.name=khalid -c user.email=khalidinter87@gmail.com commit -q -m "Restore live edgefinder-pro source from Vercel deployment $DEPLOYMENT"

echo "4/4 creating private repo $REPO and pushing ..."
gh repo create "$REPO" --private --source=. --push
echo
echo "Done. Source is at $DEST and pushed to https://github.com/$REPO"
