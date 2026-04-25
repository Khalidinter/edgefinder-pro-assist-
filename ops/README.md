# EdgeFinder daily ops

Local launchd cron that runs the assists + rebounds predict & resolve pipelines
once a day at noon local time. Replaces the GitHub Actions `daily-refresh.yml`
cron (which was killed 2026-04-25 by stats.nba.com WAF rate-limiting GHA IPs).

## One-time install

```bash
mkdir -p ops/logs
chmod +x ops/run_daily.sh

# Install plist into LaunchAgents and load it
cp ops/com.edgefinder.daily.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.edgefinder.daily.plist
```

Verify:
```bash
launchctl list | grep edgefinder
# expect: <pid>  0  com.edgefinder.daily   (or - 0 com.edgefinder.daily when not running)
```

## Run on demand

```bash
launchctl start com.edgefinder.daily
tail -f ops/logs/$(date +%Y-%m-%d).log
```

## Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.edgefinder.daily.plist
rm ~/Library/LaunchAgents/com.edgefinder.daily.plist
```

## Why local instead of GHA

stats.nba.com fronts a Cloudflare WAF that flags GitHub Actions runner IPs.
Every cron run since 2026-04-17 was cancelled at 30→45 min timeout. The library
fetches succeed instantly from a residential IP (this Mac), so the cleanest fix
for a personal-use tool is to run the pipeline here.

If the Mac is asleep at noon, the run is skipped that day. The dashboard's
status pill turns yellow to indicate stale data; resolution catches up the next
time the Mac is awake at noon.
