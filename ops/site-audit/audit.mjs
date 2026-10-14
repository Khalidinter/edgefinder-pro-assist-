// Live audit of the EdgeFinder site: loads the app in headless Chromium, passes the
// access gate, then presses every tab and non-destructive control and reports
// console errors, uncaught exceptions and failed network calls per step.
//
//   npm install && npx playwright install chromium
//   PASSCODE=1234 node audit.mjs                 (PASSCODE optional)
//   HEADED=1 ... to watch it; ALLOW_ACTIONS=1 ... to also press Scan/Resolve/Export buttons
//
// Each top-level control is pressed on a freshly loaded page, so filters that only
// exist on one tab are pressed while that tab is open. Controls that appear after a
// press (sub-tabs, toggles, filters) are pressed next, in place.
// Output: report.txt (Odds API keys in URLs are redacted) and shots/*.png
import { chromium } from "playwright";
import fs from "node:fs";

const SITE = process.env.SITE || "https://edgefinder.cognivaultlabs.com";
const PASSCODE = process.env.PASSCODE || "";
const HEADED = process.env.HEADED === "1";
const ALLOW_ACTIONS = process.env.ALLOW_ACTIONS === "1";
const STEP_WAIT = Number(process.env.STEP_WAIT || 1800);
// Never pressed: anything that deletes or rewrites a record (trash icons, result marks).
const DESTRUCTIVE = /reset|delete|untrack|remove|clear|wipe|logout|sign out|paper|\u{1F5D1}|won|lost|push|undo|void/iu;
const ACTIONS = /scan|resolve|run|reconcile|backfill|update sets|notify|export|sync/i;

const redact = (u) => String(u).replace(/apiKey=[^&\s]+/gi, "apiKey=REDACTED");
const norm = (s) => s.trim().replace(/\s+/g, " ").replace(/\d+/g, "#");
const lines = [];
const log = (s = "") => { lines.push(s); console.log(s); };

fs.mkdirSync("shots", { recursive: true });
const browser = await chromium.launch({ headless: !HEADED });
const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
const page = await context.newPage();

// Read-only by default: this headless profile starts with empty storage, so the app
// would otherwise push default settings to the shared Supabase store and overwrite
// the real ones. Writes to Supabase are answered locally and never sent unless
// ALLOW_WRITES=1. Worker calls are left alone (ALLOW_ACTIONS controls those buttons).
const ALLOW_WRITES = process.env.ALLOW_WRITES === "1";
let blockedWrites = 0;
if (!ALLOW_WRITES) {
  await page.route(/supabase\.co\/rest\//, (route) => {
    const m = route.request().method();
    if (m === "GET" || m === "HEAD" || m === "OPTIONS") return route.continue();
    blockedWrites++;
    return route.fulfill({ status: 204, headers: { "access-control-allow-origin": "*" }, body: "" });
  });
}

const events = [];
const apiHits = {};
const push = (type, text) => events.push({ type, text: redact(text) });
page.on("console", (m) => { if (["error", "warning"].includes(m.type())) push("console." + m.type(), m.text()); });
page.on("pageerror", (e) => push("pageerror", e.message));
page.on("requestfailed", (r) => push("requestfailed", `${r.method()} ${r.url()} :: ${r.failure()?.errorText}`));
page.on("response", (r) => {
  const u = r.url();
  if (r.status() >= 400) push("http" + r.status(), `${r.request().method()} ${u}`);
  if (/supabase|workers\.dev|the-odds-api|espn|ntfy|\/api\//.test(u)) {
    const host = new URL(u).host;
    apiHits[host] = apiHits[host] || {};
    apiHits[host][r.status()] = (apiHits[host][r.status()] || 0) + 1;
  }
});
page.on("download", (d) => log(`  download started: ${d.suggestedFilename()}`));
let cursor = 0;
function drain(label) {
  const mine = events.slice(cursor); cursor = events.length;
  const seen = new Set();
  const out = mine.filter((e) => { const k = e.type + e.text; if (seen.has(k)) return false; seen.add(k); return true; });
  if (out.length) { log(`  problems during "${label}":`); out.forEach((e) => log(`    [${e.type}] ${e.text.slice(0, 220)}`)); }
  else log(`  ok`);
}
let shot = 0;
async function snap(label) {
  const name = `shots/${String(++shot).padStart(2, "0")}-${label.replace(/[^a-z0-9]+/gi, "-").slice(0, 40)}.png`;
  await page.screenshot({ path: name }).catch(() => {});
}

async function passGate() {
  if (!PASSCODE) return false;
  const input = page.locator('input[type="password"], input[inputmode="numeric"], input[type="text"], input[type="tel"]').first();
  if (!(await input.count()) || !(await input.isVisible().catch(() => false))) return false;
  await input.fill(PASSCODE);
  await input.press("Enter").catch(() => {});
  const btn = page.getByRole("button", { name: /enter|unlock|go|submit|continue|access/i }).first();
  if (await btn.count()) await btn.click().catch(() => {});
  await page.waitForTimeout(2500);
  return true;
}
async function freshPage() {
  await page.goto(SITE, { waitUntil: "load", timeout: 60000 });
  await page.waitForTimeout(3000);
  const gated = await passGate();
  await page.waitForTimeout(gated ? 4000 : 2500);
}
async function visibleControls() {
  const loc = page.locator('button, [role="tab"], [role="button"]');
  const n = await loc.count();
  const out = []; const seen = new Set();
  for (let i = 0; i < n && out.length < 120; i++) {
    const el = loc.nth(i);
    if (!(await el.isVisible().catch(() => false))) continue;
    const raw = ((await el.innerText().catch(() => "")) || (await el.getAttribute("aria-label").catch(() => "")) || (await el.getAttribute("title").catch(() => "")) || "").trim().replace(/\s+/g, " ");
    if (!raw || raw.length > 40) continue;
    const key = norm(raw);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ el, label: raw, key, enabled: await el.isEnabled().catch(() => true) });
  }
  return out;
}
function allowed(c, pad) {
  if (DESTRUCTIVE.test(c.label)) { log(`${pad}[skip] ${c.label} (destructive)`); return false; }
  if (!ALLOW_ACTIONS && ACTIONS.test(c.label)) { log(`${pad}[skip] ${c.label} (action, set ALLOW_ACTIONS=1 to include)`); return false; }
  if (!c.enabled) { log(`${pad}[disabled] ${c.label}`); return false; }
  return true;
}
async function press(c, depth) {
  const pad = "  ".repeat(depth);
  if (!allowed(c, pad)) return false;
  log(`${pad}[click] ${c.label}`);
  await c.el.click({ timeout: 5000 }).catch((e) => log(`${pad}  click failed: ${e.message.split("\n")[0]}`));
  await page.waitForTimeout(STEP_WAIT);
  await snap(c.label);
  drain(c.label);
  await page.keyboard.press("Escape").catch(() => {});
  return true;
}

log(`EdgeFinder site audit ${new Date().toISOString()}`);
log(`site: ${SITE}`);
await freshPage();
log(`title: ${await page.title()}`);
await snap("landing");
drain("initial load");

const top = await visibleControls();
const topKeys = new Set(top.map((c) => c.key));
log(`\ntop-level controls: ${top.length}: ${top.map((c) => c.label).join(" | ")}`);
log(`\npressing (destructive skipped${ALLOW_ACTIONS ? ", actions allowed" : ", actions skipped: set ALLOW_ACTIONS=1 to include"})`);
const done = new Set();
for (const t of top) {
  if (done.has(t.key)) continue;
  done.add(t.key);
  await freshPage();
  const now = await visibleControls();
  const c = now.find((x) => x.key === t.key);
  log(`\n== ${t.label}`);
  if (!c) { log(`  not visible after a fresh load, skipped`); continue; }
  if (!(await press(c, 0))) continue;
  // Controls that are visible now but were not on the initial screen: sub-tabs, filters, toggles.
  const inner = (await visibleControls()).filter((x) => !topKeys.has(x.key) && !done.has(x.key));
  for (const d of inner) {
    done.add(d.key);
    await press(d, 1);
    // A sub-tab may reveal a further layer (e.g. Health -> Players -> pager).
    const deeper = (await visibleControls()).filter((x) => !topKeys.has(x.key) && !done.has(x.key));
    for (const e of deeper) { done.add(e.key); await press(e, 2); }
  }
}

log(`\nSupabase writes intercepted and not sent: ${blockedWrites}${ALLOW_WRITES ? " (ALLOW_WRITES=1)" : ""}`);
log(`\nAPI hosts contacted (status: count):`);
for (const [host, codes] of Object.entries(apiHits)) log(`  ${host}: ${Object.entries(codes).map(([c, k]) => `${c}:${k}`).join(" ")}`);
const lsKeys = await page.evaluate(() => Object.keys(localStorage)).catch(() => []);
log(`\nlocalStorage keys: ${lsKeys.join(", ") || "(none)"}`);
const problems = events.filter((e) => e.type !== "console.warning");
log(`\nTOTAL problems (errors, exceptions, failed requests): ${problems.length}`);
fs.writeFileSync("report.txt", lines.join("\n") + "\n");
log(`\nreport written to report.txt, screenshots in shots/`);
await browser.close();
