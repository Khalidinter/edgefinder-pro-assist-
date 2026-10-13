// Live audit of the EdgeFinder site: loads the app in headless Chromium, passes the
// access gate, clicks every tab and non-destructive button, and reports console
// errors, uncaught exceptions and failed network calls per step.
//
//   npm install && npx playwright install chromium
//   PASSCODE=1234 node audit.mjs                 (PASSCODE optional)
//   HEADED=1 ... to watch it; ALLOW_ACTIONS=1 ... to also press Scan/Resolve-style buttons
//
// Output: report.txt (safe to share: Odds API keys in URLs are redacted) and shots/*.png
import { chromium } from "playwright";
import fs from "node:fs";

const SITE = process.env.SITE || "https://edgefinder.cognivaultlabs.com";
const PASSCODE = process.env.PASSCODE || "";
const HEADED = process.env.HEADED === "1";
const ALLOW_ACTIONS = process.env.ALLOW_ACTIONS === "1";
const STEP_WAIT = Number(process.env.STEP_WAIT || 1800);
const DESTRUCTIVE = /reset|delete|untrack|remove|clear|wipe|logout|sign out|paper/i;
const ACTIONS = /scan|resolve|run|reconcile|backfill|update sets|push|notify|export/i;

const redact = (u) => String(u).replace(/apiKey=[^&\s]+/gi, "apiKey=REDACTED");
const lines = [];
const log = (s = "") => { lines.push(s); console.log(s); };

fs.mkdirSync("shots", { recursive: true });
const browser = await chromium.launch({ headless: !HEADED });
const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
const page = await context.newPage();

const events = [];
const push = (type, text) => events.push({ t: Date.now(), type, text: redact(text) });
page.on("console", (m) => { if (["error", "warning"].includes(m.type())) push("console." + m.type(), m.text()); });
page.on("pageerror", (e) => push("pageerror", e.message));
page.on("requestfailed", (r) => push("requestfailed", `${r.method()} ${r.url()} :: ${r.failure()?.errorText}`));
page.on("response", (r) => {
  const u = r.url();
  if (r.status() >= 400) push("http" + r.status(), `${r.request().method()} ${u}`);
  const host = new URL(u).host;
  if (/supabase|workers\.dev|the-odds-api|espn|ntfy|\/api\//.test(u)) {
    apiHits[host] = apiHits[host] || {};
    apiHits[host][r.status()] = (apiHits[host][r.status()] || 0) + 1;
  }
});
const apiHits = {};
let cursor = 0;
function drain(label) {
  const mine = events.slice(cursor); cursor = events.length;
  const seen = new Set();
  const out = mine.filter((e) => { const k = e.type + e.text; if (seen.has(k)) return false; seen.add(k); return true; });
  if (out.length) { log(`  problems during "${label}":`); out.forEach((e) => log(`    [${e.type}] ${e.text.slice(0, 220)}`)); }
  else log(`  ok: no console errors, exceptions or failed requests`);
}

let shot = 0;
async function snap(label) {
  const name = `shots/${String(++shot).padStart(2, "0")}-${label.replace(/[^a-z0-9]+/gi, "-").slice(0, 40)}.png`;
  await page.screenshot({ path: name, fullPage: false }).catch(() => {});
}

log(`EdgeFinder site audit ${new Date().toISOString()}`);
log(`site: ${SITE}`);
await page.goto(SITE, { waitUntil: "load", timeout: 60000 });
await page.waitForTimeout(3000);
log(`title: ${await page.title()}`);
await snap("landing");
drain("initial load");

// Access gate
if (PASSCODE) {
  const input = page.locator('input[type="password"], input[inputmode="numeric"], input[type="text"], input[type="tel"]').first();
  if (await input.count()) {
    await input.fill(PASSCODE);
    await input.press("Enter").catch(() => {});
    const btn = page.getByRole("button", { name: /enter|unlock|go|submit|continue|access/i }).first();
    if (await btn.count()) await btn.click().catch(() => {});
    await page.waitForTimeout(2500);
    log("access gate: passcode submitted");
    await snap("after-gate");
    drain("access gate");
  } else log("access gate: no input found (already unlocked?)");
}

// Let the splash screen finish
await page.waitForTimeout(4000);

// Discover tabs and buttons
const candidates = page.locator('[role="tab"], nav button, nav a, header button, footer button, [class*="tab" i] button, button');
const n = await candidates.count();
const seenLabels = new Set();
const targets = [];
for (let i = 0; i < n && targets.length < 60; i++) {
  const el = candidates.nth(i);
  if (!(await el.isVisible().catch(() => false))) continue;
  const label = ((await el.innerText().catch(() => "")) || (await el.getAttribute("aria-label").catch(() => "")) || "").trim().replace(/\s+/g, " ");
  if (!label || label.length > 40 || seenLabels.has(label)) continue;
  seenLabels.add(label);
  targets.push({ i, label });
}
log(`\ncontrols found: ${targets.length}`);
targets.forEach((t) => log(`  - ${t.label}`));

log(`\nclicking through controls (destructive ones skipped${ALLOW_ACTIONS ? ", actions allowed" : ", actions skipped: set ALLOW_ACTIONS=1 to include"})`);
for (const t of targets) {
  if (DESTRUCTIVE.test(t.label)) { log(`\n[skip] ${t.label} (destructive)`); continue; }
  if (!ALLOW_ACTIONS && ACTIONS.test(t.label)) { log(`\n[skip] ${t.label} (action)`); continue; }
  const el = candidates.nth(t.i);
  if (!(await el.isVisible().catch(() => false))) { log(`\n[gone] ${t.label}`); continue; }
  log(`\n[click] ${t.label}`);
  await el.click({ timeout: 5000 }).catch((e) => log(`  click failed: ${e.message.split("\n")[0]}`));
  await page.waitForTimeout(STEP_WAIT);
  await snap(t.label);
  drain(t.label);
  // close any dialog we opened
  await page.keyboard.press("Escape").catch(() => {});
}

// Summary
log(`\nAPI hosts contacted (status: count):`);
for (const [host, codes] of Object.entries(apiHits)) log(`  ${host}: ${Object.entries(codes).map(([c, k]) => `${c}:${k}`).join(" ")}`);
const lsKeys = await page.evaluate(() => Object.keys(localStorage)).catch(() => []);
log(`\nlocalStorage keys: ${lsKeys.join(", ") || "(none)"}`);
const allProblems = events.filter((e) => e.type !== "console.warning");
log(`\nTOTAL problems (errors, exceptions, failed requests): ${allProblems.length}`);
fs.writeFileSync("report.txt", lines.join("\n") + "\n");
log(`\nreport written to report.txt, screenshots in shots/`);
await browser.close();
