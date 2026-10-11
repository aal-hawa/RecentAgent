// demo/deploy.mjs — deploys the RecentAgent WEB DEMO (this folder) to Vercel
// as a Python (FastAPI) serverless function.
//
// Run from the repo root:   node demo/deploy.mjs
// Requires in your environment (values are never printed):
//   VERCEL_API_TOKEN  — Vercel API token
//   GEMINI_API        — free-tier Google Gemini key (powers the demo agent)
//
// Only the files in this demo/ folder are uploaded — the Streamlit app itself
// stays local-first and untouched.
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const TOKEN = process.env.VERCEL_API_TOKEN;
if (!TOKEN) { console.error("FATAL: VERCEL_API_TOKEN is not set"); process.exit(1); }
const GEMINI = process.env.GEMINI_API || "";
const ROOT = path.dirname(fileURLToPath(import.meta.url)); // the demo folder itself
const NAME = "recentagent"; // Vercel project name -> recentagent-<suffix>.vercel.app
let TEAM = null;

// ---- API helper with automatic team-scope retry ----
async function api(method, url, body) {
  const opts = { method, headers: { Authorization: `Bearer ${TOKEN}` } };
  if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  let res = await fetch(teamUrl(url), opts);
  let json = await res.json().catch(() => null);
  if ((res.status === 403 || res.status === 400) && !TEAM && json?.error?.message?.toLowerCase().includes("team")) {
    const me = await fetch("https://api.vercel.com/v2/user", { headers: { Authorization: `Bearer ${TOKEN}` } }).then(r => r.json()).catch(() => null);
    TEAM = me?.user?.defaultTeamId || null;
    if (TEAM) {
      res = await fetch(teamUrl(url), opts);
      json = await res.json().catch(() => null);
    }
  }
  return { status: res.status, json };
}
function teamUrl(url) {
  const u = new URL("https://api.vercel.com" + url);
  if (TEAM) u.searchParams.set("teamId", TEAM);
  return u;
}

function walk(dir, base = "") {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (e.isDirectory()) {
      if (e.name !== "__pycache__" && !e.name.startsWith(".")) {
        out.push(...walk(path.join(dir, e.name), base ? `${base}/${e.name}` : e.name));
      }
    } else if (!e.name.startsWith(".") && e.name !== "deploy.mjs") {
      out.push({ rel: base ? `${base}/${e.name}` : e.name, abs: path.join(dir, e.name) });
    }
  }
  return out;
}

async function setEnv(key, value, targets) {
  let r = await api("POST", `/v10/projects/${NAME}/env`, { key, value, target: targets, type: "encrypted" });
  if (r.status >= 300) {
    const list = await api("GET", `/v9/projects/${NAME}/env`);
    const found = (list.json?.envs || []).find(e => e.key === key);
    if (found) {
      await api("DELETE", `/v10/projects/${NAME}/env/${found.id}`);
      r = await api("POST", `/v10/projects/${NAME}/env`, { key, value, target: targets, type: "encrypted" });
    }
  }
  return r.status < 300;
}

async function main() {
  console.log("== Vercel demo deploy: RecentAgent (web demo) ==");
  console.log(`deploy root: ${ROOT}`);
  console.log(`GEMINI_API: ${GEMINI ? "present (hidden)" : "MISSING"}`);

  // 1. create project (or reuse)
  let proj = await api("POST", "/v9/projects", { name: NAME });
  if (proj.status === 409) {
    console.log(`project '${NAME}' already exists — reusing`);
  } else if (proj.status < 300) {
    console.log(`project '${NAME}' created`);
  } else {
    console.error("project create failed:", proj.status, JSON.stringify(proj.json).slice(0, 300));
    process.exit(1);
  }

  // 1b. direct API deployments don't auto-detect the framework — set it
  // explicitly so the build resolves the Python entrypoint (index.py) and
  // routes every request through the FastAPI app.
  const cur = await api("GET", `/v9/projects/${NAME}`);
  if (cur.json?.framework !== "fastapi") {
    const patch = await api("PATCH", `/v9/projects/${NAME}`, { framework: "fastapi" });
    if (patch.status >= 300) {
      console.error("framework PATCH failed:", patch.status, JSON.stringify(patch.json).slice(0, 300));
      process.exit(1);
    }
    console.log("project framework set: fastapi");
  } else {
    console.log("project framework: fastapi (already set)");
  }

  // 2. env vars (values never printed)
  if (GEMINI) {
    const ok = await setEnv("GEMINI_API", GEMINI, ["production", "preview"]);
    console.log(`env: GEMINI_API=${ok ? "set" : "FAILED"}`);
    if (!ok) process.exit(1);
  } else {
    console.log("env: GEMINI_API skipped (not set) — the demo will reply with an LLM error");
  }

  // 3. collect demo files
  const files = walk(ROOT);
  const totalBytes = files.reduce((s, f) => s + fs.statSync(f.abs).size, 0);
  console.log(`files: ${files.length} (${(totalBytes / 1024).toFixed(0)} KB) — ${files.map(f => f.rel).join(", ")}`);
  const mustHave = ["index.py", "index.html", "requirements.txt", "vercel.json"];
  for (const m of mustHave) {
    if (!files.some(f => f.rel === m)) { console.error(`FATAL: ${m} is missing from the demo folder`); process.exit(1); }
  }
  const payload = files.map(f => ({ file: f.rel, data: fs.readFileSync(f.abs).toString("base64"), encoding: "base64" }));

  // 4. deploy (framework auto-detected: FastAPI via requirements.txt + index.py entrypoint)
  console.log("creating deployment (production)…");
  const dep = await api("POST", "/v13/deployments", { name: NAME, project: NAME, target: "production", files: payload });
  if (dep.status >= 300) {
    console.error("deployment create failed:", dep.status, JSON.stringify(dep.json).slice(0, 500));
    process.exit(1);
  }
  const id = dep.json.id;
  console.log(`deployment ${id} queued`);

  // 5. poll
  const t0 = Date.now();
  while (Date.now() - t0 < 8 * 60 * 1000) {
    await new Promise(r => setTimeout(r, 6000));
    const d = await api("GET", `/v13/deployments/${id}`);
    const state = d.json?.readyState;
    const elapsed = Math.round((Date.now() - t0) / 1000);
    process.stdout.write(`\r  ${state || "…"} (${elapsed}s)   `);
    if (state === "READY") {
      console.log("\n== DEPLOYMENT READY ==");
      const urls = d.json.urls || [];
      for (const u of urls) console.log("  https://" + u.replace(/^https?:\/\//, ""));
      return;
    }
    if (state === "ERROR" || state === "CANCELED") {
      console.log(`\n== DEPLOYMENT ${state} ==`);
      console.log("errorMessage:", d.json.errorMessage || "(none)");
      const ev = await fetch(teamUrl(`/v2/deployments/${id}/events?limit=500`), { headers: { Authorization: `Bearer ${TOKEN}` } }).then(r => r.text()).catch(() => "");
      const lines = ev.split("\n").filter(l => /error|Error|ERROR|failed|FAILED/.test(l));
      console.log("error events (tail):\n" + (lines.slice(-15).join("\n") || ev.split("\n").slice(-15).join("\n")));
      process.exit(1);
    }
  }
  console.log("\ntimed out waiting for build — check the Vercel dashboard");
  process.exit(1);
}
main().catch(e => { console.error("FATAL", e); process.exit(1); });
