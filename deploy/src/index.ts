/**
 * Hyperiad's public API, at arity.science/api/hyperiad/.
 *
 *   GET  /info  the server's budget; answered here, so loading a notebook never wakes a container
 *   POST /run   {code, replay}: run a cell after replaying the earlier ones, in a container
 *
 * Guards, sized for low traffic from anonymous visitors:
 * - request size, and exact rate limits per visitor and overall, counted by one Durable Object (Gate);
 * - at most INSTANCES containers, each running one job at a time; a busy instance answers 503
 *   and the other is tried once;
 * - in the container (hyperiad serve --isolate): the cost estimate refuses heavy jobs before they
 *   run, and each job has a wall-clock limit and a memory limit.
 */
import { Container } from "@cloudflare/containers";
import { DurableObject } from "cloudflare:workers";

export const LIMITS = {
  budget: 2e9, // operations a server job may estimate (the browser's is 2e7)
  timeoutSeconds: 60,
  memoryMb: 1024,
};
const INSTANCES = 2;
// Jobs allowed per visitor (by IP address) and overall, over sliding windows.
const RATE = { visitorPerMinute: 4, visitorPerHour: 30, overallPerMinute: 30 };
const MAX_BODY = 200_000;
const ALLOWED_ORIGINS = [
  /^https:\/\/(www\.)?arity\.science$/,
  /^https:\/\/([a-z0-9-]+\.)?arity-science\.pages\.dev$/,
  /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/,
];

export class HyperiadServer extends Container {
  defaultPort = 8080;
  sleepAfter = "5m";
  envVars = {
    HYPERIAD_BUDGET: String(LIMITS.budget),
    HYPERIAD_JOB_TIMEOUT: String(LIMITS.timeoutSeconds),
    HYPERIAD_JOB_MEMORY: String(LIMITS.memoryMb),
  };
}

/**
 * Counts jobs. A single instance sees every request, so the limits are exact (Cloudflare's rate
 * limiting binding counts per machine and let bursts through). Counts live in memory: if the
 * instance is evicted they restart from zero, which only errs towards letting a job run.
 */
export class Gate extends DurableObject {
  private visitors = new Map<string, number[]>();
  private overall: number[] = [];

  admit(visitor: string): boolean {
    const now = Date.now();
    const minute = now - 60_000;
    const hour = now - 3_600_000;
    this.overall = this.overall.filter((t) => t > minute);
    const mine = (this.visitors.get(visitor) ?? []).filter((t) => t > hour);
    const mineLastMinute = mine.filter((t) => t > minute).length;
    if (
      this.overall.length >= RATE.overallPerMinute ||
      mine.length >= RATE.visitorPerHour ||
      mineLastMinute >= RATE.visitorPerMinute
    ) {
      this.visitors.set(visitor, mine);
      return false;
    }
    mine.push(now);
    this.overall.push(now);
    this.visitors.set(visitor, mine);
    if (this.visitors.size > 10_000) {
      for (const [k, v] of this.visitors) if (!v.some((t) => t > hour)) this.visitors.delete(k);
    }
    return true;
  }
}

interface Env {
  HYPERIAD: DurableObjectNamespace<HyperiadServer>;
  GATE: DurableObjectNamespace<Gate>;
}

function cors(origin: string | null): Record<string, string> {
  if (!origin || !ALLOWED_ORIGINS.some((re) => re.test(origin))) return {};
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

function json(value: unknown, status: number, headers: Record<string, string>): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store", ...headers },
  });
}

const BUSY = "The Hyperiad server is busy with other jobs. Try again in a minute.";
const TOO_MANY = "Too many requests to the Hyperiad server. Wait a minute and try again.";

async function run(request: Request, env: Env, headers: Record<string, string>): Promise<Response> {
  const declared = Number(request.headers.get("Content-Length") ?? "0");
  if (declared > MAX_BODY) return json({ error: "This notebook is too large to send to the server." }, 413, headers);
  const body = await request.text();
  if (body.length > MAX_BODY) return json({ error: "This notebook is too large to send to the server." }, 413, headers);

  const visitor = request.headers.get("CF-Connecting-IP") ?? "unknown";
  if (!(await env.GATE.getByName("gate").admit(visitor))) return json({ error: TOO_MANY }, 429, headers);

  const first = Math.floor(Math.random() * INSTANCES);
  for (let i = 0; i < INSTANCES; i++) {
    const instance = env.HYPERIAD.getByName(`hyperiad-${(first + i) % INSTANCES}`);
    const answer = await instance.fetch(
      new Request("http://container/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body })
    );
    if (answer.status === 503) continue; // that instance is running a job: try the next
    return new Response(answer.body, {
      status: answer.status,
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store", ...headers },
    });
  }
  return json({ error: BUSY }, 503, headers);
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const path = new URL(request.url).pathname.replace(/^\/api\/hyperiad/, "").replace(/\/$/, "");
    const headers = cors(request.headers.get("Origin"));
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers });
    if (path === "/info" && request.method === "GET") {
      return json({ name: "hyperiad", budget: LIMITS.budget, timeout: LIMITS.timeoutSeconds, memoryMb: LIMITS.memoryMb }, 200, headers);
    }
    if (path === "/run" && request.method === "POST") {
      try {
        return await run(request, env, headers);
      } catch (err) {
        console.error(err);
        return json({ error: "The Hyperiad server could not start. Try again in a minute." }, 502, headers);
      }
    }
    return json({ error: "not found" }, 404, headers);
  },
};
