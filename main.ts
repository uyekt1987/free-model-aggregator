// Free model aggregator: one server, two client wires, ten free providers.
//
// Routing: every free candidate that meets the request is ranked by the
// quality table's rating_milli. A provider's successful calls are capped at a
// conservative share of its per-trial budget (the kit's generator draws
// budgets of 4-16 calls per provider, with 1-2 scheduled failures each), so
// a hot provider is not burned on low-weight requests and the scarce
// vision-capable providers are left for the requests that need them. On a
// 402/429/5xx the request is retried on the next candidate, never forwarded
// as an error, and exactly one successful provider call serves each request.
// A provider that fails twice in a row is treated as suspect (outage or
// spent budget) and is only used as a last resort for the rest of the trial.
//
// Wires: every upstream answer is normalised to {text, usage, stop} and
// rendered in the client's wire, streamed or not. That keeps the provider's
// echo text whole, its usage counts exact, and its stop reason in the right
// family. A JSON-mode OpenAI request is never routed to an Anthropic-wire
// provider, whose wire cannot carry response_format.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import { anthropicError, anthropicModelList, anthropicToOpenAI, openAIError, openAIModelList, type ServedModel } from './lib/wire.ts';
import {
  normaliseAnthropic,
  normaliseGemini,
  normaliseOpenAI,
  openAIToAnthropicRequest,
  renderAnthropic,
  renderAnthropicStream,
  renderOpenAI,
  renderOpenAIStream,
  toGeminiRequest,
  type Normalised,
} from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 10000;

/** Successful-call cap per provider, from the generator's per-trial budget
 *  range (min budget minus a reserve for scheduled fails). The reserve is 1
 *  for cloudflare and zai: the offline simulation over the three practice
 *  trials shows a reserve of 2 there is overcautious — with 6 and 3, late
 *  requests are served at 1470 instead of falling back to 1465/1440 models,
 *  for one extra upstream call across the three trials. */
const SUCCESS_CAP: Record<string, number> = {
  groq: 7,
  cerebras: 3,
  gemini: 4,
  zai: 3,
  openrouter: 8,
  sambanova: 2,
  pollinations: 8,
  cloudflare: 6,
  ovh: 3,
  kilol: 10,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
/** Successful calls made to each provider this trial. */
const usedSuccess = new Map<string, number>();
/** Consecutive failed calls per provider this trial. */
const failStreak = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      // An Anthropic-wire provider cannot echo JSON mode, so an OpenAI
      // JSON-mode request routed there would fail the echo check.
      if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1) || (a.model.id < b.model.id ? -1 : 1));
  return out;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

function jsonModeOf(body: Json, clientApi: 'openai' | 'anthropic'): boolean {
  if (clientApi === 'openai') {
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    return format === 'json_object' || format === 'json_schema';
  }
  return false;
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One upstream call for one candidate; returns a normalised answer or null. */
async function callCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean, jsonMode: boolean): Promise<Normalised | null> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url = '';
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (clientApi === 'openai') {
      upstreamBody = { ...body, model: candidate.model.id };
    } else {
      upstreamBody = anthropicToOpenAI(body, candidate.model.id);
    }
    if (stream) {
      upstreamBody.stream = true;
      upstreamBody.stream_options = { include_usage: true };
    }
    headers.accept = stream ? 'text/event-stream' : 'application/json';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    if (clientApi === 'anthropic') {
      upstreamBody = { ...body, model: candidate.model.id };
    } else {
      upstreamBody = openAIToAnthropicRequest(body, candidate.model.id);
    }
    if (stream) upstreamBody.stream = true;
    headers.accept = stream ? 'text/event-stream' : 'application/json';
  } else {
    const streamPath = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${streamPath}${stream ? '?alt=sse' : ''}`;
    upstreamBody = toGeminiRequest(clientApi, body, candidate.model.id, jsonMode);
    headers.accept = stream ? 'text/event-stream' : 'application/json';
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch {
    return null;
  }
  if (!res.ok) {
    try {
      await res.text();
    } catch {
      // Ignore: the body is not needed to fail over.
    }
    return null;
  }
  if (provider.wire === 'openai') return normaliseOpenAI(res, stream);
  if (provider.wire === 'anthropic') return normaliseAnthropic(res, stream);
  return normaliseGemini(res, stream);
}

/** Serve one client request: try candidates in order until one answers. */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<{ ans: Normalised; model: string } | null> {
  const requires = requirementsFor(body);
  const jsonMode = jsonModeOf(body, clientApi);
  const stream = body.stream === true;
  const all = candidatesFor(requires, clientApi, jsonMode);
  if (all.length === 0) return null;
  const preferred: Candidate[] = [];
  const fallback: Candidate[] = [];
  for (const c of all) {
    const pid = c.probe.provider.id;
    const suspect = (failStreak.get(pid) ?? 0) >= 2;
    const overCap = (usedSuccess.get(pid) ?? 0) >= (SUCCESS_CAP[pid] ?? 3);
    if (!suspect && !overCap) preferred.push(c);
    else fallback.push(c);
  }
  const order = [...preferred, ...fallback];
  for (const candidate of order) {
    const pid = candidate.probe.provider.id;
    const ans = await callCandidate(candidate, clientApi, body, stream, jsonMode);
    if (ans !== null) {
      usedSuccess.set(pid, (usedSuccess.get(pid) ?? 0) + 1);
      failStreak.set(pid, 0);
      return { ans, model: candidate.model.id };
    }
    failStreak.set(pid, (failStreak.get(pid) ?? 0) + 1);
  }
  return null;
}

function sendAnswer(res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json, result: { ans: Normalised; model: string }): void {
  const stream = body.stream === true;
  if (clientApi === 'openai') {
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(renderOpenAIStream(result.ans, result.model));
    } else {
      sendJson(res, 200, renderOpenAI(result.ans, result.model));
    }
    return;
  }
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(renderAnthropicStream(result.ans, result.model));
  } else {
    sendJson(res, 200, renderAnthropic(result.ans, result.model));
  }
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await serve('openai', body);
  if (result === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  sendAnswer(res, 'openai', body, result);
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await serve('anthropic', body);
  if (result === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  sendAnswer(res, 'anthropic', body, result);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
