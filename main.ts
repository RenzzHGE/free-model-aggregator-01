// free-model-aggregator-01: one server, two client wires, ten simulated free-tier providers.
//
// Routing: every free model that meets a request's requirements is ranked by the quality table's
// rating_milli, best first; ties go to the provider used least so far, then to the brief's order.
// Failover: a provider that refuses with 402/429/5xx costs the request nothing but the call — the
// next candidate serves it. The refusal's own message names the cause: "budget is spent" and "is
// out" mark that provider dead for the trial (its budget is gone, or its outage has begun); a
// one-off "refused this call" leaves it live for later requests; a 400 spends no budget. Every
// provider call is counted, because the trial counts it too, and calls are never made in parallel
// (a provider serves one inference call at a time). All three upstream wires are spoken — openai,
// anthropic and gemini — in JSON and in SSE, with the request's system text, user text, images,
// tools, stops, max_tokens and JSON mode passed through untouched, because the providers echo what
// they parsed and the trial checks that echo. A JSON-mode request is never routed to an
// anthropic-wire provider: that wire cannot carry JSON mode, so the echo would fail.
//
// Everything here is a pure function of the brief and the providers' answers: no clock, no
// unseeded randomness, so a replay leaves the same state export.

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
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAIText,
  anthropicModelList,
  anthropicToAnthropic,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToGemini,
  openAIToOpenAI,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream inference call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 3000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** One provider's live state: how many inference calls it has taken, and whether it is dead. */
interface ProviderState {
  probe: Probe;
  calls: number;
  dead: boolean;
  badModels: Set<string>;
}
let states: ProviderState[] = [];

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  states = probes.map((probe) => ({ probe, calls: 0, dead: false, badModels: new Set<string>() }));
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  state: ProviderState;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that meets the requirements, best first. A request that asked for JSON mode on
 * the OpenAI wire is never offered to an anthropic-wire provider: that wire cannot carry JSON mode,
 * so its echo would report json_mode false and the request would fail its checks.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const state of states) {
    if (state.dead || !state.probe.ok) continue;
    if (jsonMode && state.probe.provider.wire === 'anthropic') continue;
    for (const model of state.probe.models) {
      if (state.badModels.has(model.id)) continue;
      const row = ratingOfModel(brief.ratings, state.probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ state, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      a.state.calls - b.state.calls ||
      (a.state.probe.provider.id < b.state.probe.provider.id ? -1 : a.state.probe.provider.id > b.state.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
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

/** One client request body, or null when it is not one JSON object or is too large. */
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

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream URL for one provider call: the model travels in Gemini's path, not its body. */
function upstreamUrl(state: ProviderState, modelId: string, stream: boolean): string {
  const base = joinBase(ENV_URL, state.probe.provider.base);
  const wire = state.probe.provider.wire;
  if (wire === 'openai') return `${base}/v1/chat/completions`;
  if (wire === 'anthropic') return `${base}/v1/messages`;
  return `${base}/v1beta/models/${encodeURIComponent(modelId)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
}

/** The upstream request body for one (client wire, provider wire) pair, naming the routed model. */
function upstreamBody(clientWire: 'openai' | 'anthropic', providerWire: string, body: Json, modelId: string): Json {
  if (providerWire === 'openai') return clientWire === 'openai' ? openAIToOpenAI(body, modelId) : anthropicToOpenAI(body, modelId);
  if (providerWire === 'anthropic') return clientWire === 'openai' ? openAIToAnthropic(body, modelId) : anthropicToAnthropic(body, modelId);
  return clientWire === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
}

/** The refusal message inside one provider's error envelope, in its own wire's shape. */
function errorMessageOf(wire: string, text: string): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return '';
  }
  if (!isRecord(parsed) || !isRecord(parsed.error)) return '';
  return typeof parsed.error.message === 'string' ? parsed.error.message : '';
}

type RefusalKind = 'budget' | 'outage' | 'oneoff' | 'concurrency' | 'badmodel' | 'badrequest' | 'transient';

interface Attempt {
  ok: boolean;
  status: number;
  text: string;
  message: string;
  kind: RefusalKind;
}

/**
 * One inference call to one provider, and what it cost. A 402/429/5xx spends the provider's budget
 * (its call counter moves before the refusal), a 400 or 401 does not. A refusal that says the budget
 * is spent, or that the provider is out, marks it dead for the trial; a one-off refusal leaves it
 * live; an unknown model is struck from this provider only.
 */
async function callUpstream(state: ProviderState, modelId: string, clientWire: 'openai' | 'anthropic', body: Json): Promise<Attempt> {
  const provider = state.probe.provider;
  const wire = provider.wire;
  const stream = body.stream === true;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let res: Response;
  try {
    res = await fetch(upstreamUrl(state, modelId, stream), {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody(clientWire, wire, body, modelId)),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    state.calls += 1;
    return { ok: false, status: 0, text: '', message: (err as Error).message, kind: 'transient' };
  }
  const text = await res.text();
  if (res.ok) {
    state.calls += 1;
    return { ok: true, status: res.status, text, message: '', kind: 'transient' };
  }
  const message = errorMessageOf(wire, text);
  if (res.status === 400) {
    if (message.includes('invalid_model') || message.includes('unknown model')) {
      state.badModels.add(modelId);
      return { ok: false, status: res.status, text, message, kind: 'badmodel' };
    }
    return { ok: false, status: res.status, text, message, kind: 'badrequest' };
  }
  if (res.status === 401) {
    state.dead = true;
    return { ok: false, status: res.status, text, message, kind: 'badrequest' };
  }
  state.calls += 1;
  let kind: RefusalKind = 'transient';
  if (message.includes('budget is spent')) {
    kind = 'budget';
    state.dead = true;
  } else if (message.includes('is out')) {
    kind = 'outage';
    state.dead = true;
  } else if (message.includes('concurrent_requests')) {
    kind = 'concurrency';
  } else if (message.includes('refused this call')) {
    kind = 'oneoff';
  }
  return { ok: false, status: res.status, text, message, kind };
}

/**
 * One provider's answer rewritten into the client's wire. JSON answers are translated object to
 * object; SSE answers are read whole and re-emitted in the client's event order. A same-wire answer
 * is passed through byte for byte.
 */
function translateResponse(clientWire: 'openai' | 'anthropic', providerWire: string, modelId: string, stream: boolean, text: string): string {
  if (!stream) {
    const parsed: unknown = JSON.parse(text);
    if (!isRecord(parsed)) throw new Error('the provider answered with a body that is not one JSON object');
    if (clientWire === 'openai') {
      if (providerWire === 'openai') return text;
      if (providerWire === 'anthropic') return JSON.stringify(anthropicToOpenAIResponse(parsed, modelId));
      return JSON.stringify(geminiToOpenAIResponse(parsed, modelId));
    }
    if (providerWire === 'anthropic') return text;
    if (providerWire === 'openai') return JSON.stringify(openAIToAnthropic(parsed, modelId));
    return JSON.stringify(geminiToAnthropicResponse(parsed, modelId));
  }
  if (clientWire === providerWire) return text;
  if (clientWire === 'openai') {
    if (providerWire === 'anthropic') return openAIEventsFromAnthropic(text, modelId);
    return openAIEventsFromGemini(text, modelId);
  }
  if (providerWire === 'openai') return anthropicEventsFromOpenAIText(text, modelId);
  return anthropicEventsFromGemini(text, modelId);
}

/** One client request: the best free model that meets it, failing over while providers refuse. */
async function serve(req: IncomingMessage, res: ServerResponse, clientWire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const jsonMode = isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const stream = body.stream === true;
  const candidates = candidatesFor(requires, jsonMode);
  let lastError = 'no free model on the directory meets this request';
  for (const candidate of candidates) {
    const attempt = await callUpstream(candidate.state, candidate.model.id, clientWire, body);
    if (!attempt.ok) {
      lastError = attempt.message !== '' ? attempt.message : `the provider answered HTTP ${attempt.status}`;
      log(`${candidate.state.probe.provider.id}/${candidate.model.id} refused (${attempt.kind}): ${lastError}`);
      continue;
    }
    let out: string;
    try {
      out = translateResponse(clientWire, candidate.state.probe.provider.wire, candidate.model.id, stream, attempt.text);
    } catch (err) {
      log(`${candidate.state.probe.provider.id}/${candidate.model.id} answered, but the answer could not be translated: ${(err as Error).message}`);
      return sendError(res, clientWire, 502, 'upstream_error', 'the provider answered in a shape that could not be translated');
    }
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(out);
    } else {
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(out) });
      res.end(out);
    }
    return;
  }
  return sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', lastError);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
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

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
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

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
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
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
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
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
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
