#!/usr/bin/env bun
/**
 * Stand-in for the `agy` binary (v1.5.18 shapes from real captures). Plays agy over real protocols.
 *
 *   fake-agy.ts models                          -> `agy models` fixture
 *   fake-agy.ts -p=/usage --output-format json  -> `/usage` fixture
 *   fake-agy.ts acp [flags]                     -> ACP agent, NDJSON JSON-RPC 2.0 on stdio
 *
 * Env:
 *   FAKE_AGY_SCENARIO        JSON file: array of turns, each turn an array of steps. One turn per `session/prompt`,
 *                            consumed in order, also across restarts. Missing turn = `[{ "stop": "end_turn" }]`.
 *   FAKE_AGY_LOG             JSONL log: {"in":method,"id","params"} per received message,
 *                            {"permission":{"id"?,"title","answer"}} per permission answer, {"mcp":{"name","result"|"error"}} per tool call.
 *                            The cursor lives in `<log or scenario>.cursor`.
 *   FAKE_AGY_VERSION         agentInfo.version (default 1.5.18)
 *   FAKE_AGY_MCP_TIMEOUT_MS  MCP call deadline (default 180000, agy's hard 3m0s)
 *   FAKE_AGY_AUTH_FAIL=1     first `session/new` fails -32000 "Authentication required"
 *   FAKE_AGY_EXHAUSTED=1     `/usage` reports the Gemini weekly bucket at 0
 *
 * Steps:
 *   { "thought": "..." } | { "text": "..." } | { "usage": { "used": 1, "size": 1048576 } }
 *   { "tool": { "name": "<short kilo tool name>", "args": {...} } }  tool_call + request_permission + MCP tools/call
 *   { "permission": { "title": "..." } }                              non-Kilo permission request; answer is logged
 *   { "error": { "code": -32603, "message": "..." } }                 prompt fails
 *   { "exit": true }                                                  process exits mid-turn
 *   { "wait": 5000 }                                                  pauses the turn; cancel ends it
 *   { "stop": "end_turn", "usage"?: {...PromptResponse.usage} }        final PromptResponse
 */
import { appendFileSync, existsSync, readFileSync, writeFileSync } from "node:fs"
import { randomUUID } from "node:crypto"
import { createInterface } from "node:readline"

type Json = Record<string, any>
type Step = Json

const env = process.env
const log = env.FAKE_AGY_LOG
const scenario = env.FAKE_AGY_SCENARIO
const timeout = Number(env.FAKE_AGY_MCP_TIMEOUT_MS ?? 180_000)
const cursor = `${log ?? scenario ?? "/tmp/fake-agy"}.cursor`

const record = (entry: Json) => {
  if (log) appendFileSync(log, JSON.stringify(entry) + "\n")
}

const MODELS = [
  ["gemini-4-hydrogen-high", "Gemini 4 Hydrogen (High)"],
  ["gemini-4-hydrogen-xhigh", "Gemini 4 Hydrogen (XHigh)"],
  ["gemini-4-hydrogen-medium", "Gemini 4 Hydrogen (Medium)"],
  ["gemini-4-hydrogen-low", "Gemini 4 Hydrogen (Low)"],
  ["gemini-3.8-flash-high", "Gemini 3.8 Flash (High)"],
  ["gemini-3.8-flash-medium", "Gemini 3.8 Flash (Medium)"],
  ["gemini-3.8-flash-low", "Gemini 3.8 Flash (Low)"],
  ["claude-sonnet-4-6", "Claude Sonnet 4.6 (Thinking)"],
  ["claude-opus-4-6-thinking", "Claude Opus 4.6 (Thinking)"],
  ["gpt-oss-120b-medium", "GPT-OSS 120B (Medium)"],
]

const USAGE = {
  status: "SUCCESS",
  command: {
    name: "usage",
    data: {
      description:
        "Within each group, models share a weekly limit and a 5-hour limit. When either limit is reached, models in that group are unavailable until it resets.",
      groups: [
        {
          name: "Gemini Models",
          description: "Models within this group: Gemini 4 Hydrogen, Gemini 3.8 Flash",
          buckets: [
            { id: "gemini-weekly", name: "Weekly Limit Remaining", window: "weekly", remaining_fraction: 0.9138, reset_time: "2026-12-04T01:37:45Z" },
          ],
        },
        {
          name: "Claude and GPT models",
          description: "Models within this group: Claude Sonnet 4.6 (Thinking), Claude Opus 4.6 (Thinking), GPT-OSS 120B (Medium)",
          buckets: [
            { id: "3p-weekly", name: "Weekly Limit Remaining", window: "weekly", remaining_fraction: 1, reset_time: "1970-01-01T00:00:00Z" },
            { id: "3p-5h", name: "5-Hour Limit Remaining", window: "5h", remaining_fraction: 1, reset_time: "1970-01-01T00:00:00Z" },
          ],
        },
      ],
    },
  },
}

const cmd = process.argv[2] ?? ""

if (cmd === "models") {
  console.error("Fetching available models...")
  console.log(MODELS.map((m) => m.join("\t")).join("\n"))
  process.exit(0)
}

if (cmd.startsWith("-p=/usage")) {
  if (env.FAKE_AGY_EXHAUSTED === "1") USAGE.command.data.groups[0].buckets[0].remaining_fraction = 0
  console.log(JSON.stringify(USAGE))
  process.exit(0)
}

if (cmd !== "acp") {
  console.error(`fake-agy: unsupported args ${process.argv.slice(2).join(" ")}`)
  process.exit(2)
}

const state: { turn: number; auth: boolean } = existsSync(cursor)
  ? JSON.parse(readFileSync(cursor, "utf8"))
  : { turn: 0, auth: false }
const save = () => writeFileSync(cursor, JSON.stringify(state))

const send = (msg: Json) => process.stdout.write(JSON.stringify({ jsonrpc: "2.0", ...msg }) + "\n")
const update = (sessionId: string, body: Json) => send({ method: "session/update", params: { sessionId, update: body } })

let seq = 1
const waiting = new Map<string, (msg: Json) => void>()
const ask = (method: string, params: Json) => {
  const id = `jetski-acp-front:${++seq}`
  const done = Promise.withResolvers<Json>()
  waiting.set(id, done.resolve)
  send({ id, method, params })
  return done.promise
}

const options = () => [
  {
    id: "model",
    name: "Model",
    category: "model",
    type: "select",
    currentValue: "gemini-4-hydrogen",
    options: [
      { value: "gemini-4-hydrogen", name: "Gemini 4 Hydrogen" },
      { value: "gemini-3.8-flash", name: "Gemini 3.8 Flash" },
      { value: "claude-sonnet-4-6", name: "Claude Sonnet 4.6 (Thinking)" },
      { value: "claude-opus-4-6-thinking", name: "Claude Opus 4.6 (Thinking)" },
      { value: "gpt-oss-120b-medium", name: "GPT-OSS 120B (Medium)" },
    ],
  },
  {
    id: "thought_level",
    name: "Thinking Level",
    category: "thought_level",
    type: "select",
    currentValue: "high",
    options: [
      { value: "low", name: "Low" },
      { value: "medium", name: "Medium" },
      { value: "high", name: "High" },
      { value: "xhigh", name: "Extra" },
    ],
  },
  {
    id: "mode",
    name: "Mode",
    category: "mode",
    type: "select",
    currentValue: "default",
    options: [
      { value: "default", name: "Default", description: "Ask before edits and commands, per your permission settings." },
      { value: "accept-edits", name: "Accept edits", description: "Approve file edits in the workspace without asking." },
      { value: "auto", name: "Auto", description: "Approve edits and safe commands; ask before irreversible actions." },
      { value: "plan", name: "Plan", description: "Plan carefully before executing a task." },
    ],
  },
]

interface Session {
  servers: Json[]
  config: ReturnType<typeof options>
  abort?: AbortController
}
const sessions = new Map<string, Session>()

const describe = (session: Session) => ({
  models: {
    availableModels: MODELS.map(([modelId, name]) => ({ modelId, name })),
    currentModelId: "gemini-4-hydrogen-high",
  },
  modes: {
    currentModeId: "default",
    availableModes: options()[2].options.map((o: Json) => ({ id: o.value, name: o.name, description: o.description })),
  },
  configOptions: session.config,
  _meta: { antigravity: { tier: { name: "Antigravity Starter Quota", id: "free-tier", upgradeUri: "https://antigravity.google/g1-upgrade" } } },
})

const open = (params: Json) => {
  const session: Session = { servers: params.mcpServers ?? [], config: options() }
  sessions.set(params.sessionId, session)
  return session
}

class Fail extends Error {
  constructor(
    readonly code: number,
    message: string,
  ) {
    super(message)
  }
}

// Plain-fetch Streamable HTTP MCP client; accepts JSON or SSE responses.
const mcp = async (server: Json, name: string, args: Json, signal: AbortSignal) => {
  const headers: Record<string, string> = {
    "content-type": "application/json",
    accept: "application/json, text/event-stream",
    ...Object.fromEntries((server.headers ?? []).map((h: Json) => [h.name, h.value])),
  }
  const post = async (body: Json) => {
    const res = await fetch(server.url, { method: "POST", headers, body: JSON.stringify({ jsonrpc: "2.0", ...body }), signal })
    const sid = res.headers.get("mcp-session-id")
    if (sid) headers["mcp-session-id"] = sid
    if (!res.ok && res.status !== 202) throw new Error(`MCP HTTP ${res.status}: ${await res.text()}`)
    if (body.id == null) return undefined
    const text = await res.text()
    const msgs = (res.headers.get("content-type") ?? "").includes("text/event-stream")
      ? text
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => JSON.parse(line.slice(5)))
      : [JSON.parse(text)]
    const msg = msgs.find((m) => m.id === body.id)
    if (!msg) throw new Error(`MCP: no response to ${body.method}`)
    if (msg.error) throw new Error(`MCP error ${msg.error.code}: ${msg.error.message}`)
    return msg.result
  }
  await post({
    id: 1,
    method: "initialize",
    params: { protocolVersion: "2025-11-25", capabilities: { roots: { listChanged: true } }, clientInfo: { name: "antigravity", version: "1.5.18" } },
  })
  await post({ method: "notifications/initialized" })
  return post({ id: 2, method: "tools/call", params: { name, arguments: args } })
}

const text = (value: string) => [{ type: "content", content: { type: "text", text: value } }]

const tool = async (sid: string, session: Session, step: Json, signal: AbortSignal) => {
  const short = step.name as string
  const id = `call_${String(Math.random()).slice(2)}`
  const title = `Run mcp_kilo_${short}`
  const input = { ...step.args, toolSummary: `Run ${short}` }
  const base = { toolCallId: id, name: `mcp_kilo_${short}`, args: input, kind: "other", title, rawInput: input }
  update(sid, { sessionUpdate: "tool_call", ...base, status: "in_progress" })
  update(sid, { sessionUpdate: "tool_call_update", ...base, status: "pending" })
  const answer = await abortable(
    ask("session/request_permission", {
      sessionId: sid,
      toolCall: { sessionUpdate: "tool_call", ...base, status: "pending" },
      options: [
        { optionId: "allow", name: "Allow", kind: "allow_once" },
        { optionId: "deny", name: "Deny", kind: "reject_once" },
      ],
    }),
    signal,
  )
  const outcome = answer.result?.outcome
  record({ permission: { id, title, answer: outcome ?? answer.error } })
  if (outcome?.outcome !== "selected" || outcome.optionId !== "allow") {
    const msg = "Permission denied by user"
    update(sid, { sessionUpdate: "tool_call_update", ...base, status: "failed", rawOutput: msg, content: text(msg) })
    return
  }
  const server = session.servers.find((s) => s.type === "http" && s.name === "kilo") ?? session.servers.find((s) => s.type === "http")
  if (!server) throw new Fail(-32603, "fake-agy: no http MCP server in session")
  const deadline = AbortSignal.timeout(timeout)
  const result = await mcp(server, short, step.args ?? {}, AbortSignal.any([signal, deadline])).catch((err: Error) => {
    if (signal.aborted) throw err
    const msg = deadline.aborted ? `MCP tool call to server "kilo" timed out after 3m0s: context deadline exceeded` : err.message
    record({ mcp: { name: short, error: msg } })
    update(sid, { sessionUpdate: "tool_call_update", ...base, status: "failed", rawOutput: msg, content: text(msg) })
    return undefined
  })
  if (!result) return
  record({ mcp: { name: short, result } })
  const out = (result.content ?? [])
    .filter((c: Json) => c.type === "text")
    .map((c: Json) => c.text)
    .join("\n")
  update(sid, { sessionUpdate: "tool_call_update", ...base, status: result.isError ? "failed" : "completed", rawOutput: out, content: text(out) })
}

const abortable = <T>(promise: Promise<T>, signal: AbortSignal) => {
  const stop = Promise.withResolvers<never>()
  if (signal.aborted) stop.reject(signal.reason)
  signal.addEventListener("abort", () => stop.reject(signal.reason), { once: true })
  return Promise.race([promise, stop.promise])
}

const USAGE_DEFAULT = { inputTokens: 12969, outputTokens: 104, cachedReadTokens: 8769, cachedWriteTokens: 0, thoughtTokens: 0, totalTokens: 21842 }

const prompt = async (params: Json) => {
  const sid = params.sessionId as string
  const session = sessions.get(sid)
  if (!session) throw new Fail(-32602, `Session not found: ${sid}`)
  if (session.abort?.signal.aborted)
    throw new Fail(-32603, `Can't add this prompt to session ${sid}: the running prompt is being cancelled; wait for it to finish, then send this prompt again`)
  const turns: Step[][] = scenario ? JSON.parse(readFileSync(scenario, "utf8")) : []
  const steps = turns.at(state.turn) ?? [{ stop: "end_turn" }]
  state.turn++
  save()
  const abort = new AbortController()
  session.abort = abort
  const run = async () => {
    for (const step of steps) {
      if (abort.signal.aborted) return { stopReason: "cancelled" }
      if (step.thought != null) update(sid, { sessionUpdate: "agent_thought_chunk", content: { type: "text", text: step.thought } })
      if (step.text != null) update(sid, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: step.text } })
      if (step.usage) update(sid, { sessionUpdate: "usage_update", used: step.usage.used, size: step.usage.size })
      if (step.tool) await tool(sid, session, step.tool, abort.signal)
      if (step.permission) {
        const id = `call_${String(Math.random()).slice(2)}`
        const answer = await abortable(
          ask("session/request_permission", {
            sessionId: sid,
            toolCall: { toolCallId: id, title: step.permission.title, kind: "execute", status: "pending", rawInput: {} },
            options: [
              { optionId: "allow", name: "Allow", kind: "allow_once" },
              { optionId: "deny", name: "Deny", kind: "reject_once" },
            ],
          }),
          abort.signal,
        )
        record({ permission: { title: step.permission.title, answer: answer.result?.outcome ?? answer.error } })
      }
      if (step.error) throw new Fail(step.error.code, step.error.message)
      if (step.exit) process.exit(1)
      if (step.wait) await abortable(new Promise((done) => setTimeout(done, step.wait)), abort.signal).catch(() => undefined)
      if (step.stop) return { stopReason: step.stop, usage: step.usage ?? USAGE_DEFAULT }
    }
    return { stopReason: abort.signal.aborted ? "cancelled" : "end_turn", usage: USAGE_DEFAULT }
  }
  return run()
    .catch((err) => {
      if (abort.signal.aborted) return { stopReason: "cancelled" }
      throw err
    })
    .finally(() => {
      session.abort = undefined
    })
}

const handle = async (method: string, params: Json) => {
  if (method === "initialize") {
    const caps = { resume: {}, close: {}, load: {}, list: {}, additionalDirectories: {} }
    return {
      protocolVersion: 1,
      agentCapabilities: {
        loadSession: true,
        load_session: true,
        promptCapabilities: { image: true, audio: false, embeddedContext: true },
        sessionCapabilities: caps,
        session_capabilities: caps,
        mcpCapabilities: { http: true },
        mcp_capabilities: { http: true },
        auth: { logout: {} },
        _meta: { antigravity: { tier: true, systemPrompt: { append: true } } },
      },
      agentInfo: { name: "Antigravity ACP Server", version: env.FAKE_AGY_VERSION ?? "1.5.18" },
      authMethods: [
        { id: "oauth-personal", name: "Log in with Google", description: "Log in with your Google account" },
        { id: "oauth-business", name: "Log in with Gemini Enterprise", description: "Log in with your Gemini Enterprise account" },
        { id: "gemini-api-key", name: "Gemini API key", description: "Use an API key with Gemini Developer API" },
        {
          id: "agent-platform",
          name: "Gemini Enterprise Agent Platform",
          description: "Use Gemini Enterprise Agent Platform (formerly Vertex AI) with Application Default Credentials",
        },
      ],
    }
  }
  if (method === "session/new") {
    if (env.FAKE_AGY_AUTH_FAIL === "1" && !state.auth) {
      state.auth = true
      save()
      throw new Fail(-32000, "Authentication required")
    }
    const sessionId = randomUUID()
    return { sessionId, ...describe(open({ ...params, sessionId })) }
  }
  if (method === "session/resume") return describe(open(params))
  if (method === "session/close") {
    sessions.get(params.sessionId)?.abort?.abort()
    sessions.delete(params.sessionId)
    return {}
  }
  if (method === "session/set_config_option") {
    const session = sessions.get(params.sessionId)
    if (!session) throw new Fail(-32602, `Session not found: ${params.sessionId}`)
    const opt = session.config.find((o) => o.id === params.configId)
    if (!opt || !opt.options.some((o) => o.value === params.value))
      throw new Fail(-32602, `Invalid value for config option ${params.configId}: ${params.value}`)
    opt.currentValue = params.value
    return { configOptions: session.config }
  }
  if (method === "session/prompt") return prompt(params)
  throw new Fail(-32601, `Method not found: ${method}`)
}

const rl = createInterface({ input: process.stdin })
rl.on("close", () => process.exit(0))
rl.on("line", (line) => {
  if (!line.trim()) return
  const msg = JSON.parse(line)
  if (msg.method == null) {
    waiting.get(String(msg.id))?.(msg)
    waiting.delete(String(msg.id))
    return
  }
  record({ in: msg.method, id: msg.id, params: msg.params })
  if (msg.method === "session/cancel") {
    sessions.get(msg.params?.sessionId)?.abort?.abort()
    return
  }
  if (msg.id == null) return
  handle(msg.method, msg.params ?? {}).then(
    (result) => send({ id: msg.id, result }),
    (err) => send({ id: msg.id, error: { code: err instanceof Fail ? err.code : -32603, message: err.message } }),
  )
})
