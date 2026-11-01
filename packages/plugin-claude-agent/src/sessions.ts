import fs from "node:fs/promises"
import path from "node:path"
import type {
  LanguageModelV3FunctionTool,
  LanguageModelV3Prompt,
  LanguageModelV3TextPart,
  LanguageModelV3ToolCallPart,
} from "@ai-sdk/provider"
import type { Options, SDKMessage } from "@anthropic-ai/claude-agent-sdk"
import { DATA } from "./constants"
import { content, type Block } from "./message"
import { executable, session as sessionOptions } from "./options"
import { sdk, start, type Session } from "./session"
import { failed, failure, spent, succeeded, type Failure } from "./stream"
import { boundary, catchUp, checkpoints, decide, type Limit, type SessionRecord } from "./sync"
import { SERVER, collect, original, pending, type Pending, type Result } from "./tools"
import { lifted } from "./usage"

const IDLE_MS = 5 * 60_000
const READY_MS = 30_000
const MINUTE_MS = 60_000
const HOUR_MS = 60 * MINUTE_MS
// Claude already has the failed input, so it only needs a nudge.
const NUDGE: Block[] = [
  { type: "text", text: "The previous request failed before you finished. Continue where you left off." },
]

export interface Request {
  // Only primary calls have one. Side calls run in a throwaway process.
  sessionID?: string
  directory: string
  executable?: string
  model: string
  effort?: Options["effort"]
  maxOutputTokens?: number
  connection?: Record<string, string>
  system: string
  prompt: LanguageModelV3Prompt
  tools: readonly LanguageModelV3FunctionTool[]
  emit: (message: SDKMessage, cost?: number) => void
}

interface Live {
  session: Session
  pending: Pending
  model: string
  effort?: Options["effort"]
  // Settings that need a new process to change. Model and effort change on the running one.
  launch: string
  waiting?: string[]
}

interface Entry {
  directory: string
  queue: Promise<unknown>
  active: boolean
  process?: Live
  record?: SessionRecord
  idle?: ReturnType<typeof setTimeout>
}

interface Continuation {
  end: number
  results: Array<{ id: string; result: Result }>
}

interface Prepared {
  first: number
  continuation?: Continuation
  // Kilo sent the input of a stalled Claude turn again.
  resume?: boolean
}

const entries = new Map<string, Entry>()

const launch = (request: Request) =>
  JSON.stringify([request.maxOutputTokens, request.connection, request.system, request.tools, request.directory])

const file = (id: string) => path.join(DATA, "sessions", `${id}.json`)

const load = (id: string) =>
  fs.readFile(file(id), "utf8").then(
    (text): SessionRecord | undefined => JSON.parse(text),
    () => undefined,
  )

async function save(id: string, record: SessionRecord) {
  await fs.mkdir(path.dirname(file(id)), { recursive: true })
  await fs.writeFile(file(id), JSON.stringify(record))
}

function warn(message: string, error: unknown) {
  console.warn(`[claude-agent] ${message}:`, error instanceof Error ? error.message : error)
}

function locked<T>(entry: Entry, work: () => Promise<T>) {
  const run = entry.queue.then(work, work)
  entry.queue = run.catch(() => undefined)
  return run
}

// Resolves or rejects with the work, but rejects as soon as the signal aborts. The work itself keeps going.
function abortable<T>(work: Promise<T>, signal: AbortSignal) {
  if (signal.aborted) work.catch(() => undefined)
  if (signal.aborted) return Promise.reject(signal.reason)
  const aborted = Promise.withResolvers<never>()
  const listener = () => aborted.reject(signal.reason)
  signal.addEventListener("abort", listener, { once: true })
  return Promise.race([work, aborted.promise]).finally(() => signal.removeEventListener("abort", listener))
}

async function interrupt(entry: Entry) {
  const process = entry.process
  if (!process) return
  await process.session.interrupt()
  process.pending.clear()
  process.waiting = undefined
}

async function stop(entry: Entry) {
  const process = entry.process
  if (!process) return
  await process.session.stop().finally(() => {
    process.pending.clear()
    entry.process = undefined
  })
}

function arm(entry: Entry) {
  clearTimeout(entry.idle)
  entry.idle = setTimeout(() => void locked(entry, () => stop(entry)).catch((err) => warn("idle stop failed", err)), IDLE_MS)
  entry.idle.unref()
}

// Checks every minute for the first 5 hours, then once the reported reset is due, then every minute again.
// Timers stop while the computer sleeps, so each tick is at most a minute and the schedule follows the wall clock.
async function wait(entry: Entry, limit: Limit, signal: AbortSignal) {
  const reset = (limit.resetsAt ?? 0) * 1000
  const next = () => {
    const now = Date.now()
    return now - limit.since < 5 * HOUR_MS || reset <= now ? now + MINUTE_MS : reset
  }
  let due = next()
  while (true) {
    const tick = Promise.withResolvers<void>()
    const timer = setTimeout(tick.resolve, Math.max(0, Math.min(due - Date.now(), MINUTE_MS)))
    timer.unref()
    await abortable(tick.promise, signal).finally(() => clearTimeout(timer))
    if (Date.now() < due) continue
    const state = await abortable(lifted(limit.type, entry.process?.session.usage).catch(() => undefined), signal)
    // Without plan limits to read, the reported reset time decides.
    if (state ?? reset <= Date.now()) return
    due = next()
  }
}

async function open(entry: Entry, request: Request, signal: AbortSignal, persist: boolean) {
  if (!entry.process) {
    const calls = pending()
    const session = await start(
      await sessionOptions({
        executable: await executable(request.executable),
        connection: request.connection,
        directory: request.directory,
        model: request.model,
        effort: request.effort,
        maxOutputTokens: request.maxOutputTokens,
        system: request.system,
        tools: request.tools,
        pending: calls,
        persist,
        resume: entry.record?.claudeSessionID,
      }),
    )
    entry.process = {
      session,
      pending: calls,
      model: request.model,
      effort: request.effort,
      launch: launch(request),
    }
  }
  const process = entry.process
  const timer = AbortSignal.timeout(READY_MS)
  await abortable(process.session.ready, AbortSignal.any([signal, timer])).catch(async (error) => {
    await stop(entry).catch((err) => warn("stop failed", err))
    throw timer.aborted ? new Error(`claude did not start within ${READY_MS / 1000} seconds`) : error
  })
  // Apply controls before sending input or releasing parked tool results.
  if (process.model !== request.model) {
    await process.session.setModel(request.model)
    process.model = request.model
  }
  if (process.effort !== request.effort) {
    await process.session.setEffort(request.effort)
    process.effort = request.effort
  }
  return process
}

async function turn(entry: Entry, request: Request & { sessionID: string }, signal: AbortSignal, prepared: Prepared) {
  const continuation = prepared.continuation
  // Validate before opening a process: a rejected rebuild must not pair an empty process with the old record.
  const blocks = catchUp(request.prompt, continuation?.end ?? prepared.first)
  const live = await open(entry, request, signal, true)
  const session = live.session
  // A resumed turn keeps the checkpoint of the input Claude already has.
  const userID = !prepared.resume && (!continuation || blocks.length > 0) ? crypto.randomUUID() : undefined

  const reply: Array<LanguageModelV3TextPart | LanguageModelV3ToolCallPart> = []
  const toolMessages = new Map<string, string>()
  const seen = new Set<string>()
  const waiting: string[] = []
  const check = failure()
  const held: Failure[] = []
  let messageID: string | undefined
  let cost = entry.record?.cost

  const emit = (message: SDKMessage) => {
    const verdict = check(message)
    // Claude keeps the failed input, so the turn continues after a limit wait or Kilo's retry instead of failing.
    if (verdict && (verdict.limit || verdict.error.isRetryable)) {
      held.push(verdict)
      return
    }
    if (message.type === "system" && message.subtype === "init" && request.tools.length > 0) {
      const connected = message.mcp_servers.some((item) => item.name === SERVER && item.status === "connected")
      if (!connected) throw new Error("The Kilo tool server did not connect to Claude Code")
    }
    if (message.type === "assistant" && message.parent_tool_use_id === null && !message.error && !seen.has(message.uuid)) {
      seen.add(message.uuid)
      messageID = message.uuid
      message.message.content.forEach((part) => {
        if (part.type === "text") reply.push({ type: "text", text: part.text })
        if (part.type !== "tool_use") return
        const name = original(part.name, request.tools)
        if (!name) return
        live.pending.expect(part.id, part.name)
        waiting.push(part.id)
        reply.push({ type: "tool-call", toolCallId: part.id, toolName: name, input: part.input })
      })
    }
    if (message.type === "user" && message.uuid && Array.isArray(message.message.content)) {
      message.message.content.forEach((part) => {
        if (part.type === "tool_result") toolMessages.set(part.tool_use_id, message.uuid!)
      })
    }
    // A failed or cancelled turn reports nothing, so its cost stays in the next turn's delta.
    if (message.type === "result" && succeeded(message) && !signal.aborted) {
      const delta = spent(message.total_cost_usd, cost ?? 0)
      cost = message.total_cost_usd
      request.emit(message, delta)
      return
    }
    request.emit(message)
  }

  let completed = false
  let parked = false
  let started = false
  let stall = prepared.resume ? entry.record?.stalled : undefined
  const persist = async () => {
    if (!started || !session.persisted) return
    // Like a cancelled turn, a stalled one leaves Claude owning the reply.
    const interrupted = !completed && (signal.aborted || !!stall)
    const history = completed || interrupted
      ? checkpoints({
          prompt: request.prompt,
          previous: entry.record,
          userID,
          reply: completed ? reply : [],
          messageID: completed ? messageID : undefined,
          toolMessages,
        })
      : undefined
    entry.record = { claudeSessionID: session.id, history, interrupted, cost, stalled: stall }
    await save(request.sessionID, entry.record).catch((err) => warn("could not save session", err))
  }
  try {
    if (stall?.limit) await wait(entry, stall.limit, signal)
    if (prepared.resume) session.begin(NUDGE)
    if (!continuation && !prepared.resume) session.begin(blocks, userID)
    if (continuation && userID) session.addUser(blocks, userID)
    continuation?.results.forEach((item) => live.pending.resolve(item.id, item.result))
    started = true
    while (true) {
      const outcome = await abortable(session.read(emit, request.tools), signal)
      const verdict = held.pop()
      if (!verdict) {
        parked = outcome === "tools"
        completed = true
        stall = undefined
        live.waiting = parked ? waiting : undefined
        break
      }
      const since = entry.record?.stalled?.limit?.since ?? Date.now()
      stall = { limit: verdict.limit && { since, ...verdict.limit } }
      await persist()
      if (!stall.limit) throw verdict.error
      const until = stall.limit.resetsAt ? new Date(stall.limit.resetsAt * 1000).toISOString() : "it resets"
      console.warn(`[claude-agent] limit reached, waiting until ${until}`)
      await wait(entry, stall.limit, signal)
      session.begin(NUDGE)
    }
  } catch (error) {
    if (signal.aborted) return
    // An exited process keeps its transcript, so Kilo's retry resumes it.
    if (!started || !session.ended) throw error
    stall ??= {}
    throw failed(error, true)
  } finally {
    // A tool boundary ends only Kilo's step; the Claude turn waits for the results. Anything else interrupts and drains it.
    if (!parked) await interrupt(entry)
    await persist()
  }
}

// The SDK gives forked messages fresh UUIDs. Verify the copied prefix before mapping checkpoints onto them.
async function fork(sessionID: string, messageID: string) {
  const { deleteSession, forkSession, getSessionMessages } = await sdk()
  const source = await getSessionMessages(sessionID)
  const end = source.findIndex((message) => message.uuid === messageID)
  if (end < 0) return
  const created = await forkSession(sessionID, { upToMessageId: messageID })
  try {
    const copied = await getSessionMessages(created.sessionId)
    if (copied.length !== end + 1) throw new Error("Claude fork did not preserve the requested history boundary")
    const messages = new Map<string, string>()
    copied.forEach((after, i) => {
      const before = source[i]!
      if (before.type !== after.type || JSON.stringify(before.message) !== JSON.stringify(after.message)) {
        throw new Error("Claude fork changed the copied conversation")
      }
      messages.set(before.uuid, after.uuid)
    })
    return { sessionID: created.sessionId, messages }
  } catch (error) {
    await deleteSession(created.sessionId)
    throw error
  }
}

async function branch(entry: Entry, request: Request, source?: SessionRecord) {
  const found = boundary(request.prompt, source)
  if (!source || !found) return 0
  // Validate catch-up input before creating a forked transcript on disk.
  catchUp(request.prompt, found.start)
  const created = await fork(source.claudeSessionID, found.messageID)
  if (!created) return 0
  entry.record = {
    claudeSessionID: created.sessionID,
    // A fork continues from the transcript's saved total.
    cost: source.cost,
    history: found.history.map((item) => ({
      hash: item.hash,
      claudeMessageID: item.claudeMessageID ? created.messages.get(item.claudeMessageID) : undefined,
    })),
  }
  return found.start
}

async function firstMissed(entry: Entry, request: Request & { sessionID: string }) {
  entry.record ??= await load(request.sessionID)
  const decision = decide(request.prompt, entry.record)
  if (decision.kind === "fresh" || decision.kind === "changed") {
    await stop(entry)
    const previous = entry.record
    entry.record = undefined
    return decision.kind === "changed" ? branch(entry, request, previous) : decision.start
  }
  if (entry.process || !entry.record) return decision.start
  const { getSessionInfo } = await sdk()
  if (await getSessionInfo(entry.record.claudeSessionID)) return decision.start
  entry.record = undefined
  return 0
}

async function prepare(entry: Entry, request: Request & { sessionID: string }): Promise<Prepared> {
  // Retire an exited or incompatible process first: once we continue its parked turn, open() must not replace it.
  if (entry.process && (entry.process.session.ended || entry.process.launch !== launch(request))) await stop(entry)
  const first = await firstMissed(entry, request)
  const live = entry.process
  // A stalled turn is never parked: it ended on its result and the interrupt cleared the waiting calls.
  if (!live || !live.session.running || !live.waiting) {
    return { first, resume: !!entry.record?.stalled && first === request.prompt.length }
  }
  const collected = collect(request.prompt, first)
  const results = live.waiting.map((id) => ({ id, result: collected.results.get(id) }))
  if (results.some((item) => !item.result)) {
    await stop(entry)
    entry.record = undefined
    return { first: 0 }
  }
  if (collected.aborted) {
    // A stopped Kilo turn needs a fresh Claude turn; addUser would only steer the old one.
    await interrupt(entry)
    return { first }
  }
  return {
    first,
    continuation: {
      end: collected.end,
      results: results.map((item) => ({ id: item.id, result: item.result! })),
    },
  }
}

async function utility(request: Request, signal: AbortSignal) {
  const entry: Entry = { directory: request.directory, queue: Promise.resolve(), active: true }
  try {
    const live = await open(entry, request, signal, false)
    live.session.begin(content(request.prompt))
    await abortable(live.session.read(request.emit, request.tools), signal)
  } catch (error) {
    if (!signal.aborted) throw error
  } finally {
    await stop(entry).catch((err) => warn("stop failed", err))
  }
}

export function run(request: Request, signal: AbortSignal): Promise<void> {
  const sessionID = request.sessionID
  if (!sessionID) return utility(request, signal)
  const entry = entries.get(sessionID) ?? { directory: request.directory, queue: Promise.resolve(), active: false }
  entries.set(sessionID, entry)
  return locked(entry, async () => {
    clearTimeout(entry.idle)
    entry.active = true
    try {
      await turn(entry, { ...request, sessionID }, signal, await prepare(entry, { ...request, sessionID }))
    } finally {
      entry.active = false
      // A turn waiting for tool results is still busy. Subagents, permission prompts and questions can take longer
      // than IDLE_MS, so idle() starts the timer once Kilo stops running the session.
      if (!entry.process?.waiting) arm(entry)
    }
  })
}

// Kilo stopped running the session, so a turn still waiting for tool results will not get them.
export function idle(sessionID: string) {
  const entry = entries.get(sessionID)
  if (entry?.process?.waiting && !entry.active) arm(entry)
}

export async function forget(sessionID: string) {
  const entry = entries.get(sessionID)
  entries.delete(sessionID)
  if (entry) {
    clearTimeout(entry.idle)
    await locked(entry, () => stop(entry)).catch((err) => warn("stop failed", err))
  }
  await fs.rm(file(sessionID), { force: true }).catch((err) => warn("could not remove session", err))
}

export async function close(directory: string) {
  await Promise.all(
    [...entries].map(async ([id, entry]) => {
      if (entry.directory !== directory) return
      entries.delete(id)
      clearTimeout(entry.idle)
      await locked(entry, () => stop(entry)).catch((err) => warn("stop failed", err))
    }),
  )
}
