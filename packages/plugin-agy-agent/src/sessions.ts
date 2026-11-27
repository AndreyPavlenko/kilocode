import { APICallError } from "@ai-sdk/provider"
import type {
  LanguageModelV3FunctionTool,
  LanguageModelV3Prompt,
  LanguageModelV3StreamPart,
  LanguageModelV3TextPart,
  LanguageModelV3ToolCallPart,
} from "@ai-sdk/provider"
import type * as schema from "@agentclientprotocol/sdk"
import { agent, remove, retire, type Agent } from "./acp"
import { learn } from "./catalog"
import { content } from "./message"
import { executable } from "./options"
import { code, converter, failed, quota } from "./stream"
import { decide, digest, drop, hashes, load, save, type Record } from "./sync"
import { bridge, fit, type Bridge, type Call } from "./tools"
import { exhausted, limits } from "./usage"

const IDLE_MS = 5 * 60_000
// agy aborts an MCP call 180 s after it arrives, so a parked call is handed off before that.
const DEADLINE_MS = 170_000

export interface Request {
  // Only primary calls have one. Utility calls run in a throwaway agy session.
  sessionID?: string
  directory: string
  executable?: string
  model: string
  effort?: string
  system: string
  prompt: LanguageModelV3Prompt
  tools: readonly LanguageModelV3FunctionTool[]
  emit: (part: LanguageModelV3StreamPart) => void
}

// An agy prompt that can span several Kilo steps, one per tool call.
interface Turn {
  prompt: Promise<schema.PromptResponse>
  step: number
  // Last context occupancy, carried into the next step's converter.
  used?: number
  deadline?: ReturnType<typeof setTimeout>
}

// An ACP session open in an agy process.
interface Live {
  id: string
  agent: Agent
  system: string
  tools: string
  bridge?: Bridge
  calls?: AsyncIterator<Call>
  next?: Promise<IteratorResult<Call>>
  sink?: (update: schema.SessionUpdate) => void
  turn?: Turn
}

interface Entry {
  id: string
  directory: string
  queue: Promise<unknown>
  active: boolean
  live?: Live
  record?: Record
  idle?: ReturnType<typeof setTimeout>
  uses: number
  // Forgotten or closed: queued runs must not recreate its record.
  gone: boolean
}

type Reply = Array<LanguageModelV3TextPart | LanguageModelV3ToolCallPart>
type Step = { reply: Reply; seen: boolean } & (
  | { type: "call"; call: Call }
  | { type: "done"; response: schema.PromptResponse }
)

const entries = new Map<string, Entry>()
const never = new Promise<never>(() => undefined)
// Open or opening ACP sessions per agy process. A process stops when its count reaches zero.
const users = new WeakMap<Agent, number>()

function warn(message: string, error: unknown) {
  console.warn(`[agy-agent] ${message}:`, error instanceof Error ? error.message : error)
}

function locked<T>(entry: Entry, work: () => Promise<T>) {
  const run = entry.queue.then(work, work)
  entry.queue = run.catch(() => undefined)
  return run
}

// Resolves or rejects with the work, but rejects as soon as the signal aborts. The work itself keeps going.
function abortable<T>(work: Promise<T>, signal: AbortSignal) {
  if (signal.aborted) {
    work.catch(() => undefined)
    return Promise.reject(signal.reason)
  }
  const aborted = Promise.withResolvers<never>()
  const listener = () => aborted.reject(signal.reason)
  signal.addEventListener("abort", listener, { once: true })
  return Promise.race([work, aborted.promise]).finally(() => signal.removeEventListener("abort", listener))
}

async function acquire(exe: string): Promise<Agent> {
  const item = await agent(exe)
  // Its last owner may have stopped it while this caller waited.
  if (item.closed) return acquire(exe)
  users.set(item, (users.get(item) ?? 0) + 1)
  return item
}

function release(item: Agent) {
  const count = (users.get(item) ?? 1) - 1
  users.set(item, count)
  if (count === 0) void item.stop().catch((err) => warn("stop failed", err))
}

async function open(request: Request, exe: string, tools: boolean, resume?: string) {
  const item = await acquire(exe)
  const live: Live = { id: "", agent: item, system: digest(request.system), tools: digest(request.tools) }
  try {
    const fitted = tools && request.tools.length > 0 ? fit(request.tools) : undefined
    live.bridge = fitted && (await bridge(fitted.tools))
    live.calls = live.bridge?.calls[Symbol.asyncIterator]()
    live.id = await item.open(
      {
        directory: request.directory,
        system: request.system + (fitted?.manual ?? ""),
        mcp: live.bridge && { url: live.bridge.url, headers: live.bridge.headers },
        resume,
      },
      {
        update: (update) => {
          live.bridge?.observe(update)
          live.sink?.(update)
        },
        permission: (call) => (live.bridge?.announce(call) ? "allow" : "deny"),
      },
    )
    return live
  } catch (error) {
    // Checked before release, which may stop the process itself.
    const exited = item.closed
    live.bridge?.close()
    release(item)
    // The process went away: the conversation is still on disk, so a retry resumes it.
    throw exited ? failed(error, true) : error
  }
}

// Ends the turn without closing the session. Kilo still runs any parked call; its result comes back as text.
async function detach(live: Live | undefined) {
  const turn = live?.turn
  if (!live || !turn) return
  clearTimeout(turn.deadline)
  // Cancel before resolving: a result would let agy continue the turn it is about to lose.
  if (!live.agent.closed) await live.agent.cancel(live.id).catch((err) => warn("cancel failed", err))
  await turn.prompt.catch(() => undefined)
  live.bridge?.expire()
  live.turn = undefined
}

async function shut(entry: Entry) {
  const live = entry.live
  if (!live) return
  await detach(live)
  entry.live = undefined
  live.bridge?.close()
  if (!live.agent.closed) await live.agent.close(live.id).catch((err) => warn("close failed", err))
  release(live.agent)
}

function arm(entry: Entry) {
  clearTimeout(entry.idle)
  const uses = entry.uses
  entry.idle = setTimeout(
    () =>
      void locked(entry, async () => {
        if (entry.uses === uses) await shut(entry)
      }).catch((err) => warn("idle close failed", err)),
    IDLE_MS,
  )
  entry.idle.unref()
}

// Read at park time so tests can change it per case.
function park(entry: Entry, turn: Turn, at: number) {
  const step = turn.step
  const limit = Number(process.env.KILO_AGY_DEADLINE_MS) || DEADLINE_MS
  turn.deadline = setTimeout(
    () =>
      void locked(entry, async () => {
        if (entry.live?.turn !== turn || turn.step !== step) return
        await detach(entry.live)
        arm(entry)
      }).catch((err) => warn("deadline handoff failed", err)),
    Math.max(0, at + limit - Date.now()),
  )
  turn.deadline.unref()
}

// Reads one Kilo step: up to the next Kilo tool call or the end of the agy turn.
async function read(live: Live, turn: Turn, request: Request, signal: AbortSignal): Promise<Step> {
  const convert = converter(turn.used)
  const reply: Reply = []
  let text = ""
  let seen = false
  const out = (parts: LanguageModelV3StreamPart[]) =>
    parts.forEach((part) => {
      if (part.type === "text-delta") text += part.delta
      if (part.type === "text-delta" || part.type === "reasoning-delta") seen = true
      request.emit(part)
    })
  turn.step++
  live.sink = (update) => out(convert.update(update))
  const done = turn.prompt.then((response) => ({ type: "done" as const, response }))
  try {
    while (true) {
      const next = live.calls ? (live.next ??= live.calls.next()) : never
      const outcome = await abortable(
        Promise.race([done, next.then((item) => (item.done ? never : { type: "call" as const, call: item.value }))]),
        signal,
      )
      if (outcome.type === "done") {
        out(convert.done(outcome.response))
        if (text) reply.push({ type: "text", text })
        return { ...outcome, reply, seen }
      }
      live.next = undefined
      // A call agy already gave up on was expired; its result can no longer be delivered.
      if (!live.bridge?.parked().includes(outcome.call.id)) continue
      out(convert.call(outcome.call))
      if (text) reply.push({ type: "text", text })
      reply.push({ type: "tool-call", toolCallId: outcome.call.id, toolName: outcome.call.name, input: outcome.call.input })
      return { ...outcome, reply, seen: true }
    }
  } finally {
    live.sink = undefined
    turn.used = convert.used
    if (convert.size) learn(request.model, convert.size)
  }
}

async function fail(error: unknown, live: Live) {
  // The ACP stream closed: the process exited. The conversation is on disk, so a retry resumes it.
  if (live.agent.closed) return failed(error, true)
  const message = error instanceof Error ? error.message : String(error)
  if (!quota.test(message)) return error
  const extra = await limits(true).then(exhausted, (err) => warn("could not read limits", err))
  return extra ? failed(new Error(`${message}\n${extra}`)) : error
}

async function primary(entry: Entry, request: Request, exe: string, signal: AbortSignal) {
  if (signal.aborted) return
  entry.record ??= await load(entry.id)
  const live = entry.live
  const stale = !!live && (live.agent.closed || live.system !== digest(request.system) || live.tools !== digest(request.tools))
  const parked = live?.turn && !stale ? (live.bridge?.parked() ?? []) : []
  const first = decide(request.prompt, entry.record, parked)
  if (stale || first.type === "rebuild") await shut(entry)
  // Changed settings need a resumed session, which ends the parked turn: its results become text.
  const decision = stale && first.type === "results" ? decide(request.prompt, entry.record, []) : first
  if (decision.type === "rebuild" && entry.record) {
    const old = entry.record.conversation
    entry.record = undefined
    await remove(old).catch((err) => warn("could not remove conversation", err))
  }
  // New input while parked: agy must finish the old turn before it takes a new prompt.
  if (decision.type === "continue") await detach(entry.live)

  // Validate before opening: a rejected attachment must not leave an empty conversation behind.
  const blocks = decision.type === "results" ? [] : content(decision.type === "continue" ? decision.tail : request.prompt)
  const resume = entry.record?.conversation
  if (!entry.live && resume) {
    entry.live = await open(request, exe, true, resume).catch(async (error) => {
      // Only agy's own answer proves the conversation is unusable; anything else is retried as is.
      if (APICallError.isInstance(error) || typeof code(error) !== "number" || code(error) === -32000) throw error
      warn("could not resume conversation, replaying history", error)
      entry.record = undefined
      await remove(resume).catch((err) => warn("could not remove conversation", err))
      return undefined
    })
  }
  const full = !entry.record && decision.type === "continue" ? content(request.prompt) : blocks
  entry.live ??= await open(request, exe, true)
  const current = entry.live
  const record = (entry.record ??= { conversation: current.id, hashes: [], system: current.system, tools: current.tools })
  record.system = current.system
  record.tools = current.tools

  const held = decision.type === "results" ? current.turn : undefined
  if (held && decision.type === "results") {
    await Promise.resolve()
      .then(() => current.bridge?.resolve(decision.results))
      .catch(async (error) => {
        // agy still waits on the call; end its turn so the next prompt finds the session free.
        await detach(current)
        throw error
      })
    clearTimeout(held.deadline)
  }
  const turn =
    held ??
    (await (async () => {
      await current.agent.configure(current.id, request.model, request.effort)
      const prompt = current.agent.prompt(current.id, full)
      // A parked turn's failure is seen by the next step, or never if Kilo moves on.
      prompt.catch(() => undefined)
      return { prompt, step: 0 }
    })())
  current.turn = turn

  const sent = hashes(request.prompt)
  const persist = async (reply: Reply, seen: boolean) => {
    // Kilo stores each step as one assistant message, so agy already has its hash.
    record.hashes = seen ? [...sent, ...hashes([{ role: "assistant", content: reply }])] : sent
    await save(entry.id, record).catch((err) => warn("could not save session", err))
  }

  const step = await read(current, turn, request, signal).catch(async (error) => {
    if (signal.aborted) {
      // Kilo stopped the step. agy keeps the input it was sent.
      await detach(current)
      await persist([], false)
      return undefined
    }
    const reason = await fail(error, current)
    await (current.agent.closed ? shut(entry) : detach(current))
    throw reason
  })
  if (!step) return
  if (step.type === "call") {
    await persist(step.reply, step.seen)
    park(entry, turn, step.call.at)
    return
  }
  current.turn = undefined
  // Kilo retries a failed step with the same input, so agy must not be recorded as having answered it.
  if (step.response.stopReason === "cancelled") throw failed(new Error("Antigravity cancelled the turn"), true)
  await persist(step.reply, step.seen)
}

async function utility(request: Request, exe: string, signal: AbortSignal) {
  if (signal.aborted) return
  const blocks = content(request.prompt)
  const live = await open(request, exe, false)
  try {
    await live.agent.configure(live.id, request.model, request.effort)
    const prompt = live.agent.prompt(live.id, blocks)
    prompt.catch(() => undefined)
    const turn: Turn = { prompt, step: 0 }
    live.turn = turn
    const step = await read(live, turn, request, signal).catch(async (error) => {
      const reason = signal.aborted ? undefined : await fail(error, live)
      await detach(live)
      if (reason !== undefined) throw reason
      return undefined
    })
    if (step?.type === "done" && step.response.stopReason === "cancelled") {
      throw failed(new Error("Antigravity cancelled the turn"), true)
    }
  } finally {
    if (!live.agent.closed) await live.agent.close(live.id).catch((err) => warn("close failed", err))
    release(live.agent)
    await remove(live.id).catch((err) => warn("could not remove conversation", err))
  }
}

// An expired login can recover in a new process, so -32000 retries once on a fresh one.
// Other sessions keep the old process until they release it.
async function retry(signal: AbortSignal, work: () => Promise<void>, leave?: () => Promise<void>) {
  return work().catch(async (error) => {
    if (code(error) !== -32000 || signal.aborted) throw error
    retire()
    await leave?.()
    return work()
  })
}

export async function run(request: Request, signal: AbortSignal): Promise<void> {
  const exe = await executable(request.executable)
  const sessionID = request.sessionID
  if (!sessionID) return retry(signal, () => utility(request, exe, signal))
  const entry = entries.get(sessionID) ?? {
    id: sessionID,
    directory: request.directory,
    queue: Promise.resolve(),
    active: false,
    uses: 0,
    gone: false,
  }
  entries.set(sessionID, entry)
  return locked(entry, async () => {
    if (entry.gone) throw new Error("This Kilo session was deleted")
    clearTimeout(entry.idle)
    entry.active = true
    entry.uses++
    try {
      await retry(signal, () => primary(entry, request, exe, signal), () => shut(entry))
    } finally {
      entry.active = false
      // A parked turn waits for Kilo; its deadline or idle() decides when it ends.
      if (!entry.live?.turn) arm(entry)
    }
  })
}

// Kilo stopped running the session, so parked calls will get their results only with the next input.
export function idle(sessionID: string) {
  const entry = entries.get(sessionID)
  if (!entry?.live?.turn || entry.active) return
  void locked(entry, async () => {
    if (entry.active) return
    await detach(entry.live)
    arm(entry)
  }).catch((err) => warn("idle detach failed", err))
}

export async function forget(sessionID: string) {
  const entry = entries.get(sessionID)
  const work = async () => {
    if (entry) {
      entry.gone = true
      clearTimeout(entry.idle)
      await shut(entry)
      if (entries.get(sessionID) === entry) entries.delete(sessionID)
    }
    const record = entry?.record ?? (await load(sessionID))
    if (record) await remove(record.conversation).catch((err) => warn("could not remove conversation", err))
    await drop(sessionID)
  }
  await (entry ? locked(entry, work) : work()).catch((err) => warn("could not forget session", err))
}

export async function close(directory: string) {
  await Promise.all(
    [...entries].map(([id, entry]) => {
      if (entry.directory !== directory) return Promise.resolve()
      return locked(entry, async () => {
        entry.gone = true
        clearTimeout(entry.idle)
        await shut(entry)
        if (entries.get(id) === entry) entries.delete(id)
      }).catch((err) => warn("close failed", err))
    }),
  )
}
