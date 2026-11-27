import { spawn } from "node:child_process"
import { readdir, rm } from "node:fs/promises"
import path from "node:path"
import { createInterface } from "node:readline"
import { Readable, Writable } from "node:stream"
import { ClientSideConnection, ndJsonStream } from "@agentclientprotocol/sdk"
import type * as schema from "@agentclientprotocol/sdk"
import { GEMINI, SERVER } from "./constants"
import { prepare } from "./options"

export interface Handlers {
  update(update: schema.SessionUpdate): void
  permission(call: schema.ToolCallUpdate): "allow" | "deny"
}

export interface Agent {
  readonly version: string
  readonly ended: Promise<void>
  // True from the moment the connection is over (stop, stream close or exit), before `ended` callbacks run.
  readonly closed: boolean
  open(
    input: { directory: string; system: string; mcp?: { url: string; headers: Record<string, string> }; resume?: string },
    on: Handlers,
  ): Promise<string>
  configure(session: string, model: string, effort?: string): Promise<void>
  prompt(session: string, blocks: schema.ContentBlock[]): Promise<schema.PromptResponse>
  cancel(session: string): Promise<void>
  close(session: string): Promise<void>
  stop(): Promise<void>
}

const MIN = [1, 5, 18]

export async function start(exe: string, args: string[]): Promise<Agent> {
  const child = spawn(exe, args, { stdio: ["pipe", "pipe", "pipe"] })
  const exit = Promise.withResolvers<void>()
  child.once("exit", () => exit.resolve())
  child.once("error", (err) => {
    console.warn("[agy-agent] process error:", err.message)
    exit.resolve()
  })
  createInterface({ input: child.stderr }).on("line", (line) => console.debug("[agy-agent]", line))

  const sessions = new Map<string, { on: Handlers; model?: string; effort?: string }>()
  const conn = new ClientSideConnection(
    () => ({
      async sessionUpdate(params) {
        sessions.get(params.sessionId)?.on.update(params.update)
      },
      async requestPermission(params) {
        const allow = sessions.get(params.sessionId)?.on.permission(params.toolCall) === "allow"
        const option = params.options.find((o) => o.kind === (allow ? "allow_once" : "reject_once"))
        if (!option) return { outcome: { outcome: "cancelled" } }
        return { outcome: { outcome: "selected", optionId: option.optionId } }
      },
    }),
    ndJsonStream(
      Writable.toWeb(child.stdin) as WritableStream<Uint8Array>,
      Readable.toWeb(child.stdout) as ReadableStream<Uint8Array>,
    ),
  )
  // Ends on stop, a closed stream or exit, whichever comes first: a closed connection is useless even while the process lingers.
  const end = Promise.withResolvers<void>()
  let closed = false
  const finish = () => {
    closed = true
    end.resolve()
  }
  void exit.promise.then(finish)
  conn.signal.addEventListener("abort", finish, { once: true })
  const stop = async () => {
    finish()
    if (child.exitCode == null && child.signalCode == null) child.kill()
    await exit.promise
  }

  const init = await Promise.race([
    conn.initialize({ protocolVersion: 1, clientCapabilities: {}, clientInfo: { name: "kilo", version: "1" } }),
    exit.promise.then(() => {
      throw new Error(`Antigravity CLI (${exe}) exited during startup`)
    }),
  ]).catch(async (err) => {
    await stop()
    throw err
  })
  const version = init.agentInfo?.version ?? "unknown"
  const parts = version.split(/[.-]/, 3).map(Number)
  const index = MIN.findIndex((min, i) => (parts[i] ?? 0) !== min)
  if (parts.some(Number.isNaN) || (index !== -1 && (parts[index] ?? 0) < MIN[index])) {
    await stop()
    throw new Error(`Antigravity CLI ${version} is too old; version ${MIN.join(".")} or newer is required. Update agy.`)
  }

  return {
    version,
    ended: end.promise,
    get closed() {
      return closed
    },
    async open(input, on) {
      const params = {
        cwd: input.directory,
        mcpServers: input.mcp
          ? [
              {
                type: "http" as const,
                name: SERVER,
                url: input.mcp.url,
                headers: Object.entries(input.mcp.headers).map(([name, value]) => ({ name, value })),
              },
            ]
          : [],
        _meta: { antigravity: { systemPrompt: { append: input.system } } },
      }
      const id = input.resume
        ? (await conn.resumeSession({ ...params, sessionId: input.resume }), input.resume)
        : (await conn.newSession(params)).sessionId
      sessions.set(id, { on })
      return id
    },
    async configure(session, model, effort) {
      const state = sessions.get(session)
      if (!state) throw new Error(`Unknown agy session ${session}`)
      if (state.model !== model) {
        await conn.setSessionConfigOption({ sessionId: session, configId: "model", value: model })
        state.model = model
        // effort options depend on the model, so reapply after a switch
        state.effort = undefined
      }
      if (effort == null || state.effort === effort) return
      await conn.setSessionConfigOption({ sessionId: session, configId: "thought_level", value: effort })
      state.effort = effort
    },
    prompt: (session, blocks) => conn.prompt({ sessionId: session, prompt: blocks }),
    cancel: (session) => conn.cancel({ sessionId: session }),
    async close(session) {
      sessions.delete(session)
      await conn.closeSession({ sessionId: session })
    },
    stop,
  }
}

const state: { current?: Promise<Agent>; ready?: Agent } = {}

// `closed` is checked synchronously, so no caller gets a stopping process.
export function agent(exe: string) {
  if (state.current && !state.ready?.closed) return state.current
  const next = prepare().then((args) => start(exe, args))
  state.current = next
  state.ready = undefined
  next.then(
    (a) => {
      if (state.current === next) state.ready = a
    },
    () => {
      if (state.current === next) state.current = undefined
    },
  )
  return next
}

// New callers get a fresh process; the retired one keeps serving its sessions until their owners stop it.
export function retire() {
  if (state.ready) state.current = state.ready = undefined
}

export async function remove(conversation: string) {
  // an empty or path-like id would widen the prefix match to unrelated conversations
  if (!/^[\w-]+$/.test(conversation)) throw new Error(`Invalid agy conversation id: ${conversation}`)
  const root = path.join(GEMINI, "antigravity-cli")
  const list = async (dir: string, prefix: string) =>
    (await readdir(dir).catch(() => [] as string[])).filter((n) => n.startsWith(prefix)).map((n) => path.join(dir, n))
  const targets = [
    ...(await list(path.join(root, "conversations"), `${conversation}.db`)),
    path.join(root, "conversations", conversation),
    path.join(root, "brain", conversation),
    ...(await list(path.join(root, "annotations"), conversation)),
  ]
  await Promise.all(targets.map((t) => rm(t, { recursive: true, force: true })))
}
