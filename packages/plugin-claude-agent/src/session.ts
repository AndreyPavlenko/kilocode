import type { LanguageModelV3FunctionTool } from "@ai-sdk/provider"
import type { AccountInfo, Options, SDKMessage, SDKResultMessage, SDKUserMessage } from "@anthropic-ai/claude-agent-sdk"
import { original } from "./tools"

// Older binaries silently ignore verbatimPrompts and can interpret user text as Claude commands.
const MINIMUM = "2.1.248"

interface Turn {
  drained: PromiseWithResolvers<void>
  messages: SDKMessage[]
  wake?: () => void
  finished: boolean
}

export type Session = Awaited<ReturnType<typeof start>>

export const sdk = () => import("@anthropic-ai/claude-agent-sdk")

export function loggedIn(account: AccountInfo) {
  return account.tokenSource !== "none" || account.apiKeySource !== undefined
}

// Compares release numbers like 2.1.248. A pre-release or build suffix is ignored.
export function atLeast(version: string, minimum: string) {
  const parse = (value: string) => value.split(/[-+]/)[0]!.split(".").map(Number)
  const current = parse(version)
  const required = parse(minimum)
  const index = required.findIndex((part, position) => (current[position] ?? 0) !== part)
  return index === -1 || (current[index] ?? 0) > required[index]!
}

export async function start(options: Options) {
  const { query } = await sdk()
  let input!: ReadableStreamDefaultController<SDKUserMessage>
  const prompt = new ReadableStream<SDKUserMessage>({
    start(controller) {
      input = controller
    },
  })
  const session = query({ prompt: prompt as AsyncIterable<SDKUserMessage>, options })
  let turn: Turn | undefined
  let initialized = false
  let ended = false
  let failure: unknown
  let stopping: Promise<void> | undefined
  let persisted = false
  const ready = session.initializationResult().then((init) => {
    const version = init.claude_code_version
    if (!version || !atLeast(version, MINIMUM)) {
      throw new Error(`Claude Code ${version ?? "(unknown)"} is too old; ${MINIMUM} or later is required. Run \`claude update\`.`)
    }
    if (!loggedIn(init.account)) throw new Error("Claude Code is not logged in. Run `claude auth login` in a terminal.")
    initialized = true
  })
  // A cancel during startup can stop the session before anything awaits ready.
  ready.catch(() => undefined)

  function finish() {
    if (!turn) return
    turn.finished = true
    turn.drained.resolve()
  }

  // The only reader of the SDK iterator, including while a turn is being cancelled.
  async function pumpAll() {
    try {
      for await (const message of session) {
        if (turn) {
          turn.messages.push(message)
          turn.wake?.()
        }
        if (message.type === "assistant" || message.type === "result") persisted = true
        if (message.type === "result") finish()
      }
    } catch (error) {
      failure = error
    } finally {
      ended = true
      finish()
      turn?.wake?.()
    }
  }
  const pump = pumpAll()

  async function interrupt() {
    const active = turn
    if (!active || active.finished || ended) return
    await session.interrupt()
    // interrupt() resolves before the result and transcript flush, so wait for the drain before another turn starts.
    await active.drained.promise
  }

  async function shutdown() {
    try {
      if (initialized && !ended) {
        await session.interrupt()
        if (turn && !turn.finished) await turn.drained.promise
      }
      input.close()
      // An uninitialized process has no conversation to flush.
      if (!initialized) session.close()
      await pump
    } finally {
      session.close()
    }
  }

  return {
    id: options.resume ?? options.sessionId!,
    ready,
    get ended() {
      return ended
    },
    get persisted() {
      return persisted
    },
    get running() {
      return !!turn && !turn.finished
    },
    begin(content: SDKUserMessage["message"]["content"], uuid: NonNullable<SDKUserMessage["uuid"]> = crypto.randomUUID()) {
      if (ended || stopping) throw failure ?? new Error("Claude Code session has stopped")
      if (turn && !turn.finished) throw new Error("Claude Code is still finishing the previous turn")
      turn = { drained: Promise.withResolvers<void>(), messages: [], finished: false }
      input.enqueue({ type: "user", uuid, message: { role: "user", content }, parent_tool_use_id: null })
    },
    addUser(content: SDKUserMessage["message"]["content"], uuid: NonNullable<SDKUserMessage["uuid"]>) {
      input.enqueue({ type: "user", uuid, priority: "next", message: { role: "user", content }, parent_tool_use_id: null })
    },
    async read(
      emit: (message: SDKMessage) => void,
      tools: readonly LanguageModelV3FunctionTool[],
    ): Promise<SDKResultMessage | "tools"> {
      const active = turn
      if (!active) throw new Error("No Claude turn to read")
      let called = false
      while (true) {
        const message = active.messages.shift()
        if (!message) {
          if (ended) throw failure ?? new Error("Claude Code stopped before finishing its reply")
          await new Promise<void>((resolve) => {
            active.wake = resolve
          })
          continue
        }
        emit(message)
        if (message.type === "assistant" && message.parent_tool_use_id === null) {
          // Calls to tools we did not offer get Claude Code's own error, and the turn continues without Kilo.
          if (message.message.content.some((part) => part.type === "tool_use" && original(part.name, tools))) called = true
        }
        const stop =
          message.type === "stream_event" && message.parent_tool_use_id === null && message.event.type === "message_stop"
        if (called && stop) return "tools"
        if (message.type === "result") return message
      }
    },
    interrupt,
    stop() {
      stopping ??= shutdown()
      return stopping
    },
    setModel: (model: string) => session.setModel(model),
    setEffort: (effort: Options["effort"]) => session.applyFlagSettings({ effortLevel: effort ?? null }),
  }
}
