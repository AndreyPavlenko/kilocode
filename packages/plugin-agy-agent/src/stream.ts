import { APICallError } from "@ai-sdk/provider"
import type { LanguageModelV3FinishReason, LanguageModelV3StreamPart, LanguageModelV3Usage } from "@ai-sdk/provider"
import type * as schema from "@agentclientprotocol/sdk"
import type { Call } from "./tools"

const reasons: { [key in schema.StopReason]?: LanguageModelV3FinishReason["unified"] } = {
  end_turn: "stop",
  max_tokens: "length",
  max_turn_requests: "other",
  refusal: "content-filter",
}

// `used` carries the context occupancy over from the previous step of the same agy turn.
export function converter(used?: number) {
  let open: { kind: "text" | "reasoning"; id: string } | undefined
  let count = 0
  let size: number | undefined

  const close = (): LanguageModelV3StreamPart[] => {
    if (!open) return []
    const part: LanguageModelV3StreamPart = { type: `${open.kind}-end`, id: open.id }
    open = undefined
    return [part]
  }

  const chunk = (kind: "text" | "reasoning", block: schema.ContentBlock): LanguageModelV3StreamPart[] => {
    if (block.type !== "text" || !block.text) return []
    const parts = open?.kind === kind ? [] : close()
    if (!open) {
      open = { kind, id: String(count++) }
      parts.push({ type: `${kind}-start`, id: open.id })
    }
    parts.push({ type: `${kind}-delta`, id: open.id, delta: block.text })
    return parts
  }

  // agy's per-turn input counts sum every internal model call, so only the context occupancy is a step's input.
  const usage = (output?: schema.Usage | null): LanguageModelV3Usage => ({
    inputTokens: { total: used, noCache: undefined, cacheRead: undefined, cacheWrite: undefined },
    outputTokens: {
      total: output?.outputTokens ?? undefined,
      text: output?.outputTokens == null ? undefined : output.outputTokens - (output.thoughtTokens ?? 0),
      reasoning: output?.thoughtTokens ?? undefined,
    },
  })

  return {
    get size() {
      return size
    },
    get used() {
      return used
    },
    update(update: schema.SessionUpdate): LanguageModelV3StreamPart[] {
      if (update.sessionUpdate === "agent_thought_chunk") return chunk("reasoning", update.content)
      if (update.sessionUpdate === "agent_message_chunk") return chunk("text", update.content)
      if (update.sessionUpdate === "usage_update") {
        used = update.used
        size = update.size
      }
      return []
    },
    call(call: Call): LanguageModelV3StreamPart[] {
      return [
        ...close(),
        { type: "tool-call", toolCallId: call.id, toolName: call.name, input: JSON.stringify(call.input) },
        { type: "finish", finishReason: { unified: "tool-calls", raw: undefined }, usage: usage() },
      ]
    },
    // Output tokens are turn totals. Earlier steps of the turn report none, so their sum is not counted twice.
    done(response: schema.PromptResponse): LanguageModelV3StreamPart[] {
      // A cancelled turn only ends when Kilo aborted or a deadline handed off; neither is a finished step.
      if (response.stopReason === "cancelled") return close()
      return [
        ...close(),
        {
          type: "finish",
          finishReason: { unified: reasons[response.stopReason] ?? "other", raw: response.stopReason },
          usage: usage(response.usage),
        },
      ]
    },
  }
}

export const quota = /quota|resource_exhausted|429/i
const transient = /network|timeout|timed out|econn|unavailable|socket|wait for it to finish/i

export const code = (error: unknown) =>
  error !== null && typeof error === "object" && "code" in error ? error.code : undefined

// Kilo only stops retrying on an API error marked non-retryable, whatever its text says.
export function failed(error: unknown, retryable = false) {
  if (APICallError.isInstance(error)) return error
  const message = error instanceof Error ? error.message : String(error)
  const make = (text: string, isRetryable: boolean) =>
    new APICallError({ message: text, url: "", requestBodyValues: {}, isRetryable, cause: error })
  if (quota.test(message)) return make(message, false)
  if (code(error) === -32000) return make("Antigravity is not logged in. Run `agy` in a terminal and sign in.", false)
  if (code(error) === -32603) return make(message, transient.test(message))
  return make(message, retryable)
}
