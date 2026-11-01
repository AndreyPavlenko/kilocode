import { APICallError } from "@ai-sdk/provider"
import type {
  LanguageModelV3FunctionTool,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
} from "@ai-sdk/provider"
import type {
  SDKAssistantMessageError,
  SDKMessage,
  SDKRateLimitInfo,
  SDKResultMessage,
} from "@anthropic-ai/claude-agent-sdk"
import { original } from "./tools"

interface Usage {
  input_tokens?: number | null
  cache_read_input_tokens?: number | null
  cache_creation_input_tokens?: number | null
  output_tokens?: number
  output_tokens_details?: { thinking_tokens?: number } | null
}

export function usage() {
  const messages = new Map<string, { input: number; read: number; write: number; output: number; reasoning: number }>()
  let last: string | undefined

  function update(id: string, value: Usage) {
    const counts = messages.get(id) ?? { input: 0, read: 0, write: 0, output: 0, reasoning: 0 }
    if (!messages.has(id)) {
      messages.set(id, counts)
      last = id
    }
    counts.input = Math.max(counts.input, value.input_tokens ?? 0)
    counts.read = Math.max(counts.read, value.cache_read_input_tokens ?? 0)
    counts.write = Math.max(counts.write, value.cache_creation_input_tokens ?? 0)
    counts.output = Math.max(counts.output, value.output_tokens ?? 0)
    counts.reasoning = Math.max(counts.reasoning, value.output_tokens_details?.thinking_tokens ?? 0)
  }

  return {
    update,
    result(value: Usage) {
      // Result usage sums the whole turn; only apply it when there was a single model call.
      if (messages.size <= 1) update(last ?? "result", value)
    },
    snapshot(): LanguageModelV3Usage {
      const all = [...messages.values()]
      const output = all.reduce((sum, item) => sum + item.output, 0)
      const reasoning = all.reduce((sum, item) => sum + item.reasoning, 0)
      // Context occupancy is the last API call's input, not the sum of repeated contexts.
      const call = last === undefined ? undefined : messages.get(last)
      return {
        inputTokens: {
          total: call ? call.input + call.read + call.write : undefined,
          noCache: call?.input,
          cacheRead: call?.read,
          cacheWrite: call?.write,
        },
        outputTokens: { total: output, reasoning, text: output - reasoning },
      }
    },
  }
}

// Claude Code already retries API failures, and replaying a whole turn here would also retry permanent errors.
// Kilo only stops retrying on an API error marked non-retryable, whatever its text says.
export function failed(error: unknown) {
  if (APICallError.isInstance(error)) return error
  const message = error instanceof Error ? error.message : String(error)
  return new APICallError({ message, url: "", requestBodyValues: {}, isRetryable: false, cause: error })
}

export function succeeded(message: SDKResultMessage) {
  if (message.subtype !== "success" || message.is_error) return false
  return (message.api_error_status ?? 0) < 400
}

// Claude's total runs across the whole query and Kilo sums what it gets. A lower total means the counter restarted.
export const spent = (total: number, last: number) => (total < last ? total : total - last)

// Keep request-local diagnostics until the terminal result. A usage warning alone is not a failed request.
export function failure() {
  let code: SDKAssistantMessageError | undefined
  let rejected: SDKRateLimitInfo | undefined
  return (message: SDKMessage) => {
    if (message.type === "rate_limit_event") {
      rejected = message.rate_limit_info.status === "rejected" ? message.rate_limit_info : undefined
    }
    if (message.type === "assistant" && message.parent_tool_use_id === null && message.error) code = message.error
    if (message.type !== "result") return

    if (succeeded(message)) return
    const status = message.subtype === "success" ? (message.api_error_status ?? undefined) : undefined
    const result = message.subtype === "success" ? message.result : message.errors.join("\n")
    const server =
      status !== undefined && status >= 500
        ? `${status === 529 ? "Anthropic is overloaded (529)." : `Anthropic returned HTTP ${status}.`} Claude Code ran out of retries.`
        : result
    const limited = code === "rate_limit" || (rejected && status === 429)
    const window = rejected?.rateLimitType?.replaceAll("_", " ")
    const limit =
      limited && !server ? `Claude Code subscription limit reached${window ? ` (${window})` : ""}.` : server
    const reset = limited && rejected?.resetsAt ? ` Resets at ${new Date(rejected.resetsAt * 1000).toISOString()}.` : ""
    const billing = code === "billing_error" ? " Check your Claude plan and extra-usage settings." : ""
    const auth = code === "authentication_failed" ? " Run `claude auth login` in a terminal." : ""
    const text = limit + reset + billing + auth
    if (text) return failed(new Error(text))
    return failed(new Error(`Claude Code could not finish its reply${status ? ` (HTTP ${status})` : ""}`))
  }
}

export function converter(tools: readonly LanguageModelV3FunctionTool[]) {
  const counts = usage()
  const check = failure()
  const streamed = new Set<string>()
  const blocks = new Map<number, "text" | "reasoning">()
  let messageID = ""
  let called = false

  // A throwaway process has no earlier turns, so without a delta its whole total is this call's cost.
  return (message: SDKMessage, cost?: number): LanguageModelV3StreamPart[] => {
    const parts: LanguageModelV3StreamPart[] = []
    const error = check(message)
    if (error) return [{ type: "error", error }]
    if (message.type === "stream_event" && message.parent_tool_use_id === null) {
      const event = message.event
      const id = "index" in event ? `${messageID}:${event.index}` : ""
      if (event.type === "message_start") {
        messageID = event.message.id
        streamed.add(messageID)
        counts.update(messageID, event.message.usage)
        parts.push({ type: "response-metadata", id: messageID, modelId: event.message.model })
      }
      if (event.type === "content_block_start" && event.content_block.type === "text") {
        blocks.set(event.index, "text")
        parts.push({ type: "text-start", id })
        if (event.content_block.text) parts.push({ type: "text-delta", id, delta: event.content_block.text })
      }
      if (event.type === "content_block_start" && event.content_block.type === "thinking") {
        blocks.set(event.index, "reasoning")
        parts.push({ type: "reasoning-start", id })
        if (event.content_block.thinking) {
          parts.push({ type: "reasoning-delta", id, delta: event.content_block.thinking })
        }
      }
      if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
        parts.push({ type: "text-delta", id, delta: event.delta.text })
      }
      if (event.type === "content_block_delta" && event.delta.type === "thinking_delta" && event.delta.thinking) {
        parts.push({ type: "reasoning-delta", id, delta: event.delta.thinking })
      }
      if (event.type === "content_block_stop") {
        const kind = blocks.get(event.index)
        if (kind) parts.push({ type: `${kind}-end`, id })
        blocks.delete(event.index)
      }
      if (event.type === "message_delta") counts.update(messageID, event.usage)
      // Must match the tool boundary in Session.read, or Kilo's step ends without a finish event.
      if (event.type === "message_stop" && called) {
        parts.push({
          type: "finish",
          finishReason: { unified: "tool-calls", raw: "tool_use" },
          usage: counts.snapshot(),
        })
      }
    }
    if (message.type === "assistant" && message.parent_tool_use_id === null) {
      // Synthetic API errors are diagnostics, not assistant answers.
      if (message.error) return parts
      counts.update(message.message.id, message.message.usage)
      message.message.content.forEach((block) => {
        if (block.type !== "tool_use") return
        const toolName = original(block.name, tools)
        if (!toolName) return
        called = true
        parts.push({ type: "tool-call", toolCallId: block.id, toolName, input: JSON.stringify(block.input) })
      })
      // Complete assistant messages repeat the streamed content. Only use them when nothing was streamed.
      if (!streamed.has(message.message.id)) {
        message.message.content.forEach((block, index) => {
          const id = `${message.uuid}:${index}`
          if (block.type === "text") {
            parts.push(
              { type: "text-start", id },
              { type: "text-delta", id, delta: block.text },
              { type: "text-end", id },
            )
          }
          if (block.type === "thinking") {
            parts.push(
              { type: "reasoning-start", id },
              { type: "reasoning-delta", id, delta: block.thinking },
              { type: "reasoning-end", id },
            )
          }
        })
      }
    }
    if (message.type === "result") {
      if (message.num_turns === 1) counts.result(message.usage)
      parts.push({
        type: "finish",
        finishReason: {
          unified: message.stop_reason === "max_tokens" ? "length" : "stop",
          raw: message.stop_reason ?? undefined,
        },
        usage: {
          ...counts.snapshot(),
          // Kilo reads this key for a provider-reported cost, whatever the provider ID.
          raw: { cost_details: { upstream_inference_cost: cost ?? message.total_cost_usd } },
        },
      })
    }
    return parts
  }
}
