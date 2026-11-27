import { execFile } from "node:child_process"
import type { LanguageModelV3Message, LanguageModelV3Prompt, LanguageModelV3StreamPart } from "@ai-sdk/provider"
import { executable } from "./options"

export const MARKER = "[[agy-agent:usage]]"
const TTL_MS = 60_000
const TIMEOUT_MS = 15_000

export interface Bucket {
  id: string
  name: string
  window?: string
  remaining_fraction: number
  reset_time?: string
}

export interface Usage {
  groups: Array<{ name: string; description?: string; buckets: Bucket[] }>
}

let cached: { at: number; value: Usage } | undefined
let flight: Promise<Usage> | undefined

// Print mode with /usage makes no model call and creates no conversation.
const probe = async () => {
  const exe = await executable()
  const done = Promise.withResolvers<string>()
  execFile(exe, ["-p=/usage", "--output-format", "json"], { timeout: TIMEOUT_MS }, (err, stdout) =>
    err ? done.reject(err) : done.resolve(stdout),
  )
  const body = JSON.parse(await done.promise)
  const groups = body?.command?.data?.groups
  if (body?.status !== "SUCCESS" || !Array.isArray(groups)) throw new Error(body?.error ?? "unexpected /usage answer")
  return { groups } as Usage
}

// Limits belong to the account, so every session shares one result.
export function limits(fresh = false) {
  if (!fresh && cached && Date.now() - cached.at < TTL_MS) return Promise.resolve(cached.value)
  flight ??= probe()
    .then((value) => {
      cached = { at: Date.now(), value }
      return value
    })
    .finally(() => {
      flight = undefined
    })
  return flight
}

// The epoch means the bucket was never used, so it has no reset.
const reset = (bucket: Bucket) =>
  bucket.reset_time && new Date(bucket.reset_time).getUTCFullYear() > 1970
    ? `, resets ${new Date(bucket.reset_time).toLocaleString()}`
    : ""

export function format(usage: Usage) {
  return [
    "**Antigravity limits**",
    ...usage.groups.flatMap((group) => [
      "",
      `**${group.name}**`,
      ...group.buckets.map(
        (bucket) => `- ${bucket.name}: ${Math.round(bucket.remaining_fraction * 100)}% remaining${reset(bucket)}`,
      ),
    ]),
  ].join("\n")
}

export function exhausted(usage: Usage) {
  const rows = usage.groups.flatMap((group) =>
    group.buckets.filter((bucket) => bucket.remaining_fraction <= 0).map((bucket) => `${group.name}: ${bucket.name}${reset(bucket)}`),
  )
  if (rows.length === 0) return undefined
  return `Antigravity quota exhausted. ${rows.join("; ")}`
}

// Kilo may append text to the command's message, but the template stays first. Title prompts quote it mid-text.
export const marked = (message: LanguageModelV3Message) =>
  message.role === "user" && message.content.some((part) => part.type === "text" && part.text.trimStart().startsWith(MARKER))

// Usage turns are not conversation: agy never saw them, and they must not break the history comparison.
export function strip(prompt: LanguageModelV3Prompt): LanguageModelV3Prompt {
  let skipping = false
  return prompt
    .filter((message) => {
      if (message.role === "system") return true
      if (message.role === "user") skipping = marked(message)
      return !skipping
    })
    .map((message) =>
      message.role !== "user"
        ? message
        : {
            ...message,
            content: message.content.map((part) =>
              part.type === "text" ? { ...part, text: part.text.replaceAll(MARKER, "") } : part,
            ),
          },
    )
}

export function answer() {
  return new ReadableStream<LanguageModelV3StreamPart>({
    async start(controller) {
      controller.enqueue({ type: "stream-start", warnings: [] })
      const text = await limits()
        .then(format)
        .catch((error) => `Could not read Antigravity limits: ${error instanceof Error ? error.message : error}`)
      const id = "usage"
      controller.enqueue({ type: "text-start", id })
      controller.enqueue({ type: "text-delta", id, delta: text })
      controller.enqueue({ type: "text-end", id })
      controller.enqueue({
        type: "finish",
        finishReason: { unified: "stop", raw: undefined },
        usage: {
          inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 0, text: 0, reasoning: 0 },
        },
      })
      controller.close()
    },
  })
}
