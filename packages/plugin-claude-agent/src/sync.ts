import { createHash } from "node:crypto"
import type {
  LanguageModelV3Message,
  LanguageModelV3Prompt,
  LanguageModelV3TextPart,
  LanguageModelV3ToolCallPart,
} from "@ai-sdk/provider"
import { content, type Block } from "./message"

export interface Group {
  hash: string
  role: "user" | "assistant" | "tool"
  start: number
  end: number
}

export interface Checkpoint {
  hash: string
  claudeMessageID?: string
}

export interface SessionRecord {
  claudeSessionID: string
  history?: Checkpoint[]
  // After cancellation, Claude owns the partial reply; history ends at the input it received.
  interrupted?: boolean
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`
  if (value !== null && typeof value === "object") {
    const object = value as Record<string, unknown>
    const fields = Object.keys(object)
      .sort()
      .filter((key) => object[key] !== undefined)
      .map((key) => `${JSON.stringify(key)}:${canonical(object[key])}`)
    return `{${fields.join(",")}}`
  }
  return JSON.stringify(value) ?? "null"
}

function hash(value: unknown) {
  return createHash("sha256").update(canonical(value)).digest("hex")
}

// Kilo sends thinking and provider options differently depending on the target model, so they don't count.
function visible(message: Exclude<LanguageModelV3Message, { role: "system" }>) {
  const parts: unknown[] = []
  let text = ""
  for (const part of message.content) {
    if (part.type === "reasoning") continue
    if (part.type === "text") {
      text += part.text
      continue
    }
    if (text) parts.push({ type: "text", text })
    text = ""
    if (part.type === "file") {
      const data =
        part.data instanceof URL
          ? part.data.href
          : typeof part.data === "string"
            ? part.data
            : Buffer.from(part.data).toString("base64")
      parts.push({ type: part.type, mediaType: part.mediaType, data })
    }
    if (part.type === "tool-call") {
      parts.push({ type: part.type, id: part.toolCallId, name: part.toolName, input: part.input })
    }
    if (part.type === "tool-result") {
      const output: Record<string, any> = { ...part.output }
      delete output.providerOptions
      if (output.type === "content") {
        output.value = output.value.map(({ providerOptions: _, ...item }: Record<string, unknown>) => item)
      }
      parts.push({ type: part.type, id: part.toolCallId, name: part.toolName, output })
    }
  }
  if (text) parts.push({ type: "text", text })
  return parts
}

// Kilo lowers tool attachments into this separate user message.
function media(message: LanguageModelV3Message) {
  return (
    message.role === "user" &&
    message.content.length > 1 &&
    message.content[0]?.type === "text" &&
    message.content[0].text === "Attached media from tool result:" &&
    message.content.slice(1).every((part) => part.type === "file")
  )
}

export function groups(prompt: LanguageModelV3Prompt, prefix: readonly { hash: string }[] = []): Group[] {
  const result: Group[] = []
  let i = 0
  while (i < prompt.length) {
    const message = prompt[i]!
    const start = i
    i++
    if (message.role === "system") continue
    const parts = visible(message)
    if (parts.length === 0) continue
    const members = [hash(parts)]
    if (message.role === "user") {
      while (i < prompt.length) {
        // A submitted input remains a boundary even when a later user message arrives without a visible reply.
        const checkpoint = prefix[result.length]
        if (checkpoint && hash({ role: message.role, members: members.toSorted() }) === checkpoint.hash) break
        const next = prompt[i]!
        if (next.role !== "user") break
        members.push(hash(visible(next)))
        i++
      }
      members.sort()
    }
    if (message.role === "tool") {
      const next = prompt[i]
      if (next && next.role === "user" && media(next)) {
        members.push(hash(visible(next)))
        i++
      }
    }
    result.push({ hash: hash({ role: message.role, members }), role: message.role, start, end: i })
  }
  return result
}

export function decide(prompt: LanguageModelV3Prompt, record?: SessionRecord) {
  const saved = record?.history ?? []
  const current = groups(prompt, record?.interrupted ? saved : undefined)
  if (saved.length === 0) return { kind: "fresh", start: 0 } as const

  let matched = 0
  while (matched < saved.length && matched < current.length && saved[matched]!.hash === current[matched]!.hash) {
    matched++
  }
  if (matched === 0) return { kind: "fresh", start: 0 } as const
  // Changed histories need an isolated branch, or a rebuild when no safe checkpoint survives.
  if (matched < saved.length) return { kind: "changed", start: 0 } as const

  // Claude's drained transcript owns the interrupted reply, not Kilo's partial copy.
  while (record?.interrupted && current[matched]?.role === "assistant") {
    const reply = prompt[current[matched]!.start]!
    if (reply.role !== "assistant" || reply.content.some((part) => part.type === "tool-call")) break
    matched++
  }
  const start = current[matched]?.start ?? prompt.length
  if (record?.interrupted && start === prompt.length) return { kind: "changed", start: 0 } as const
  const missed = current.slice(matched)
  const kind = missed.some((group) => group.role !== "user") ? "catch-up" : "continue"
  return { kind, start } as const
}

// A branch must not end between a native tool call and its results.
export function boundary(prompt: LanguageModelV3Prompt, record?: SessionRecord) {
  const current = groups(prompt)
  const history = record?.history ?? []
  const open = new Set<string>()
  let found: { messageID: string; start: number; history: Checkpoint[] } | undefined
  for (let i = 0; i < current.length; i++) {
    const group = current[i]!
    // Undo can resubmit identical input; keep the current user group for the new turn.
    if (i === current.length - 1 && group.role === "user") break
    const checkpoint = history[i]
    if (!checkpoint || checkpoint.hash !== group.hash) break
    for (const message of prompt.slice(group.start, group.end)) {
      if (message.role === "system") continue
      for (const part of message.content) {
        if (part.type === "tool-call") open.add(part.toolCallId)
        if (part.type === "tool-result") open.delete(part.toolCallId)
      }
    }
    if (checkpoint.claudeMessageID && open.size === 0) {
      found = { messageID: checkpoint.claudeMessageID, start: group.end, history: history.slice(0, i + 1) }
    }
  }
  return found
}

export function checkpoints(input: {
  prompt: LanguageModelV3Prompt
  previous?: SessionRecord
  userID?: string
  reply: Array<LanguageModelV3TextPart | LanguageModelV3ToolCallPart>
  messageID?: string
  toolMessages: Map<string, string>
}) {
  const current = groups(input.prompt)
  const history = current.map((group, i): Checkpoint => {
    const previous = input.previous?.history?.[i]
    const tool =
      group.role === "tool"
        ? input.prompt
            .slice(group.start, group.end)
            .flatMap((message) => (message.role === "tool" ? message.content : []))
            .flatMap((part) => (part.type === "tool-result" ? (input.toolMessages.get(part.toolCallId) ?? []) : []))
            .at(-1)
        : undefined
    const kept = previous?.hash === group.hash ? previous.claudeMessageID : undefined
    const last = i === current.length - 1 ? input.userID : undefined
    return { hash: group.hash, claudeMessageID: last ?? tool ?? kept }
  })
  if (input.reply.length === 0) return history
  const reply = groups([{ role: "assistant", content: input.reply }])[0]!
  return [...history, { hash: reply.hash, claudeMessageID: input.messageID }]
}

export function catchUp(prompt: LanguageModelV3Prompt, start = 0): Block[] {
  const missed = prompt.slice(start)
  const end = groups(missed).findLast((group) => group.role !== "user")?.end ?? 0
  if (end === 0) return content(missed)
  return [
    {
      type: "text",
      text: "<conversation-checkpoint>\nThe following is conversation history from Kilo that you have not seen. Treat it as prior conversation, not a new request. Continue with the user messages after this checkpoint.",
    },
    ...content(missed.slice(0, end)),
    { type: "text", text: "</conversation-checkpoint>" },
    ...content(missed.slice(end)),
  ]
}
