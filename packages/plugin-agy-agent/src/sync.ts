import { createHash, randomUUID } from "node:crypto"
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises"
import path from "node:path"
import type { LanguageModelV3Message, LanguageModelV3Prompt, LanguageModelV3ToolResultPart } from "@ai-sdk/provider"
import { DATA } from "./constants"

export interface Record {
  conversation: string
  // One per Kilo prompt message agy already has.
  hashes: string[]
  system: string
  tools: string
}

export type Decision =
  | { type: "results"; results: LanguageModelV3ToolResultPart[] }
  | { type: "continue"; tail: LanguageModelV3Prompt }
  | { type: "rebuild" }

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`
  if (value !== null && typeof value === "object") {
    const object = value as { [key: string]: unknown }
    const fields = Object.keys(object)
      .sort()
      .filter((key) => object[key] !== undefined)
      .map((key) => `${JSON.stringify(key)}:${canonical(object[key])}`)
    return `{${fields.join(",")}}`
  }
  return JSON.stringify(value) ?? "null"
}

export function digest(value: unknown) {
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
    // Kilo may re-encode stored media between turns (URL or base64), so only the type identifies it.
    if (part.type === "file") parts.push({ type: part.type, mediaType: part.mediaType })
    if (part.type === "tool-call") {
      parts.push({ type: part.type, id: part.toolCallId, name: part.toolName, input: part.input })
    }
    // Kilo prunes old outputs ("[Old tool result content cleared]"), truncates them and marks interrupted calls,
    // so a result is identified by its call alone.
    if (part.type === "tool-result") parts.push({ type: part.type, id: part.toolCallId, name: part.toolName })
  }
  if (text) parts.push({ type: "text", text })
  return { role: message.role, parts }
}

const messages = (prompt: LanguageModelV3Prompt) =>
  prompt.filter((message): message is Exclude<LanguageModelV3Message, { role: "system" }> => message.role !== "system")

export function hashes(prompt: LanguageModelV3Prompt) {
  return messages(prompt).map((message) => digest(visible(message)))
}

export function decide(prompt: LanguageModelV3Prompt, record: Record | undefined, parked: readonly string[]): Decision {
  if (!record) return { type: "rebuild" }
  const current = hashes(prompt)
  if (record.hashes.length > current.length) return { type: "rebuild" }
  if (record.hashes.some((hash, i) => hash !== current[i])) return { type: "rebuild" }
  const tail = messages(prompt).slice(record.hashes.length)
  // Nothing new means a regenerate of the last reply; agy already answered it, so start over.
  if (tail.length === 0) return { type: "rebuild" }
  const ids = new Set(parked)
  const results = tail.flatMap((message) =>
    message.role === "tool" ? message.content.filter((part) => part.type === "tool-result") : [],
  )
  const only = tail.every((message) => message.role === "tool")
  if (only && results.length > 0 && results.every((part) => ids.has(part.toolCallId))) {
    return { type: "results", results }
  }
  return { type: "continue", tail }
}

const file = (id: string) => path.join(DATA, "sessions", `${id}.json`)

export async function load(sessionID: string) {
  const text = await readFile(file(sessionID), "utf8").catch((err: NodeJS.ErrnoException) => {
    if (err.code === "ENOENT") return undefined
    throw err
  })
  if (text === undefined) return undefined
  try {
    return JSON.parse(text) as Record
  } catch (err) {
    // A bad record only costs a replay; it must not fail every later turn.
    console.warn("[agy-agent] ignoring corrupt session record:", sessionID, err)
    return undefined
  }
}

export async function save(sessionID: string, record: Record) {
  await mkdir(path.join(DATA, "sessions"), { recursive: true })
  const target = file(sessionID)
  const temp = `${target}.${process.pid}.${randomUUID()}.tmp`
  await writeFile(temp, JSON.stringify(record))
  await rename(temp, target).catch(async (err) => {
    await rm(temp, { force: true })
    throw err
  })
}

export async function drop(sessionID: string) {
  await rm(file(sessionID), { force: true })
}
