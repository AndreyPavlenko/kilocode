import type { LanguageModelV3FilePart, LanguageModelV3Prompt } from "@ai-sdk/provider"
import type { ContentBlock } from "@agentclientprotocol/sdk"

const images = new Set(["image/png", "image/jpeg", "image/gif", "image/webp"])

// A new agy session can't be given earlier turns, so their roles become text tags. Images stay real content blocks.
export function content(prompt: LanguageModelV3Prompt): ContentBlock[] {
  const blocks: ContentBlock[] = []
  const history = prompt.some((message) => message.role === "assistant" || message.role === "tool")
  for (const message of prompt) {
    if (message.role === "system") continue
    if (history) blocks.push({ type: "text", text: `<${message.role}>` })
    for (const part of message.content) {
      if (part.type === "text") blocks.push({ type: "text", text: part.text })
      if (part.type === "file") blocks.push(file(part))
      if (part.type === "tool-call") {
        const call = JSON.stringify({ id: part.toolCallId, name: part.toolName, input: part.input })
        blocks.push({ type: "text", text: `<tool-call>\n${call}\n</tool-call>` })
      }
      if (part.type === "tool-result") {
        const result = JSON.stringify({ id: part.toolCallId, name: part.toolName, output: part.output })
        blocks.push({ type: "text", text: `<tool-result>\n${result}\n</tool-result>` })
      }
    }
    if (history) blocks.push({ type: "text", text: `</${message.role}>` })
  }
  return blocks
}

function file(part: LanguageModelV3FilePart): ContentBlock {
  const type = part.mediaType
  if (!images.has(type)) throw new Error(`Antigravity does not support ${type} attachments`)
  if (!(part.data instanceof URL)) return { type: "image", mimeType: type, data: base64(part.data) }
  // ACP image blocks need inline data; only data: URLs carry it.
  const match = /^data:[^;,]*;base64,(.*)$/.exec(part.data.href)
  if (!match) throw new Error("Antigravity does not support image URLs")
  return { type: "image", mimeType: type, data: match[1] }
}

function base64(data: string | Uint8Array) {
  if (typeof data === "string") return data
  return Buffer.from(data).toString("base64")
}
