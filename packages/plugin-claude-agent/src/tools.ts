import type {
  LanguageModelV3FilePart,
  LanguageModelV3FunctionTool,
  LanguageModelV3Prompt,
  LanguageModelV3ToolResultOutput,
  LanguageModelV3ToolResultPart,
} from "@ai-sdk/provider"
import { createHash } from "node:crypto"
import type { HookCallback, McpSdkServerConfigWithInstance, Options } from "@anthropic-ai/claude-agent-sdk"
import type { CallToolResult, Tool } from "@modelcontextprotocol/sdk/types.js"
import { content, type Block } from "./message"
import { groups } from "./sync"

export const SERVER = "kilocode"
export const PREFIX = `mcp__${SERVER}__`

// Claude Code cuts each MCP tool description at this many characters.
const LIMIT = 2048
const CONTINUED = "\n[Continued in the system prompt]"
const HEAD = LIMIT - CONTINUED.length
// Kilo records a stopped tool with one of these.
const ABORTED = new Set(["Tool execution aborted", "[Tool execution was interrupted]"])

export function short(name: string) {
  const limit = 64 - PREFIX.length
  if (name.length <= limit && /^[a-zA-Z0-9_-]+$/.test(name)) return name
  const hash = createHash("sha256").update(name).digest("hex").slice(0, 10)
  return name.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, limit - 11) + "_" + hash
}

// Undefined for a tool we did not offer. Claude Code answers those calls itself, without our hooks or server.
export function original(name: string, tools: readonly LanguageModelV3FunctionTool[]) {
  return tools.find((tool) => PREFIX + short(tool.name) === name)?.name
}

function brief(description = "") {
  return description.length > LIMIT ? description.slice(0, HEAD) + CONTINUED : description
}

// The cut text can hold required data, such as the Task tool's agent list, so the system prompt carries it instead.
export function overflow(tools: readonly LanguageModelV3FunctionTool[]) {
  const parts = tools.flatMap(({ name, description = "" }) =>
    description.length > LIMIT
      ? [`<${PREFIX + short(name)}>\n${description.slice(HEAD)}\n</${PREFIX + short(name)}>`]
      : [],
  )
  return parts.length > 0 ? `\n\nContinuation of tool descriptions cut off above:\n${parts.join("\n")}` : ""
}

export interface Result {
  mcp: CallToolResult
  replacement: Block[] | undefined
}

export type Pending = ReturnType<typeof pending>

export function pending() {
  const calls = new Map<string, { name: string; result: PromiseWithResolvers<Result> }>()
  const replacements = new Map<string, Block[]>()
  let next: string | undefined

  function expect(id: string, name: string) {
    const call = calls.get(id) ?? { name, result: Promise.withResolvers<Result>() }
    calls.set(id, call)
    if (call.name !== name) throw new Error(`Mismatched Claude tool call ${id}`)
    return call
  }

  return {
    expect,
    before(id: string, name: string) {
      expect(id, name)
      next = id
    },
    async call(name: string) {
      const id = next
      next = undefined
      if (!id) throw new Error("Claude invoked a Kilo tool without its PreToolUse ID")
      const result = await expect(id, name).result.promise
      if (result.replacement) replacements.set(id, result.replacement)
      calls.delete(id)
      return result.mcp
    },
    resolve(id: string, result: Result) {
      const call = calls.get(id)
      if (!call) throw new Error(`No waiting Claude tool call ${id}`)
      call.result.resolve(result)
    },
    after(id: string) {
      const replacement = replacements.get(id)
      replacements.delete(id)
      return replacement
    },
    // Call only after interrupt-and-drain; resolving these handlers earlier can start another model call.
    clear() {
      calls.forEach((call) =>
        call.result.resolve({
          mcp: { isError: true, content: [{ type: "text", text: "Cancelled by user." }] },
          replacement: undefined,
        }),
      )
      calls.clear()
      replacements.clear()
      next = undefined
    },
  }
}

export function hooks(calls: Pending): Options["hooks"] {
  const before: HookCallback = async (input) => {
    if (input.hook_event_name === "PreToolUse") calls.before(input.tool_use_id, input.tool_name)
    return {}
  }
  const after: HookCallback = async (input) => {
    if (input.hook_event_name !== "PostToolUse") return {}
    const replacement = calls.after(input.tool_use_id)
    if (!replacement) return {}
    return { hookSpecificOutput: { hookEventName: "PostToolUse", updatedMCPToolOutput: replacement } }
  }
  return { PreToolUse: [{ hooks: [before] }], PostToolUse: [{ hooks: [after] }] }
}

export async function server(
  tools: readonly LanguageModelV3FunctionTool[],
  calls: Pending,
): Promise<McpSdkServerConfigWithInstance> {
  const [{ Server }, { ListToolsRequestSchema, CallToolRequestSchema }] = await Promise.all([
    import("@modelcontextprotocol/sdk/server/index.js"),
    import("@modelcontextprotocol/sdk/types.js"),
  ])
  const instance = new Server({ name: SERVER, version: "1.0.0" }, { capabilities: { tools: {} } })
  const definitions = tools.map(
    (tool): Tool => ({
      name: short(tool.name),
      description: brief(tool.description),
      inputSchema: tool.inputSchema as Tool["inputSchema"],
      // Claude serializes non-read-only tools, so each handler takes the ID from the preceding hook.
      annotations: { readOnlyHint: false },
      _meta: { "anthropic/alwaysLoad": true, "anthropic/maxResultSizeChars": 500_000 },
    }),
  )
  instance.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: definitions }))
  instance.setRequestHandler(CallToolRequestSchema, async (request) => {
    const name = PREFIX + request.params.name
    if (!original(name, tools)) throw new Error(`Claude Code called an unregistered Kilo tool: ${name}`)
    return calls.call(name)
  })
  return {
    type: "sdk",
    name: SERVER,
    instance: instance as unknown as McpSdkServerConfigWithInstance["instance"],
  }
}

export function convert(output: LanguageModelV3ToolResultOutput, files: readonly LanguageModelV3FilePart[] = []) {
  const parts = [...text(output), ...files]
  const blocks = content([{ role: "user", content: parts }])
  const failed = output.type === "error-text" || output.type === "error-json" || output.type === "execution-denied"
  const mcp: CallToolResult = { content: [], isError: failed }
  let replace = false
  blocks.forEach((block) => {
    if (block.type === "text") {
      mcp.content.push({ type: "text", text: block.text })
      return
    }
    if (block.type === "image" && block.source.type === "base64") {
      mcp.content.push({ type: "image", mimeType: block.source.media_type, data: block.source.data })
      return
    }
    replace = true
    mcp.content.push({ type: "text", text: "Attached document or image follows." })
  })
  // The PostToolUse hook replaces tool_result.content, so it needs content blocks rather than an MCP envelope.
  return { mcp, replacement: replace ? blocks : undefined } satisfies Result
}

function text(
  output: LanguageModelV3ToolResultOutput,
): Array<{ type: "text"; text: string } | LanguageModelV3FilePart> {
  if (output.type === "text" || output.type === "error-text") return [{ type: "text", text: output.value }]
  if (output.type === "json" || output.type === "error-json") {
    return [{ type: "text", text: JSON.stringify(output.value) }]
  }
  if (output.type === "execution-denied") {
    return [{ type: "text", text: output.reason ?? "Tool execution was denied by the user." }]
  }
  return output.value.map((part) => {
    if (part.type === "text") return { type: "text", text: part.text }
    if (part.type === "image-data" || part.type === "file-data") {
      return { type: "file", mediaType: part.mediaType, data: part.data }
    }
    // Only picks an image block. URL images are sent without a media type.
    if (part.type === "image-url") return { type: "file", mediaType: "image/png", data: new URL(part.url) }
    throw new Error(`Claude Code does not support ${part.type} tool output`)
  })
}

export function collect(prompt: LanguageModelV3Prompt, start: number) {
  const results = new Map<string, Result>()
  let end = start
  let aborted = false
  for (const group of groups(prompt.slice(start))) {
    if (group.role !== "tool") break
    const messages = prompt.slice(start + group.start, start + group.end)
    const tools = messages.flatMap((message) =>
      message.role === "tool"
        ? message.content.filter((part): part is LanguageModelV3ToolResultPart => part.type === "tool-result")
        : [],
    )
    // Kilo puts the group's media in a following user message; attach it to the group's last result.
    const files = messages.flatMap((message) =>
      message.role === "user"
        ? message.content.filter((part): part is LanguageModelV3FilePart => part.type === "file")
        : [],
    )
    tools.forEach((tool, index) => {
      if (tool.output.type === "error-text" && ABORTED.has(tool.output.value)) aborted = true
      results.set(tool.toolCallId, convert(tool.output, index === tools.length - 1 ? files : []))
    })
    end = start + group.end
  }
  return { results, end, aborted }
}
