import os from "node:os"
import path from "node:path"
import type { LanguageModelV3FunctionTool } from "@ai-sdk/provider"
import type { Options } from "@anthropic-ai/claude-agent-sdk"
import which from "which"
import { hooks, overflow, server, type Pending, SERVER } from "./tools"

// Claude Code features that duplicate Kilo's. Kilo's version wins.
const ISOLATION = {
  CLAUDE_CODE_DISABLE_CLAUDE_MDS: "1",
  CLAUDE_CODE_DISABLE_AUTO_MEMORY: "1",
  DISABLE_AUTO_COMPACT: "1",
  CLAUDE_CODE_ENABLE_AWAY_SUMMARY: "0",
  ENABLE_TOOL_SEARCH: "false",
  CLAUDE_CODE_DISABLE_ADVISOR_TOOL: "1",
  // Raised so Kilo's own truncation applies instead
  MAX_MCP_OUTPUT_TOKENS: "200000",
  // Skips OAuth, so the subscription would not be used
  CLAUDE_CODE_SIMPLE: undefined,
  // Would override the effort chosen in Kilo
  CLAUDE_CODE_EFFORT_LEVEL: undefined,
  // Would fail subagents and permission prompts that run longer
  MCP_TOOL_TIMEOUT: undefined,
}

export function connection(input: Record<string, any>) {
  const env: Record<string, string> = {}
  if (typeof input.baseURL === "string") env.ANTHROPIC_BASE_URL = input.baseURL
  if (typeof input.apiKey === "string") env.ANTHROPIC_API_KEY = input.apiKey
  if (typeof input.authToken === "string") env.ANTHROPIC_AUTH_TOKEN = input.authToken
  const headers = Object.entries(input.headers ?? {}).map(([name, value]) => `${name}: ${value}`)
  if (headers.length > 0) env.ANTHROPIC_CUSTOM_HEADERS = headers.join("\n")
  if (Object.keys(input.body ?? {}).length > 0) env.CLAUDE_CODE_EXTRA_BODY = JSON.stringify(input.body)
  return env
}

// The native installer's directory is often missing from PATH in GUI-launched IDEs.
export async function executable(configured?: string) {
  const found =
    configured ??
    process.env.KILO_CLAUDE_PATH ??
    process.env.OPENCODE_CLAUDE_PATH ??
    (await which("claude", {
      nothrow: true,
      path: [process.env.PATH, path.join(os.homedir(), ".local", "bin")].filter(Boolean).join(path.delimiter),
    }))
  if (!found) throw new Error("Claude Code is not installed. Install it, then run `claude auth login`.")
  return found
}

export function base(input: { executable: string; connection?: Record<string, string> }): Options {
  return {
    pathToClaudeCodeExecutable: input.executable,
    env: { ...process.env, ...input.connection, ...ISOLATION },
    settingSources: [],
    settings: { disableAllHooks: true },
    // Without summaries, Kilo has no reasoning text to display
    thinking: { type: "adaptive", display: "summarized" },
    strictMcpConfig: true,
    skills: [],
    tools: [],
    permissionMode: "dontAsk",
    allowedTools: [`mcp__${SERVER}__*`],
    verbatimPrompts: true,
  }
}

export async function session(input: {
  executable: string
  connection?: Record<string, string>
  directory: string
  model: string
  effort?: Options["effort"]
  maxOutputTokens?: number
  system: string
  tools: readonly LanguageModelV3FunctionTool[]
  pending: Pending
  persist: boolean
  resume?: string
}): Promise<Options> {
  const options: Options = {
    ...base(input),
    cwd: input.directory,
    model: input.model,
    effort: input.effort,
    systemPrompt: input.system + overflow(input.tools),
    includePartialMessages: true,
    persistSession: input.persist,
    hooks: hooks(input.pending),
    mcpServers: input.tools.length > 0 ? { [SERVER]: await server(input.tools, input.pending) } : {},
  }
  if (input.maxOutputTokens !== undefined) {
    options.env = { ...options.env, CLAUDE_CODE_MAX_OUTPUT_TOKENS: String(input.maxOutputTokens) }
  }
  if (input.resume !== undefined) return { ...options, resume: input.resume }
  return { ...options, sessionId: crypto.randomUUID() }
}
