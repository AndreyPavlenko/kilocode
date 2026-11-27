import os from "node:os"
import path from "node:path"

export const ID = "agy-agent"
export const KIND_HEADER = "x-kilocode-agy-kind"
export const SESSION_HEADER = "x-kilocode-agy-session"
export const DIRECTORY_HEADER = "x-kilocode-working-directory"
// MCP server name. agy names the tools mcp_${SERVER}_${short}.
export const SERVER = "kilo"
// Custom agent file name and --agent value.
export const AGENT = "kilo"

export const DATA = path.join(
  process.env.XDG_DATA_HOME ?? path.join(os.homedir(), ".local", "share"),
  "kilo",
  "plugin-agy-agent",
)

// Own gemini dir, so the user's ~/.gemini config, plugins and history stay untouched.
export const GEMINI = path.join(DATA, "gemini")

// These run outside the conversation, so they must not touch the session's agy conversation.
const UTILITY = new Set(["title", "summary", "compaction"])

export function kind(agent: string) {
  return UTILITY.has(agent) ? "utility" : "primary"
}
