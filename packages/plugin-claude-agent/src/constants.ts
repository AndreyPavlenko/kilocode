import os from "node:os"
import path from "node:path"

export const ID = "claude-agent"
export const KIND_HEADER = "x-kilocode-claude-kind"
export const SESSION_HEADER = "x-kilocode-claude-session"
export const DIRECTORY_HEADER = "x-kilocode-working-directory"

export const DATA = path.join(
  process.env.XDG_DATA_HOME ?? path.join(os.homedir(), ".local", "share"),
  "kilo",
  "plugin-claude-agent",
)

// These run outside the conversation, so they must not touch the session's Claude process.
const UTILITY = new Set(["title", "summary", "compaction"])

export function kind(agent: string) {
  return UTILITY.has(agent) ? "utility" : "primary"
}
