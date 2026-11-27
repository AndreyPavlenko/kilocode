import { mkdir, readFile, writeFile } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import which from "which"
import { AGENT, GEMINI } from "./constants"

export const EFFORTS = ["low", "medium", "high", "xhigh", "max"] as const

const CUSTOM = [
  "---",
  `name: ${AGENT}`,
  "description: Kilo Code bridge agent",
  "kind: local",
  "excludeDefaultComponents: true",
  "---",
  "",
].join("\n")

// The native installer's directory is often missing from PATH in GUI-launched IDEs.
export async function executable(configured?: string) {
  const found =
    configured ??
    process.env.KILO_AGY_PATH ??
    (await which("agy", {
      nothrow: true,
      path: [process.env.PATH, path.join(os.homedir(), ".local", "bin")].filter(Boolean).join(path.delimiter),
    }))
  if (!found) throw new Error("Antigravity CLI (agy) is not installed. Install it, then run `agy` and sign in.")
  return found
}

// The custom agent drops agy's default prompt and tools. The system prompt comes per session.
export async function prepare() {
  const file = path.join(GEMINI, "agents", `${AGENT}.md`)
  const current = await readFile(file, "utf8").catch(() => "")
  if (current !== CUSTOM) {
    await mkdir(path.dirname(file), { recursive: true })
    await writeFile(file, CUSTOM)
  }
  return [
    "acp",
    "--process-per-session",
    "--gemini-dir",
    GEMINI,
    "--agent",
    AGENT,
    "--include-default-customizations=false",
    "--disable-telemetry",
  ]
}

export function effort(value: unknown) {
  if (value === undefined) return undefined
  const found = EFFORTS.find((item) => item === value)
  if (found) return found
  throw new Error("Antigravity Agent effort must be low, medium, high, xhigh or max")
}
