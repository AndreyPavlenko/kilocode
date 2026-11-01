import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs"
import path from "node:path"
import { DATA } from "./constants"
import { base, executable } from "./options"
import { atLeast, sdk } from "./session"

const FILE = path.join(DATA, "models.json")
const MAX_AGE_MS = 24 * 60 * 60_000
const TIMEOUT_MS = 30_000
// Older binaries do not support summary-only context usage.
const DISCOVERY = "2.1.257"
// Until the first check finishes, and whenever Claude Code can't be asked.
const FALLBACK: Item[] = [
  { id: "opus", name: "Claude Opus", efforts: [] },
  { id: "sonnet", name: "Claude Sonnet", efforts: [] },
  { id: "haiku", name: "Claude Haiku", efforts: [] },
]
const DEFAULT_CONTEXT = 200_000

interface Item {
  id: string
  name: string
  efforts: readonly string[]
  context?: number
}

function saved(): Item[] {
  try {
    const items = JSON.parse(readFileSync(FILE, "utf8"))
    if (Array.isArray(items) && items.length > 0) return items
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "ENOENT") console.warn("[claude-agent] bad model list:", err)
  }
  return FALLBACK
}

export function models() {
  return Object.fromEntries(
    saved().map((item) => [
      item.id,
      {
        id: item.id,
        name: item.name,
        tool_call: true,
        reasoning: true,
        attachment: true,
        temperature: false,
        modalities: { input: ["text", "image", "pdf"], output: ["text"] },
        // The SDK reports context windows but not output limits. 0 means unknown in Kilo.
        limit: { context: item.context ?? DEFAULT_CONTEXT, output: 0 },
        variants: Object.fromEntries(item.efforts.map((effort) => [effort, { effort }])),
      },
    ]),
  )
}

// The input never sends a message, so no model call is made and no tokens are used.
async function probe(): Promise<Item[]> {
  const { query } = await sdk()
  const done = Promise.withResolvers<void>()
  const session = query({
    prompt: (async function* (): AsyncGenerator<never> {
      await done.promise
    })(),
    options: { ...base({ executable: await executable() }), persistSession: false },
  })
  try {
    const init = await session.initializationResult()
    const version = init.claude_code_version
    const discover = !!version && atLeast(version, DISCOVERY)
    const items: Item[] = []
    // "default" follows Claude Code's own default, which can change under a running session.
    for (const model of init.models.filter((item) => item.value !== "default")) {
      const usage = discover
        ? await session
            .setModel(model.value)
            .then(() => session.getContextUsage({ detail: "summary" }))
            // Normal for a model the plan can't select. It keeps the default window.
            .catch(() => undefined)
        : undefined
      items.push({
        id: model.value,
        name: model.displayName,
        efforts: model.supportedEffortLevels ?? [],
        // rawMaxTokens is capacity; maxTokens can reflect Claude's own compaction window.
        context: usage?.rawMaxTokens,
      })
    }
    return items
  } finally {
    done.resolve()
    session.close()
  }
}

let checked = false

// Runs in the background because the config hook blocks startup. A new list applies from the next start.
export function refresh() {
  if (checked) return
  checked = true
  const age = (() => {
    try {
      return Date.now() - statSync(FILE).mtimeMs
    } catch {
      return Infinity
    }
  })()
  if (age < MAX_AGE_MS) return
  const timer = setTimeout(() => console.warn("[claude-agent] model check is slow"), TIMEOUT_MS)
  timer.unref()
  void probe()
    .then((items) => {
      if (items.length === 0) return
      mkdirSync(DATA, { recursive: true })
      writeFileSync(FILE, JSON.stringify(items))
    })
    .catch((err) => console.warn("[claude-agent] model check failed:", err instanceof Error ? err.message : err))
    .finally(() => clearTimeout(timer))
}
