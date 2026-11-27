import { execFile } from "node:child_process"
import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs"
import path from "node:path"
import { DATA } from "./constants"
import { EFFORTS, executable } from "./options"

const FILE = path.join(DATA, "models.json")
const MAX_AGE_MS = 24 * 60 * 60_000
const TIMEOUT_MS = 30_000
const DEFAULT_CONTEXT = 200_000
// Gemini models have a 1M window; a lower guess makes Kilo compact long before agy needs it.
const GEMINI_CONTEXT = 1_048_576
const SUFFIX = new RegExp(`-(${EFFORTS.join("|")})$`)
// Until the first check finishes, and whenever agy can't be asked.
const FALLBACK: Item[] = [
  { id: "gemini-4-hydrogen", name: "Gemini 4 Hydrogen", efforts: ["low", "medium", "high", "xhigh"] },
  { id: "gemini-3.8-flash", name: "Gemini 3.8 Flash", efforts: ["low", "medium", "high"] },
]

export interface Item {
  id: string
  name: string
  efforts: readonly string[]
  context?: number
}

// Ids sharing a base after stripping an effort suffix are one model; a lone suffixed id keeps its full id.
export function group(lines: string): Item[] {
  const rows = lines.split("\n").flatMap((line) => {
    const [id, name] = line.trim().split("\t")
    return id && name ? [{ id, name: name.trim() }] : []
  })
  const bases = new Map<string, typeof rows>()
  rows.forEach((row) => {
    const base = row.id.replace(SUFFIX, "")
    bases.set(base, [...(bases.get(base) ?? []), row])
  })
  return [...bases].flatMap(([base, items]): Item[] => {
    if (items.length < 2) return items.map((item) => ({ id: item.id, name: item.name, efforts: [] }))
    const efforts = items.map((item) => item.id.slice(base.length + 1))
    return [
      {
        id: base,
        name: items[0].name.replace(/\s*\([^)]*\)$/, ""),
        efforts: EFFORTS.filter((effort) => efforts.includes(effort)),
      },
    ]
  })
}

function cached(): Item[] | undefined {
  try {
    const items = JSON.parse(readFileSync(FILE, "utf8"))
    if (Array.isArray(items) && items.length > 0) return items
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "ENOENT") console.warn("[agy-agent] bad model list:", err)
  }
  return undefined
}

const save = (items: Item[]) => {
  mkdirSync(DATA, { recursive: true })
  writeFileSync(FILE, JSON.stringify(items))
}

export function models() {
  return Object.fromEntries(
    (cached() ?? FALLBACK).map((item) => [
      item.id,
      {
        id: item.id,
        name: item.name,
        tool_call: true,
        reasoning: true,
        attachment: true,
        temperature: false,
        modalities: { input: ["text" as const, "image" as const], output: ["text" as const] },
        // agy reports the context window only at runtime and never an output limit. 0 means unknown in Kilo.
        limit: {
          context: item.context ?? (item.id.startsWith("gemini") ? GEMINI_CONTEXT : DEFAULT_CONTEXT),
          output: 0,
        },
        variants: Object.fromEntries(item.efforts.map((effort) => [effort, { effort }])),
      },
    ]),
  )
}

export function learn(id: string, context: number) {
  const items = cached()
  const item = items?.find((entry) => entry.id === id)
  if (!items || !item || item.context === context) return
  item.context = context
  try {
    save(items)
  } catch (err) {
    console.warn("[agy-agent] could not save context size:", err)
  }
}

const list = (exe: string) => {
  const done = Promise.withResolvers<string>()
  // The "Fetching available models..." banner lacks a tab, so group() drops it wherever it lands.
  execFile(exe, ["models"], { timeout: TIMEOUT_MS }, (err, stdout) => (err ? done.reject(err) : done.resolve(stdout)))
  return done.promise
}

let checked = false

// Runs in the background because the config hook blocks startup. A new list applies from the next start.
export function refresh() {
  if (checked) return
  checked = true
  const time = (file: string) => {
    try {
      return statSync(file).mtimeMs
    } catch {
      return undefined
    }
  }
  void executable()
    .then(async (exe) => {
      const at = time(FILE)
      // An updated agy can offer new models, so its install invalidates the list.
      if (at != null && Date.now() - at < MAX_AGE_MS && (time(exe) ?? 0) <= at) return
      const items = group(await list(exe))
      if (items.length === 0) return
      const known = new Map((cached() ?? []).map((item) => [item.id, item.context]))
      save(items.map((item) => ({ ...item, context: known.get(item.id) })))
    })
    .catch((err) => console.warn("[agy-agent] model check failed:", err instanceof Error ? err.message : err))
}
