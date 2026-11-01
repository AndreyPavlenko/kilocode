import type { LanguageModelV3Message, LanguageModelV3Prompt, LanguageModelV3StreamPart } from "@ai-sdk/provider"
import type { SDKControlGetUsageResponse } from "@anthropic-ai/claude-agent-sdk"
import { inspect } from "./catalog"
import { loggedIn } from "./session"

export const MARKER = "[[claude-agent:usage]]"
const TTL_MS = 60_000
const TIMEOUT_MS = 15_000
const WINDOWS = {
  five_hour: "5-hour",
  seven_day: "7-day",
  seven_day_opus: "7-day Opus",
  seven_day_sonnet: "7-day Sonnet",
  seven_day_oauth_apps: "7-day OAuth apps",
} as const

type Response = SDKControlGetUsageResponse
type Load = () => Promise<Response>

let cached: { at: number; value: Response } | undefined
let flight: Promise<Response> | undefined

async function timed<T>(work: Promise<T>) {
  const timeout = Promise.withResolvers<never>()
  const timer = setTimeout(() => timeout.reject(new Error(`no answer within ${TIMEOUT_MS / 1000} seconds`)), TIMEOUT_MS)
  timer.unref()
  return Promise.race([work, timeout.promise]).finally(() => clearTimeout(timer))
}

// The input never sends a message, so no model call is made and no tokens are used.
const probe = () =>
  inspect((session) =>
    timed(
      session.initializationResult().then((init) => {
        if (!loggedIn(init.account)) throw new Error("Claude Code is not logged in. Run `claude auth login` in a terminal.")
        return session.usage_EXPERIMENTAL_MAY_CHANGE_DO_NOT_RELY_ON_THIS_API_YET({ skipBehaviors: true })
      }),
    ),
  )

// Limits belong to the account, so every session shares one result.
export function limits(live?: Load) {
  if (cached && Date.now() - cached.at < TTL_MS) return Promise.resolve(cached.value)
  flight ??= (live ? timed(live()).catch(probe) : probe())
    .then((value) => {
      cached = { at: Date.now(), value }
      return value
    })
    .finally(() => {
      flight = undefined
    })
  return flight
}

export function format(response: Response) {
  const plan = response.subscription_type ? `**Claude plan:** ${response.subscription_type}` : "**Claude plan limits**"
  const rates = response.rate_limits
  if (!response.rate_limits_available || !rates) {
    return `${plan}\n\nPlan limits unavailable (API key, Bedrock/Vertex, or missing profile scope).`
  }
  const windows: Array<{ name: string; utilization?: number | null; resets_at?: string | null }> = [
    ...Object.entries(WINDOWS).map(([key, name]) => ({ name, ...rates[key as keyof typeof WINDOWS] })),
    ...(rates.model_scoped ?? []).map((item) => ({ name: `7-day ${item.display_name}`, ...item })),
  ]
  const rows = windows.flatMap((row) => {
    if (row.utilization === null || row.utilization === undefined) return []
    const reset = row.resets_at ? `, resets ${new Date(row.resets_at).toLocaleString()}` : ""
    return [`- ${row.name}: ${Math.round(row.utilization)}%${reset}`]
  })
  const extra = rates.extra_usage
  if (extra?.is_enabled) {
    const used = extra.used_credits ?? 0
    const cap = extra.monthly_limit === null ? "" : ` of ${extra.monthly_limit}`
    rows.push(`- Extra usage: ${used}${cap}${extra.currency ? ` ${extra.currency}` : ""}`)
  }
  return [plan, "", ...rows].join("\n")
}

export function prefix(response: Response) {
  const week = response.rate_limits?.seven_day?.utilization
  const hour = response.rate_limits?.five_hour?.utilization
  const parts = [
    typeof week === "number" ? `W${Math.round(week)}` : "",
    typeof hour === "number" ? `H${Math.round(hour)}` : "",
  ].filter(Boolean)
  return parts.length ? `${parts.join(" ")} · ` : ""
}

export const retitle = (title: string, value: string) => value + title.replace(/^(W\d+ )?(H\d+ )?· /, "")

// Kilo may append text to the command's message, but the template stays first. Title prompts quote it mid-text.
export const marked = (message: LanguageModelV3Message) =>
  message.role === "user" && message.content.some((part) => part.type === "text" && part.text.trimStart().startsWith(MARKER))

// Usage turns are not conversation: Claude never saw them, and they must not break the history comparison.
export function strip(prompt: LanguageModelV3Prompt): LanguageModelV3Prompt {
  let skipping = false
  return prompt
    .filter((message) => {
      if (message.role === "system") return true
      if (message.role === "user") skipping = marked(message)
      return !skipping
    })
    .map((message) =>
      message.role !== "user"
        ? message
        : {
            ...message,
            content: message.content.map((part) =>
              part.type === "text" ? { ...part, text: part.text.replaceAll(MARKER, "") } : part,
            ),
          },
    )
}

export function answer() {
  return new ReadableStream<LanguageModelV3StreamPart>({
    async start(controller) {
      controller.enqueue({ type: "stream-start", warnings: [] })
      const text = await limits()
        .then(format)
        .catch((error) => `Could not read Claude limits: ${error instanceof Error ? error.message : error}`)
      const id = "usage"
      controller.enqueue({ type: "text-start", id })
      controller.enqueue({ type: "text-delta", id, delta: text })
      controller.enqueue({ type: "text-end", id })
      controller.enqueue({
        type: "finish",
        finishReason: { unified: "stop", raw: undefined },
        usage: {
          inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 0, text: 0, reasoning: 0 },
        },
      })
      controller.close()
    },
  })
}

// The prefix would make a default title look custom and permanently block Kilo's title generation.
export async function updateTitle(
  client: {
    session: {
      get(options: { path: { id: string }; query: { directory: string } }): Promise<{ data?: { title: string; parentID?: string } }>
      update(options: { path: { id: string }; query: { directory: string }; body: { title: string } }): Promise<unknown>
    }
  },
  directory: string,
  sessionID: string,
  live?: Load,
) {
  const value = prefix(await limits(live))
  if (!value) return
  const target = { path: { id: sessionID }, query: { directory } }
  const session = (await client.session.get(target)).data
  if (!session || session.parentID || session.title.startsWith("New session - ")) return
  const title = retitle(session.title, value)
  if (title !== session.title) await client.session.update({ ...target, body: { title } })
}
