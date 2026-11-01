import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3FunctionTool,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider"
import type { EffortLevel } from "@anthropic-ai/claude-agent-sdk"
import { DIRECTORY_HEADER, ID, KIND_HEADER, SESSION_HEADER } from "./constants"
import { connection } from "./options"
import { run } from "./sessions"
import { converter, failed } from "./stream"
import { answer, marked, strip } from "./usage"

const EFFORTS: readonly EffortLevel[] = ["low", "medium", "high", "xhigh", "max"]

function effort(value: unknown) {
  if (value === undefined) return undefined
  const found = EFFORTS.find((item) => item === value)
  if (found) return found
  throw new Error("Claude Agent effort must be low, medium, high, xhigh or max")
}

class ClaudeAgentModel implements LanguageModelV3 {
  readonly specificationVersion = "v3"
  readonly provider = ID
  readonly supportedUrls = {}

  constructor(
    readonly modelId: string,
    readonly settings: Record<string, any>,
  ) {}

  async doGenerate(): Promise<never> {
    throw new Error("Claude Agent models only support streaming")
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const abort = new AbortController()
    const signal = options.abortSignal ? AbortSignal.any([abort.signal, options.abortSignal]) : abort.signal
    const headers = options.headers ?? {}
    const settings = { ...this.settings, ...options.providerOptions?.[ID] }
    const primary = headers[KIND_HEADER] === "primary"
    const last = options.prompt.findLast((message) => message.role !== "system")
    if (primary && last && marked(last)) return { stream: answer() }
    const tools =
      !primary || options.toolChoice?.type === "none"
        ? []
        : (options.tools ?? []).filter((tool): tool is LanguageModelV3FunctionTool => tool.type === "function")
    const convert = converter(tools)
    const system = options.prompt.flatMap((message) => (message.role === "system" ? [message.content] : [])).join("\n\n")
    let cancelled = false
    let task: Promise<void>
    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      start: (controller) => {
        controller.enqueue({ type: "stream-start", warnings: [] })
        task = Promise.resolve()
          .then(() =>
            run(
              {
                sessionID: primary ? headers[SESSION_HEADER] : undefined,
                directory: headers[DIRECTORY_HEADER] ?? process.cwd(),
                executable: typeof settings.claudePath === "string" ? settings.claudePath : undefined,
                model: this.modelId,
                effort: effort(settings.effort),
                maxOutputTokens: typeof settings.maxOutputTokens === "number" ? settings.maxOutputTokens : undefined,
                connection: connection(this.settings),
                // Titles are standalone tasks, so they need a helper prompt rather than an empty one.
                system: system || "You are a text generation helper. Return only the requested output.",
                prompt: strip(options.prompt),
                tools,
                emit: (message, cost) => {
                  if (cancelled) return
                  convert(message, cost).forEach((part) => controller.enqueue(part))
                },
              },
              signal,
            ),
          )
          .then(
            () => {
              if (!cancelled) controller.close()
            },
            (error) => {
              if (!cancelled) controller.error(failed(error))
            },
          )
      },
      cancel: () => {
        cancelled = true
        abort.abort()
        return task
      },
    })
    return { stream }
  }
}

export function createClaudeAgent(settings: Record<string, any> = {}) {
  return {
    languageModel: (id: string) => new ClaudeAgentModel(id, settings),
  }
}
