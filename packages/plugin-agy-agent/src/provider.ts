import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3FunctionTool,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider"
import { DIRECTORY_HEADER, ID, KIND_HEADER, SESSION_HEADER } from "./constants"
import { effort } from "./options"
import { run } from "./sessions"
import { failed } from "./stream"
import { answer, marked, strip } from "./usage"

class AgyAgentModel implements LanguageModelV3 {
  readonly specificationVersion = "v3"
  readonly provider = ID
  readonly supportedUrls = {}

  constructor(
    readonly modelId: string,
    readonly settings: Record<string, unknown>,
  ) {}

  async doGenerate(): Promise<never> {
    throw new Error("Antigravity Agent models only support streaming")
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
                executable: typeof settings.agyPath === "string" ? settings.agyPath : undefined,
                model: this.modelId,
                effort: effort(settings.effort),
                // Titles are standalone tasks, so they need a helper prompt rather than an empty one.
                system: system || "You are a text generation helper. Return only the requested output.",
                prompt: strip(options.prompt),
                tools,
                emit: (part) => {
                  if (!cancelled) controller.enqueue(part)
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

export function createAgyAgent(settings: Record<string, unknown> = {}) {
  return {
    languageModel: (id: string) => new AgyAgentModel(id, settings),
  }
}
