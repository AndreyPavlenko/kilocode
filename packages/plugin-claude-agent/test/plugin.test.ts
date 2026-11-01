import fs from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { APICallError, type LanguageModelV3Message, type LanguageModelV3Prompt } from "@ai-sdk/provider"
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { KIND_HEADER } from "../src/constants"
import { createClaudeAgent } from "../src/provider"
import { atLeast } from "../src/session"
import { converter, failed, failure, spent, succeeded, usage } from "../src/stream"
import { MARKER, format, limits, marked, strip } from "../src/usage"
import { boundary, catchUp, checkpoints, decide, groups } from "../src/sync"
import { PREFIX, collect, convert, original, overflow, pending, short } from "../src/tools"
import { ClaudeAgentPlugin } from "../src/index"

const user = (text: string): LanguageModelV3Message => ({ role: "user", content: [{ type: "text", text }] })
const assistant = (text: string): LanguageModelV3Message => ({ role: "assistant", content: [{ type: "text", text }] })
const call = (id: string): LanguageModelV3Message => ({
  role: "assistant",
  content: [{ type: "tool-call", toolCallId: id, toolName: "read", input: { a: 1 } }],
})
const result = (id: string, value = "ok"): LanguageModelV3Message => ({
  role: "tool",
  content: [{ type: "tool-result", toolCallId: id, toolName: "read", output: { type: "text", value } }],
})
const tool = { type: "function", name: "read", inputSchema: {} } as const

function record(prompt: LanguageModelV3Prompt) {
  return {
    claudeSessionID: "c",
    history: groups(prompt).map((group, i) => ({ hash: group.hash, claudeMessageID: `m${i}` })),
  }
}

const limitsResponse = (rates: Record<string, unknown> | null) =>
  ({ subscription_type: "max", rate_limits_available: !!rates, rate_limits: rates }) as any

const zero = { input_tokens: 0, output_tokens: 0 }
const finished = (total: number, subtype = "success") =>
  ({ type: "result", subtype, is_error: subtype !== "success", errors: ["x"], num_turns: 1, usage: zero, total_cost_usd: total }) as any

describe("usage", () => {
  const window = (utilization: number | null) => ({ utilization, resets_at: "2026-11-03T00:00:00Z" })

  // Fills the cache, so nothing here starts a real Claude process.
  beforeAll(() => limits(async () => limitsResponse({ seven_day: { utilization: 1, resets_at: null } })))

  it("formats known windows and skips empty ones", () => {
    const text = format(limitsResponse({ five_hour: window(17.4), seven_day: window(null), seven_day_opus: null }))
    expect(text).toContain("5-hour: 17%")
    expect(text).not.toContain("7-day")
    expect(format(limitsResponse(null))).toContain("unavailable")
  })

  it("drops usage turns and keeps the rest of the history comparable", () => {
    const seen = [user("a"), assistant("b")]
    const prompt = [...seen, user(`${MARKER}\n<env/>`), assistant("usage"), user("c")]
    expect(marked(prompt[2]!)).toBe(true)
    expect(strip(prompt)).toEqual([...seen, user("c")])
    expect(decide(strip(prompt), record(seen))).toEqual({ kind: "continue", start: 2 })
  })

  it("only removes the marker from a prompt that quotes it", () => {
    const title = user(`Generate a title:\n1. ${MARKER}\n2. real`)
    expect(marked(title)).toBe(false)
    expect(strip([title])).toEqual([user("Generate a title:\n1. \n2. real")])
  })

  it("answers the marker without a Claude turn", async () => {
    const { stream } = await createClaudeAgent()
      .languageModel("opus")
      .doStream({ prompt: [user(MARKER)], headers: { [KIND_HEADER]: "primary" } } as any)
    const parts: any[] = []
    await stream.pipeTo(new WritableStream({ write: (part) => void parts.push(part) }))
    expect(parts.find((part) => part.type === "text-delta").delta).toContain("7-day: 1%")
    expect(parts.at(-1).usage.outputTokens.total).toBe(0)
  })

  it("registers the command without replacing the user's", async () => {
    const hooks = await ClaudeAgentPlugin({ directory: "/tmp" } as any)
    const config: any = { command: { other: { template: "x" } } }
    await hooks.config!(config)
    expect(config.command["claude-usage"].template).toBe(MARKER)
    expect(config.command["claude-usage"].model).toMatch(/^claude-agent\//)
    const custom: any = { command: { "claude-usage": { template: "mine" } } }
    await hooks.config!(custom)
    expect(custom.command["claude-usage"].template).toBe("mine")
  })
})

describe("cost", () => {
  const raw = (parts: any[]) => parts.find((part) => part.type === "finish")?.usage.raw?.cost_details.upstream_inference_cost

  it("reports the delta, or the whole total after a restart", () => {
    expect(spent(0.5, 0.2)).toBeCloseTo(0.3)
    expect(spent(0.1, 0.5)).toBe(0.1)
    expect(spent(0.4, 0)).toBe(0.4)
  })

  it("puts cost only on a successful result", () => {
    expect(raw(converter([])(finished(2), 0.25))).toBe(0.25)
    expect(raw(converter([])(finished(2)))).toBe(2)
    expect(succeeded(finished(2, "error_during_execution"))).toBe(false)
    expect(raw(converter([])(finished(2, "error_during_execution")))).toBeUndefined()
  })
})

describe("decide", () => {
  it("starts fresh without history", () => {
    expect(decide([user("a")])).toEqual({ kind: "fresh", start: 0 })
  })

  it("sends only the new input when Claude has seen the rest", () => {
    const seen = [user("a"), assistant("b")]
    expect(decide([...seen, user("c")], record(seen))).toEqual({ kind: "continue", start: 2 })
  })

  it("catches up on replies from another model", () => {
    const seen = [user("a"), assistant("b")]
    expect(decide([...seen, user("c"), assistant("d"), user("e")], record(seen))).toEqual({
      kind: "catch-up",
      start: 2,
    })
  })

  it("flags an edit as changed", () => {
    const seen = [user("a"), assistant("b"), user("c")]
    expect(decide([user("a"), assistant("x"), user("c")], record(seen)).kind).toBe("changed")
  })

  it("restarts when the first message differs, as after compaction", () => {
    expect(decide([user("summary")], record([user("a"), assistant("b")])).kind).toBe("fresh")
  })

  it("ignores reasoning and provider options", () => {
    const plain = [user("a"), assistant("b")]
    const rich: LanguageModelV3Prompt = [
      user("a"),
      {
        role: "assistant",
        content: [
          { type: "reasoning", text: "hm" },
          { type: "text", text: "b", providerOptions: { x: { y: 1 } } },
        ],
      },
    ]
    expect(groups(rich).map((group) => group.hash)).toEqual(groups(plain).map((group) => group.hash))
  })

  it("lets Claude own an interrupted reply", () => {
    const seen = [user("a")]
    const saved = { ...record(seen), interrupted: true }
    expect(decide([user("a"), assistant("partial"), user("b")], saved)).toEqual({ kind: "continue", start: 2 })
  })
})

describe("boundary", () => {
  it("never splits a tool call from its result", () => {
    const prompt = [user("a"), call("1"), result("1"), assistant("b"), user("c")]
    const found = boundary(prompt, record(prompt))
    expect(found?.start).toBe(4)
    const open = [user("a"), call("1"), result("1")]
    const saved = record(open)
    saved.history[1]!.claudeMessageID = "call"
    saved.history[2]!.claudeMessageID = undefined as unknown as string
    expect(boundary(open, saved)?.messageID).toBe("m0")
  })
})

describe("checkpoints", () => {
  it("records the sent input and the reply", () => {
    const prompt = [user("a")]
    const history = checkpoints({
      prompt,
      userID: "u",
      reply: [{ type: "text", text: "b" }],
      messageID: "m",
      toolMessages: new Map(),
    })
    expect(history.map((item) => item.claudeMessageID)).toEqual(["u", "m"])
    expect(decide([user("a"), assistant("b"), user("c")], { claudeSessionID: "c", history })).toEqual({
      kind: "continue",
      start: 2,
    })
  })

  it("matches a tool step on the next request", () => {
    const history = checkpoints({
      prompt: [user("a")],
      userID: "u",
      reply: [{ type: "tool-call", toolCallId: "1", toolName: "read", input: { a: 1 } }],
      messageID: "m",
      toolMessages: new Map(),
    })
    const next = decide([user("a"), call("1"), result("1")], { claudeSessionID: "c", history })
    expect(next).toEqual({ kind: "catch-up", start: 2 })
  })
})

describe("catchUp", () => {
  it("wraps missed replies and leaves the new input outside", () => {
    const text = catchUp([user("a"), assistant("b"), user("c")]).map((block) =>
      block.type === "text" ? block.text : "",
    )
    expect(text[0]).toContain("<conversation-checkpoint>")
    expect(text.at(-2)).toBe("</conversation-checkpoint>")
    expect(text.at(-1)).toBe("c")
  })

  it("sends plain input when nothing was missed", () => {
    expect(catchUp([user("a")])).toEqual([{ type: "text", text: "a" }])
  })
})

describe("tools", () => {
  it("carries description text past the MCP limit in the system prompt", () => {
    const tool = (name: string, description: string) => ({
      type: "function" as const,
      name,
      description,
      inputSchema: {},
    })
    const text = overflow([tool("task", "x".repeat(2048) + "- routine: agent"), tool("read", "y".repeat(2048))])
    expect(text).toContain(`<${PREFIX}task>`)
    expect(text).toContain("- routine: agent")
    expect(text).not.toContain("read")
  })

  it("round-trips names and refuses tools not offered", () => {
    expect(original(PREFIX + "read", [tool])).toBe("read")
    expect(original("glob", [tool])).toBeUndefined()
    const long = "x".repeat(100) + ".y"
    expect((PREFIX + short(long)).length).toBeLessThanOrEqual(64)
    expect(short(long)).toMatch(/^[a-zA-Z0-9_-]+$/)
  })

  it("holds a call until its result arrives", async () => {
    const calls = pending()
    calls.before("1", "n")
    const held = calls.call("n")
    calls.resolve("1", convert({ type: "text", value: "done" }))
    expect((await held).content).toEqual([{ type: "text", text: "done" }])
  })

  it("rejects a call without a hook and cancels held calls", async () => {
    const calls = pending()
    await expect(calls.call("n")).rejects.toThrow("PreToolUse")
    calls.before("1", "n")
    const held = calls.call("n")
    calls.clear()
    expect((await held).isError).toBe(true)
  })

  it("marks failures and replaces content that MCP can't carry", () => {
    expect(convert({ type: "error-text", value: "bad" }).mcp.isError).toBe(true)
    const pdf = convert({ type: "text", value: "a" }, [{ type: "file", mediaType: "application/pdf", data: "AA" }])
    expect(pdf.replacement?.some((block) => block.type === "document")).toBe(true)
    const png = convert({ type: "text", value: "a" }, [{ type: "file", mediaType: "image/png", data: "AA" }])
    expect(png.replacement).toBeUndefined()
    expect(png.mcp.content.at(-1)).toEqual({ type: "image", mimeType: "image/png", data: "AA" })
  })

  it("collects results and spots stopped tools", () => {
    const done = collect([result("1")], 0)
    expect(done.results.has("1")).toBe(true)
    expect(done.aborted).toBe(false)
    const stopped: LanguageModelV3Prompt = [
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "1",
            toolName: "read",
            output: { type: "error-text", value: "Tool execution aborted" },
          },
        ],
      },
    ]
    expect(collect(stopped, 0).aborted).toBe(true)
  })
})

describe("stream", () => {
  it("keeps the highest count per response and the last input", () => {
    const counts = usage()
    counts.update("a", { input_tokens: 10, output_tokens: 1 })
    counts.update("a", { input_tokens: 10, output_tokens: 5 })
    counts.update("b", { input_tokens: 20, cache_read_input_tokens: 5, output_tokens: 2 })
    const snap = counts.snapshot()
    expect(snap.inputTokens.total).toBe(25)
    expect(snap.outputTokens.total).toBe(7)
  })

  it("converts text, ends tool steps and ignores subagents", () => {
    const convert = converter([{ ...tool, name: "read" }])
    const event = (value: unknown) => ({ type: "stream_event", parent_tool_use_id: null, event: value }) as any
    const parts = [
      event({ type: "message_start", message: { id: "m", model: "x", usage: {} } }),
      event({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
      event({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "hi" } }),
      event({ type: "content_block_stop", index: 0 }),
      {
        type: "assistant",
        parent_tool_use_id: null,
        uuid: "u",
        message: {
          id: "m",
          usage: {},
          content: [{ type: "tool_use", id: "1", name: PREFIX + "read", input: { a: 1 } }],
        },
      } as any,
      event({ type: "message_stop" }),
    ].flatMap(convert)
    expect(parts.map((part) => part.type)).toEqual([
      "response-metadata",
      "text-start",
      "text-delta",
      "text-end",
      "tool-call",
      "finish",
    ])
    expect(parts.at(-1)).toMatchObject({ finishReason: { unified: "tool-calls" } })
    expect(convert({ ...event({ type: "message_stop" }), parent_tool_use_id: "x" })).toEqual([])
  })

  it("tells limits, transient and permanent failures apart", () => {
    const check = failure()
    const failedResult = (result: string, status?: number) =>
      ({ type: "result", subtype: "success", is_error: true, result, api_error_status: status }) as any
    const failedReply = (error: string) =>
      ({ type: "assistant", parent_tool_use_id: null, error, message: { content: [] } }) as any
    expect(check({ type: "result", subtype: "success", is_error: false } as any)).toBeUndefined()
    const overloaded = check(failedResult("", 529))
    expect(APICallError.isInstance(overloaded?.error)).toBe(true)
    expect(overloaded?.error.isRetryable).toBe(true)
    expect(overloaded?.error.message).toContain("overloaded")
    check(failedReply("server_error"))
    const offline = check(failedResult("API Error: Unable to connect to API"))
    expect(offline?.error.isRetryable).toBe(true)
    expect((offline?.error.cause as any).code).toBe("ENETUNREACH")
    check({ type: "rate_limit_event", rate_limit_info: { status: "rejected", resetsAt: 5, rateLimitType: "five_hour" } } as any)
    check(failedReply("rate_limit"))
    expect(check(failedResult("limit", 429))?.limit).toEqual({ resetsAt: 5, type: "five_hour" })
    check(failedReply("authentication_failed"))
    const auth = check(failedResult("API Error: invalid token", 401))
    expect(auth?.limit).toBeUndefined()
    expect(auth?.error.isRetryable).toBe(false)
  })

  it("wraps anything thrown so Kilo won't retry on its wording", () => {
    const error = failed(new Error("connection refused"))
    expect(APICallError.isInstance(error) && error.isRetryable).toBe(false)
  })
})

describe("plugin", () => {
  const load = () => ClaudeAgentPlugin({ directory: "/d" } as any)

  it("registers the provider and keeps user settings", async () => {
    const hooks = await load()
    const config: any = { provider: { "claude-agent": { name: "Mine", models: { extra: { id: "extra" } } } } }
    await hooks.config?.(config)
    const section = config.provider["claude-agent"]
    expect(section.name).toBe("Mine")
    expect(section.npm).toMatch(/^file:.*provider\.ts$/)
    expect(section.models.extra).toEqual({ id: "extra" })
  })

  it("tags requests only for this provider and tells title calls apart", async () => {
    const hooks = await load()
    const headers = async (providerID: string, agent: string) => {
      const output = { headers: {} as Record<string, string> }
      await hooks["chat.headers"]?.({ sessionID: "s", agent, model: { providerID } } as any, output)
      return output.headers
    }
    expect(await headers("other", "code")).toEqual({})
    expect(await headers("claude-agent", "code")).toMatchObject({ "x-kilocode-claude-kind": "primary" })
    expect(await headers("claude-agent", "title")).toMatchObject({ "x-kilocode-claude-kind": "utility" })
  })

  it("drops Kilo's output cap", async () => {
    const hooks = await load()
    const output: any = { maxOutputTokens: 32000 }
    await hooks["chat.params"]?.({ model: { providerID: "claude-agent" } } as any, output)
    expect(output.maxOutputTokens).toBeUndefined()
  })
})

describe("limits and disconnects", () => {
  const MINUTE = 60_000
  const HOUR = 60 * MINUTE
  const reply = (fields: Record<string, unknown>) => ({
    type: "assistant",
    parent_tool_use_id: null,
    uuid: crypto.randomUUID(),
    message: { id: crypto.randomUUID(), content: [], usage: {} },
    ...fields,
  })
  const broken = (result: string, status?: number) => ({ ...finished(0), is_error: true, result, api_error_status: status })
  // Claude reports whole epoch seconds.
  const limited = (after: number) => [
    {
      type: "rate_limit_event",
      rate_limit_info: { status: "rejected", rateLimitType: "five_hour", resetsAt: Math.floor((Date.now() + after) / 1000) },
    },
    reply({ error: "rate_limit" }),
    broken("limit", 429),
  ]
  const answered = (text: string) => [
    reply({ message: { id: crypto.randomUUID(), content: [{ type: "text", text }], usage: {} } }),
    finished(1),
  ]
  const nudged = (content: unknown) => (content as any[])[0].text.includes("Continue where you left off")

  const launched: any[] = []
  const claudes: any[] = []
  const sent: unknown[] = []
  const replies: any[][] = []
  const checks = { count: 0, used: 100 }
  const env = process.env.XDG_DATA_HOME
  const warned = vi.fn()
  let run: typeof import("../src/sessions").run

  async function start(options: any) {
    launched.push(options)
    const turn = { batch: [] as any[], running: false }
    const claude = {
      id: options.resume ?? options.sessionId,
      ready: Promise.resolve(),
      persisted: true,
      ended: false,
      get running() {
        return turn.running
      },
      begin(content: unknown) {
        sent.push(content)
        turn.batch = replies.shift() ?? answered("?")
        turn.running = true
      },
      addUser: () => undefined,
      async read(emit: (message: any) => void) {
        turn.batch.forEach(emit)
        turn.running = false
        return turn.batch.at(-1)
      },
      interrupt: async () => undefined,
      stop: async () => undefined,
      usage: async () => {
        checks.count++
        return limitsResponse({ five_hour: { utilization: checks.used, resets_at: null } })
      },
      setModel: async () => undefined,
      setEffort: async () => undefined,
    }
    claudes.push(claude)
    return claude
  }

  const request = (sessionID: string, prompt: LanguageModelV3Prompt, emitted: any[] = []) => ({
    sessionID,
    directory: "/d",
    model: "opus",
    system: "s",
    prompt,
    tools: [],
    emit: (message: any) => void emitted.push(message),
  })
  const flush = async (done: () => boolean) => {
    for (let i = 0; i < 1000 && !done(); i++) await new Promise((resolve) => setImmediate(resolve))
  }
  const signal = () => new AbortController().signal

  beforeAll(async () => {
    process.env.XDG_DATA_HOME = await fs.mkdtemp(path.join(os.tmpdir(), "claude-agent-"))
    vi.spyOn(console, "warn").mockImplementation(warned)
    vi.resetModules()
    vi.doMock("../src/session", async (actual) => ({
      ...(await actual<typeof import("../src/session")>()),
      start,
      sdk: async () => ({ getSessionInfo: async () => ({}) }),
    }))
    vi.doMock("../src/options", async (actual) => ({
      ...(await actual<typeof import("../src/options")>()),
      executable: async () => "claude",
    }))
    run = (await import("../src/sessions")).run
  })

  beforeEach(() => {
    launched.length = 0
    claudes.length = 0
    sent.length = 0
    replies.length = 0
    checks.count = 0
    checks.used = 100
    warned.mockClear()
  })

  afterEach(() => vi.useRealTimers())

  afterAll(() => {
    vi.doUnmock("../src/session")
    vi.doUnmock("../src/options")
    vi.mocked(console.warn).mockRestore()
    process.env.XDG_DATA_HOME = env
  })

  it("waits out a limit and continues the same Claude session", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] })
    replies.push(limited(3 * HOUR), answered("hi"))
    const emitted: any[] = []
    const done = run(request("limit", [user("a")], emitted), signal())
    await flush(() => warned.mock.calls.length > 0)
    expect(warned.mock.calls[0]?.[0]).toContain("limit reached, waiting until")
    await vi.advanceTimersByTimeAsync(MINUTE)
    expect(sent).toHaveLength(1)
    checks.used = 50
    await vi.advanceTimersByTimeAsync(MINUTE)
    await done
    expect(sent).toHaveLength(2)
    expect(nudged(sent[1])).toBe(true)
    expect(emitted.some((message) => message.type === "result" && message.is_error)).toBe(false)
    expect(emitted.at(-1)).toMatchObject({ type: "result", is_error: false })

    await run(request("limit", [user("a"), assistant("hi"), user("b")]), signal())
    expect(sent.at(-1)).toEqual([{ type: "text", text: "b" }])
    expect(launched).toHaveLength(1)
  })

  it("sleeps until the reported reset after an hour of checks", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] })
    replies.push(limited(3 * HOUR), answered("ok"))
    const done = run(request("week", [user("a")]), signal())
    await flush(() => warned.mock.calls.length > 0)
    await vi.advanceTimersByTimeAsync(HOUR)
    const hourly = checks.count
    expect(hourly).toBe(60)
    await vi.advanceTimersByTimeAsync(HOUR)
    expect(checks.count).toBe(hourly)
    checks.used = 50
    await vi.advanceTimersByTimeAsync(HOUR)
    await done
    expect(checks.count).toBe(hourly + 1)
    expect(nudged(sent.at(-1))).toBe(true)
  })

  it("stops waiting on cancel and resumes the wait on the same input", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] })
    replies.push(limited(HOUR))
    const abort = new AbortController()
    const done = run(request("abort", [user("a")]), abort.signal)
    await flush(() => warned.mock.calls.length > 0)
    abort.abort()
    await done
    const saved = JSON.parse(
      await fs.readFile(path.join(process.env.XDG_DATA_HOME!, "kilo", "plugin-claude-agent", "sessions", "abort.json"), "utf8"),
    )
    expect(saved.stalled.limit.resetsAt).toBeGreaterThan(0)

    checks.used = 50
    const next = run(request("abort", [user("a")]), signal())
    await flush(() => false)
    expect(sent).toHaveLength(1)
    await vi.advanceTimersByTimeAsync(MINUTE)
    await next
    expect(sent).toHaveLength(2)
    expect(nudged(sent[1])).toBe(true)
    expect(launched).toHaveLength(1)
  })

  it("hands a lost connection to Kilo's retry and resumes without a rebuild", async () => {
    replies.push([reply({ error: "server_error" }), broken("API Error: Unable to connect to API")])
    const error = await run(request("net", [user("a")]), signal()).catch((err) => err)
    expect(APICallError.isInstance(error) && error.isRetryable).toBe(true)
    // Claude Code exits as well.
    claudes[0].ended = true
    await run(request("net", [user("a")]), signal())
    expect(launched).toHaveLength(2)
    expect(launched[1].resume).toBe(launched[0].sessionId)
    expect(sent).toHaveLength(2)
    expect(nudged(sent[1])).toBe(true)
  })
})

describe("version", () => {
  it("compares release numbers", () => {
    expect(atLeast("2.1.316", "2.1.248")).toBe(true)
    expect(atLeast("2.1.100", "2.1.248")).toBe(false)
    expect(atLeast("2.1.248-beta", "2.1.248")).toBe(true)
  })
})
