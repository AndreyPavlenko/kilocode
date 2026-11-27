import { execFileSync } from "node:child_process"
import { randomUUID } from "node:crypto"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import type {
  LanguageModelV3CallOptions,
  LanguageModelV3FunctionTool,
  LanguageModelV3Message,
  LanguageModelV3Prompt,
  LanguageModelV3StreamPart,
} from "@ai-sdk/provider"
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest"

const root = fs.mkdtempSync(path.join(os.tmpdir(), "agy-test-"))
process.env.XDG_DATA_HOME = path.join(root, "data")
const exe = path.join(root, "agy")
fs.writeFileSync(exe, `#!/bin/sh\nexec bun ${path.join(import.meta.dirname, "fake-agy.ts")} "$@"\n`)
fs.chmodSync(exe, 0o755)
process.env.KILO_AGY_PATH = exe

const { createAgyAgent } = await import("../src/provider")
const { close, idle, run: execute } = await import("../src/sessions")
const { agent } = await import("../src/acp")
const { group, models } = await import("../src/catalog")
const { MARKER, exhausted, format } = await import("../src/usage")
const { decide, hashes } = await import("../src/sync")
const { fit, short } = await import("../src/tools")
const { failed } = await import("../src/stream")
const { KIND_HEADER, SESSION_HEADER, DIRECTORY_HEADER } = await import("../src/constants")

type Json = Record<string, any>

const user = (text: string): LanguageModelV3Message => ({ role: "user", content: [{ type: "text", text }] })
const assistant = (text: string): LanguageModelV3Message => ({ role: "assistant", content: [{ type: "text", text }] })
const result = (id: string, value: string): LanguageModelV3Message => ({
  role: "tool",
  content: [{ type: "tool-result", toolCallId: id, toolName: "read", output: { type: "text", value } }],
})
const tool = { type: "function", name: "read", inputSchema: { type: "object", properties: {} } } as const

const sleep = (ms: number) => new Promise((done) => setTimeout(done, ms))

let dir = ""
let log = ""
let scenario = ""
let session = ""

const logs = (): Json[] =>
  fs.existsSync(log)
    ? fs
        .readFileSync(log, "utf8")
        .split("\n")
        .filter(Boolean)
        .map((line) => JSON.parse(line))
    : []
const received = (method: string) => logs().filter((entry) => entry.in === method)
const until = async (check: () => boolean) => {
  const end = Date.now() + 5000
  while (!check()) {
    if (Date.now() > end) throw new Error("timed out waiting")
    await sleep(20)
  }
}
const script = (turns: Json[][]) => fs.writeFileSync(scenario, JSON.stringify(turns))

async function run(
  prompt: LanguageModelV3Prompt,
  opts: { kind?: string; model?: string; effort?: string; signal?: AbortSignal; tools?: LanguageModelV3FunctionTool[] } = {},
) {
  const kind = opts.kind ?? "primary"
  const headers =
    kind === "primary"
      ? { [KIND_HEADER]: kind, [SESSION_HEADER]: session, [DIRECTORY_HEADER]: dir }
      : { [KIND_HEADER]: kind, [DIRECTORY_HEADER]: dir }
  const { stream } = await createAgyAgent()
    .languageModel(opts.model ?? "gemini-3.8-flash")
    .doStream({
      prompt,
      headers,
      abortSignal: opts.signal,
      tools: kind === "primary" ? (opts.tools ?? [tool]) : undefined,
      providerOptions: opts.effort ? { "agy-agent": { effort: opts.effort } } : undefined,
    } as LanguageModelV3CallOptions)
  const parts: LanguageModelV3StreamPart[] = []
  const error = await stream
    .pipeTo(new WritableStream({ write: (part) => void parts.push(part) }))
    .then(() => undefined, (err: unknown) => err)
  return { parts, error, text: parts.flatMap((p) => (p.type === "text-delta" ? [p.delta] : [])).join("") }
}

const finish = (parts: LanguageModelV3StreamPart[]) => parts.find((p) => p.type === "finish") as Extract<LanguageModelV3StreamPart, { type: "finish" }>
const called = (parts: LanguageModelV3StreamPart[]) => parts.find((p) => p.type === "tool-call") as Extract<LanguageModelV3StreamPart, { type: "tool-call" }>
// The assistant message Kilo stores for a step that ended in a tool call.
const reply = (parts: LanguageModelV3StreamPart[], text: string): LanguageModelV3Message => {
  const call = called(parts)
  return {
    role: "assistant",
    content: [
      { type: "text", text },
      { type: "tool-call", toolCallId: call.toolCallId, toolName: call.toolName, input: JSON.parse(call.input) },
    ],
  }
}
const texts = (entry: Json) => (entry.params.prompt as Json[]).flatMap((b) => (b.type === "text" ? [b.text] : [])).join("\n")

describe("provider against fake agy", { timeout: 30_000 }, () => {
  beforeEach(() => {
    const id = randomUUID()
    dir = fs.mkdtempSync(path.join(root, "work-"))
    log = path.join(root, `${id}.log`)
    scenario = path.join(root, `${id}.json`)
    session = `ses_${id}`
    process.env.FAKE_AGY_LOG = log
    process.env.FAKE_AGY_SCENARIO = scenario
    script([])
  })

  afterEach(async () => {
    await close(dir)
    await (await agent(exe)).stop()
    delete process.env.KILO_AGY_DEADLINE_MS
    delete process.env.FAKE_AGY_AUTH_FAIL
    delete process.env.FAKE_AGY_EXHAUSTED
  })

  it("streams thought, text and final usage", async () => {
    script([
      [
        { thought: "hm" },
        { text: "hi" },
        { usage: { used: 100, size: 1048576 } },
        { stop: "end_turn", usage: { inputTokens: 10, outputTokens: 5, cachedReadTokens: 3, cachedWriteTokens: 0, thoughtTokens: 2 } },
      ],
    ])
    const out = await run([user("a")])
    expect(out.error).toBeUndefined()
    expect(out.parts.map((p) => p.type)).toEqual([
      "stream-start",
      "reasoning-start",
      "reasoning-delta",
      "reasoning-end",
      "text-start",
      "text-delta",
      "text-end",
      "finish",
    ])
    expect(out.text).toBe("hi")
    const end = finish(out.parts)
    expect(end.finishReason.unified).toBe("stop")
    // agy's turn-wide input sums its internal calls, so only the context occupancy is reported.
    expect(end.usage.inputTokens).toEqual({ total: 100, noCache: undefined, cacheRead: undefined, cacheWrite: undefined })
    expect(end.usage.outputTokens).toEqual({ total: 5, text: 3, reasoning: 2 })
  })

  it("parks a Kilo tool call, correlates it, and resolves it on the next step", async () => {
    script([[{ text: "let me" }, { usage: { used: 77, size: 1000 } }, { tool: { name: "read", args: { path: "a" } } }, { text: "done" }, { stop: "end_turn" }]])
    const first = await run([user("a")])
    expect(first.error).toBeUndefined()
    const call = called(first.parts)
    const permission = logs().find((entry) => entry.permission)!.permission
    expect(call.toolCallId).toBe(permission.id)
    expect(permission.answer).toEqual({ outcome: "selected", optionId: "allow" })
    expect(call.toolName).toBe("read")
    expect(JSON.parse(call.input)).toEqual({ path: "a" })
    expect(finish(first.parts).finishReason.unified).toBe("tool-calls")
    expect(finish(first.parts).usage.inputTokens.total).toBe(77)

    const second = await run([user("a"), reply(first.parts, "let me"), result(call.toolCallId, "content A")])
    expect(second.error).toBeUndefined()
    expect(second.text).toBe("done")
    expect(finish(second.parts).finishReason.unified).toBe("stop")
    expect(logs().find((entry) => entry.mcp)!.mcp.result.content).toEqual([{ type: "text", text: "content A" }])
    expect(received("session/prompt")).toHaveLength(1)
    expect(received("session/new")).toHaveLength(1)
  })

  it("denies permissions that are not Kilo tools", async () => {
    script([[{ permission: { title: "Run shell" } }, { text: "ok" }, { stop: "end_turn" }]])
    const out = await run([user("a")])
    expect(out.text).toBe("ok")
    expect(logs().find((entry) => entry.permission)!.permission.answer).toEqual({ outcome: "selected", optionId: "deny" })
  })

  it("hands a parked call off at the deadline and sends its result as text", async () => {
    process.env.KILO_AGY_DEADLINE_MS = "200"
    script([[{ tool: { name: "read", args: { path: "a" } } }, { stop: "end_turn" }], [{ text: "thanks" }, { stop: "end_turn" }]])
    const first = await run([user("a")])
    const call = called(first.parts)
    await until(() => received("session/cancel").length === 1)

    const second = await run([user("a"), reply(first.parts, ""), result(call.toolCallId, "late A")])
    expect(second.text).toBe("thanks")
    const prompts = received("session/prompt")
    expect(prompts).toHaveLength(2)
    expect(texts(prompts[1])).toContain("<tool-result>")
    expect(texts(prompts[1])).toContain(call.toolCallId)
    expect(texts(prompts[1])).toContain("late A")
    expect(received("session/new")).toHaveLength(1)
  })

  it("detaches parked calls on idle and carries them with the next input", async () => {
    script([[{ tool: { name: "read", args: { path: "a" } } }, { stop: "end_turn" }], [{ text: "next" }, { stop: "end_turn" }]])
    const first = await run([user("a")])
    const call = called(first.parts)
    idle(session)
    await until(() => received("session/cancel").length === 1)
    const second = await run([user("a"), reply(first.parts, ""), result(call.toolCallId, "late B"), user("go on")])
    expect(second.text).toBe("next")
    const last = texts(received("session/prompt").at(-1)!)
    expect(last).toContain("late B")
    expect(last).toContain("go on")
  })

  it("sets model and effort only when they change", async () => {
    script([[{ text: "r1" }], [{ text: "r2" }], [{ text: "r3" }]])
    const options = () => received("session/set_config_option").map((entry) => [entry.params.configId, entry.params.value])
    const prompt = [user("a")]
    await run(prompt, { effort: "low" })
    expect(options()).toEqual([
      ["model", "gemini-3.8-flash"],
      ["thought_level", "low"],
    ])
    prompt.push(assistant("r1"), user("b"))
    await run(prompt, { effort: "low" })
    expect(options()).toHaveLength(2)
    prompt.push(assistant("r2"), user("c"))
    await run(prompt, { effort: "high" })
    expect(options().at(-1)).toEqual(["thought_level", "high"])
    expect(options()).toHaveLength(3)
    expect(received("session/new")).toHaveLength(1)
  })

  it("replays tagged history in a new session when history diverges", async () => {
    script([[{ text: "r1" }], [{ text: "r2" }]])
    await run([user("a")])
    const out = await run([user("a"), assistant("edited"), user("b")])
    expect(out.text).toBe("r2")
    expect(received("session/new")).toHaveLength(2)
    const replay = texts(received("session/prompt").at(-1)!)
    expect(replay).toContain("<assistant>")
    expect(replay).toContain("edited")
    expect(replay).toContain("<user>")
  })

  it("resumes with the new system prompt when it changes", async () => {
    script([[{ text: "r1" }], [{ text: "r2" }]])
    const prompt: LanguageModelV3Prompt = [{ role: "system", content: "S1" }, user("a")]
    await run(prompt)
    const id = received("session/prompt")[0].params.sessionId
    const out = await run([{ role: "system", content: "S2" }, user("a"), assistant("r1"), user("b")])
    expect(out.error).toBeUndefined()
    expect(out.text).toBe("r2")
    expect(received("session/close").map((entry) => entry.params.sessionId)).toContain(id)
    const resume = received("session/resume")
    expect(resume).toHaveLength(1)
    expect(resume[0].params.sessionId).toBe(id)
    expect(resume[0].params._meta.antigravity.systemPrompt.append).toBe("S2")
    expect(texts(received("session/prompt").at(-1)!)).toBe("b")
  })

  it("appends long tool descriptions to the system prompt", async () => {
    script([[{ text: "r1" }]])
    const long = { ...tool, name: "task", description: `Launch an agent.\n\n${"x".repeat(2000)}` }
    await run([{ role: "system", content: "S" }, user("a")], { tools: [tool, long] })
    expect(received("session/new")[0].params._meta.antigravity.systemPrompt.append).toBe(
      `S\n\n# Kilo tool instructions\n\n## mcp_kilo_task\n\n${long.description}`,
    )
  })

  it("fails retryably when agy exits mid-turn and resumes on the next call", async () => {
    script([[{ text: "a" }, { exit: true }], [{ text: "back" }, { stop: "end_turn" }]])
    const first = await run([user("q")])
    expect(first.error).toMatchObject({ isRetryable: true })
    const id = received("session/prompt")[0].params.sessionId
    const second = await run([user("q")])
    expect(second.error).toBeUndefined()
    expect(second.text).toBe("back")
    expect(received("initialize")).toHaveLength(2)
    expect(received("session/new")).toHaveLength(1)
    expect(received("session/resume").map((entry) => entry.params.sessionId)).toEqual([id])
  })

  it("runs utility calls in a throwaway session without tools", async () => {
    script([[{ text: "Title" }, { stop: "end_turn" }]])
    const out = await run([{ role: "system", content: "make a title" }, user("a")], { kind: "utility" })
    expect(out.text).toBe("Title")
    const open = received("session/new")
    expect(open).toHaveLength(1)
    expect(open[0].params.mcpServers).toEqual([])
    expect(open[0].params._meta.antigravity.systemPrompt.append).toBe("make a title")
    const id = received("session/prompt")[0].params.sessionId
    expect(received("session/close").map((entry) => entry.params.sessionId)).toEqual([id])
  })

  it("restarts agy once on -32000", async () => {
    process.env.FAKE_AGY_AUTH_FAIL = "1"
    script([[{ text: "ok" }, { stop: "end_turn" }]])
    const out = await run([user("a")])
    expect(out.error).toBeUndefined()
    expect(out.text).toBe("ok")
    expect(received("initialize")).toHaveLength(2)
    expect(received("session/new")).toHaveLength(2)
  })

  it("passes images through as image blocks and rejects other files", async () => {
    script([[{ text: "seen" }, { stop: "end_turn" }]])
    const data = new Uint8Array([1, 2, 3])
    const out = await run([
      { role: "user", content: [{ type: "text", text: "look" }, { type: "file", mediaType: "image/png", data }] },
    ])
    expect(out.text).toBe("seen")
    expect(received("session/prompt")[0].params.prompt).toEqual([
      { type: "text", text: "look" },
      { type: "image", mimeType: "image/png", data: Buffer.from(data).toString("base64") },
    ])
    const bad = await run([{ role: "user", content: [{ type: "file", mediaType: "application/pdf", data: "AA==" }] }])
    expect((bad.error as Error).message).toContain("Antigravity does not support application/pdf attachments")
  })

  it("answers /agy-usage without an agy turn", async () => {
    const out = await run([user(MARKER)])
    expect(out.text).toContain("Gemini Models")
    expect(out.text).toContain("91% remaining")
    expect(received("session/prompt")).toHaveLength(0)
  })

  it("cancels on Kilo abort and continues the same conversation", async () => {
    script([
      [{ text: "partial" }, { wait: 10_000 }, { stop: "end_turn" }],
      [{ tool: { name: "read", args: { path: "a" } } }, { wait: 10_000 }, { stop: "end_turn" }],
      [{ text: "after" }, { stop: "end_turn" }],
    ])
    const abort = async (prompt: LanguageModelV3Prompt, cancels: number) => {
      const controller = new AbortController()
      const pending = run(prompt, { signal: controller.signal })
      await until(() => received("session/prompt").length > 0)
      await sleep(200)
      controller.abort()
      const out = await pending
      expect(out.error).toBeUndefined()
      await until(() => received("session/cancel").length === cancels)
      return out
    }
    const text = await abort([user("a")], 1)
    expect(text.text).toBe("partial")
    expect(finish(text.parts)).toBeUndefined()

    const prompt = [user("a"), assistant("partial"), user("b")]
    const step = await run(prompt)
    const call = called(step.parts)
    // Kilo aborts while agy continues the parked call's turn.
    const parked = await abort([...prompt, reply(step.parts, ""), result(call.toolCallId, "content A")], 2)
    expect(finish(parked.parts)).toBeUndefined()
    expect(logs().find((entry) => entry.mcp)!.mcp.result.content).toEqual([{ type: "text", text: "content A" }])

    const out = await run([...prompt, reply(step.parts, ""), result(call.toolCallId, "content A"), user("c")])
    expect(out.error).toBeUndefined()
    expect(out.text).toBe("after")
    expect(received("session/new")).toHaveLength(1)
    expect(received("session/resume")).toHaveLength(0)
    expect(texts(received("session/prompt").at(-1)!)).toBe("c")
  })

  it("ends the agy turn when a step fails locally, so the next prompt finds the session free", async () => {
    script([[{ tool: { name: "read", args: { path: "a" } } }, { wait: 10_000 }, { stop: "end_turn" }], [{ text: "fine" }, { stop: "end_turn" }]])
    const failing = execute(
      {
        sessionID: session,
        directory: dir,
        model: "gemini-3.8-flash",
        system: "",
        prompt: [user("a")],
        tools: [tool],
        emit: (part) => {
          if (part.type === "tool-call") throw new Error("boom")
        },
      },
      new AbortController().signal,
    )
    await expect(failing).rejects.toThrow("boom")
    expect(received("session/cancel")).toHaveLength(1)
    // The fake rejects a prompt while a cancelled one is still running.
    const out = await run([user("a"), assistant("x"), user("b")])
    expect(out.error).toBeUndefined()
    expect(out.text).toBe("fine")
    expect(received("session/new")).toHaveLength(1)
  })

  it("sends remote image results as text and ends the turn on unsupported results", async () => {
    script([
      [{ tool: { name: "read", args: { path: "a" } } }, { tool: { name: "read", args: { path: "b" } } }, { wait: 10_000 }, { stop: "end_turn" }],
      [{ text: "recovered" }, { stop: "end_turn" }],
    ])
    const output = (value: Json[]): LanguageModelV3Message => ({
      role: "tool",
      content: [{ type: "tool-result", toolCallId: "", toolName: "read", output: { type: "content", value } as never }],
    })
    const withId = (message: LanguageModelV3Message, id: string) => {
      const part = (message.content as Json[])[0]
      part.toolCallId = id
      return message
    }
    const first = await run([user("a")])
    const one = called(first.parts).toolCallId
    const remote = withId(output([{ type: "image-url", url: "https://example.com/a.png" }]), one)
    const prompt = [user("a"), reply(first.parts, ""), remote]
    const second = await run(prompt)
    expect(second.error).toBeUndefined()
    const results = logs().filter((entry) => entry.mcp)
    expect(results[0].mcp.result.content).toEqual([{ type: "text", text: "[image: https://example.com/a.png]" }])
    const two = called(second.parts).toolCallId

    const pdf = withId(output([{ type: "file-data", mediaType: "application/pdf", data: "AA==" }]), two)
    const bad = await run([...prompt, reply(second.parts, ""), pdf])
    expect((bad.error as Error).message).toContain("Antigravity does not support application/pdf attachments")
    expect(received("session/cancel")).toHaveLength(1)

    const out = await run([...prompt, reply(second.parts, ""), pdf, user("go on")])
    expect(out.error).toBeUndefined()
    expect(out.text).toBe("recovered")
    expect(received("session/new")).toHaveLength(1)
  })

  it("keeps the conversation when Kilo prunes an old tool result", async () => {
    script([[{ tool: { name: "read", args: { path: "a" } } }, { text: "read it" }, { stop: "end_turn" }], [{ text: "r2" }, { stop: "end_turn" }]])
    const first = await run([user("a")])
    const id = called(first.parts).toolCallId
    await run([user("a"), reply(first.parts, ""), result(id, "long content")])
    const out = await run([user("a"), reply(first.parts, ""), result(id, "[Old tool result content cleared]"), assistant("read it"), user("b")])
    expect(out.text).toBe("r2")
    expect(received("session/new")).toHaveLength(1)
    expect(texts(received("session/prompt").at(-1)!)).toBe("b")
  })

  // Last in this block: it leaves an exhausted /usage answer in the shared 60 s cache.
  it("fails non-retryably on quota errors with the exhausted buckets", async () => {
    process.env.FAKE_AGY_EXHAUSTED = "1"
    script([[{ error: { code: -32603, message: "RESOURCE_EXHAUSTED: quota exceeded" } }]])
    const out = await run([user("a")])
    expect(out.error).toMatchObject({ isRetryable: false })
    const message = (out.error as Error).message
    expect(message).toContain("RESOURCE_EXHAUSTED: quota exceeded")
    expect(message).toMatch(/Antigravity quota exhausted\. Gemini Models: Weekly Limit Remaining, resets /)
    expect(message).not.toContain("Claude")
  })
})

describe("auth error", () => {
  it("is non-retryable with a login hint", () => {
    const error = failed(Object.assign(new Error("Authentication required"), { code: -32000 }))
    expect(error.isRetryable).toBe(false)
    expect(error.message).toContain("Run `agy` in a terminal")
  })
})

describe("catalog", () => {
  it("assumes a 1M window for Gemini models until agy reports one", () => {
    expect(models()["gemini-3.8-flash"].limit.context).toBe(1_048_576)
  })

  it("groups the real model list by effort suffix", () => {
    const items = group(execFileSync(exe, ["models"], { encoding: "utf8" }))
    expect(items).toEqual([
      { id: "gemini-4-hydrogen", name: "Gemini 4 Hydrogen", efforts: ["low", "medium", "high", "xhigh"] },
      { id: "gemini-3.8-flash", name: "Gemini 3.8 Flash", efforts: ["low", "medium", "high"] },
      { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6 (Thinking)", efforts: [] },
      { id: "claude-opus-4-6-thinking", name: "Claude Opus 4.6 (Thinking)", efforts: [] },
      { id: "gpt-oss-120b-medium", name: "GPT-OSS 120B (Medium)", efforts: [] },
    ])
  })
})

describe("usage", () => {
  const fixture = () => JSON.parse(execFileSync(exe, ["-p=/usage", "--output-format", "json"], { encoding: "utf8" })).command.data
  let usage: Json

  beforeAll(() => {
    usage = fixture()
  })

  it("formats groups and skips epoch resets", () => {
    const text = format(usage as never)
    expect(text).toContain("**Gemini Models**")
    expect(text).toMatch(/Weekly Limit Remaining: 91% remaining, resets /)
    expect(text).toContain("- Weekly Limit Remaining: 100% remaining\n")
    expect(text.match(/resets/g)).toHaveLength(1)
  })

  it("reports only exhausted buckets", () => {
    expect(exhausted(usage as never)).toBeUndefined()
    usage.groups[0].buckets[0].remaining_fraction = 0
    const text = exhausted(usage as never)!
    expect(text).toMatch(/^Antigravity quota exhausted\. Gemini Models: Weekly Limit Remaining, resets /)
    expect(text).not.toContain("Claude")
  })
})

describe("decide", () => {
  const record = (prompt: LanguageModelV3Prompt) => ({ conversation: "c", hashes: hashes(prompt), system: "", tools: "" })
  const call: LanguageModelV3Message = {
    role: "assistant",
    content: [{ type: "tool-call", toolCallId: "t1", toolName: "read", input: { a: 1 } }],
  }
  const seen = [user("a"), call]

  it("rebuilds without a record, on divergence, on a shorter prompt and on regenerate", () => {
    expect(decide([user("a")], undefined, [])).toEqual({ type: "rebuild" })
    expect(decide([user("b"), user("c")], record([user("a")]), [])).toEqual({ type: "rebuild" })
    expect(decide([user("a")], record(seen), [])).toEqual({ type: "rebuild" })
    expect(decide(seen, record(seen), [])).toEqual({ type: "rebuild" })
  })

  it("ignores reasoning and system messages when comparing", () => {
    const thinking: LanguageModelV3Message = { role: "assistant", content: [{ type: "reasoning", text: "x" }, ...call.content] }
    const prompt = [{ role: "system" as const, content: "s" }, user("a"), thinking, result("t1", "ok")]
    expect(decide(prompt, record(seen), ["t1"]).type).toBe("results")
  })

  it("delivers parked results only when the tail is just those results", () => {
    const tail = result("t1", "ok")
    expect(decide([...seen, tail], record(seen), ["t1"])).toEqual({ type: "results", results: tail.content })
    expect(decide([...seen, tail], record(seen), [])).toEqual({ type: "continue", tail: [tail] })
    expect(decide([...seen, tail, user("more")], record(seen), ["t1"])).toEqual({ type: "continue", tail: [tail, user("more")] })
  })
})

describe("fit", () => {
  it("stubs long descriptions, keeps short ones and drops $schema", () => {
    const schema = { $schema: "https://json-schema.org/draft/2020-12/schema", type: "object", properties: {} } as const
    const long = { ...tool, name: "task", description: `Launch an agent.\n\n${"x".repeat(2000)}`, inputSchema: schema }
    const out = fit([{ ...tool, description: "Read a file.", inputSchema: schema }, long])
    expect(out.tools.map((t) => t.description)).toEqual([
      "Read a file.",
      'Launch an agent.\n\nFull instructions: the "mcp_kilo_task" section of "Kilo tool instructions" in the system prompt.',
    ])
    expect(out.tools.map((t) => t.inputSchema)).toEqual([tool.inputSchema, tool.inputSchema])
    expect(out.manual).toBe(`\n\n# Kilo tool instructions\n\n## mcp_kilo_task\n\n${long.description}`)
    expect(fit([tool]).manual).toBe("")
  })
})

describe("short", () => {
  it("keeps valid names and fits long or invalid ones under the MCP limit", () => {
    expect(short("read")).toBe("read")
    const long = short("x".repeat(80))
    expect(long).toHaveLength(64 - "mcp_kilo_".length)
    expect(long).toMatch(/^x+_[0-9a-f]{10}$/)
    expect(short("a.b")).toMatch(/^a_b_[0-9a-f]{10}$/)
    expect(short("a.b")).not.toBe(short("a:b"))
    expect(short("x".repeat(80))).toBe(long)
  })
})
