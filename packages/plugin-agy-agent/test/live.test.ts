import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { setTimeout as sleep } from "node:timers/promises"
import type {
  LanguageModelV3CallOptions,
  LanguageModelV3FunctionTool,
  LanguageModelV3Message,
  LanguageModelV3Prompt,
  LanguageModelV3StreamPart,
} from "@ai-sdk/provider"
import { afterAll, describe, expect, it } from "vitest"

// Real agy against the user's subscription: opt in with KILO_AGY_LIVE=1.
const live = process.env.KILO_AGY_LIVE === "1"
const root = fs.mkdtempSync(path.join(os.tmpdir(), "agy-live-"))
process.env.XDG_DATA_HOME = path.join(root, "data")

const { createAgyAgent } = await import("../src/provider")
const { close, forget } = await import("../src/sessions")
const { DATA, DIRECTORY_HEADER, KIND_HEADER, SESSION_HEADER } = await import("../src/constants")

const session = `ses_live_${Date.now()}`
const dir = fs.mkdtempSync(path.join(root, "work-"))
const tool: LanguageModelV3FunctionTool = {
  type: "function",
  name: "echo",
  description: "Echoes the given text back.",
  inputSchema: {
    type: "object",
    properties: { text: { type: "string" } },
    required: ["text"],
    additionalProperties: false,
  },
}

async function run(prompt: LanguageModelV3Prompt, utility = false, id = session) {
  const { stream } = await createAgyAgent()
    .languageModel("gemini-3.8-flash")
    .doStream({
      prompt,
      headers: utility
        ? { [KIND_HEADER]: "utility", [DIRECTORY_HEADER]: dir }
        : { [KIND_HEADER]: "primary", [SESSION_HEADER]: id, [DIRECTORY_HEADER]: dir },
      tools: utility ? undefined : [tool],
      providerOptions: { "agy-agent": { effort: "low" } },
    } as LanguageModelV3CallOptions)
  const parts: LanguageModelV3StreamPart[] = []
  await stream.pipeTo(new WritableStream({ write: (part) => void parts.push(part) }))
  return parts
}

const record = (id: string) => JSON.parse(fs.readFileSync(path.join(DATA, "sessions", `${id}.json`), "utf8"))

describe.skipIf(!live)("live agy", { timeout: 180_000 }, () => {
  afterAll(async () => {
    await forget(session)
    await forget(`${session}_late`)
    await close(dir)
    delete process.env.KILO_AGY_DEADLINE_MS
    fs.rmSync(root, { recursive: true, force: true })
  })

  // A late result arrives after agy's turn was cancelled at the deadline, as with a long-running subagent.
  it.each([
    ["in time", session, 0],
    ["after the deadline handoff", `${session}_late`, 1000],
  ])("calls a Kilo tool and continues the same conversation with its result %s", async (_, id, deadline) => {
    if (deadline) process.env.KILO_AGY_DEADLINE_MS = String(deadline)
    const system: LanguageModelV3Message = { role: "system", content: "You are a test assistant. Follow instructions exactly." }
    const user: LanguageModelV3Message = {
      role: "user",
      content: [
        {
          type: "text",
          text: 'Call the echo tool with text "kiwi-4711". After it returns, reply with exactly the text it returned.',
        },
      ],
    }
    const first = await run([system, user], false, id)
    const call = first.find((p) => p.type === "tool-call")
    expect(call).toBeDefined()
    if (call?.type !== "tool-call") return
    expect(call.toolName).toBe("echo")
    const conversation = record(id).conversation
    if (deadline) await sleep(deadline * 5)

    const text = first.flatMap((p) => (p.type === "text-delta" ? [p.delta] : [])).join("")
    const assistant: LanguageModelV3Message = {
      role: "assistant",
      content: [
        ...(text ? [{ type: "text" as const, text }] : []),
        { type: "tool-call", toolCallId: call.toolCallId, toolName: call.toolName, input: JSON.parse(call.input) },
      ],
    }
    const result: LanguageModelV3Message = {
      role: "tool",
      content: [
        { type: "tool-result", toolCallId: call.toolCallId, toolName: "echo", output: { type: "text", value: "plum-2093" } },
      ],
    }
    const second = await run([system, user, assistant, result], false, id)
    const answer = second.flatMap((p) => (p.type === "text-delta" ? [p.delta] : [])).join("")
    expect(answer).toContain("plum-2093")
    const finish = second.find((p) => p.type === "finish")
    expect(finish?.type === "finish" && finish.finishReason.unified).toBe("stop")
    expect(record(id).conversation).toBe(conversation)
  })

  it("answers a utility request without leaving a conversation behind", async () => {
    const conversations = path.join(DATA, "gemini", "antigravity-cli", "conversations")
    const before = fs.readdirSync(conversations)
    const parts = await run(
      [{ role: "user", content: [{ type: "text", text: "Generate a short title for: reading notes.txt" }] }],
      true,
    )
    expect(parts.flatMap((p) => (p.type === "text-delta" ? [p.delta] : [])).join("").trim()).not.toBe("")
    expect(fs.readdirSync(conversations)).toEqual(before)
  })
})
