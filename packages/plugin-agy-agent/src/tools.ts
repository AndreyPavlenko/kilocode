import { createHash, randomBytes, timingSafeEqual } from "node:crypto"
import http from "node:http"
import type { AddressInfo } from "node:net"
import type {
  LanguageModelV3FunctionTool,
  LanguageModelV3ToolResultOutput,
  LanguageModelV3ToolResultPart,
} from "@ai-sdk/provider"
import type * as schema from "@agentclientprotocol/sdk"
import { Server } from "@modelcontextprotocol/sdk/server/index.js"
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js"
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js"
import type { CallToolResult, Tool } from "@modelcontextprotocol/sdk/types.js"
import { SERVER } from "./constants"
import { content } from "./message"

const PREFIX = `mcp_${SERVER}_`
const TITLE = `Run ${PREFIX}`
const CANCELLED: CallToolResult = { isError: true, content: [{ type: "text", text: "cancelled" }] }
const DATA_URL = /^data:([^;,]+);base64,(.*)$/

export function short(name: string) {
  const limit = 64 - PREFIX.length
  if (name.length <= limit && /^[a-zA-Z0-9_-]+$/.test(name)) return name
  const hash = createHash("sha256").update(name).digest("hex").slice(0, 10)
  return name.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, limit - 11) + "_" + hash
}

// agy silently hides the largest MCP tools once their declarations outgrow its server-set budget
// (about 58K characters seen), so long descriptions move to the system prompt behind a short stub.
const LIMIT = 1000
const WARN = 50_000

export function fit(tools: readonly LanguageModelV3FunctionTool[]) {
  const moved = tools.filter((tool) => (tool.description?.length ?? 0) > LIMIT)
  const fitted = tools.map((tool) => ({
    ...tool,
    description: moved.includes(tool)
      ? `${tool.description!.split("\n\n")[0].slice(0, LIMIT)}\n\nFull instructions: the "${PREFIX + short(tool.name)}" section of "Kilo tool instructions" in the system prompt.`
      : tool.description,
    inputSchema: Object.fromEntries(
      Object.entries(tool.inputSchema).filter(([key]) => key !== "$schema"),
    ) as LanguageModelV3FunctionTool["inputSchema"],
  }))
  const size = fitted.reduce((sum, tool) => sum + (tool.description?.length ?? 0) + JSON.stringify(tool.inputSchema).length, 0)
  if (size > WARN)
    console.warn(`[agy-agent] ${fitted.length} tool declarations take ${size} characters; agy hides the largest ones past its budget`)
  const manual = moved.map((tool) => `## ${PREFIX + short(tool.name)}\n\n${tool.description}`).join("\n\n")
  return { tools: fitted, manual: manual && `\n\n# Kilo tool instructions\n\n${manual}` }
}

// Undefined for a tool we did not offer, such as agy's own manage_task.
export function original(name: string, tools: readonly LanguageModelV3FunctionTool[]) {
  return tools.find((tool) => PREFIX + short(tool.name) === name)?.name
}

export interface Call {
  id: string
  // Kilo tool name.
  name: string
  input: unknown
  // When the MCP call arrived; agy's 180 s deadline counts from here.
  at: number
}

export interface Bridge {
  readonly url: string
  readonly headers: Record<string, string>
  announce(call: schema.ToolCallUpdate): boolean
  // Every session update, so announcements agy dropped without calling us do not linger.
  observe(update: schema.SessionUpdate): void
  readonly calls: AsyncIterable<Call>
  parked(): string[]
  resolve(results: readonly LanguageModelV3ToolResultPart[]): void
  expire(): string[]
  close(): void
}

interface Route {
  secret: Buffer
  server: () => Server
}

const routes = new Map<string, Route>()
let shared: Promise<{ server: http.Server; url: string }> | undefined

// Stateless: agy's MCP client is served by a fresh MCP server and transport per HTTP request.
async function serve(req: http.IncomingMessage, res: http.ServerResponse) {
  const route = routes.get(new URL(req.url ?? "/", "http://localhost").pathname.slice(1))
  if (!route) {
    res.writeHead(404).end()
    return
  }
  const given = Buffer.from(req.headers.authorization ?? "")
  const expected = Buffer.concat([Buffer.from("Bearer "), route.secret])
  if (given.length !== expected.length || !timingSafeEqual(given, expected)) {
    res.writeHead(401).end()
    return
  }
  const server = route.server()
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined })
  // Closing aborts the handlers' signals, so a call agy gave up on leaves the parked set.
  res.on("close", () => {
    void transport.close()
    void server.close()
  })
  await server.connect(transport)
  await transport.handleRequest(req, res)
}

function listen() {
  if (shared) return shared
  const next = new Promise<{ server: http.Server; url: string }>((resolve, reject) => {
    const server = http.createServer((req, res) =>
      serve(req, res).catch((err) => {
        console.warn("[agy-agent] MCP request failed:", err instanceof Error ? err.message : err)
        if (!res.headersSent) res.writeHead(500)
        res.end()
      }),
    )
    server.once("error", reject)
    server.on("error", (err) => console.warn("[agy-agent] MCP server error:", err.message))
    server.listen(0, "127.0.0.1", () => {
      server.unref()
      resolve({ server, url: `http://127.0.0.1:${(server.address() as AddressInfo).port}` })
    })
  })
  shared = next
  next.catch(() => {
    if (shared === next) shared = undefined
  })
  return next
}

export async function bridge(tools: readonly LanguageModelV3FunctionTool[]): Promise<Bridge> {
  const token = randomBytes(16).toString("hex")
  const secret = randomBytes(32).toString("hex")
  const names = new Map(tools.map((tool) => [short(tool.name), tool.name]))
  const definitions = tools.map(
    (tool): Tool => ({
      name: short(tool.name),
      description: tool.description,
      inputSchema: tool.inputSchema as Tool["inputSchema"],
    }),
  )
  // agy asks permission before every MCP call (we never pass --dangerously-skip-permissions),
  // so each call takes the oldest announcement with its name.
  const announced: Array<{ id: string; name: string }> = []
  const parked = new Map<string, PromiseWithResolvers<CallToolResult>>()
  const arrived: Call[] = []
  const state: { waiter?: PromiseWithResolvers<IteratorResult<Call>>; closed: boolean } = { closed: false }

  const push = (call: Call) => {
    const waiter = state.waiter
    state.waiter = undefined
    if (waiter) return waiter.resolve({ done: false, value: call })
    arrived.push(call)
  }
  const settle = (id: string, result: CallToolResult) => {
    parked.get(id)?.resolve(result)
    parked.delete(id)
    const index = arrived.findIndex((call) => call.id === id)
    if (index >= 0) arrived.splice(index, 1)
  }
  const expire = () => {
    const ids = [...parked.keys()]
    ids.forEach((id) => settle(id, CANCELLED))
    announced.length = 0
    return ids
  }

  const server = () => {
    const instance = new Server({ name: SERVER, version: "1.0.0" }, { capabilities: { tools: {} } })
    instance.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: definitions }))
    instance.setRequestHandler(CallToolRequestSchema, async (request, extra) => {
      const name = names.get(request.params.name)
      const index = announced.findIndex((entry) => entry.name === name)
      if (!name || index < 0) throw new Error(`Antigravity called Kilo tool ${request.params.name} without permission`)
      const id = announced.splice(index, 1)[0].id
      const result = Promise.withResolvers<CallToolResult>()
      parked.set(id, result)
      extra.signal.addEventListener("abort", () => parked.get(id) === result && settle(id, CANCELLED), { once: true })
      push({ id, name, input: request.params.arguments ?? {}, at: Date.now() })
      return result.promise
    })
    return instance
  }
  // Registered before listening, so a concurrent close() of another bridge keeps the shared server.
  routes.set(token, { secret: Buffer.from(secret), server })
  const host = await listen().catch((err) => {
    routes.delete(token)
    throw err
  })

  return {
    url: `${host.url}/${token}`,
    headers: { authorization: `Bearer ${secret}` },
    announce(call) {
      const title = call.title ?? ""
      const name = names.get(title.slice(TITLE.length))
      if (!title.startsWith(TITLE) || !name) return false
      announced.push({ id: call.toolCallId, name })
      return true
    },
    observe(update) {
      if (update.sessionUpdate !== "tool_call" && update.sessionUpdate !== "tool_call_update") return
      if (update.status !== "failed" && update.status !== "completed") return
      const index = announced.findIndex((entry) => entry.id === update.toolCallId)
      if (index >= 0) announced.splice(index, 1)
    },
    calls: {
      [Symbol.asyncIterator]: () => ({
        next() {
          const call = arrived.shift()
          if (call) return Promise.resolve({ done: false, value: call })
          if (state.closed) return Promise.resolve({ done: true, value: undefined })
          state.waiter = Promise.withResolvers()
          return state.waiter.promise
        },
      }),
    },
    parked: () => [...parked.keys()],
    resolve(results) {
      // Convert all first: a bad result must not leave the others half-delivered.
      results
        .map((part) => [part.toolCallId, convert(part.output)] as const)
        .forEach(([id, result]) => settle(id, result))
    },
    expire,
    close() {
      expire()
      state.closed = true
      state.waiter?.resolve({ done: true, value: undefined })
      state.waiter = undefined
      routes.delete(token)
      const current = shared
      if (routes.size > 0 || !current) return
      shared = undefined
      void current.then((item) => {
        item.server.closeAllConnections()
        item.server.close()
      })
    },
  }
}

function convert(output: LanguageModelV3ToolResultOutput): CallToolResult {
  const failed = output.type === "error-text" || output.type === "error-json" || output.type === "execution-denied"
  return { isError: failed, content: blocks(output) }
}

function blocks(output: LanguageModelV3ToolResultOutput): CallToolResult["content"] {
  if (output.type === "text" || output.type === "error-text") return [{ type: "text", text: output.value }]
  if (output.type === "json" || output.type === "error-json") return [{ type: "text", text: JSON.stringify(output.value) }]
  if (output.type === "execution-denied") {
    return [{ type: "text", text: output.reason ?? "Tool execution was denied by the user." }]
  }
  return output.value.map((part): CallToolResult["content"][number] => {
    if (part.type === "text") return { type: "text", text: part.text }
    if (part.type === "image-url") {
      const match = DATA_URL.exec(part.url)
      // agy takes inline images only; a remote one stays a reference the model can mention.
      if (!match) return { type: "text", text: `[image: ${part.url}]` }
      return { type: "image", mimeType: match[1], data: match[2] }
    }
    if (part.type === "image-data" || part.type === "file-data") {
      // Same attachment rules as prompt files: images only.
      const block = content([{ role: "user", content: [{ type: "file", mediaType: part.mediaType, data: part.data }] }])[0]
      if (block?.type === "image") return { type: "image", mimeType: block.mimeType, data: block.data }
    }
    throw new Error(`Antigravity does not support ${part.type} tool output`)
  })
}
