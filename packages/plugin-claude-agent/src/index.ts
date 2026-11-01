import type { Hooks, Plugin } from "@kilocode/plugin"
import { models, refresh } from "./catalog"
import { DIRECTORY_HEADER, ID, KIND_HEADER, SESSION_HEADER, kind } from "./constants"
import { close, forget, idle, settled } from "./sessions"
import { MARKER, updateTitle } from "./usage"

function install(config: any) {
  const saved = config.provider?.[ID] ?? {}
  config.provider = {
    ...config.provider,
    [ID]: {
      ...saved,
      name: saved.name ?? "Claude Agent",
      npm: saved.npm ?? new URL("./provider.ts", import.meta.url).href,
      models: { ...models(), ...saved.models },
    },
  }
  config.command = {
    "claude-usage": {
      description: "Show Claude plan limits and reset times",
      template: MARKER,
      model: `${ID}/${Object.keys(config.provider[ID].models)[0]}`,
    },
    ...config.command,
  }
}

export const ClaudeAgentPlugin: Plugin = async (input): Promise<Hooks> => ({
  config: async (config) => {
    install(config)
    refresh()
  },
  "chat.params": async (hook, output) => {
    if (hook.model.providerID !== ID) return
    // Claude picks its own budget. Kilo's per-request cap would also restart the process on every change.
    output.maxOutputTokens = undefined
  },
  "chat.headers": async (hook, output) => {
    if (hook.model.providerID !== ID) return
    output.headers[KIND_HEADER] = kind(hook.agent)
    output.headers[SESSION_HEADER] = hook.sessionID
    output.headers[DIRECTORY_HEADER] = input.directory
  },
  event: async ({ event }) => {
    if (event.type === "session.idle") {
      const sessionID = event.properties.sessionID
      idle(sessionID)
      const turn = settled(sessionID)
      if (turn) {
        // Runs after the turn so it can't delay it. A failure keeps the current title.
        void updateTitle(input.client, input.directory, sessionID, turn.usage).catch((err) =>
          console.warn("[claude-agent] could not update limits:", err instanceof Error ? err.message : err),
        )
      }
    }
    if (event.type === "session.deleted") await forget(event.properties.info.id)
  },
  dispose: () => close(input.directory),
})

export default ClaudeAgentPlugin
