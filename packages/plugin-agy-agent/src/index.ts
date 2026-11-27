import type { Config, Hooks, Plugin } from "@kilocode/plugin"
import { models, refresh } from "./catalog"
import { DIRECTORY_HEADER, ID, KIND_HEADER, SESSION_HEADER, kind } from "./constants"
import { close, forget, idle } from "./sessions"
import { MARKER } from "./usage"

function install(config: Config) {
  const saved = config.provider?.[ID] ?? {}
  const list = { ...models(), ...saved.models }
  config.provider = {
    ...config.provider,
    [ID]: {
      ...saved,
      name: saved.name ?? "Antigravity Agent",
      npm: saved.npm ?? new URL("./provider.ts", import.meta.url).href,
      models: list,
    },
  }
  config.command = {
    "agy-usage": {
      description: "Show Antigravity plan limits and reset times",
      template: MARKER,
      model: `${ID}/${Object.keys(list)[0]}`,
    },
    ...config.command,
  }
}

export const AgyAgentPlugin: Plugin = async (input): Promise<Hooks> => ({
  config: async (config) => {
    install(config)
    refresh()
  },
  "chat.params": async (hook, output) => {
    if (hook.model.providerID !== ID) return
    // agy picks its own budget.
    output.maxOutputTokens = undefined
  },
  "chat.headers": async (hook, output) => {
    if (hook.model.providerID !== ID) return
    output.headers[KIND_HEADER] = kind(hook.agent)
    output.headers[SESSION_HEADER] = hook.sessionID
    output.headers[DIRECTORY_HEADER] = input.directory
  },
  event: async ({ event }) => {
    if (event.type === "session.idle") idle(event.properties.sessionID)
    if (event.type === "session.deleted") await forget(event.properties.info.id)
  },
  dispose: () => close(input.directory),
})

export default AgyAgentPlugin
