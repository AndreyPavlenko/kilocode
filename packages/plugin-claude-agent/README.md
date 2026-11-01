# @kilocode/plugin-claude-agent

Runs Claude models in Kilo Code through your own Claude Code login, using the Claude Agent SDK. Based on upstream OpenCode's built-in `claude-agent` provider, packaged as a plugin with no core changes.

Kilo's tools, permissions and history stay in charge: Claude Code's built-in tools, settings, CLAUDE.md files, hooks, MCP servers, auto memory and auto compaction are all turned off.

## Requirements

- Claude Code 2.1.248 or later on the machine running Kilo, signed in with `claude auth login`.
- `claude` on `PATH` or in `~/.local/bin`. Otherwise set `KILO_CLAUDE_PATH` or `provider.claude-agent.options.claudePath`.

## Install from this checkout

```sh
cd packages/plugin-claude-agent
dest=~/.local/share/kilo/plugins/claude-agent
mkdir -p "$dest" && bun pm pack --destination "$dest"
cd "$dest" && tar xzf kilocode-plugin-claude-agent-*.tgz --strip-components=1 && rm kilocode-plugin-claude-agent-*.tgz
npm install --omit=dev --omit=optional --no-package-lock
```

`--omit=optional` skips the SDK's own 270 MB copy of `claude`, which is never run. Then add the plugin to `kilo.jsonc`:

```json
{
  "plugin": ["file:///home/you/.local/share/kilo/plugins/claude-agent/src/index.ts"]
}
```

## Models

The `claude-agent` provider offers the models your login reports, with effort levels as variants. The list is refreshed in the background at most once a day and applies from the next start. Before the first check finishes it offers `opus`, `sonnet` and `haiku`.

## Configuration

Optional settings go under `provider.claude-agent.options`: `claudePath`, `maxOutputTokens`, `baseURL`, `apiKey`, `authToken`, `headers` and `body`. The last five map to Claude Code's environment variables.

## Differences from upstream

- Kilo's full system prompt replaces Claude Code's, rather than being appended to it. Kilo sends its prompt as one string, so a plugin can't separate the agent prompt from the instructions. Agent prompts apply in full, and Claude Code's own coding guidance does not.
- Compaction runs in a throwaway process, so it doesn't reuse the chat's prompt cache.
- Costs always show $0, including for API key logins.

## Limitations

- `claude` must be installed and signed in on the same machine as Kilo.
- Temperature, top-p, top-k and stop sequences are not supported.
- `http.request` plugin hooks never run, because Claude Code makes the API calls.
- Chats appear in `claude --resume` for that directory, because each is a saved Claude session.
