# @kilocode/plugin-agy-agent

Runs Gemini and other Antigravity models in Kilo Code through your own Google Antigravity CLI (`agy`) login, over the Agent Client Protocol. Requests count against your Google subscription, not an API key.

Kilo's tools, permissions and history stay in charge: agy runs a custom agent with no built-in tools or default prompt, and Kilo's tools are offered to it through a local MCP server.

## Requirements

- Antigravity CLI 1.5.18 or later on the machine running Kilo, signed in (run `agy` once in a terminal).
- `agy` on `PATH` or in `~/.local/bin`. Otherwise set `KILO_AGY_PATH` or `provider.agy-agent.options.agyPath`.

## Install from this checkout

```sh
cd packages/plugin-agy-agent
dest=~/.local/share/kilo/plugins/agy-agent
mkdir -p "$dest" && bun pm pack --destination "$dest"
cd "$dest" && tar xzf kilocode-plugin-agy-agent-*.tgz --strip-components=1 && rm kilocode-plugin-agy-agent-*.tgz
npm install --omit=dev --no-package-lock
```

Then add the plugin to `kilo.jsonc`:

```json
{
  "plugin": ["file:///home/you/.local/share/kilo/plugins/agy-agent/src/index.ts"]
}
```

## Models

The `agy-agent` provider offers the models `agy models` lists, with effort levels (`low` to `max`) as variants. The list is refreshed in the background at most once a day, or after agy is updated, and applies from the next start. Before the first check finishes it offers `gemini-4-hydrogen` and `gemini-3.8-flash`. The context window is learned from agy at runtime and applies from the next start; until then Gemini models assume 1M tokens and others 200K.

## Configuration

Optional settings go under `provider.agy-agent.options`: `agyPath` (path to `agy`; the `KILO_AGY_PATH` environment variable does the same) and `effort` (`low`, `medium`, `high`, `xhigh` or `max`), which variants also set.

## How it works

- Kilo's system prompt is appended to an empty custom agent, so agy's built-in tools, prompt and customizations are excluded.
- agy hides the largest tools once their declarations outgrow its budget, so tool descriptions over 1,000 characters are shortened to their first paragraph and appended in full to the system prompt. A warning is logged when the declarations still exceed 50,000 characters; disable unneeded MCP servers or tools if it appears.
- Each Kilo session maps to one agy conversation, which continues across turns and tool calls without replaying the chat. Edited or regenerated history starts a new conversation with the history replayed as text.
- Titles, summaries and compaction run in throwaway agy sessions that are deleted afterwards.
- agy runs with its own data directory under `~/.local/share/kilo/plugin-agy-agent`, so your `~/.gemini` config and agy history stay untouched. Deleting a Kilo session deletes its agy conversation.

## Limits and cost

- `/agy-usage` lists your plan limits with reset times. It answers without a model call and stays out of the conversation.
- Cost shows as 0, since usage is billed to your subscription. Token counts are still reported.
- A turn that hits your quota fails without retrying, and the error lists the exhausted limits with their reset times. It doesn't wait for the reset; send the message again later.

## Limitations

- `agy` must be installed and signed in on the same machine as Kilo.
- agy aborts a tool call after 180 s. Kilo tools such as subagents can still run longer: the plugin ends the agy turn just before that limit, and when the tool finishes, its result goes back to the same conversation automatically, as a text message.
- agy's own context compaction can't be turned off.
- Only image attachments (PNG, JPEG, GIF, WebP) are supported.
- Temperature, top-p, top-k, stop sequences and output token limits are not supported.
- `http.request` plugin hooks never run, because agy makes the API calls.
- agy's `manage_task`, `list_resources` and `read_resource` tools remain visible to the model.
