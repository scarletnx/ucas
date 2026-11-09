**Wimsy for bots and AI agents** (the full guide is attached: wimsy-ai-usage.md)
- Address me with a real mention `{MENTION}` or a Discord reply to my message. Only allowlisted bots are served; others and webhooks are ignored. Limits per bot: {BURST} requests at once, then {RATE}/min (one `rate limited: retry in N s` notice, then silence); at most {CHAIN} replies in a row to me without a human message.
- Recommended: one code block tagged json holding a JSON frontend request (as `ucas --json`), or an array of up to 8:
```json
{"id": 1, "cmd": "integrate", "args": {"f": "x^2 sin(x)"}}
```
  Code: `{"cmd": "spirv_simplify", "args": {"code": "<GLSL, SPIR-V asm or C>"}}`. Shader on images: `{"cmd": "animate", "args": {"code": "<GLSL>", "inputs": ["black", {"attachment": 1}]}}` with the image attached.
- Reply: a json code block `{"id", "ok", "cmd", "result", "notes", "checks", "files", "text", "error"}`; files are attached and listed by name. Too long: the whole response is attached as response.json and `"full": "response.json"` says so. `{"cmd": "schema"}` attaches schema.json (every command).
- Ambiguity: `ok` false, `error.code` "ambiguous" with `error.key` and `error.readings`; repeat with `"assume": {"KEY": INDEX}` or `"defaults": true` in args. Never a question.
- Text also works: English or ucas, one statement per line, code in code blocks tagged glsl, spv or c, or attached, images as shader inputs in order. A numbered question: reply to it with the number.
- Reactions: ❓ question, ✅ answered, ❌ failed. State persists per channel (`reset` clears it). {CPU} s CPU / {WALL} s per request; no host files, no network.
- Don't loop: never answer my replies automatically, wait out rate limits, keep requests self-contained.
