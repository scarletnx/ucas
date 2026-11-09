# Wimsy for bots and AI agents

Wimsy ({MENTION}, user id {ID}) runs ucas, a computer algebra system for programmers: exact algebra and calculus, ranges and identities checked over every input of machine types, cheaper implementations proved equal, SPIR-V/GLSL/C simplification proved by SAT, shader runs on images as GIFs.

## Who is served
- Only bots on Wimsy's allowlist are answered; other bots and all webhooks are ignored silently. Humans are always served.
- Address every message explicitly: mention `{MENTION}` (a real mention, not typed text), or send a Discord reply to one of Wimsy's messages. A reply to a notice (rate limit, chain limit, this help) is not a request unless it also mentions Wimsy.
- Rate limit per bot: {BURST} requests at once, then {RATE} per minute. Past it you get one `rate limited: retry in N s` reply; everything else is ignored until then.
- Reply chains: at most {CHAIN} requests in a row that are replies to Wimsy with no human message in between; then one `reply-chain limit:` notice and silence. A new message with a mention (not a reply) starts a new chain.
- Wimsy's replies never ping you; they reply to your message.

## JSON (recommended for programs)
Send one ```json block holding a request of the JSON frontend (the protocol of `ucas --json`), or an array of up to 8 run in order. Text outside the block is ignored.

```json
{"id": 1, "cmd": "integrate", "args": {"f": "x^2 sin(x)"}}
```
```json
{"id": 2, "cmd": "spirv_simplify", "args": {"code": "#version 450\nlayout(local_size_x = 64) in;\nlayout(std430, binding = 0) buffer B { uint v[]; };\nvoid main() { uint i = gl_GlobalInvocationID.x; v[i] = v[i] * 4u / 2u; }"}}
```
```json
{"id": 3, "cmd": "animate", "args": {"code": "<GLSL fragment shader>", "inputs": ["black", {"attachment": 1}]}}
```
(the third with an image attached: `{"attachment": N}` is your message's N-th attachment, from 1; it also works as `args.code` for a .frag/.glsl/.comp/.spvasm/.spv/.c file)

The reply is a ```json block with the response, as in JSON mode:
`{"id", "ok", "cmd", "ucas", "structured", "result": {"kind", ...}, "notes", "conditions", "checks", "proof", "files", "text", "error"}`.
- `ok` false means `error` = `{"code", "message", ...}`.
- Files (exec.gif, simplified.spvasm, simplified.glsl, simplified.c, plots) are attached to the reply; `files` lists them as `{"name", "mime", "bytes", "attached"}`, with no base64.
- If the response does not fit in one message, the whole of it is attached as `response.json` and the block holds the response with `text` cut (`"text_truncated": true`) or a summary `{"id", "ok", "cmd", "headline", "error", "files"}`; either way `"full": "response.json"` names the attachment. Read the attachment.
- GLSL in `args.code` is compiled to SPIR-V by Wimsy (a note says so). `args.files` is ignored (files are always attached; `"none"` skips them), as is `timeout_s` (the limits below apply).
- `{"cmd": "schema"}` (alone) attaches `schema.json`: every command, its arguments and the response format. `{"cmd": "reset"}` (alone) clears the channel's state.

## Ambiguity (deterministic)
An ambiguous request never asks back in JSON: it fails with `error.code` `"ambiguous"`, `error.key` and `error.readings` (`[{"index", "reading", "default"}]`). Repeat it with `args.assume = {"KEY": INDEX}` or `"args": {..., "defaults": true}`.

## English and ucas text
Without a ```json block the message is English or ucas input, one statement per line: `{MENTION} range isqrt(x: u4)`, `{MENTION} integrate x^2 sin(x) from 0 to pi`.
- Code goes in ```glsl, ```spv or ```c blocks, or as attachments (.frag/.comp/.glsl, .spvasm/.spv, .c): `{MENTION} simplify this` with a ```glsl block.
- Images (PNG, JPEG, GIF, BMP, PPM; attachments or an ```image gallery) are a shader's inputs in order: `{MENTION} animate exec of this` with a shader and images; say `for input 1 use pure black` to set one.
- A follow-up question comes as a numbered list: answer with a Discord reply to that question message holding the option's number (or `cancel`).

## Reading results
- Reactions on your message: ❓ a question follows, ✅ answered, ❌ failed (an error, a failed check, or a limit).
- Text answers come in ``` blocks; long ones are split or attached as ucas-output.txt. JSON answers come as above. Attachments are named by file name.

## Sessions
Definitions (functions, models, assumptions) persist per channel, shared by text and JSON requests and by everyone in the channel. `reset` (or `{"cmd": "reset"}`) clears them. Keep requests self-contained: define what you use in the same message (a JSON array works).

## Limits
- One request per message: a text message may hold several statements (one per line), a JSON array up to 8 commands. Messages up to 2000 characters; longer code as an attachment.
- Up to 10 attachments; code up to 256 KiB, images up to 8 MiB.
- Each request: {CPU} s of CPU, {WALL} s of wall clock, then it is stopped and the channel's state is unchanged.
- No host file access (`file` inputs and paths do not work), no network, no secrets: everything runs in a sandbox.

## Etiquette
- Prefer JSON; send one request and wait for the reaction or the reply before the next.
- Never answer Wimsy's replies automatically; do not loop. On `rate limited: retry in N s`, wait N seconds; on `reply-chain limit:`, stop.
- Mention `{MENTION}` with `explain AI usage` to get this text again.
