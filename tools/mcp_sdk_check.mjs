// Checks ucas --mcp with the official MCP TypeScript SDK client:
//   npm install @modelcontextprotocol/sdk   (in this directory or a parent)
//   node tools/mcp_sdk_check.mjs ./ucas [--flat]
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
const bin = process.argv[2], extra = process.argv.slice(3);
const t = new StdioClientTransport({ command: bin, args: ["--mcp", ...extra] });
const c = new Client({ name: "sdk-check", version: "1" });
await c.connect(t);
console.log("server:", c.getServerVersion(), "proto ok");
const tools = await c.listTools();
console.log("tools:", tools.tools.length, tools.tools.slice(0, 3).map(x => x.name).join(","));
let r = await c.callTool({ name: extra.includes("--flat") ? "range" : "analyze",
  arguments: extra.includes("--flat") ? { f: "isqrt(x: u4)" } : { op: "range", args: { f: "isqrt(x: u4)" } } });
console.log("range:", r.isError, r.structuredContent?.result?.max, r.content[0].text.split("\n")[0]);
r = await c.callTool({ name: "ucas", arguments: { source: "k = 3 ^ 5 & 1" } });
console.log("ambiguous:", r.isError, r.structuredContent?.error?.key);
if (!extra.includes("--flat")) {
  r = await c.callTool({ name: "plot", arguments: { op: "plot", args: { fs: "sin x" } } });
  console.log("plot:", r.content.map(x => x.type + (x.mimeType ? ":" + x.mimeType : "")).join(","));
  r = await c.callTool({ name: "session", arguments: { op: "define", args: { name: "f", params: [{ name: "x", type: "u8" }], body: "x*3" } } });
  r = await c.callTool({ name: "analyze", arguments: { op: "wrap", args: { f: "f" } } });
  console.log("wrap f:", r.isError, r.structuredContent.result.end_to_end.exact);
  r = await c.callTool({ name: "analyze", arguments: { op: "gates", args: { f: "(x: u3) * (y: u3)", max: 40, nodes: 100000000000, timeout_s: 1 } } });
  console.log("timeout:", r.isError, r.structuredContent?.error?.code);
  r = await c.callTool({ name: "analyze", arguments: { op: "show", args: { f: "f" } } });
  console.log("state after timeout:", r.content[0].text.split("\n")[0]);
}
console.log("resources:", (await c.listResources()).resources.map(x => x.uri).join(","));
console.log("state:", (await c.readResource({ uri: "ucas://state" })).contents[0].text.slice(0, 60));
console.log("prompts:", (await c.listPrompts()).prompts.map(x => x.name).join(","));
const p = await c.getPrompt({ name: "explore_function", arguments: { f: "x^3" } });
console.log("prompt:", p.messages[0].content.text.slice(0, 40));
try { await c.callTool({ name: "nope", arguments: {} }); } catch (e) { console.log("unknown tool error:", e.code); }
await c.close();
