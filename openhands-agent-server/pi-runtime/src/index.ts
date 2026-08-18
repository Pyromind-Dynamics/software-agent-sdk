import { PiAgentRuntime } from "./agent-runtime.ts";
import { JsonlRpcPeer } from "./rpc-peer.ts";

const peer = new JsonlRpcPeer(process.stdin, process.stdout);
const runtime = new PiAgentRuntime(peer);

peer.listen(runtime.handle.bind(runtime)).catch((error: unknown) => {
  process.stderr.write(error instanceof Error ? error.message : "Pi runner failed");
  process.exitCode = 1;
});
