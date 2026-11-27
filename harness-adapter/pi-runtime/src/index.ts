import { PiAgentRuntime } from "./agent-runtime.js";
import { deliverStage } from "./stage-delivery.js";
import { analyzeStage } from "./stage-analysis.js";
import { JsonlRpcPeer } from "./rpc-peer.js";

const peer = new JsonlRpcPeer(process.stdin, process.stdout);
const runtime = new PiAgentRuntime(peer);
let stageAnalysis = false;
peer.listen(async (method, params) => {
  if (method === "start" && (params.mode === "stage_analysis" || params.mode === "stage_delivery")) {
    stageAnalysis = true;
    return params.mode === "stage_delivery" ? deliverStage(params, peer) : analyzeStage(params);
  }
  if (stageAnalysis) {
    if (method === "close") return {};
    throw new Error("Stage analysis runner accepts only one request");
  }
  return runtime.handle(method, params);
}).catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : "Pi runner failed"}\n`);
  process.exitCode = 1;
});
