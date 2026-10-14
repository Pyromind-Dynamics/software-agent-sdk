import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { installChildHost } from "./subagent-child-extension.js";
export default (pi: ExtensionAPI) => installChildHost(pi, "general_purpose");
