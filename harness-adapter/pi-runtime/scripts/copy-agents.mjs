import { cpSync } from "node:fs";

cpSync(new URL("../agents/", import.meta.url), new URL("../dist/agents/", import.meta.url), { recursive: true });
