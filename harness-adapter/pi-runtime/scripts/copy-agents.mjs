import { cpSync, rmSync } from "node:fs";

const destination = new URL("../dist/agents/", import.meta.url);
rmSync(destination, { recursive: true, force: true });
cpSync(new URL("../agents/", import.meta.url), destination, { recursive: true });
