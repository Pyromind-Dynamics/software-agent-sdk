import assert from "node:assert/strict";
import { mkdtemp, readFile, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createLocalBashOperations } from "@earendil-works/pi-coding-agent";
import { createTerminalTool } from "../src/terminal-tool.js";
import { SandboxTerminalExecutionError, type TerminalExecutionEvidence } from "../src/sandbox-terminal.js";

test("terminal preserves native shell status and starts every call at its configured cwd", async (context) => {
  const cwd = await realpath(await mkdtemp(join(tmpdir(), "pi-terminal-result-")));
  context.after(() => rm(cwd, { recursive: true, force: true }));
  const tool = createTerminalTool(cwd, { operations: createLocalBashOperations() });
  for (const [command, code] of [
    ["printf ok", 0],
    ["printf 'Traceback: diagnostic text' >&2", 0],
    ["false | cat", 0],
    ["false; printf done", 0],
    ["bash -o pipefail -c 'false | cat'", 1],
    ["exit 7", 7],
    ["__pyromind_missing_executable__", 127],
    ["cd /", 0],
    ["pwd", 0],
  ] as const) {
    const result = await tool.execute(command, { command });
    assert.equal(result.details.exitCode, code, command);
    assert.equal(result.details.cwd, cwd);
    assert.equal(result.details.resultCategory, code === 0 ? "success" : "failure");
    const text = result.content.find((block) => block.type === "text")?.text ?? "";
    if (code !== 0) assert.match(text, new RegExp(`Command exited with code ${code}`));
    if (command === "pwd") assert.equal(text.trim(), cwd);
  }
});

test("concurrent terminal calls keep their exit status and sandbox evidence separate", async () => {
  let release: () => void = () => {};
  const gate = new Promise<void>((resolve) => { release = resolve; });
  const evidence = (id: string): TerminalExecutionEvidence => ({
    execution_id: id, remote_run_dir: `/runs/${id}`, phase: "finished", started: true, stopped: true,
  });
  const tool = createTerminalTool(".", {
    operations: {
      async exec(command, cwd, options) {
        if (command === "slow") await gate;
        else release();
        assert.equal(cwd, ".");
        options.onData(Buffer.from(command));
        return { exitCode: command === "slow" ? 9 : 0, execution: evidence(command) };
      },
    },
  });
  const [slow, fast] = await Promise.all([
    tool.execute("1", { command: "slow" }),
    tool.execute("2", { command: "fast" }),
  ]);
  assert.equal(slow.details.exitCode, 9);
  assert.equal(fast.details.exitCode, 0);
  assert.deepEqual(slow.details.execution, evidence("slow"));
  assert.deepEqual(fast.details.execution, evidence("fast"));
});

test("terminal keeps timeout, cancellation and uncertain execution evidence without inventing an exit code", async () => {
  for (const kind of ["timeout", "cancel", "sandbox"] as const) {
    const controller = new AbortController();
    const evidence: TerminalExecutionEvidence = {
      execution_id: "uncertain", remote_run_dir: "/runs/uncertain", phase: "observing",
      started: true, stopped: false, error_code: "termination_unconfirmed",
    };
    const tool = createTerminalTool(".", {
      operations: {
        async exec(_command, _cwd, options) {
          options.onData(Buffer.from("started\n"));
          if (kind === "cancel") {
            controller.abort();
            throw new Error("aborted");
          }
          if (kind === "timeout") throw new Error("timeout:1");
          throw new SandboxTerminalExecutionError("connection lost", evidence);
        },
      },
    });
    const result = await tool.execute(kind, { command: "work" }, controller.signal);
    assert.equal(result.details.resultCategory, "failure");
    assert.equal("exitCode" in result.details, false);
    assert.equal(result.details.timedOut, kind === "timeout" ? true : undefined);
    assert.equal(result.details.cancelled, kind === "cancel" ? true : undefined);
    if (kind === "sandbox") {
      assert.deepEqual(result.details.execution, evidence);
      assert.equal(result.details.error_code, "termination_unconfirmed");
    }
  }
});

test("terminal preserves native streaming and full output metadata even on a failed exit", async (context) => {
  const output = "a line of output\n".repeat(5000);
  for (const code of [0, 3]) {
    let updates = 0;
    const tool = createTerminalTool(".", {
      operations: {
        async exec(_command, _cwd, options) {
          options.onData(Buffer.from(output));
          return { exitCode: code };
        },
      },
    });
    const result = await tool.execute(String(code), { command: "large output" }, undefined, () => { updates++; });
    assert(updates > 0);
    assert.equal(result.details.truncation?.truncated, true);
    const path = result.details.fullOutputPath;
    assert.equal(typeof path, "string");
    assert(path);
    context.after(() => rm(path, { force: true }));
    assert.equal(await readFile(path, "utf8"), output);
    assert.equal(result.details.exitCode, code);
    assert.equal(result.details.resultCategory, code === 0 ? "success" : "failure");
  }
});

test("terminal obtains current SDK environment without retaining credentials on later calls", async () => {
  const base = { PATH: "/bin", MODEL: "existing-model" };
  const environments: Array<NodeJS.ProcessEnv | undefined> = [];
  let calls = 0;
  const tool = createTerminalTool(".", {
    environmentProvider: async () => ++calls === 1 ? {
      PYROMIND_API_KEY: "sdk-access-key",
      LLM_BASE_URL: "https://model.example",
      LLM_AUTH_TOKEN: "model-access-key",
      LLM_MODEL: "session-model",
    } : {},
    spawnHook: ({ command }) => ({ command, cwd: ".", env: base }),
    operations: {
      async exec(_command, _cwd, options) {
        environments.push(options.env);
        options.onData(Buffer.from("ok"));
        return { exitCode: 0 };
      },
    },
  });
  await tool.execute("first", { command: "python3 script.py" });
  await tool.execute("second", { command: "pwd" });
  assert.equal(environments[0]?.PYROMIND_API_KEY, "sdk-access-key");
  assert.equal(environments[0]?.LLM_AUTH_TOKEN, "model-access-key");
  assert.equal(environments[0]?.LLM_MODEL, "session-model");
  assert.deepEqual(environments[1], base);
  assert.deepEqual(base, { PATH: "/bin", MODEL: "existing-model" });
});

test("SDK and model secrets are filtered across chunks before streaming or full-output persistence", async (context) => {
  const secrets = ["derived-sdk-key-1234567890", "model-access-key-1234567890",
    "dataflow-profile-key-1234567890", "direct-model-key-1234567890"];
  for (const fail of [false, true]) {
    const updates: string[] = [];
    const tool = createTerminalTool(".", {
      environmentProvider: async () => ({ PYROMIND_API_KEY: secrets[0]!, LLM_AUTH_TOKEN: secrets[1]! }),
      spawnHook: ({ command }) => ({ command, cwd: ".", env: {
        DF_API_KEY: secrets[3]!,
        PYROMIND_DATAFLOW_PROFILES: JSON.stringify({ text: { DF_API_KEY: secrets[2] }, vision: {} }),
      } }),
      operations: {
        async exec(_command, _cwd, options) {
          for (const secret of secrets) {
            options.onData(Buffer.from(secret.slice(0, 9)));
            options.onData(Buffer.from(secret.slice(9) + "\n"));
          }
          options.onData(Buffer.from("a line\n".repeat(12000)));
          for (const secret of secrets) {
            options.onData(Buffer.from(secret.slice(0, 5)));
            options.onData(Buffer.from(secret.slice(5)));
          }
          if (fail) throw new Error("request failed: " + secrets.join(" "));
          return { exitCode: 0 };
        },
      },
    });
    const result = await tool.execute(String(fail), { command: "run SDK script" }, undefined,
      (partial) => { updates.push(JSON.stringify(partial)); });
    assert(secrets.every((secret) => !JSON.stringify(result).includes(secret)));
    assert(updates.every((update) => secrets.every((secret) => !update.includes(secret))));
    const path = result.details.fullOutputPath;
    assert(path);
    context.after(() => rm(path, { force: true }));
    const content = await readFile(path, "utf8");
    assert(secrets.every((secret) => !content.includes(secret)));
    assert(content.includes("[REDACTED]"));
    assert.equal(result.details.resultCategory, fail ? "failure" : "success");
  }
});
