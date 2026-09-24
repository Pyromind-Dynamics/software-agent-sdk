import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import type { InlineExtension } from "@earendil-works/pi-coding-agent";

/**
 * Stops tool calls that keep returning the same answer.
 *
 * A repeat only counts when both the arguments and the observation are
 * identical: the same question already produced the same answer, so asking it
 * again cannot add information. Work that changes state between calls (a poll,
 * a re-run after an edit) produces a different observation and is left alone.
 */

/** Appended to the second identical result, before any call is blocked. */
const REPEATED_RESULT_NOTICE =
  "[no new information: this call returned the same result as the identical " +
  "previous call, so repeating it cannot add anything. Change approach or " +
  "report what you have.]";

const REPEATED_RESULT_NOTICE_AT = 2;
const REPEATED_CALL_BLOCK_AT = 3;

/**
 * Polling tools report the same state until an external task moves on, so an
 * unchanged answer is expected rather than a stalled search.
 */
const UNGUARDED_TOOLS = new Set(["df_check_progress"]);

const BLOCK_REASON =
  "Blocked: this exact call already returned the same result " +
  `${REPEATED_CALL_BLOCK_AT} times, so it cannot produce new information. ` +
  "Stop repeating it and report what you have found, or ask the user how to " +
  "proceed.";

interface CallOutcome {
  digest: string;
  repeats: number;
}

type ObservationBlock = TextContent | ImageContent;

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value === null || typeof value !== "object") return value;
  const entries = Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
    .map(([key, entry]) => [key, canonicalize(entry)]);
  return Object.fromEntries(entries);
}

function callKey(toolName: string, input: unknown): string {
  return `${toolName}\u0000${JSON.stringify(canonicalize(input))}`;
}

/**
 * FNV-1a over the observation, fed block by block: results can carry megabytes
 * of text or base64 image data, and only the identity of the result matters.
 */
function digestObservation(blocks: readonly ObservationBlock[]): string {
  let hash = 0x811c9dc5;
  let length = 0;
  const feed = (value: string): void => {
    length += value.length;
    for (let index = 0; index < value.length; index += 1) {
      hash ^= value.charCodeAt(index);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
  };
  for (const block of blocks) {
    if (block.type === "text") feed(`text:${block.text}`);
    else feed(`image:${block.mimeType}:${block.data}`);
  }
  return `${length}:${hash.toString(16)}`;
}

export function createNoProgressGuardExtension(): InlineExtension {
  return (pi) => {
    const outcomes = new Map<string, CallOutcome>();
    const blockedCalls = new Set<string>();

    pi.on("tool_call", (event) => {
      if (UNGUARDED_TOOLS.has(event.toolName)) return undefined;
      const outcome = outcomes.get(callKey(event.toolName, event.input));
      if (outcome === undefined || outcome.repeats < REPEATED_CALL_BLOCK_AT) {
        return undefined;
      }
      blockedCalls.add(event.toolCallId);
      return { block: true, reason: BLOCK_REASON };
    });

    pi.on("tool_result", (event) => {
      if (blockedCalls.delete(event.toolCallId)) return undefined;
      if (UNGUARDED_TOOLS.has(event.toolName)) return undefined;
      const key = callKey(event.toolName, event.input);
      const digest = digestObservation(event.content);
      const previous = outcomes.get(key);
      const repeats = previous?.digest === digest ? previous.repeats + 1 : 1;
      outcomes.set(key, { digest, repeats });
      if (repeats !== REPEATED_RESULT_NOTICE_AT) return undefined;
      return {
        content: [
          ...event.content,
          { type: "text", text: REPEATED_RESULT_NOTICE },
        ],
      };
    });
  };
}
