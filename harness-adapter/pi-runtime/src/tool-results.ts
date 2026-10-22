// Adapted from pi-agent-browser-native's buildAgentBrowserToolResultPatch:
// https://github.com/fitchmultz/pi-agent-browser-native/blob/67fd7fd3f1ab1ff6dd911994c604ac15e016ce26/extensions/agent-browser/lib/pi-tool-rendering.ts#L238
// MIT License — Copyright (c) 2026 Mitch Fultz
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { isRecord } from "./protocol.js";

export function createToolResultExtension(): ExtensionFactory {
  return (pi) => {
    pi.on("tool_result", (event) => {
      const shouldMarkError = isRecord(event.details) &&
        event.details.resultCategory === "failure" && event.isError !== true;
      if (!shouldMarkError) return undefined;
      return { isError: true };
    });
  };
}
