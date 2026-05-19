import { describe, expect, it } from "vitest";
import {
  classifyRetryableError,
  extractErrorMessage,
  extractRetryDelay,
  parseProviderError,
  shouldFallbackForState,
} from "../src/fallback-policy.js";

describe("fallback-policy", () => {
  it("parses nested provider error payloads", () => {
    const payload = JSON.stringify({
      error: {
        code: 429,
        status: "RESOURCE_EXHAUSTED",
        message: "Rate limit exceeded",
      },
    });

    const parsed = parseProviderError(payload);
    expect(parsed?.code).toBe(429);
    expect(parsed?.status).toBe("RESOURCE_EXHAUSTED");
  });

  it("extracts RetryInfo delays", () => {
    const delay = extractRetryDelay({
      details: [{ "@type": "type.googleapis.com/google.rpc.RetryInfo", retryDelay: "12.5s" }],
    });
    expect(delay).toBe(12500);
  });

  it("classifies retryable provider errors", () => {
    const result = classifyRetryableError("429 Too Many Requests");
    expect(result).toEqual({ retryable: true });
  });

  it("classifies non-retryable provider errors", () => {
    const result = classifyRetryableError("401 Unauthorized");
    expect(result).toEqual({ retryable: false });
  });

  it("extracts error message from common payload shapes", () => {
    expect(extractErrorMessage(new Error("boom"))).toBe("boom");
    expect(extractErrorMessage({ errorMessage: "stream failed" })).toBe(
      "stream failed",
    );
  });

  it("only allows fallback before commit point", () => {
    expect(
      shouldFallbackForState({ hasOutput: false, hasToolExecution: false }),
    ).toBe(true);
    expect(
      shouldFallbackForState({ hasOutput: true, hasToolExecution: false }),
    ).toBe(false);
    expect(
      shouldFallbackForState({ hasOutput: false, hasToolExecution: true }),
    ).toBe(false);
  });
});
