export interface ProviderErrorDetail {
  "@type"?: string;
  retryDelay?: string;
  [key: string]: unknown;
}

export interface ProviderError {
  code?: number | string;
  status?: string;
  message?: string;
  details?: ProviderErrorDetail[];
}

export interface RetryDecision {
  retryable: boolean;
  delayMs?: number;
}

export interface AttemptCommitState {
  hasOutput: boolean;
  hasToolExecution: boolean;
}

const RETRYABLE_STATUSES = new Set([
  "RESOURCE_EXHAUSTED",
  "OVERLOADED",
  "UNAVAILABLE",
  "DEADLINE_EXCEEDED",
  "INTERNAL",
]);

const RETRYABLE_CODES = new Set([429, 529, 500, 502, 503, 504]);

const RETRYABLE_PATTERNS = [
  "rate limit",
  "overloaded",
  "too many requests",
  "fetch failed",
  "network error",
  "connection refused",
  "timeout",
  "temporarily unavailable",
  "service unavailable",
  "resource_exhausted",
  "internal server error",
  "socket hang up",
  "econnreset",
  "bad gateway",
  "gateway timeout",
  "aborted",
  "abort",
  "429",
  "529",
  "500",
  "502",
  "503",
  "504",
] as const;

export const FALLBACK_DEFAULTS = {
  maxRetriesPerModel: 3,
  initialRetryDelayMs: 1000,
  maxRetryDelayMs: 30_000,
} as const;

export function parseProviderError(errorMessage: string): ProviderError | null {
  try {
    const parsed = JSON.parse(errorMessage);
    if (parsed?.error && typeof parsed.error === "object") {
      return {
        code: parsed.error.code,
        status: parsed.error.status,
        message: parsed.error.message,
        details: parsed.error.details,
      };
    }
    return {
      code: parsed?.code,
      status: parsed?.status,
      message: parsed?.message,
      details: parsed?.details,
    };
  } catch {
    return null;
  }
}

export function extractRetryDelay(error: ProviderError): number | null {
  if (!error.details) return null;

  for (const detail of error.details) {
    if (!detail?.["@type"]?.includes("RetryInfo") || !detail.retryDelay) {
      continue;
    }
    const match = detail.retryDelay.match(/^([\d.]+)s$/);
    if (match) {
      return Math.round(Number.parseFloat(match[1]) * 1000);
    }
  }

  return null;
}

export function classifyRetryableError(errorMessage: string): RetryDecision {
  const parsed = parseProviderError(errorMessage);

  if (parsed) {
    const code =
      typeof parsed.code === "string"
        ? Number.parseInt(parsed.code, 10)
        : parsed.code;
    if (typeof code === "number" && RETRYABLE_CODES.has(code)) {
      const delay = extractRetryDelay(parsed);
      return { retryable: true, delayMs: delay ?? undefined };
    }

    if (parsed.status && RETRYABLE_STATUSES.has(parsed.status)) {
      const delay = extractRetryDelay(parsed);
      return { retryable: true, delayMs: delay ?? undefined };
    }
  }

  const lower = errorMessage.toLowerCase();
  return {
    retryable: RETRYABLE_PATTERNS.some((pattern) => lower.includes(pattern)),
  };
}

export function extractErrorMessage(err: unknown): string {
  if (typeof err === "string") return err;
  if (err instanceof Error) return err.message;

  if (err && typeof err === "object") {
    const fromErrorMessage = (err as { errorMessage?: unknown }).errorMessage;
    if (typeof fromErrorMessage === "string" && fromErrorMessage.trim()) {
      return fromErrorMessage;
    }

    const fromMessage = (err as { message?: unknown }).message;
    if (typeof fromMessage === "string" && fromMessage.trim()) {
      return fromMessage;
    }

    try {
      return JSON.stringify(err);
    } catch {
      return String(err);
    }
  }

  return String(err);
}

export function shouldFallbackForState(state: AttemptCommitState): boolean {
  return !state.hasOutput && !state.hasToolExecution;
}
