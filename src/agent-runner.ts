/**
 * agent-runner.ts — Core execution engine: creates sessions, runs agents, collects results.
 */

import type { Model } from "@mariozechner/pi-ai";
import type { ExtensionContext } from "@mariozechner/pi-coding-agent";
import {
  type AgentSession,
  type AgentSessionEvent,
  createAgentSession,
  DefaultResourceLoader,
  type ExtensionAPI,
  getAgentDir,
  SessionManager,
  SettingsManager,
} from "@mariozechner/pi-coding-agent";
import {
  getAgentConfig,
  getConfig,
  getMemoryToolNames,
  getReadOnlyMemoryToolNames,
  getToolNamesForType,
} from "./agent-types.js";
import { buildParentContext, extractText } from "./context.js";
import { DEFAULT_AGENTS } from "./default-agents.js";
import { detectEnv } from "./env.js";
import {
  classifyRetryableError,
  extractErrorMessage,
  FALLBACK_DEFAULTS,
  shouldFallbackForState,
} from "./fallback-policy.js";
import { buildMemoryBlock, buildReadOnlyMemoryBlock } from "./memory.js";
import { buildAgentPrompt, type PromptExtras } from "./prompts.js";
import { preloadSkills } from "./skill-loader.js";
import type { SubagentType, ThinkingLevel } from "./types.js";

/** Names of tools registered by this extension that subagents must NOT inherit. */
const EXCLUDED_TOOL_NAMES = ["Agent", "get_subagent_result", "steer_subagent"];

/** Default max turns. undefined = unlimited (no turn limit). */
let defaultMaxTurns: number | undefined;

/** Normalize max turns. undefined or 0 = unlimited, otherwise minimum 1. */
export function normalizeMaxTurns(n: number | undefined): number | undefined {
  if (n == null || n === 0) return undefined;
  return Math.max(1, n);
}

/** Get the default max turns value. undefined = unlimited. */
export function getDefaultMaxTurns(): number | undefined {
  return defaultMaxTurns;
}
/** Set the default max turns value. undefined or 0 = unlimited, otherwise minimum 1. */
export function setDefaultMaxTurns(n: number | undefined): void {
  defaultMaxTurns = normalizeMaxTurns(n);
}

/** Additional turns allowed after the soft limit steer message. */
let graceTurns = 5;

/** Get the grace turns value. */
export function getGraceTurns(): number {
  return graceTurns;
}
/** Set the grace turns value (minimum 1). */
export function setGraceTurns(n: number): void {
  graceTurns = Math.max(1, n);
}

/** Global fallback model IDs from settings (ordered). */
let globalFallbackModels: string[] = [];

/** Get current global fallback model IDs. */
export function getGlobalFallbackModels(): string[] {
  return [...globalFallbackModels];
}
/** Set global fallback model IDs (ordered, deduplicated). */
export function setGlobalFallbackModels(models: string[] | undefined): void {
  if (!Array.isArray(models)) {
    globalFallbackModels = [];
    return;
  }
  const next = models.map((m) => m.trim()).filter(Boolean);
  globalFallbackModels = [...new Set(next)];
}

type ModelRegistryLike = {
  find(provider: string, modelId: string): Model<any> | undefined;
  getAvailable?(): Model<any>[];
};

function getModelKey(model: Model<any>): string {
  const m = model as any;
  return `${String(m.provider ?? "")}/${String(m.id ?? m.name ?? "")}`.toLowerCase();
}

function resolveConfiguredModel(
  registry: ModelRegistryLike,
  modelRef: string | undefined,
): Model<any> | undefined {
  if (!modelRef) return undefined;
  const slashIdx = modelRef.indexOf("/");
  if (slashIdx === -1) return undefined;

  const provider = modelRef.slice(0, slashIdx);
  const modelId = modelRef.slice(slashIdx + 1);

  const available = registry.getAvailable?.();
  const availableKeys = available
    ? new Set(available.map((m: any) => `${m.provider}/${m.id}`.toLowerCase()))
    : undefined;
  if (
    availableKeys &&
    !availableKeys.has(`${provider}/${modelId}`.toLowerCase())
  ) {
    return undefined;
  }

  return registry.find(provider, modelId);
}

function buildModelAttemptChain(
  primaryModel: Model<any> | undefined,
  parentModel: Model<any> | undefined,
  registry: ModelRegistryLike,
  agentFallbackModels: string[] | undefined,
): Model<any>[] {
  const chain: Model<any>[] = [];
  const push = (model: Model<any> | undefined) => {
    if (!model) return;
    const key = getModelKey(model);
    if (chain.some((existing) => getModelKey(existing) === key)) return;
    chain.push(model);
  };

  push(primaryModel);
  for (const ref of agentFallbackModels ?? []) {
    push(resolveConfiguredModel(registry, ref));
  }
  for (const ref of globalFallbackModels) {
    push(resolveConfiguredModel(registry, ref));
  }
  push(parentModel);
  return chain;
}

function getModelName(model: Model<any> | undefined): string {
  if (!model) return "<default>";
  const m = model as any;
  const provider = String(m.provider ?? "unknown");
  const id = String(m.id ?? m.name ?? "unknown");
  return `${provider}/${id}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Info about a tool event in the subagent. */
export interface ToolActivity {
  type: "start" | "end";
  toolName: string;
}

export interface RunOptions {
  /** ExtensionAPI instance — used for pi.exec() instead of execSync. */
  pi: ExtensionAPI;
  /** Manager-assigned id; suffixes session name to disambiguate parallel spawns (e.g. `Explore#a1b2c3d4`). */
  agentId?: string;
  model?: Model<any>;
  maxTurns?: number;
  signal?: AbortSignal;
  isolated?: boolean;
  inheritContext?: boolean;
  thinkingLevel?: ThinkingLevel;
  /** Override working directory (e.g. for worktree isolation). */
  cwd?: string;
  /** Called on tool start/end with activity info. */
  onToolActivity?: (activity: ToolActivity) => void;
  /** Called on streaming text deltas from the assistant response. */
  onTextDelta?: (delta: string, fullText: string) => void;
  onSessionCreated?: (session: AgentSession) => void;
  /** Called at the end of each agentic turn with the cumulative count. */
  onTurnEnd?: (turnCount: number) => void;
  /**
   * Called once per assistant message_end with that message's usage delta.
   * Lets callers maintain a lifetime accumulator that survives compaction
   * (which replaces session.state.messages and resets stats-derived sums).
   */
  onAssistantUsage?: (usage: {
    input: number;
    output: number;
    cacheWrite: number;
  }) => void;
  /**
   * Called when the session successfully compacts. `tokensBefore` is upstream's
   * pre-compaction context size estimate. Aborted compactions don't fire.
   */
  onCompaction?: (info: {
    reason: "manual" | "threshold" | "overflow";
    tokensBefore: number;
  }) => void;
}

export interface RunResult {
  responseText: string;
  session: AgentSession;
  /** True if the agent was hard-aborted (max_turns + grace exceeded). */
  aborted: boolean;
  /** True if the agent was steered to wrap up (hit soft turn limit) but finished in time. */
  steered: boolean;
}

/**
 * Subscribe to a session and collect the last assistant message text.
 * Returns an object with a `getText()` getter and an `unsubscribe` function.
 */
function collectResponseText(session: AgentSession) {
  let text = "";
  const unsubscribe = session.subscribe((event: AgentSessionEvent) => {
    if (event.type === "message_start") {
      text = "";
    }
    if (
      event.type === "message_update" &&
      event.assistantMessageEvent.type === "text_delta"
    ) {
      text += event.assistantMessageEvent.delta;
    }
  });
  return { getText: () => text, unsubscribe };
}

/** Get the last assistant text from the completed session history. */
function getLastAssistantText(session: AgentSession): string {
  for (let i = session.messages.length - 1; i >= 0; i--) {
    const msg = session.messages[i];
    if (msg.role !== "assistant") continue;
    const text = extractText(msg.content).trim();
    if (text) return text;
  }
  return "";
}

/**
 * Wire an AbortSignal to abort a session.
 * Returns a cleanup function to remove the listener.
 */
function forwardAbortSignal(
  session: AgentSession,
  signal?: AbortSignal,
): () => void {
  if (!signal) return () => {};
  const onAbort = () => session.abort();
  signal.addEventListener("abort", onAbort, { once: true });
  return () => signal.removeEventListener("abort", onAbort);
}

export async function runAgent(
  ctx: ExtensionContext,
  type: SubagentType,
  prompt: string,
  options: RunOptions,
): Promise<RunResult> {
  const config = getConfig(type);
  const agentConfig = getAgentConfig(type);

  // Resolve working directory: worktree override > parent cwd
  const effectiveCwd = options.cwd ?? ctx.cwd;

  const env = await detectEnv(options.pi, effectiveCwd);

  // Get parent system prompt for append-mode agents
  const parentSystemPrompt = ctx.getSystemPrompt();

  // Build prompt extras (memory, skill preloading)
  const extras: PromptExtras = {};

  // Resolve extensions/skills: isolated overrides to false
  const extensions = options.isolated ? false : config.extensions;
  const skills = options.isolated ? false : config.skills;

  // Skill preloading: when skills is string[], preload their content into prompt
  if (Array.isArray(skills)) {
    const loaded = preloadSkills(skills, effectiveCwd);
    if (loaded.length > 0) {
      extras.skillBlocks = loaded;
    }
  }

  let toolNames = getToolNamesForType(type);

  // Persistent memory: detect write capability and branch accordingly.
  // Account for disallowedTools — a tool in the base set but on the denylist is not truly available.
  if (agentConfig?.memory) {
    const existingNames = new Set(toolNames);
    const denied = agentConfig.disallowedTools
      ? new Set(agentConfig.disallowedTools)
      : undefined;
    const effectivelyHas = (name: string) =>
      existingNames.has(name) && !denied?.has(name);
    const hasWriteTools = effectivelyHas("write") || effectivelyHas("edit");

    if (hasWriteTools) {
      // Read-write memory: add any missing memory tool names (read/write/edit)
      const extraNames = getMemoryToolNames(existingNames);
      if (extraNames.length > 0) toolNames = [...toolNames, ...extraNames];
      extras.memoryBlock = buildMemoryBlock(
        agentConfig.name,
        agentConfig.memory,
        effectiveCwd,
      );
    } else {
      // Read-only memory: only add read tool name, use read-only prompt
      const extraNames = getReadOnlyMemoryToolNames(existingNames);
      if (extraNames.length > 0) toolNames = [...toolNames, ...extraNames];
      extras.memoryBlock = buildReadOnlyMemoryBlock(
        agentConfig.name,
        agentConfig.memory,
        effectiveCwd,
      );
    }
  }

  // Build system prompt from agent config
  let systemPrompt: string;
  if (agentConfig) {
    systemPrompt = buildAgentPrompt(
      agentConfig,
      effectiveCwd,
      env,
      parentSystemPrompt,
      extras,
    );
  } else {
    // Unknown type fallback: spread the canonical general-purpose config (defensive —
    // unreachable in practice since index.ts resolves unknown types before calling runAgent).
    const fallback = DEFAULT_AGENTS.get("general-purpose");
    if (!fallback)
      throw new Error(
        `No fallback config available for unknown type "${type}"`,
      );
    systemPrompt = buildAgentPrompt(
      { ...fallback, name: type },
      effectiveCwd,
      env,
      parentSystemPrompt,
      extras,
    );
  }

  // When skills is string[], we've already preloaded them into the prompt.
  // Still pass noSkills: true since we don't need the skill loader to load them again.
  const noSkills = skills === false || Array.isArray(skills);

  // Build the effective prompt once: optionally prepend parent context.
  let effectivePrompt = prompt;
  if (options.inheritContext) {
    const parentContext = buildParentContext(ctx);
    if (parentContext) {
      effectivePrompt = parentContext + prompt;
    }
  }

  const agentDir = getAgentDir();

  // Resolve model candidates in descending priority.
  const configModel = resolveConfiguredModel(
    ctx.modelRegistry,
    agentConfig?.model,
  );
  const primaryModel = options.model ?? configModel ?? ctx.model;
  const modelAttempts = buildModelAttemptChain(
    primaryModel,
    ctx.model,
    ctx.modelRegistry,
    agentConfig?.fallbackModels,
  );

  // Resolve thinking level: explicit option > agent config > undefined (inherit)
  const thinkingLevel = options.thinkingLevel ?? agentConfig?.thinking;

  const maxTurns = normalizeMaxTurns(
    options.maxTurns ?? agentConfig?.maxTurns ?? defaultMaxTurns,
  );
  const attemptList = modelAttempts.length > 0 ? modelAttempts : [undefined];
  const errors: string[] = [];

  for (let i = 0; i < attemptList.length; i++) {
    const model = attemptList[i];
    const modelName = getModelName(model);
    let modelTerminalError: string | null = null;
    let retryDelayMs = FALLBACK_DEFAULTS.initialRetryDelayMs;

    for (let retry = 0; retry <= FALLBACK_DEFAULTS.maxRetriesPerModel; retry++) {
      let hasOutput = false;
      let hasToolExecution = false;
      let streamErrorReason: string | null = null;
      let streamErrorRetryable = false;
      let streamErrorDelayMs: number | undefined;

      try {
      // Load extensions/skills: true or string[] → load; false → don't.
      // Suppress AGENTS.md/CLAUDE.md and APPEND_SYSTEM.md — upstream's
      // buildSystemPrompt() re-appends both AFTER systemPromptOverride, which
      // would defeat prompt_mode: replace and isolated: true.
      const loader = new DefaultResourceLoader({
        cwd: effectiveCwd,
        agentDir,
        noExtensions: extensions === false,
        noSkills,
        noPromptTemplates: true,
        noThemes: true,
        noContextFiles: true,
        systemPromptOverride: () => systemPrompt,
        appendSystemPromptOverride: () => [],
      });
      await loader.reload();

      const sessionOpts: Parameters<typeof createAgentSession>[0] = {
        cwd: effectiveCwd,
        agentDir,
        sessionManager: SessionManager.inMemory(effectiveCwd),
        settingsManager: SettingsManager.create(effectiveCwd, agentDir),
        modelRegistry: ctx.modelRegistry,
        model,
        tools: toolNames,
        resourceLoader: loader,
      };
      if (thinkingLevel) {
        sessionOpts.thinkingLevel = thinkingLevel;
      }

      const { session } = await createAgentSession(sessionOpts);

      const baseSessionName = agentConfig?.name ?? type;
      session.setSessionName(
        options.agentId
          ? `${baseSessionName}#${options.agentId.slice(0, 8)}`
          : baseSessionName,
      );

      // Build disallowed tools set from agent config
      const disallowedSet = agentConfig?.disallowedTools
        ? new Set(agentConfig.disallowedTools)
        : undefined;

      // Filter active tools: remove our own tools to prevent nesting,
      // apply extension allowlist if specified, and apply disallowedTools denylist
      if (extensions !== false) {
        const builtinToolNameSet = new Set(toolNames);
        const activeTools = session.getActiveToolNames().filter((t) => {
          if (EXCLUDED_TOOL_NAMES.includes(t)) return false;
          if (disallowedSet?.has(t)) return false;
          if (builtinToolNameSet.has(t)) return true;
          if (Array.isArray(extensions)) {
            return extensions.some(
              (ext) => t.startsWith(ext) || t.includes(ext),
            );
          }
          return true;
        });
        session.setActiveToolsByName(activeTools);
      } else if (disallowedSet) {
        // Even with extensions disabled, apply denylist to built-in tools
        const activeTools = session
          .getActiveToolNames()
          .filter((t) => !disallowedSet.has(t));
        session.setActiveToolsByName(activeTools);
      }

      // Bind extensions so that session_start fires and extensions can initialize.
      await session.bindExtensions({
        onError: (err) => {
          options.onToolActivity?.({
            type: "end",
            toolName: `extension-error:${err.extensionPath}`,
          });
        },
      });

      options.onSessionCreated?.(session);

      // Track turns for graceful max_turns enforcement
      let turnCount = 0;
      let softLimitReached = false;
      let aborted = false;

      let currentMessageText = "";
      const unsubTurns = session.subscribe((event: AgentSessionEvent) => {
        if (event.type === "turn_end") {
          turnCount++;
          options.onTurnEnd?.(turnCount);
          if (maxTurns != null) {
            if (!softLimitReached && turnCount >= maxTurns) {
              softLimitReached = true;
              session.steer(
                "You have reached your turn limit. Wrap up immediately — provide your final answer now.",
              );
            } else if (softLimitReached && turnCount >= maxTurns + graceTurns) {
              aborted = true;
              session.abort();
            }
          }
        }
        if (event.type === "message_start") {
          currentMessageText = "";
        }
        if (
          event.type === "message_update" &&
          event.assistantMessageEvent.type === "text_delta"
        ) {
          hasOutput = true;
          currentMessageText += event.assistantMessageEvent.delta;
          options.onTextDelta?.(
            event.assistantMessageEvent.delta,
            currentMessageText,
          );
        }
        if (
          event.type === "message_update" &&
          (event.assistantMessageEvent as any).type === "error"
        ) {
          const reason = extractErrorMessage(
            (event.assistantMessageEvent as any).error ?? event.assistantMessageEvent,
          );
          const decision = classifyRetryableError(reason);
          streamErrorReason = reason;
          streamErrorRetryable = decision.retryable;
          streamErrorDelayMs = decision.delayMs;
        }
        if (event.type === "tool_execution_start") {
          hasToolExecution = true;
          options.onToolActivity?.({ type: "start", toolName: event.toolName });
        }
        if (event.type === "tool_execution_end") {
          options.onToolActivity?.({ type: "end", toolName: event.toolName });
        }
        if (
          event.type === "message_end" &&
          event.message.role === "assistant"
        ) {
          const u = (event.message as any).usage;
          if (u)
            options.onAssistantUsage?.({
              input: u.input ?? 0,
              output: u.output ?? 0,
              cacheWrite: u.cacheWrite ?? 0,
            });
        }
        if (event.type === "compaction_end" && !event.aborted && event.result) {
          options.onCompaction?.({
            reason: event.reason,
            tokensBefore: event.result.tokensBefore,
          });
        }
      });

      const collector = collectResponseText(session);
      const cleanupAbort = forwardAbortSignal(session, options.signal);

      try {
        await session.prompt(effectivePrompt);
      } finally {
        unsubTurns();
        collector.unsubscribe();
        cleanupAbort();
      }

      if (streamErrorReason) {
        const canFallback = shouldFallbackForState({
          hasOutput,
          hasToolExecution,
        });
        if (streamErrorRetryable && canFallback) {
          throw new Error(streamErrorReason);
        }
        throw new Error(
          canFallback
            ? streamErrorReason
            : `Cannot fallback after commit point: ${streamErrorReason}`,
        );
      }

      const responseText =
        collector.getText().trim() || getLastAssistantText(session);
      return { responseText, session, aborted, steered: softLimitReached };
      } catch (err) {
        if (options.signal?.aborted) throw err;

        const reason = extractErrorMessage(err);
        const retryDecision = streamErrorReason
          ? {
              retryable: streamErrorRetryable,
              delayMs: streamErrorDelayMs,
            }
          : classifyRetryableError(reason);
        const canFallback = shouldFallbackForState({
          hasOutput,
          hasToolExecution,
        });

        if (!canFallback) {
          throw new Error(reason);
        }

        if (retryDecision.retryable && retry < FALLBACK_DEFAULTS.maxRetriesPerModel) {
          const delay = retryDecision.delayMs ?? retryDelayMs;
          await sleep(delay);
          retryDelayMs = Math.min(
            retryDelayMs * 2,
            FALLBACK_DEFAULTS.maxRetryDelayMs,
          );
          continue;
        }

        modelTerminalError = reason;
        break;
      }
    }

    errors.push(
      `attempt ${i + 1}/${attemptList.length} (${modelName}): ${modelTerminalError ?? "unknown failure"}`,
    );

    if (i === attemptList.length - 1) {
      throw new Error(errors.join("\n"));
    }
  }

  throw new Error("Failed to run agent: no model attempts were possible.");
}

/**
 * Send a new prompt to an existing session (resume).
 */
export async function resumeAgent(
  session: AgentSession,
  prompt: string,
  options: {
    onToolActivity?: (activity: ToolActivity) => void;
    onAssistantUsage?: (usage: {
      input: number;
      output: number;
      cacheWrite: number;
    }) => void;
    onCompaction?: (info: {
      reason: "manual" | "threshold" | "overflow";
      tokensBefore: number;
    }) => void;
    signal?: AbortSignal;
  } = {},
): Promise<string> {
  const collector = collectResponseText(session);
  const cleanupAbort = forwardAbortSignal(session, options.signal);

  const unsubEvents =
    options.onToolActivity || options.onAssistantUsage || options.onCompaction
      ? session.subscribe((event: AgentSessionEvent) => {
          if (event.type === "tool_execution_start")
            options.onToolActivity?.({
              type: "start",
              toolName: event.toolName,
            });
          if (event.type === "tool_execution_end")
            options.onToolActivity?.({ type: "end", toolName: event.toolName });
          if (
            event.type === "message_end" &&
            event.message.role === "assistant"
          ) {
            const u = (event.message as any).usage;
            if (u)
              options.onAssistantUsage?.({
                input: u.input ?? 0,
                output: u.output ?? 0,
                cacheWrite: u.cacheWrite ?? 0,
              });
          }
          if (
            event.type === "compaction_end" &&
            !event.aborted &&
            event.result
          ) {
            options.onCompaction?.({
              reason: event.reason,
              tokensBefore: event.result.tokensBefore,
            });
          }
        })
      : () => {};

  try {
    await session.prompt(prompt);
  } finally {
    collector.unsubscribe();
    unsubEvents();
    cleanupAbort();
  }

  return collector.getText().trim() || getLastAssistantText(session);
}

/**
 * Send a steering message to a running subagent.
 * The message will interrupt the agent after its current tool execution.
 */
export async function steerAgent(
  session: AgentSession,
  message: string,
): Promise<void> {
  await session.steer(message);
}

/**
 * Get the subagent's conversation messages as formatted text.
 */
export function getAgentConversation(session: AgentSession): string {
  const parts: string[] = [];

  for (const msg of session.messages) {
    if (msg.role === "user") {
      const text =
        typeof msg.content === "string"
          ? msg.content
          : extractText(msg.content);
      if (text.trim()) parts.push(`[User]: ${text.trim()}`);
    } else if (msg.role === "assistant") {
      const textParts: string[] = [];
      const toolCalls: string[] = [];
      for (const c of msg.content) {
        if (c.type === "text" && c.text) textParts.push(c.text);
        else if (c.type === "toolCall")
          toolCalls.push(
            `  Tool: ${(c as any).name ?? (c as any).toolName ?? "unknown"}`,
          );
      }
      if (textParts.length > 0)
        parts.push(`[Assistant]: ${textParts.join("\n")}`);
      if (toolCalls.length > 0)
        parts.push(`[Tool Calls]:\n${toolCalls.join("\n")}`);
    } else if (msg.role === "toolResult") {
      const text = extractText(msg.content);
      const truncated = text.length > 200 ? text.slice(0, 200) + "..." : text;
      parts.push(`[Tool Result (${msg.toolName})]: ${truncated}`);
    }
  }

  return parts.join("\n\n");
}
