/**
 * NanoClaw Agent Runner
 * Runs inside a container, receives config via stdin, outputs result to stdout
 *
 * Input protocol:
 *   Stdin: Full ContainerInput JSON (read until EOF, like before)
 *   IPC:   Follow-up messages written as JSON files to /workspace/ipc/input/
 *          Files: {type:"message", text:"..."}.json — polled and consumed
 *          Sentinel: /workspace/ipc/input/_close — signals session end
 *
 * Stdout protocol:
 *   Each result is wrapped in OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs.
 *   Multiple results may be emitted (one per agent teams result).
 *   Final marker after loop ends signals completion.
 */

import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { query, HookCallback, PreCompactHookInput, PreToolUseHookInput } from '@anthropic-ai/claude-agent-sdk';
import OpenAI from 'openai';
import { fileURLToPath } from 'url';
import { CronExpressionParser } from 'cron-parser';

interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  secrets?: Record<string, string>;
}

interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

interface SessionEntry {
  sessionId: string;
  fullPath: string;
  summary: string;
  firstPrompt: string;
}

interface SessionsIndex {
  entries: SessionEntry[];
}

interface SDKUserMessage {
  type: 'user';
  message: { role: 'user'; content: string };
  parent_tool_use_id: null;
  session_id: string;
}

const IPC_INPUT_DIR = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const IPC_POLL_MS = 500;
const IPC_DIR = '/workspace/ipc';
const MESSAGES_DIR = path.join(IPC_DIR, 'messages');
const TASKS_DIR = path.join(IPC_DIR, 'tasks');
const DEFAULT_OPENAI_MODEL = 'gpt-5-codex';
const MAX_OPENAI_TOOL_TURNS = 24;

type AgentProvider = 'claude' | 'openai';

function normalizeProvider(raw: string | undefined): AgentProvider {
  if (raw?.trim().toLowerCase() === 'claude') return 'claude';
  return 'openai';
}

interface OpenAIFunctionTool {
  type: 'function';
  name: string;
  description: string;
  strict: boolean;
  parameters: {
    type: 'object';
    properties: Record<string, unknown>;
    required?: string[];
    additionalProperties?: boolean;
  };
}

type OpenAITool = OpenAIFunctionTool | { type: 'local_shell' } | { type: 'web_search_preview' };

interface OpenAIResponseOutputItem {
  type?: string;
  call_id?: string;
  name?: string;
  arguments?: string;
  id?: string;
  action?: {
    type?: string;
    command?: unknown;
    env?: unknown;
    timeout_ms?: number | null;
    working_directory?: string | null;
  };
  content?: Array<{ type?: string; text?: string }>;
}

interface OpenAIResponseLike {
  id: string;
  output_text?: string;
  output?: OpenAIResponseOutputItem[];
}

interface OpenAIFunctionCallItem {
  call_id: string;
  name: string;
  arguments: string;
}

interface OpenAILocalShellCallItem {
  id: string;
  action: {
    type?: string;
    command?: unknown;
    env?: unknown;
    timeout_ms?: number | null;
    working_directory?: string | null;
  };
}

type OpenAIToolCallItem =
  | { kind: 'function'; call: OpenAIFunctionCallItem }
  | { kind: 'local_shell'; call: OpenAILocalShellCallItem };

interface OpenAIFunctionCallOutputInput {
  type: 'function_call_output';
  call_id: string;
  output: string;
}

interface OpenAILocalShellCallOutputInput {
  type: 'local_shell_call_output';
  id: string;
  output: string;
}

type OpenAIToolOutputInput = OpenAIFunctionCallOutputInput | OpenAILocalShellCallOutputInput;

const OPENAI_FUNCTION_TOOLS: OpenAIFunctionTool[] = [
  {
    type: 'function',
    name: 'send_message',
    description: "Send a message to the user or group immediately while you're still running.",
    strict: true,
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        text: { type: 'string', description: 'The message text to send' },
        sender: { type: 'string', description: 'Optional role/identity name (e.g., "Researcher")' },
      },
      required: ['text'],
    },
  },
  {
    type: 'function',
    name: 'schedule_task',
    description:
      'Schedule a recurring or one-time task. context_mode can be "group" (with chat context) or "isolated" (fresh context).',
    strict: true,
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        prompt: { type: 'string', description: 'Task instructions for the scheduled run' },
        schedule_type: { type: 'string', enum: ['cron', 'interval', 'once'] },
        schedule_value: {
          type: 'string',
          description:
            'cron: "0 9 * * *" | interval: milliseconds like "300000" | once: local timestamp like "2026-02-01T15:30:00"',
        },
        context_mode: { type: 'string', enum: ['group', 'isolated'] },
        target_group_jid: { type: 'string' },
      },
      required: ['prompt', 'schedule_type', 'schedule_value'],
    },
  },
  {
    type: 'function',
    name: 'list_tasks',
    description: "List scheduled tasks. Main group sees all; other groups only see their own tasks.",
    strict: true,
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {},
      required: [],
    },
  },
  {
    type: 'function',
    name: 'pause_task',
    description: 'Pause a scheduled task.',
    strict: true,
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        task_id: { type: 'string', description: 'Task ID to pause' },
      },
      required: ['task_id'],
    },
  },
  {
    type: 'function',
    name: 'resume_task',
    description: 'Resume a paused task.',
    strict: true,
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        task_id: { type: 'string', description: 'Task ID to resume' },
      },
      required: ['task_id'],
    },
  },
  {
    type: 'function',
    name: 'cancel_task',
    description: 'Cancel and delete a scheduled task.',
    strict: true,
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        task_id: { type: 'string', description: 'Task ID to cancel' },
      },
      required: ['task_id'],
    },
  },
  {
    type: 'function',
    name: 'register_group',
    description: 'Register a new WhatsApp group so the agent can respond there. Main group only.',
    strict: true,
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        jid: { type: 'string', description: 'WhatsApp JID (e.g., "120363336345536173@g.us")' },
        name: { type: 'string', description: 'Display name for the group' },
        folder: { type: 'string', description: 'Folder name (lowercase, hyphens)' },
        trigger: { type: 'string', description: 'Trigger word (e.g., "@Andy")' },
      },
      required: ['jid', 'name', 'folder', 'trigger'],
    },
  },
];

const OPENAI_TOOLS: OpenAITool[] = [
  { type: 'local_shell' },
  { type: 'web_search_preview' },
  ...OPENAI_FUNCTION_TOOLS,
];

/**
 * Push-based async iterable for streaming user messages to the SDK.
 * Keeps the iterable alive until end() is called, preventing isSingleUserTurn.
 */
class MessageStream {
  private queue: SDKUserMessage[] = [];
  private waiting: (() => void) | null = null;
  private done = false;

  push(text: string): void {
    this.queue.push({
      type: 'user',
      message: { role: 'user', content: text },
      parent_tool_use_id: null,
      session_id: '',
    });
    this.waiting?.();
  }

  end(): void {
    this.done = true;
    this.waiting?.();
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<SDKUserMessage> {
    while (true) {
      while (this.queue.length > 0) {
        yield this.queue.shift()!;
      }
      if (this.done) return;
      await new Promise<void>(r => { this.waiting = r; });
      this.waiting = null;
    }
  }
}

async function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => { data += chunk; });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

function writeOutput(output: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

function log(message: string): void {
  console.error(`[agent-runner] ${message}`);
}

function getSessionSummary(sessionId: string, transcriptPath: string): string | null {
  const projectDir = path.dirname(transcriptPath);
  const indexPath = path.join(projectDir, 'sessions-index.json');

  if (!fs.existsSync(indexPath)) {
    log(`Sessions index not found at ${indexPath}`);
    return null;
  }

  try {
    const index: SessionsIndex = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
    const entry = index.entries.find(e => e.sessionId === sessionId);
    if (entry?.summary) {
      return entry.summary;
    }
  } catch (err) {
    log(`Failed to read sessions index: ${err instanceof Error ? err.message : String(err)}`);
  }

  return null;
}

/**
 * Archive the full transcript to conversations/ before compaction.
 */
function createPreCompactHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const preCompact = input as PreCompactHookInput;
    const transcriptPath = preCompact.transcript_path;
    const sessionId = preCompact.session_id;

    if (!transcriptPath || !fs.existsSync(transcriptPath)) {
      log('No transcript found for archiving');
      return {};
    }

    try {
      const content = fs.readFileSync(transcriptPath, 'utf-8');
      const messages = parseTranscript(content);

      if (messages.length === 0) {
        log('No messages to archive');
        return {};
      }

      const summary = getSessionSummary(sessionId, transcriptPath);
      const name = summary ? sanitizeFilename(summary) : generateFallbackName();

      const conversationsDir = '/workspace/group/conversations';
      fs.mkdirSync(conversationsDir, { recursive: true });

      const date = new Date().toISOString().split('T')[0];
      const filename = `${date}-${name}.md`;
      const filePath = path.join(conversationsDir, filename);

      const markdown = formatTranscriptMarkdown(messages, summary);
      fs.writeFileSync(filePath, markdown);

      log(`Archived conversation to ${filePath}`);
    } catch (err) {
      log(`Failed to archive transcript: ${err instanceof Error ? err.message : String(err)}`);
    }

    return {};
  };
}

// Secrets to strip from Bash tool subprocess environments.
// These are needed by claude-code for API auth but should never
// be visible to commands Kit runs.
const SECRET_ENV_VARS = ['ANTHROPIC_API_KEY', 'CLAUDE_CODE_OAUTH_TOKEN', 'OPENAI_API_KEY'];

function createSanitizeBashHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const preInput = input as PreToolUseHookInput;
    const command = (preInput.tool_input as { command?: string })?.command;
    if (!command) return {};

    const unsetPrefix = `unset ${SECRET_ENV_VARS.join(' ')} 2>/dev/null; `;
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse',
        updatedInput: {
          ...(preInput.tool_input as Record<string, unknown>),
          command: unsetPrefix + command,
        },
      },
    };
  };
}

function sanitizeFilename(summary: string): string {
  return summary
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 50);
}

function generateFallbackName(): string {
  const time = new Date();
  return `conversation-${time.getHours().toString().padStart(2, '0')}${time.getMinutes().toString().padStart(2, '0')}`;
}

interface ParsedMessage {
  role: 'user' | 'assistant';
  content: string;
}

function parseTranscript(content: string): ParsedMessage[] {
  const messages: ParsedMessage[] = [];

  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (entry.type === 'user' && entry.message?.content) {
        const text = typeof entry.message.content === 'string'
          ? entry.message.content
          : entry.message.content.map((c: { text?: string }) => c.text || '').join('');
        if (text) messages.push({ role: 'user', content: text });
      } else if (entry.type === 'assistant' && entry.message?.content) {
        const textParts = entry.message.content
          .filter((c: { type: string }) => c.type === 'text')
          .map((c: { text: string }) => c.text);
        const text = textParts.join('');
        if (text) messages.push({ role: 'assistant', content: text });
      }
    } catch {
    }
  }

  return messages;
}

function formatTranscriptMarkdown(messages: ParsedMessage[], title?: string | null): string {
  const now = new Date();
  const formatDateTime = (d: Date) => d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true
  });

  const lines: string[] = [];
  lines.push(`# ${title || 'Conversation'}`);
  lines.push('');
  lines.push(`Archived: ${formatDateTime(now)}`);
  lines.push('');
  lines.push('---');
  lines.push('');

  for (const msg of messages) {
    const sender = msg.role === 'user' ? 'User' : 'Andy';
    const content = msg.content.length > 2000
      ? msg.content.slice(0, 2000) + '...'
      : msg.content;
    lines.push(`**${sender}**: ${content}`);
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * Check for _close sentinel.
 */
function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }
    return true;
  }
  return false;
}

/**
 * Drain all pending IPC input messages.
 * Returns messages found, or empty array.
 */
function drainIpcInput(): string[] {
  try {
    fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
    const files = fs.readdirSync(IPC_INPUT_DIR)
      .filter(f => f.endsWith('.json'))
      .sort();

    const messages: string[] = [];
    for (const file of files) {
      const filePath = path.join(IPC_INPUT_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        fs.unlinkSync(filePath);
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
        }
      } catch (err) {
        log(`Failed to process input file ${file}: ${err instanceof Error ? err.message : String(err)}`);
        try { fs.unlinkSync(filePath); } catch { /* ignore */ }
      }
    }
    return messages;
  } catch (err) {
    log(`IPC drain error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

/**
 * Wait for a new IPC message or _close sentinel.
 * Returns the messages as a single string, or null if _close.
 */
function waitForIpcMessage(): Promise<string | null> {
  return new Promise((resolve) => {
    const poll = () => {
      if (shouldClose()) {
        resolve(null);
        return;
      }
      const messages = drainIpcInput();
      if (messages.length > 0) {
        resolve(messages.join('\n'));
        return;
      }
      setTimeout(poll, IPC_POLL_MS);
    };
    poll();
  });
}

/**
 * Run a single query and stream results via writeOutput.
 * Uses MessageStream (AsyncIterable) to keep isSingleUserTurn=false,
 * allowing agent teams subagents to run to completion.
 * Also pipes IPC messages into the stream during the query.
 */
async function runClaudeQuery(
  prompt: string,
  sessionId: string | undefined,
  mcpServerPath: string,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
  resumeAt?: string,
): Promise<{ newSessionId?: string; lastAssistantUuid?: string; closedDuringQuery: boolean }> {
  const stream = new MessageStream();
  stream.push(prompt);

  // Poll IPC for follow-up messages and _close sentinel during the query
  let ipcPolling = true;
  let closedDuringQuery = false;
  const pollIpcDuringQuery = () => {
    if (!ipcPolling) return;
    if (shouldClose()) {
      log('Close sentinel detected during query, ending stream');
      closedDuringQuery = true;
      stream.end();
      ipcPolling = false;
      return;
    }
    const messages = drainIpcInput();
    for (const text of messages) {
      log(`Piping IPC message into active query (${text.length} chars)`);
      stream.push(text);
    }
    setTimeout(pollIpcDuringQuery, IPC_POLL_MS);
  };
  setTimeout(pollIpcDuringQuery, IPC_POLL_MS);

  let newSessionId: string | undefined;
  let lastAssistantUuid: string | undefined;
  let messageCount = 0;
  let resultCount = 0;

  // Load global CLAUDE.md as additional system context (shared across all groups)
  const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
  let globalClaudeMd: string | undefined;
  if (!containerInput.isMain && fs.existsSync(globalClaudeMdPath)) {
    globalClaudeMd = fs.readFileSync(globalClaudeMdPath, 'utf-8');
  }

  // Discover additional directories mounted at /workspace/extra/*
  // These are passed to the SDK so their CLAUDE.md files are loaded automatically
  const extraDirs: string[] = [];
  const extraBase = '/workspace/extra';
  if (fs.existsSync(extraBase)) {
    for (const entry of fs.readdirSync(extraBase)) {
      const fullPath = path.join(extraBase, entry);
      if (fs.statSync(fullPath).isDirectory()) {
        extraDirs.push(fullPath);
      }
    }
  }
  if (extraDirs.length > 0) {
    log(`Additional directories: ${extraDirs.join(', ')}`);
  }

  for await (const message of query({
    prompt: stream,
    options: {
      cwd: '/workspace/group',
      additionalDirectories: extraDirs.length > 0 ? extraDirs : undefined,
      resume: sessionId,
      resumeSessionAt: resumeAt,
      systemPrompt: globalClaudeMd
        ? { type: 'preset' as const, preset: 'claude_code' as const, append: globalClaudeMd }
        : undefined,
      allowedTools: [
        'Bash',
        'Read', 'Write', 'Edit', 'Glob', 'Grep',
        'WebSearch', 'WebFetch',
        'Task', 'TaskOutput', 'TaskStop',
        'TeamCreate', 'TeamDelete', 'SendMessage',
        'TodoWrite', 'ToolSearch', 'Skill',
        'NotebookEdit',
        'mcp__nanoclaw__*'
      ],
      env: sdkEnv,
      permissionMode: 'bypassPermissions',
      allowDangerouslySkipPermissions: true,
      settingSources: ['project', 'user'],
      mcpServers: {
        nanoclaw: {
          command: 'node',
          args: [mcpServerPath],
          env: {
            NANOCLAW_CHAT_JID: containerInput.chatJid,
            NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
            NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
          },
        },
      },
      hooks: {
        PreCompact: [{ hooks: [createPreCompactHook()] }],
        PreToolUse: [{ matcher: 'Bash', hooks: [createSanitizeBashHook()] }],
      },
    }
  })) {
    messageCount++;
    const msgType = message.type === 'system' ? `system/${(message as { subtype?: string }).subtype}` : message.type;
    log(`[msg #${messageCount}] type=${msgType}`);

    if (message.type === 'assistant' && 'uuid' in message) {
      lastAssistantUuid = (message as { uuid: string }).uuid;
    }

    if (message.type === 'system' && message.subtype === 'init') {
      newSessionId = message.session_id;
      log(`Session initialized: ${newSessionId}`);
    }

    if (message.type === 'system' && (message as { subtype?: string }).subtype === 'task_notification') {
      const tn = message as { task_id: string; status: string; summary: string };
      log(`Task notification: task=${tn.task_id} status=${tn.status} summary=${tn.summary}`);
    }

    if (message.type === 'result') {
      resultCount++;
      const textResult = 'result' in message ? (message as { result?: string }).result : null;
      log(`Result #${resultCount}: subtype=${message.subtype}${textResult ? ` text=${textResult.slice(0, 200)}` : ''}`);
      writeOutput({
        status: 'success',
        result: textResult || null,
        newSessionId
      });
    }
  }

  ipcPolling = false;
  log(`Query done. Messages: ${messageCount}, results: ${resultCount}, lastAssistantUuid: ${lastAssistantUuid || 'none'}, closedDuringQuery: ${closedDuringQuery}`);
  return { newSessionId, lastAssistantUuid, closedDuringQuery };
}

function extractOpenAIText(response: unknown): string {
  const responseObj = response as {
    output_text?: string;
    output?: Array<{
      type?: string;
      content?: Array<{ type?: string; text?: string }>;
    }>;
  };

  if (responseObj.output_text) {
    return responseObj.output_text.trim();
  }

  const parts: string[] = [];
  for (const item of responseObj.output || []) {
    if (item.type !== 'message') continue;
    for (const content of item.content || []) {
      if (content.type === 'output_text' && content.text) {
        parts.push(content.text);
      }
    }
  }
  return parts.join('\n').trim();
}

function extractOpenAIToolCalls(response: unknown): OpenAIToolCallItem[] {
  const responseObj = response as OpenAIResponseLike;
  const calls: OpenAIToolCallItem[] = [];

  for (const item of responseObj.output || []) {
    if (
      item.type === 'function_call' &&
      typeof item.call_id === 'string' &&
      typeof item.name === 'string' &&
      typeof item.arguments === 'string'
    ) {
      calls.push({
        kind: 'function',
        call: {
          call_id: item.call_id,
          name: item.name,
          arguments: item.arguments,
        },
      });
      continue;
    }

    if (
      item.type === 'local_shell_call' &&
      typeof item.id === 'string' &&
      item.action &&
      typeof item.action === 'object'
    ) {
      calls.push({
        kind: 'local_shell',
        call: {
          id: item.id,
          action: item.action,
        },
      });
    }
  }

  return calls;
}

function writeIpcFile(dir: string, data: object): string {
  fs.mkdirSync(dir, { recursive: true });

  const filename = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}.json`;
  const filepath = path.join(dir, filename);

  const tempPath = `${filepath}.tmp`;
  fs.writeFileSync(tempPath, JSON.stringify(data, null, 2));
  fs.renameSync(tempPath, filepath);

  return filename;
}

function parseOpenAIFunctionArgs(raw: string): Record<string, unknown> {
  if (!raw || !raw.trim()) return {};
  const parsed = JSON.parse(raw);
  if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
    throw new Error('Function arguments must be a JSON object');
  }
  return parsed as Record<string, unknown>;
}

function readRequiredString(args: Record<string, unknown>, key: string): string {
  const value = args[key];
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`Missing or invalid "${key}"`);
  }
  return value.trim();
}

function readOptionalString(args: Record<string, unknown>, key: string): string | undefined {
  const value = args[key];
  if (value == null) return undefined;
  if (typeof value !== 'string') {
    throw new Error(`"${key}" must be a string`);
  }
  const trimmed = value.trim();
  return trimmed || undefined;
}

function handleSendMessageTool(args: Record<string, unknown>, containerInput: ContainerInput): string {
  const text = readRequiredString(args, 'text');
  const sender = readOptionalString(args, 'sender');

  const data: Record<string, string | undefined> = {
    type: 'message',
    chatJid: containerInput.chatJid,
    text,
    sender,
    groupFolder: containerInput.groupFolder,
    timestamp: new Date().toISOString(),
  };

  writeIpcFile(MESSAGES_DIR, data);
  return 'Message sent.';
}

function handleScheduleTaskTool(args: Record<string, unknown>, containerInput: ContainerInput): string {
  const prompt = readRequiredString(args, 'prompt');
  const scheduleType = readRequiredString(args, 'schedule_type');
  const scheduleValue = readRequiredString(args, 'schedule_value');
  const contextModeRaw = readOptionalString(args, 'context_mode');
  const targetGroupJid = readOptionalString(args, 'target_group_jid');

  if (!['cron', 'interval', 'once'].includes(scheduleType)) {
    throw new Error(`Invalid schedule_type: "${scheduleType}". Use cron, interval, or once.`);
  }

  if (scheduleType === 'cron') {
    try {
      CronExpressionParser.parse(scheduleValue);
    } catch {
      throw new Error(
        `Invalid cron: "${scheduleValue}". Use format like "0 9 * * *" or "*/5 * * * *".`,
      );
    }
  } else if (scheduleType === 'interval') {
    const ms = Number.parseInt(scheduleValue, 10);
    if (Number.isNaN(ms) || ms <= 0) {
      throw new Error(`Invalid interval: "${scheduleValue}". Must be positive milliseconds.`);
    }
  } else if (scheduleType === 'once') {
    const date = new Date(scheduleValue);
    if (Number.isNaN(date.getTime())) {
      throw new Error(`Invalid timestamp: "${scheduleValue}". Use ISO format like "2026-02-01T15:30:00".`);
    }
  }

  const contextMode = contextModeRaw || 'group';
  if (!['group', 'isolated'].includes(contextMode)) {
    throw new Error(`Invalid context_mode: "${contextMode}". Use "group" or "isolated".`);
  }

  const targetJid = containerInput.isMain && targetGroupJid
    ? targetGroupJid
    : containerInput.chatJid;

  const data = {
    type: 'schedule_task',
    prompt,
    schedule_type: scheduleType,
    schedule_value: scheduleValue,
    context_mode: contextMode,
    targetJid,
    createdBy: containerInput.groupFolder,
    timestamp: new Date().toISOString(),
  };

  const filename = writeIpcFile(TASKS_DIR, data);
  return `Task scheduled (${filename}): ${scheduleType} - ${scheduleValue}`;
}

interface CurrentTaskRecord {
  id?: string;
  prompt?: string;
  schedule_type?: string;
  schedule_value?: string;
  status?: string;
  next_run?: string;
  groupFolder?: string;
}

function handleListTasksTool(containerInput: ContainerInput): string {
  const tasksFile = path.join(IPC_DIR, 'current_tasks.json');
  if (!fs.existsSync(tasksFile)) {
    return 'No scheduled tasks found.';
  }

  const raw = JSON.parse(fs.readFileSync(tasksFile, 'utf-8')) as unknown;
  const allTasks = Array.isArray(raw) ? raw as CurrentTaskRecord[] : [];
  const tasks = containerInput.isMain
    ? allTasks
    : allTasks.filter((task) => task.groupFolder === containerInput.groupFolder);

  if (tasks.length === 0) {
    return 'No scheduled tasks found.';
  }

  const formatted = tasks
    .map((task) => {
      const id = task.id || 'unknown';
      const prompt = (task.prompt || '').slice(0, 50);
      const scheduleType = task.schedule_type || 'unknown';
      const scheduleValue = task.schedule_value || 'unknown';
      const status = task.status || 'unknown';
      const nextRun = task.next_run || 'N/A';
      return `- [${id}] ${prompt}... (${scheduleType}: ${scheduleValue}) - ${status}, next: ${nextRun}`;
    })
    .join('\n');

  return `Scheduled tasks:\n${formatted}`;
}

function handleTaskMutationTool(
  type: 'pause_task' | 'resume_task' | 'cancel_task',
  args: Record<string, unknown>,
  containerInput: ContainerInput,
): string {
  const taskId = readRequiredString(args, 'task_id');

  const data = {
    type,
    taskId,
    groupFolder: containerInput.groupFolder,
    isMain: containerInput.isMain,
    timestamp: new Date().toISOString(),
  };

  writeIpcFile(TASKS_DIR, data);
  if (type === 'pause_task') return `Task ${taskId} pause requested.`;
  if (type === 'resume_task') return `Task ${taskId} resume requested.`;
  return `Task ${taskId} cancellation requested.`;
}

function handleRegisterGroupTool(args: Record<string, unknown>, containerInput: ContainerInput): string {
  if (!containerInput.isMain) {
    throw new Error('Only the main group can register new groups.');
  }

  const jid = readRequiredString(args, 'jid');
  const name = readRequiredString(args, 'name');
  const folder = readRequiredString(args, 'folder');
  const trigger = readRequiredString(args, 'trigger');

  const data = {
    type: 'register_group',
    jid,
    name,
    folder,
    trigger,
    timestamp: new Date().toISOString(),
  };

  writeIpcFile(TASKS_DIR, data);
  return `Group "${name}" registered. It will start receiving messages immediately.`;
}

function executeOpenAITool(name: string, args: Record<string, unknown>, containerInput: ContainerInput): string {
  switch (name) {
    case 'send_message':
      return handleSendMessageTool(args, containerInput);
    case 'schedule_task':
      return handleScheduleTaskTool(args, containerInput);
    case 'list_tasks':
      return handleListTasksTool(containerInput);
    case 'pause_task':
      return handleTaskMutationTool('pause_task', args, containerInput);
    case 'resume_task':
      return handleTaskMutationTool('resume_task', args, containerInput);
    case 'cancel_task':
      return handleTaskMutationTool('cancel_task', args, containerInput);
    case 'register_group':
      return handleRegisterGroupTool(args, containerInput);
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

function sanitizeShellEnv(
  sdkEnv: Record<string, string | undefined>,
  extraEnv: unknown,
): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(sdkEnv)) {
    if (typeof value === 'string') env[key] = value;
  }

  if (extraEnv && typeof extraEnv === 'object') {
    for (const [key, value] of Object.entries(extraEnv)) {
      if (typeof value === 'string') env[key] = value;
    }
  }

  for (const secretName of SECRET_ENV_VARS) {
    delete env[secretName];
  }

  return env;
}

function truncateShellOutput(text: string, maxChars = 64000): string {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}\n...truncated...`;
}

async function executeLocalShellToolCall(
  call: OpenAILocalShellCallItem,
  sdkEnv: Record<string, string | undefined>,
): Promise<OpenAILocalShellCallOutputInput> {
  const action = call.action;
  if (action.type !== 'exec') {
    return {
      type: 'local_shell_call_output',
      id: call.id,
      output: JSON.stringify({ error: `Unsupported local shell action type: ${String(action.type)}` }),
    };
  }

  if (!Array.isArray(action.command) || action.command.length === 0 || action.command.some((part) => typeof part !== 'string')) {
    return {
      type: 'local_shell_call_output',
      id: call.id,
      output: JSON.stringify({ error: 'Invalid local shell command array' }),
    };
  }

  const commandParts = action.command as string[];
  const command = commandParts[0];
  const args = commandParts.slice(1);
  const timeoutMs = Math.max(1_000, Math.min(action.timeout_ms ?? 120_000, 600_000));
  const cwd = action.working_directory || '/workspace/group';
  const env = sanitizeShellEnv(sdkEnv, action.env);

  log(`Executing local shell tool: ${commandParts.join(' ')}`);

  const output = await new Promise<string>((resolve) => {
    const child = spawn(command, args, {
      cwd,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      try {
        child.kill('SIGKILL');
      } catch {
        // ignore
      }
    }, timeoutMs);

    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');

    child.stdout.on('data', (chunk) => {
      stdout += chunk;
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk;
    });

    child.on('error', (err) => {
      clearTimeout(timer);
      resolve(
        JSON.stringify({
          error: err.message,
        }),
      );
    });

    child.on('close', (code, signal) => {
      clearTimeout(timer);
      resolve(
        JSON.stringify({
          stdout: truncateShellOutput(stdout),
          stderr: truncateShellOutput(stderr),
          exit_code: code ?? -1,
          signal: signal || null,
          timed_out: timedOut,
        }),
      );
    });
  });

  return {
    type: 'local_shell_call_output',
    id: call.id,
    output,
  };
}

function buildOpenAIInstructions(containerInput: ContainerInput): string | undefined {
  const promptParts: string[] = [];

  if (!containerInput.isMain) {
    const globalCodexMdPath = '/workspace/global/CODEX.md';
    if (fs.existsSync(globalCodexMdPath)) {
      promptParts.push(fs.readFileSync(globalCodexMdPath, 'utf-8'));
    }

    const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
    if (fs.existsSync(globalClaudeMdPath)) {
      promptParts.push(fs.readFileSync(globalClaudeMdPath, 'utf-8'));
    }
  }

  if (promptParts.length === 0) return undefined;
  return promptParts.join('\n\n');
}

async function runOpenAIQuery(
  prompt: string,
  sessionId: string | undefined,
  _mcpServerPath: string,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
  _resumeAt?: string,
): Promise<{ newSessionId?: string; lastAssistantUuid?: string; closedDuringQuery: boolean }> {
  const apiKey = sdkEnv.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY is required when NANOCLAW_AGENT_PROVIDER=openai');
  }

  const client = new OpenAI({
    apiKey,
    baseURL: sdkEnv.OPENAI_BASE_URL,
  });

  const model = sdkEnv.OPENAI_MODEL || DEFAULT_OPENAI_MODEL;
  const instructions = buildOpenAIInstructions(containerInput);

  const createResponse = async (
    input: string | OpenAIToolOutputInput[],
    previousResponseId: string | undefined,
    allowResumeFallback: boolean,
  ): Promise<OpenAIResponseLike> => {
    try {
      return await client.responses.create({
        model,
        input,
        previous_response_id: previousResponseId,
        instructions,
        tools: OPENAI_TOOLS,
        tool_choice: 'auto',
        parallel_tool_calls: true,
      }) as OpenAIResponseLike;
    } catch (err) {
      if (!allowResumeFallback || !previousResponseId) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      log(`OpenAI resume failed for session ${previousResponseId}: ${msg}. Retrying without previous_response_id.`);
      return await client.responses.create({
        model,
        input,
        instructions,
        tools: OPENAI_TOOLS,
        tool_choice: 'auto',
        parallel_tool_calls: true,
      }) as OpenAIResponseLike;
    }
  };

  log(`Running OpenAI query with model ${model} (session: ${sessionId || 'new'})`);
  let previousResponseId = sessionId;
  let input: string | OpenAIToolOutputInput[] = prompt;
  let response = await createResponse(input, previousResponseId, true);
  previousResponseId = response.id;

  for (let turn = 0; turn < MAX_OPENAI_TOOL_TURNS; turn++) {
    const calls = extractOpenAIToolCalls(response);
    if (calls.length === 0) break;

    log(`OpenAI requested ${calls.length} tool call(s) on turn ${turn + 1}`);
    const outputs: OpenAIToolOutputInput[] = [];

    for (const item of calls) {
      if (item.kind === 'function') {
        const call = item.call;
        try {
          const parsedArgs = parseOpenAIFunctionArgs(call.arguments);
          const outputText = executeOpenAITool(call.name, parsedArgs, containerInput);
          outputs.push({
            type: 'function_call_output',
            call_id: call.call_id,
            output: outputText,
          });
          log(`Function tool succeeded: ${call.name}`);
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          outputs.push({
            type: 'function_call_output',
            call_id: call.call_id,
            output: `Tool error: ${message}`,
          });
          log(`Function tool failed: ${call.name} (${message})`);
        }
        continue;
      }

      try {
        const localShellOutput = await executeLocalShellToolCall(item.call, sdkEnv);
        outputs.push(localShellOutput);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        outputs.push({
          type: 'local_shell_call_output',
          id: item.call.id,
          output: JSON.stringify({ error: message }),
        });
        log(`Local shell tool failed: ${message}`);
      }
    }

    input = outputs;
    response = await createResponse(input, previousResponseId, false);
    previousResponseId = response.id;
  }

  const remainingCalls = extractOpenAIToolCalls(response);
  if (remainingCalls.length > 0) {
    throw new Error(`OpenAI tool loop exceeded ${MAX_OPENAI_TOOL_TURNS} turns`);
  }

  const text = extractOpenAIText(response);
  writeOutput({
    status: 'success',
    result: text || null,
    newSessionId: response.id,
  });

  return {
    newSessionId: response.id,
    lastAssistantUuid: undefined,
    closedDuringQuery: false,
  };
}

async function runQuery(
  prompt: string,
  sessionId: string | undefined,
  mcpServerPath: string,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
  resumeAt?: string,
): Promise<{ newSessionId?: string; lastAssistantUuid?: string; closedDuringQuery: boolean }> {
  const provider = normalizeProvider(sdkEnv.NANOCLAW_AGENT_PROVIDER);
  if (provider === 'claude') {
    return runClaudeQuery(prompt, sessionId, mcpServerPath, containerInput, sdkEnv, resumeAt);
  }
  return runOpenAIQuery(prompt, sessionId, mcpServerPath, containerInput, sdkEnv, resumeAt);
}

async function main(): Promise<void> {
  let containerInput: ContainerInput;

  try {
    const stdinData = await readStdin();
    containerInput = JSON.parse(stdinData);
    // Delete the temp file the entrypoint wrote — it contains secrets
    try { fs.unlinkSync('/tmp/input.json'); } catch { /* may not exist */ }
    log(`Received input for group: ${containerInput.groupFolder}`);
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`
    });
    process.exit(1);
  }

  // Build SDK env: merge secrets into process.env for the SDK only.
  // Secrets never touch process.env itself, so Bash subprocesses can't see them.
  const sdkEnv: Record<string, string | undefined> = { ...process.env };
  for (const [key, value] of Object.entries(containerInput.secrets || {})) {
    sdkEnv[key] = value;
  }
  const provider = normalizeProvider(sdkEnv.NANOCLAW_AGENT_PROVIDER);
  log(`Agent provider: ${provider}`);

  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'ipc-mcp-stdio.js');

  let sessionId = containerInput.sessionId;
  fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });

  // Clean up stale _close sentinel from previous container runs
  try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }

  // Build initial prompt (drain any pending IPC messages too)
  let prompt = containerInput.prompt;
  if (containerInput.isScheduledTask) {
    prompt = `[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n${prompt}`;
  }
  const pending = drainIpcInput();
  if (pending.length > 0) {
    log(`Draining ${pending.length} pending IPC messages into initial prompt`);
    prompt += '\n' + pending.join('\n');
  }

  // Query loop: run query → wait for IPC message → run new query → repeat
  let resumeAt: string | undefined;
  try {
    while (true) {
      log(`Starting query (provider: ${provider}, session: ${sessionId || 'new'}, resumeAt: ${resumeAt || 'latest'})...`);

      const queryResult = await runQuery(prompt, sessionId, mcpServerPath, containerInput, sdkEnv, resumeAt);
      if (queryResult.newSessionId) {
        sessionId = queryResult.newSessionId;
      }
      if (queryResult.lastAssistantUuid) {
        resumeAt = queryResult.lastAssistantUuid;
      }

      // If _close was consumed during the query, exit immediately.
      // Don't emit a session-update marker (it would reset the host's
      // idle timer and cause a 30-min delay before the next _close).
      if (queryResult.closedDuringQuery) {
        log('Close sentinel consumed during query, exiting');
        break;
      }

      // Emit session update so host can track it
      writeOutput({ status: 'success', result: null, newSessionId: sessionId });

      log('Query ended, waiting for next IPC message...');

      // Wait for the next message or _close sentinel
      const nextMessage = await waitForIpcMessage();
      if (nextMessage === null) {
        log('Close sentinel received, exiting');
        break;
      }

      log(`Got new message (${nextMessage.length} chars), starting new query`);
      prompt = nextMessage;
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    log(`Agent error: ${errorMessage}`);
    writeOutput({
      status: 'error',
      result: null,
      newSessionId: sessionId,
      error: errorMessage
    });
    process.exit(1);
  }
}

main();
