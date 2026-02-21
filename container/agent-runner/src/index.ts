/**
 * NanoClaw Agent Runner
 * Runs inside a container, receives config via stdin, outputs result to stdout.
 */

import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
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
  usage?: OpenAITokenUsage;
  model?: string;
}

interface OpenAITokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

const IPC_INPUT_DIR = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const IPC_POLL_MS = 500;
const IPC_DIR = '/workspace/ipc';
const MESSAGES_DIR = path.join(IPC_DIR, 'messages');
const TASKS_DIR = path.join(IPC_DIR, 'tasks');
const DEFAULT_OPENAI_MODEL = 'gpt-5-codex';
const CHATGPT_CODEX_BASE_URL = 'https://chatgpt.com/backend-api/codex';
const CHATGPT_ORIGINATOR = 'pi';
const MAX_OPENAI_TOOL_TURNS = 24;
const OPENAI_OAUTH_TOKEN_URL = 'https://auth.openai.com/oauth/token';
const DEFAULT_OPENAI_OAUTH_CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann';
const OPENAI_PARALLEL_TOOL_CALLS_ENV = 'OPENAI_PARALLEL_TOOL_CALLS';

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
  model?: string;
  output_text?: string;
  output?: OpenAIResponseOutputItem[];
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
    prompt_tokens?: number;
    completion_tokens?: number;
  };
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

interface OpenAIOAuthTokenResponse {
  access_token?: string;
  refresh_token?: string;
  expires_in?: number;
}

function decodeBase64Url(input: string): string {
  const normalized = input.replace(/-/g, '+').replace(/_/g, '/');
  const pad = normalized.length % 4;
  const padded = pad ? normalized + '='.repeat(4 - pad) : normalized;
  return Buffer.from(padded, 'base64').toString('utf-8');
}

function extractChatgptAccountIdFromJwt(token: string): string | undefined {
  try {
    const parts = token.split('.');
    if (parts.length < 2) return undefined;
    const payloadRaw = decodeBase64Url(parts[1]);
    const payload = JSON.parse(payloadRaw) as {
      [key: string]: unknown;
      'https://api.openai.com/auth'?: {
        chatgpt_account_id?: string;
      };
    };
    const auth = payload['https://api.openai.com/auth'];
    if (!auth || typeof auth !== 'object') return undefined;
    const accountId = (auth as { chatgpt_account_id?: unknown }).chatgpt_account_id;
    return typeof accountId === 'string' && accountId ? accountId : undefined;
  } catch {
    return undefined;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function errorStatusCode(err: unknown): number | undefined {
  if (!isRecord(err)) return undefined;
  const status = err.status;
  return typeof status === 'number' ? status : undefined;
}

function truncate(value: string, max = 280): string {
  if (value.length <= max) return value;
  return `${value.slice(0, max)}...`;
}

function formatAgentError(err: unknown): string {
  if (err instanceof Error) {
    const record = err as Error & {
      status?: number;
      request_id?: string;
      error?: unknown;
      headers?: Record<string, unknown>;
    };
    const details: string[] = [err.message];
    if (typeof record.status === 'number') {
      details.push(`status=${record.status}`);
    }
    if (typeof record.request_id === 'string' && record.request_id) {
      details.push(`request_id=${record.request_id}`);
    }
    if (isRecord(record.error) && typeof record.error.message === 'string') {
      details.push(`api_error=${truncate(record.error.message)}`);
    }
    return details.join(' | ');
  }
  return String(err);
}

const OPENAI_FUNCTION_TOOLS: OpenAIFunctionTool[] = [
  {
    type: 'function',
    name: 'send_message',
    description: "Send a message to the user or group immediately while you're still running.",
    strict: false,
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
    strict: false,
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
    description: 'List scheduled tasks. Main group sees all; other groups only see their own tasks.',
    strict: false,
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
    strict: false,
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
    strict: false,
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
    strict: false,
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
    description: 'Register a new chat so the agent can respond there. Main group only.',
    strict: false,
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        jid: { type: 'string', description: 'Chat ID (e.g., "tg:-123456789")' },
        name: { type: 'string', description: 'Display name for the chat' },
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

// Secrets to strip from local shell subprocess environments.
const SECRET_ENV_VARS = [
  'OPENAI_API_KEY',
  'OPENAI_OAUTH_ACCESS_TOKEN',
  'OPENAI_OAUTH_REFRESH_TOKEN',
  'OPENAI_OAUTH_CLIENT_ID',
  'OPENAI_OAUTH_EXPIRES_AT',
];

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

function extractOpenAIUsage(response: OpenAIResponseLike): OpenAITokenUsage {
  const usage = response.usage || {};
  const promptTokens = usage.input_tokens ?? usage.prompt_tokens ?? 0;
  const completionTokens = usage.output_tokens ?? usage.completion_tokens ?? 0;
  const totalTokens = usage.total_tokens ?? promptTokens + completionTokens;

  return {
    promptTokens: Number.isFinite(promptTokens) ? promptTokens : 0,
    completionTokens: Number.isFinite(completionTokens) ? completionTokens : 0,
    totalTokens: Number.isFinite(totalTokens) ? totalTokens : 0,
  };
}

function addUsageTotals(
  current: OpenAITokenUsage,
  next: OpenAITokenUsage,
): OpenAITokenUsage {
  return {
    promptTokens: current.promptTokens + next.promptTokens,
    completionTokens: current.completionTokens + next.completionTokens,
    totalTokens: current.totalTokens + next.totalTokens,
  };
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
  if (containerInput.isMain) return undefined;

  const globalCodexMdPath = '/workspace/global/CODEX.md';
  if (!fs.existsSync(globalCodexMdPath)) return undefined;

  return fs.readFileSync(globalCodexMdPath, 'utf-8');
}

function parseOptionalEpochMs(raw: string | undefined): number | undefined {
  if (!raw) return undefined;
  const value = Number.parseInt(raw, 10);
  if (!Number.isFinite(value) || value <= 0) return undefined;
  return value;
}

async function refreshOpenAIAccessToken(
  refreshToken: string,
  clientId: string,
): Promise<OpenAIOAuthTokenResponse> {
  const body = new URLSearchParams({
    grant_type: 'refresh_token',
    refresh_token: refreshToken,
    client_id: clientId,
  });

  const response = await fetch(OPENAI_OAUTH_TOKEN_URL, {
    method: 'POST',
    headers: {
      'content-type': 'application/x-www-form-urlencoded',
      accept: 'application/json',
    },
    body,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`OAuth refresh failed (${response.status}): ${text.slice(0, 300)}`);
  }

  return await response.json() as OpenAIOAuthTokenResponse;
}

async function resolveOpenAIApiKey(
  sdkEnv: Record<string, string | undefined>,
): Promise<string> {
  if (sdkEnv.OPENAI_API_KEY) {
    return sdkEnv.OPENAI_API_KEY;
  }

  const refreshToken = sdkEnv.OPENAI_OAUTH_REFRESH_TOKEN;
  const accessToken = sdkEnv.OPENAI_OAUTH_ACCESS_TOKEN;
  const expiresAt = parseOptionalEpochMs(sdkEnv.OPENAI_OAUTH_EXPIRES_AT);
  const now = Date.now();

  if (accessToken && (!expiresAt || now + 60_000 < expiresAt)) {
    return accessToken;
  }

  if (!refreshToken) {
    throw new Error(
      'OpenAI auth not configured. Set OPENAI_API_KEY or run setup to configure OAuth tokens.',
    );
  }

  const clientId = sdkEnv.OPENAI_OAUTH_CLIENT_ID || DEFAULT_OPENAI_OAUTH_CLIENT_ID;
  const tokenResponse = await refreshOpenAIAccessToken(refreshToken, clientId);

  if (!tokenResponse.access_token) {
    throw new Error('OAuth refresh did not return access_token');
  }

  sdkEnv.OPENAI_OAUTH_ACCESS_TOKEN = tokenResponse.access_token;
  if (tokenResponse.refresh_token) {
    sdkEnv.OPENAI_OAUTH_REFRESH_TOKEN = tokenResponse.refresh_token;
  }
  if (typeof tokenResponse.expires_in === 'number' && tokenResponse.expires_in > 0) {
    sdkEnv.OPENAI_OAUTH_EXPIRES_AT = String(Date.now() + tokenResponse.expires_in * 1000);
  }

  return tokenResponse.access_token;
}

async function runOpenAIQuery(
  prompt: string,
  sessionId: string | undefined,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
): Promise<{ newSessionId?: string }> {
  const apiKey = await resolveOpenAIApiKey(sdkEnv);
  const hasApiKey = Boolean(sdkEnv.OPENAI_API_KEY && sdkEnv.OPENAI_API_KEY.trim());
  const baseURL =
    sdkEnv.OPENAI_BASE_URL ||
    (hasApiKey ? undefined : CHATGPT_CODEX_BASE_URL);
  const allowParallelToolCalls = sdkEnv[OPENAI_PARALLEL_TOOL_CALLS_ENV]?.toLowerCase() === 'true'
    ? true
    : hasApiKey;
  const tools: OpenAITool[] = hasApiKey
    ? OPENAI_TOOLS
    : OPENAI_FUNCTION_TOOLS;
  const chatgptAccountId = hasApiKey
    ? undefined
    : extractChatgptAccountIdFromJwt(apiKey);
  const defaultHeaders: Record<string, string> = {};
  if (chatgptAccountId) {
    defaultHeaders['ChatGPT-Account-ID'] = chatgptAccountId;
    defaultHeaders.originator = CHATGPT_ORIGINATOR;
  }

  const client = new OpenAI({
    apiKey,
    baseURL,
    defaultHeaders: Object.keys(defaultHeaders).length > 0
      ? defaultHeaders
      : undefined,
  });

  const model = sdkEnv.OPENAI_MODEL || DEFAULT_OPENAI_MODEL;
  let activeModel = model;
  const instructions = buildOpenAIInstructions(containerInput);

  const createResponse = async (
    input: string | OpenAIToolOutputInput[],
    previousResponseId: string | undefined,
    allowResumeFallback: boolean,
  ): Promise<OpenAIResponseLike> => {
    try {
      return await client.responses.create({
        model: activeModel,
        input,
        previous_response_id: previousResponseId,
        instructions,
        tools,
        tool_choice: 'auto',
        parallel_tool_calls: allowParallelToolCalls,
      }) as OpenAIResponseLike;
    } catch (err) {
      const status = errorStatusCode(err);
      if (
        !hasApiKey &&
        status === 400 &&
        activeModel !== DEFAULT_OPENAI_MODEL &&
        typeof input === 'string' &&
        !previousResponseId
      ) {
        log(
          `OpenAI request failed with model ${activeModel} (status 400). Retrying once with fallback model ${DEFAULT_OPENAI_MODEL}.`,
        );
        activeModel = DEFAULT_OPENAI_MODEL;
        return await client.responses.create({
          model: activeModel,
          input,
          instructions,
          tools,
          tool_choice: 'auto',
          parallel_tool_calls: false,
        }) as OpenAIResponseLike;
      }
      if (!allowResumeFallback || !previousResponseId) throw err;
      const msg = formatAgentError(err);
      log(`OpenAI resume failed for session ${previousResponseId}: ${msg}. Retrying without previous_response_id.`);
      return await client.responses.create({
        model: activeModel,
        input,
        instructions,
        tools,
        tool_choice: 'auto',
        parallel_tool_calls: allowParallelToolCalls,
      }) as OpenAIResponseLike;
    }
  };

  log(`Running OpenAI query with model ${activeModel} (session: ${sessionId || 'new'})`);
  let previousResponseId = sessionId;
  let input: string | OpenAIToolOutputInput[] = prompt;
  let response = await createResponse(input, previousResponseId, true);
  let usageTotals = extractOpenAIUsage(response);
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
    usageTotals = addUsageTotals(usageTotals, extractOpenAIUsage(response));
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
    usage: usageTotals,
    model: activeModel,
  });

  return {
    newSessionId: response.id,
  };
}

async function main(): Promise<void> {
  let containerInput: ContainerInput;

  try {
    const stdinData = await readStdin();
    containerInput = JSON.parse(stdinData);
    // Delete the temp file the entrypoint wrote.
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
  const sdkEnv: Record<string, string | undefined> = { ...process.env };
  for (const [key, value] of Object.entries(containerInput.secrets || {})) {
    sdkEnv[key] = value;
  }

  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'ipc-mcp-stdio.js');
  if (!fs.existsSync(mcpServerPath)) {
    log('MCP server helper not found; continuing because tools are handled in-process');
  }

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

  try {
    while (true) {
      log(`Starting query (session: ${sessionId || 'new'})...`);

      const queryResult = await runOpenAIQuery(prompt, sessionId, containerInput, sdkEnv);
      if (queryResult.newSessionId) {
        sessionId = queryResult.newSessionId;
      }

      // Emit session update so host can track it even when result text is empty.
      writeOutput({ status: 'success', result: null, newSessionId: sessionId });

      log('Query ended, waiting for next IPC message...');

      const nextMessage = await waitForIpcMessage();
      if (nextMessage === null) {
        log('Close sentinel received, exiting');
        break;
      }

      log(`Got new message (${nextMessage.length} chars), starting new query`);
      prompt = nextMessage;
    }
  } catch (err) {
    const errorMessage = formatAgentError(err);
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
