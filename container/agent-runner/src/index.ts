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
import { createHash } from 'crypto';
import type {
  ResponseCreateParamsNonStreaming,
  ResponseCreateParamsStreaming,
  ResponseIncludable,
  ResponseInput,
} from 'openai/resources/responses/responses';

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
const MAX_OPENAI_TOOL_TURNS = 24;
const OPENAI_OAUTH_TOKEN_URL = 'https://auth.openai.com/oauth/token';
const DEFAULT_OPENAI_OAUTH_CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann';
const DEFAULT_OPENAI_OAUTH_INSTRUCTIONS = [
  'You are NanoClaw, a coding assistant running inside a containerized execution environment.',
  'When users ask you to build, change, or deploy something, act directly by using available tools.',
  'Prefer execution over instructions. Do not just list steps unless a required credential/permission is missing.',
  'You can access the public internet and external APIs through shell commands (for example curl, git, npm, apt) when network is available.',
  'When users ask to interact with a website or endpoint, attempt it directly via shell tools first instead of saying you cannot access websites.',
  'Only report inability to access a website after an actual command attempt fails, and include the concrete failure.',
  'Never claim external access is blocked unless you have first run a command that proves it.',
  'For deployment/build tasks, start by creating/changing files and running commands immediately; ask for auth only at the exact blocking step.',
  'If blocked by missing auth/secrets, ask only for the minimum required information and continue execution.',
].join(' ');
const DEFAULT_OPENAI_OAUTH_ORIGINATOR = 'codex_cli';
const OPENAI_PARALLEL_TOOL_CALLS_ENV = 'OPENAI_PARALLEL_TOOL_CALLS';
const OPENAI_OAUTH_ORIGINATOR_ENV = 'OPENAI_OAUTH_ORIGINATOR';
const OPENAI_OAUTH_ENABLE_FUNCTION_TOOLS_ENV = 'OPENAI_OAUTH_ENABLE_FUNCTION_TOOLS';
const OPENAI_OAUTH_ENABLE_WEB_SEARCH_ENV = 'OPENAI_OAUTH_ENABLE_WEB_SEARCH';
const NANOCLAW_AGENT_VERSION = process.env.NANOCLAW_VERSION || process.env.npm_package_version || '1.0.0';

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

function toUuidLikeFromSeed(seed: string): string {
  const hex = createHash('sha256').update(seed).digest('hex').slice(0, 32);
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20, 32)}`;
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
      cf_ray?: string;
      body_preview?: string;
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
    if (typeof record.cf_ray === 'string' && record.cf_ray) {
      details.push(`cf_ray=${record.cf_ray}`);
    }
    if (typeof record.body_preview === 'string' && record.body_preview) {
      details.push(`body=${truncate(record.body_preview, 300)}`);
    }
    if (isRecord(record.error) && typeof record.error.message === 'string') {
      details.push(`api_error=${truncate(record.error.message)}`);
    }
    return details.join(' | ');
  }
  return String(err);
}

function buildOAuthUserAgent(originator: string): string {
  return `${originator}/${NANOCLAW_AGENT_VERSION} (${process.platform}; ${process.arch}) nanoclaw-agent-runner`;
}

function createHttpStatusError(
  status: number,
  message: string,
  requestId?: string,
  cfRay?: string,
  bodyPreview?: string,
): Error {
  const err = new Error(message) as Error & {
    status?: number;
    request_id?: string;
    cf_ray?: string;
    body_preview?: string;
  };
  err.status = status;
  if (requestId) {
    err.request_id = requestId;
  }
  if (cfRay) {
    err.cf_ray = cfRay;
  }
  if (bodyPreview) {
    err.body_preview = bodyPreview;
  }
  return err;
}

async function parseOpenAIStreamingResponse(response: Response): Promise<OpenAIResponseLike> {
  if (!response.body) {
    throw new Error('OpenAI streaming response did not include a body');
  }
  const reader = response.body as unknown as AsyncIterable<Uint8Array>;
  const decoder = new TextDecoder();
  let buffer = '';
  let completedResponse: OpenAIResponseLike | undefined;

  const handleBlock = (block: string): void => {
    if (!block.trim()) return;
    const dataLines = block
      .split('\n')
      .filter(line => line.startsWith('data:'))
      .map(line => line.slice(5).trimStart());
    if (dataLines.length === 0) return;
    const rawData = dataLines.join('\n').trim();
    if (!rawData || rawData === '[DONE]') return;

    let eventPayload: unknown;
    try {
      eventPayload = JSON.parse(rawData);
    } catch (err) {
      throw new Error(`Failed to parse OpenAI SSE event: ${formatAgentError(err)}`);
    }
    if (!isRecord(eventPayload)) return;
    const eventType = typeof eventPayload.type === 'string' ? eventPayload.type : '';
    if (eventType === 'response.completed' && isRecord(eventPayload.response)) {
      completedResponse = eventPayload.response as unknown as OpenAIResponseLike;
      return;
    }
    if (eventType === 'response.failed') {
      const eventResponse = isRecord(eventPayload.response) ? eventPayload.response : undefined;
      const eventError = eventResponse && isRecord(eventResponse.error) ? eventResponse.error : undefined;
      const message = eventError && typeof eventError.message === 'string'
        ? eventError.message
        : 'OpenAI streaming response failed';
      throw new Error(message);
    }
    if (eventType === 'error') {
      const message = typeof eventPayload.message === 'string'
        ? eventPayload.message
        : 'OpenAI streaming error';
      throw new Error(message);
    }
  };

  for await (const chunk of reader) {
    buffer += decoder.decode(chunk, { stream: true });
    buffer = buffer.replace(/\r\n/g, '\n');
    let splitAt = buffer.indexOf('\n\n');
    while (splitAt >= 0) {
      const block = buffer.slice(0, splitAt);
      buffer = buffer.slice(splitAt + 2);
      handleBlock(block);
      splitAt = buffer.indexOf('\n\n');
    }
  }
  const tail = decoder.decode();
  if (tail) {
    buffer += tail;
    buffer = buffer.replace(/\r\n/g, '\n');
  }
  if (buffer.trim()) {
    handleBlock(buffer);
  }
  if (!completedResponse) {
    throw new Error('OpenAI streaming response ended without response.completed');
  }
  return completedResponse;
}

async function createOpenAIOAuthStreamingResponse(
  baseURL: string,
  bearerToken: string,
  headers: Record<string, string>,
  body: ResponseCreateParamsStreaming,
): Promise<OpenAIResponseLike> {
  const requestUrl = `${baseURL.replace(/\/+$/, '')}/responses`;
  const response = await fetch(requestUrl, {
    method: 'POST',
    headers: {
      ...headers,
      authorization: `Bearer ${bearerToken}`,
      accept: 'text/event-stream',
      'content-type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const rawBody = await response.text();
    const bodyPreview = truncate(rawBody || '(empty)', 700);
    const requestId = response.headers.get('x-request-id') || response.headers.get('x-oai-request-id') || undefined;
    const cfRay = response.headers.get('cf-ray') || undefined;
    throw createHttpStatusError(
      response.status,
      `OpenAI OAuth responses request failed (${response.status})`,
      requestId,
      cfRay,
      bodyPreview,
    );
  }

  return await parseOpenAIStreamingResponse(response);
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

const OPENAI_OAUTH_TOOLS: OpenAITool[] = [
  { type: 'local_shell' },
];

function isEnvFalse(value: string | undefined): boolean {
  if (!value) return false;
  const normalized = value.trim().toLowerCase();
  return normalized === '0' || normalized === 'false' || normalized === 'no' || normalized === 'off';
}

function isEnvTrue(value: string | undefined): boolean {
  if (!value) return false;
  const normalized = value.trim().toLowerCase();
  return normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on';
}

function buildOpenAIOAuthTools(sdkEnv: Record<string, string | undefined>): OpenAITool[] {
  const tools: OpenAITool[] = [...OPENAI_OAUTH_TOOLS];
  const includeFunctionTools = !isEnvFalse(sdkEnv[OPENAI_OAUTH_ENABLE_FUNCTION_TOOLS_ENV]);
  if (includeFunctionTools) {
    tools.push(...OPENAI_FUNCTION_TOOLS);
  }
  const includeWebSearch = isEnvTrue(sdkEnv[OPENAI_OAUTH_ENABLE_WEB_SEARCH_ENV]);
  if (includeWebSearch) {
    tools.push({ type: 'web_search_preview' });
  }
  return tools;
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

function requireOAuthInstructions(instructions: string | null | undefined): string {
  const trimmed = instructions?.trim();
  if (trimmed) return trimmed;
  return DEFAULT_OPENAI_OAUTH_INSTRUCTIONS;
}

function extractFirstHttpUrl(text: string): string | undefined {
  const match = text.match(/\bhttps?:\/\/[^\s<>"')]+/i);
  return match?.[0];
}

function augmentOAuthPromptForExecution(prompt: string): string {
  const url = extractFirstHttpUrl(prompt);
  if (!url) return prompt;
  return [
    prompt,
    '',
    '[Execution requirement]',
    `Because this request includes URL ${url}, run at least one local_shell command against it first (for example: curl -I --max-time 12 ${url}).`,
    'Do not claim website/network inaccessibility unless that command fails and you report the concrete error output.',
  ].join('\n');
}

function isLikelyActionablePrompt(prompt: string): boolean {
  return /\b(build|create|deploy|publish|push|host|website|site|api|endpoint|http|https|curl|git|npm|docker|ssh|script|install|configure|run|fix)\b/i.test(prompt);
}

function containsCapabilityRefusal(text: string): boolean {
  return /can(?:'|’)?t access external (?:websites|sites)|cannot access external (?:websites|sites)|network access appears blocked|cannot directly operate|from this environment right now/i.test(text);
}

function shouldForceOAuthExecutionRetry(prompt: string, text: string): boolean {
  return isLikelyActionablePrompt(prompt) && containsCapabilityRefusal(text);
}

function buildForcedExecutionRetryPrompt(prompt: string): string {
  const url = extractFirstHttpUrl(prompt);
  const recommendedCommand = url
    ? `curl -I --max-time 12 ${url}`
    : 'bash -lc "pwd && ls -la"';
  return [
    augmentOAuthPromptForExecution(prompt),
    '',
    '[Hard execution requirement]',
    'Your prior answer was invalid because it claimed inability without proving it.',
    `Invoke local_shell at least once before giving a final answer. Recommended first command: ${recommendedCommand}`,
    'If a command fails, report exact stderr and continue with the next best command.',
  ].join('\n');
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
  const supportsPreviousResponseId = hasApiKey;
  const baseURL =
    sdkEnv.OPENAI_BASE_URL ||
    (hasApiKey ? undefined : CHATGPT_CODEX_BASE_URL);
  const oauthBaseURL = (sdkEnv.OPENAI_BASE_URL || CHATGPT_CODEX_BASE_URL).replace(/\/+$/, '');
  const allowParallelToolCalls = sdkEnv[OPENAI_PARALLEL_TOOL_CALLS_ENV]?.toLowerCase() === 'true'
    ? true
    : hasApiKey;
  const tools: OpenAITool[] = hasApiKey
    ? OPENAI_TOOLS
    : buildOpenAIOAuthTools(sdkEnv);
  const useStreaming = !hasApiKey;
  const chatgptAccountId = hasApiKey
    ? undefined
    : extractChatgptAccountIdFromJwt(apiKey);
  const oauthOriginator = (sdkEnv[OPENAI_OAUTH_ORIGINATOR_ENV]?.trim() || DEFAULT_OPENAI_OAUTH_ORIGINATOR);
  const oauthHeaders: Record<string, string> | undefined = hasApiKey
    ? undefined
    : {
      session_id: toUuidLikeFromSeed(`nanoclaw:${containerInput.chatJid}`),
      originator: oauthOriginator,
      version: NANOCLAW_AGENT_VERSION,
      'User-Agent': buildOAuthUserAgent(oauthOriginator),
      ...(chatgptAccountId ? { 'ChatGPT-Account-ID': chatgptAccountId } : {}),
    };

  const client = hasApiKey
    ? new OpenAI({
      apiKey,
      baseURL,
    })
    : undefined;

  const model = sdkEnv.OPENAI_MODEL || DEFAULT_OPENAI_MODEL;
  let activeModel = model;
  const baseInstructions = buildOpenAIInstructions(containerInput);
  const instructions = hasApiKey
    ? baseInstructions
    : requireOAuthInstructions(baseInstructions);

  const createResponse = async (
    input: string | OpenAIToolOutputInput[],
    previousResponseId: string | undefined,
    allowResumeFallback: boolean,
  ): Promise<OpenAIResponseLike> => {
    const createResponseRaw = async (
      body: ResponseCreateParamsNonStreaming,
    ): Promise<OpenAIResponseLike> => {
      if (!client) {
        throw new Error('OpenAI API client unavailable for non-streaming request path');
      }
      return await client.responses.create(body) as OpenAIResponseLike;
    };
    const createOAuthResponseRaw = async (
      body: ResponseCreateParamsStreaming,
    ): Promise<OpenAIResponseLike> => {
      if (!oauthHeaders) {
        throw new Error('OAuth headers unavailable for OpenAI OAuth request');
      }
      return await createOpenAIOAuthStreamingResponse(oauthBaseURL, apiKey, oauthHeaders, body);
    };

    const requestInput: string | ResponseInput = (!hasApiKey && typeof input === 'string')
      ? [{
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: input }],
      }]
      : (input as unknown as ResponseInput | string);
    const include: ResponseIncludable[] = [];
    const requestBodyBase: ResponseCreateParamsNonStreaming = {
      model: activeModel,
      input: requestInput,
      instructions,
      tools,
      tool_choice: 'auto' as const,
      parallel_tool_calls: allowParallelToolCalls,
      store: false,
      include,
      stream: false,
    };
    if (supportsPreviousResponseId && previousResponseId) {
      requestBodyBase.previous_response_id = previousResponseId;
    }
    const runStreamingWithCompatibility = async (
      baseBody: ResponseCreateParamsNonStreaming,
      allowCompatibilityFallback: boolean,
    ): Promise<OpenAIResponseLike> => {
      const requiredInstructions = requireOAuthInstructions(baseBody.instructions);
      const streamBody: ResponseCreateParamsStreaming = {
        ...baseBody,
        stream: true,
      };
      try {
        return await createOAuthResponseRaw(streamBody);
      } catch (streamErr) {
        const streamStatus = errorStatusCode(streamErr);
        if (!allowCompatibilityFallback || streamStatus !== 400) {
          throw streamErr;
        }
        log('OAuth request rejected with tools (400). Retrying without tools.');
        const noToolsBody: ResponseCreateParamsStreaming = {
          model: baseBody.model,
          input: requestInput,
          instructions: requiredInstructions,
          stream: true,
          store: false,
        };
        try {
          return await createOAuthResponseRaw(noToolsBody);
        } catch (noToolsErr) {
          const noToolsStatus = errorStatusCode(noToolsErr);
          if (noToolsStatus !== 400) {
            throw noToolsErr;
          }
          log('OAuth request rejected without tools (400). Retrying minimal payload.');
          const minimalBody: ResponseCreateParamsStreaming = {
            model: baseBody.model,
            input: requestInput,
            instructions: requiredInstructions,
            stream: true,
          };
          return await createOAuthResponseRaw(minimalBody);
        }
      }
    };
    const createOnce = async (): Promise<OpenAIResponseLike> => {
      if (!useStreaming) {
        return await createResponseRaw(requestBodyBase);
      }
      return await runStreamingWithCompatibility(
        requestBodyBase,
        !previousResponseId && typeof input === 'string',
      );
    };

    try {
      return await createOnce();
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
        const fallbackBody: ResponseCreateParamsNonStreaming = {
          model: activeModel,
          input: requestInput,
          instructions,
          tools,
          tool_choice: 'auto' as const,
          parallel_tool_calls: false,
          store: false,
          include,
          stream: false,
        };
        if (!useStreaming) {
          return await createResponseRaw(fallbackBody);
        }
        return await runStreamingWithCompatibility(
          fallbackBody,
          !previousResponseId && typeof input === 'string',
        );
      }
      if (!allowResumeFallback || !previousResponseId) throw err;
      const msg = formatAgentError(err);
      log(`OpenAI resume failed for session ${previousResponseId}: ${msg}. Retrying without previous_response_id.`);
      const resumedBody: ResponseCreateParamsNonStreaming = {
        model: activeModel,
        input: requestInput,
        instructions,
        tools,
        tool_choice: 'auto' as const,
        parallel_tool_calls: allowParallelToolCalls,
        store: false,
        include,
        stream: false,
      };
      if (!useStreaming) {
        return await createResponseRaw(resumedBody);
      }
      return await runStreamingWithCompatibility(resumedBody, false);
    }
  };

  const runTurn = async (
    initialInput: string | OpenAIToolOutputInput[],
    initialPreviousResponseId: string | undefined,
    allowResumeFallback: boolean,
  ): Promise<{
    response: OpenAIResponseLike;
    usageTotals: OpenAITokenUsage;
    previousResponseId: string | undefined;
  }> => {
    let response = await createResponse(initialInput, initialPreviousResponseId, allowResumeFallback);
    let usageTotals = extractOpenAIUsage(response);
    let previousResponseId = supportsPreviousResponseId ? response.id : undefined;

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

      response = await createResponse(outputs, previousResponseId, supportsPreviousResponseId);
      usageTotals = addUsageTotals(usageTotals, extractOpenAIUsage(response));
      previousResponseId = supportsPreviousResponseId ? response.id : undefined;
    }

    const remainingCalls = extractOpenAIToolCalls(response);
    if (remainingCalls.length > 0) {
      throw new Error(`OpenAI tool loop exceeded ${MAX_OPENAI_TOOL_TURNS} turns`);
    }

    return { response, usageTotals, previousResponseId };
  };

  log(`Running OpenAI query with model ${activeModel} (session: ${supportsPreviousResponseId ? (sessionId || 'new') : 'new'})`);
  const firstInput: string | OpenAIToolOutputInput[] = hasApiKey
    ? prompt
    : augmentOAuthPromptForExecution(prompt);
  let turnResult = await runTurn(
    firstInput,
    supportsPreviousResponseId ? sessionId : undefined,
    supportsPreviousResponseId,
  );

  let response = turnResult.response;
  let usageTotals = turnResult.usageTotals;
  let text = extractOpenAIText(response);

  if (!hasApiKey && shouldForceOAuthExecutionRetry(prompt, text)) {
    log('Detected non-executing capability refusal in OAuth mode. Retrying with forced execution backstop.');
    const retryPrompt = buildForcedExecutionRetryPrompt(prompt);
    const retryResult = await runTurn(retryPrompt, undefined, false);
    response = retryResult.response;
    usageTotals = addUsageTotals(usageTotals, retryResult.usageTotals);
    text = extractOpenAIText(response);
  }

  writeOutput({
    status: 'success',
    result: text || null,
    newSessionId: supportsPreviousResponseId ? response.id : undefined,
    usage: usageTotals,
    model: activeModel,
  });

  return {
    newSessionId: supportsPreviousResponseId ? response.id : undefined,
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
