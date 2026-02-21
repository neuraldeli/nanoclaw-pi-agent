import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import readline from 'readline/promises';

import { getRegisteredGroup, initDatabase, setRegisteredGroup } from './db.js';

const DEFAULT_ASSISTANT_NAME = 'Andy';
const DEFAULT_MODEL = 'gpt-5-codex';
const DEFAULT_CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann';
const OPENAI_OAUTH_AUTHORIZE_URL = 'https://auth.openai.com/oauth/authorize';
const OPENAI_OAUTH_TOKEN_URL = 'https://auth.openai.com/oauth/token';
const OPENAI_OAUTH_REDIRECT_URI = 'http://localhost:1455/auth/callback';
const OPENAI_OAUTH_SCOPE =
  'openid profile email offline_access api.responses.read api.responses.write';

interface OAuthTokenResponse {
  access_token?: string;
  refresh_token?: string;
  expires_in?: number;
}

function base64UrlEncode(buffer: Buffer): string {
  return buffer
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/g, '');
}

function generateCodeVerifier(): string {
  return base64UrlEncode(crypto.randomBytes(64));
}

function generateCodeChallenge(verifier: string): string {
  return base64UrlEncode(crypto.createHash('sha256').update(verifier).digest());
}

function parseEnvFile(filePath: string): Record<string, string> {
  if (!fs.existsSync(filePath)) return {};

  const env: Record<string, string> = {};
  const lines = fs.readFileSync(filePath, 'utf-8').split('\n');

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    const eqIdx = trimmed.indexOf('=');
    if (eqIdx === -1) continue;

    const key = trimmed.slice(0, eqIdx).trim();
    const value = trimmed.slice(eqIdx + 1).trim();
    env[key] = value;
  }

  return env;
}

function upsertEnv(filePath: string, updates: Record<string, string>): void {
  const lines = fs.existsSync(filePath)
    ? fs.readFileSync(filePath, 'utf-8').split('\n')
    : [];

  const keyToIndex = new Map<string, number>();
  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eqIdx = trimmed.indexOf('=');
    if (eqIdx === -1) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    keyToIndex.set(key, i);
  }

  for (const [key, value] of Object.entries(updates)) {
    const line = `${key}=${value}`;
    const existingIndex = keyToIndex.get(key);
    if (existingIndex !== undefined) {
      lines[existingIndex] = line;
    } else {
      lines.push(line);
    }
  }

  const output = lines.join('\n').replace(/\n+$/, '') + '\n';
  fs.writeFileSync(filePath, output);
}

function parseYesNo(input: string, defaultYes: boolean): boolean {
  const normalized = input.trim().toLowerCase();
  if (!normalized) return defaultYes;
  if (['y', 'yes'].includes(normalized)) return true;
  if (['n', 'no'].includes(normalized)) return false;
  return defaultYes;
}

function buildOAuthUrl(clientId: string, state: string, codeChallenge: string): string {
  const params = new URLSearchParams({
    response_type: 'code',
    client_id: clientId,
    redirect_uri: OPENAI_OAUTH_REDIRECT_URI,
    scope: OPENAI_OAUTH_SCOPE,
    state,
    code_challenge: codeChallenge,
    code_challenge_method: 'S256',
    id_token_add_organizations: 'true',
    codex_cli_simplified_flow: 'true',
    originator: 'pi',
  });

  return `${OPENAI_OAUTH_AUTHORIZE_URL}?${params.toString()}`;
}

async function exchangeAuthCodeForTokens(
  code: string,
  codeVerifier: string,
  clientId: string,
): Promise<OAuthTokenResponse> {
  const body = new URLSearchParams({
    grant_type: 'authorization_code',
    code,
    redirect_uri: OPENAI_OAUTH_REDIRECT_URI,
    client_id: clientId,
    code_verifier: codeVerifier,
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
    throw new Error(`Token exchange failed (${response.status}): ${text.slice(0, 300)}`);
  }

  return await response.json() as OAuthTokenResponse;
}

function parseCallbackUrl(input: string): URL {
  try {
    return new URL(input.trim());
  } catch {
    throw new Error('Invalid URL. Paste the full callback URL from your browser address bar.');
  }
}

async function run(): Promise<void> {
  const projectRoot = process.cwd();
  const envPath = path.join(projectRoot, '.env');
  const env = parseEnvFile(envPath);

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  try {
    console.log('\nNanoClaw Setup (Telegram + OpenAI OAuth)\n');

    const assistantNameInput = await rl.question(
      `Assistant name [${env.ASSISTANT_NAME || DEFAULT_ASSISTANT_NAME}]: `,
    );
    const assistantName = assistantNameInput.trim() || env.ASSISTANT_NAME || DEFAULT_ASSISTANT_NAME;

    const modelInput = await rl.question(`OpenAI model [${env.OPENAI_MODEL || DEFAULT_MODEL}]: `);
    const openaiModel = modelInput.trim() || env.OPENAI_MODEL || DEFAULT_MODEL;

    const telegramPromptDefault = env.TELEGRAM_BOT_TOKEN ? ' (press Enter to keep current value)' : '';
    const telegramInput = await rl.question(`Telegram bot token from @BotFather${telegramPromptDefault}: `);
    const telegramToken = telegramInput.trim() || env.TELEGRAM_BOT_TOKEN || '';

    if (!telegramToken) {
      throw new Error('TELEGRAM_BOT_TOKEN is required. Create a bot in @BotFather and re-run setup.');
    }

    const existingRefresh = env.OPENAI_OAUTH_REFRESH_TOKEN || '';
    let runOAuth = true;
    if (existingRefresh) {
      const reuseAnswer = await rl.question('Reuse existing OpenAI OAuth tokens? [Y/n]: ');
      runOAuth = !parseYesNo(reuseAnswer, true);
    }

    let oauthAccessToken = env.OPENAI_OAUTH_ACCESS_TOKEN || '';
    let oauthRefreshToken = existingRefresh;
    let oauthExpiresAt = env.OPENAI_OAUTH_EXPIRES_AT || '';

    const clientId = env.OPENAI_OAUTH_CLIENT_ID || DEFAULT_CLIENT_ID;

    if (runOAuth) {
      const state = base64UrlEncode(crypto.randomBytes(24));
      const codeVerifier = generateCodeVerifier();
      const codeChallenge = generateCodeChallenge(codeVerifier);
      const authUrl = buildOAuthUrl(clientId, state, codeChallenge);

      console.log('\n1) Open this URL in your browser:');
      console.log(authUrl);
      console.log(`\n2) After login/consent, copy the full redirect URL (starts with ${OPENAI_OAUTH_REDIRECT_URI}) and paste it below.\n`);

      const callbackInput = await rl.question('Paste redirect URL: ');
      const callbackUrl = parseCallbackUrl(callbackInput);

      const returnedState = callbackUrl.searchParams.get('state');
      if (!returnedState || returnedState !== state) {
        throw new Error('OAuth state mismatch. Re-run setup and try again.');
      }

      const code = callbackUrl.searchParams.get('code');
      if (!code) {
        const oauthError = callbackUrl.searchParams.get('error') || 'missing_code';
        throw new Error(`OAuth callback did not include code (${oauthError}).`);
      }

      console.log('\nExchanging auth code...');
      const tokenResponse = await exchangeAuthCodeForTokens(code, codeVerifier, clientId);

      if (!tokenResponse.access_token) {
        throw new Error('OAuth token response missing access_token.');
      }
      if (!tokenResponse.refresh_token) {
        throw new Error('OAuth token response missing refresh_token. Ensure offline_access scope is granted.');
      }

      oauthAccessToken = tokenResponse.access_token;
      oauthRefreshToken = tokenResponse.refresh_token;
      oauthExpiresAt = tokenResponse.expires_in
        ? String(Date.now() + tokenResponse.expires_in * 1000)
        : '';
    }

    const mainChatInput = await rl.question('Main Telegram chat ID (optional, format tg:<id>): ');
    const mainChatId = mainChatInput.trim();

    upsertEnv(envPath, {
      NANOCLAW_AGENT_PROVIDER: 'openai',
      TELEGRAM_BOT_TOKEN: telegramToken,
      ASSISTANT_NAME: assistantName,
      OPENAI_MODEL: openaiModel,
      OPENAI_API_KEY: '',
      OPENAI_OAUTH_CLIENT_ID: clientId,
      OPENAI_OAUTH_ACCESS_TOKEN: oauthAccessToken,
      OPENAI_OAUTH_REFRESH_TOKEN: oauthRefreshToken,
      OPENAI_OAUTH_EXPIRES_AT: oauthExpiresAt,
    });

    if (mainChatId) {
      if (!mainChatId.startsWith('tg:')) {
        throw new Error('Main chat ID must start with tg:.');
      }

      initDatabase();
      const existing = getRegisteredGroup(mainChatId);
      setRegisteredGroup(mainChatId, {
        name: existing?.name || 'Main',
        folder: existing?.folder || 'main',
        trigger: existing?.trigger || `@${assistantName}`,
        added_at: existing?.added_at || new Date().toISOString(),
        containerConfig: existing?.containerConfig,
        requiresTrigger: false,
      });
    }

    console.log('\nSetup complete.');
    console.log('- .env was updated with Telegram + OpenAI OAuth config.');
    if (mainChatId) {
      console.log(`- Registered ${mainChatId} as the main chat.`);
    } else {
      console.log('- No main chat registered yet. Send /chatid to your bot and add one later.');
    }
    console.log('\nNext steps: npm run build && npm run dev\n');
  } finally {
    rl.close();
  }
}

run().catch((err) => {
  console.error(`\nSetup failed: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
