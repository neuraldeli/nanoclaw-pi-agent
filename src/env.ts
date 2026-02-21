import fs from 'fs';
import path from 'path';

/**
 * Parse the .env file and return values for the requested keys.
 * Does NOT load anything into process.env — callers decide what to
 * do with the values. This keeps secrets out of the process environment
 * so they don't leak to child processes.
 */
export function readEnvFile(keys: string[]): Record<string, string> {
  const envFile = path.join(process.cwd(), '.env');
  let content: string;
  try {
    content = fs.readFileSync(envFile, 'utf-8');
  } catch {
    return {};
  }

  const result: Record<string, string> = {};
  const wanted = new Set(keys);

  for (const line of content.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eqIdx = trimmed.indexOf('=');
    if (eqIdx === -1) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    if (!wanted.has(key)) continue;
    let value = trimmed.slice(eqIdx + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (value) result[key] = value;
  }

  return result;
}

/**
 * Upsert key/value pairs in the local .env file.
 * Values are written as raw strings (no shell escaping).
 */
export function upsertEnvFile(updates: Record<string, string>): void {
  const envFile = path.join(process.cwd(), '.env');
  const lines = fs.existsSync(envFile)
    ? fs.readFileSync(envFile, 'utf-8').split('\n')
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
    const idx = keyToIndex.get(key);
    if (idx === undefined) {
      lines.push(line);
    } else {
      lines[idx] = line;
    }
  }

  const output = lines.join('\n').replace(/\n+$/, '') + '\n';
  fs.writeFileSync(envFile, output);
}
