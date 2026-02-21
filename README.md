<div align="center">
  <h1>NanoClaw (Codex Edition)</h1>
  <p>Telegram-first personal assistant powered by OpenAI/Codex, running in isolated containers.</p>
</div>

## What This Repo Is

NanoClaw is a lightweight, single-process Node.js app that:
- receives messages from Telegram,
- routes them to a containerized OpenAI/Codex agent,
- supports scheduled tasks,
- persists sessions/tasks/messages in SQLite,
- keeps per-group memory files under `groups/`.

## Quick Start

1. Install dependencies:
```bash
npm ci
```

2. Run setup wizard:
```bash
npm run setup
```

The setup wizard will:
- ask for your BotFather Telegram token,
- generate and print an OpenAI OAuth URL,
- ask you to paste the localhost redirect URL,
- exchange tokens with PKCE,
- write `.env` for you.

3. Build and run:
```bash
npm run build
npm run dev
```

## Memory Model

- Global memory: `groups/global/CODEX.md`
- Main/admin memory: `groups/main/CODEX.md`
- Per-group memory: `groups/<group-folder>/...`
- Conversation artifacts: `groups/<group-folder>/conversations/`

OpenAI response continuity is tracked via stored response/session IDs in SQLite (`store/messages.db`).

## Telegram IDs

Use `tg:<id>` format, for example:
- `tg:123456789` (private chat)
- `tg:-1001234567890` (group/supergroup)

## Main Scripts

- `npm run setup`: interactive onboarding wizard
- `npm run dev`: run app with tsx
- `npm run build`: compile TypeScript
- `npm test`: run tests
- `./container/build.sh`: rebuild container image
