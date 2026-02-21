# NanoClaw

Personal Codex assistant. See `README.md` for setup and architecture.

## Quick Context

Single Node.js process that receives Telegram messages, routes them to an OpenAI/Codex agent in isolated containers, and sends replies back.

## Key Files

- `src/index.ts`: orchestrator (state, polling loop, routing)
- `src/channels/telegram.ts`: Telegram channel integration
- `src/ipc.ts`: IPC watcher and task processing
- `src/container-runner.ts`: spawns and monitors container agents
- `container/agent-runner/src/index.ts`: OpenAI tool loop runtime in container
- `src/task-scheduler.ts`: recurring/one-off scheduled task execution
- `src/db.ts`: SQLite persistence for messages, sessions, groups, tasks
- `groups/<name>/CODEX.md`: per-group working memory/instructions

## Development

Run commands directly in this repo:

```bash
npm run setup
npm run dev
npm run build
./container/build.sh
```
