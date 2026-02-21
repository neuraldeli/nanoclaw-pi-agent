# Setup

1. Install dependencies:
```bash
npm ci
```

2. Run wizard:
```bash
npm run setup
```

The wizard configures:
- `TELEGRAM_BOT_TOKEN`
- OpenAI OAuth tokens (`OPENAI_OAUTH_*`)
- `NANOCLAW_AGENT_PROVIDER=openai`

3. Build and run:
```bash
npm run build
npm run dev
```
