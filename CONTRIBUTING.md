## Contributing

This fork is focused on Telegram + OpenAI/Codex.

Guidelines:
- Keep runtime behavior simple and explicit.
- Prefer direct source changes over framework/automation abstractions.
- Include tests for behavior changes when practical.
- Avoid adding provider-specific code paths unless requested.

Before opening a PR:
```bash
npm run build
npm test
```
