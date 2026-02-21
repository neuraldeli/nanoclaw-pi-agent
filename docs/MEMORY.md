# Memory

NanoClaw uses filesystem memory plus OpenAI response continuity.

## Filesystem Memory

- Global instructions: `groups/global/CODEX.md`
- Main/admin instructions: `groups/main/CODEX.md`
- Group-specific notes/files: `groups/<group-folder>/`
- Conversation summaries/artifacts: `groups/<group-folder>/conversations/`

## Session Continuity

- Session IDs are stored in SQLite (`store/messages.db`, `sessions` table).
- The container runner passes the prior session ID as `previous_response_id` to OpenAI.
- This keeps context across turns while allowing memory files for durable facts.
