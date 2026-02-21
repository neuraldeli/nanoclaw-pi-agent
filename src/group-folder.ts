import path from 'path';

import { GROUPS_DIR } from './config.js';

const GROUP_FOLDER_PATTERN = /^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$/;

export function validateGroupFolder(folderRaw: string): string {
  const folder = folderRaw.trim();
  if (!GROUP_FOLDER_PATTERN.test(folder)) {
    throw new Error(
      'Invalid folder. Use lowercase letters, numbers, and hyphens only (1-64 chars).',
    );
  }

  const resolved = path.resolve(GROUPS_DIR, folder);
  const relative = path.relative(GROUPS_DIR, resolved);
  if (
    !relative ||
    relative.startsWith('..') ||
    path.isAbsolute(relative)
  ) {
    throw new Error('Folder must stay within the groups directory.');
  }

  return folder;
}
