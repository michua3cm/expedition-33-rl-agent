---
name: init-workspace
description: Loads core coding skills and automatically executes local branch cleanup. Trigger this immediately when the user types "Init" or "Init <branch-name>".
---

# Workspace Initialization Protocol

You must execute this setup sequence completely without stopping to ask the user any questions. Do not attempt to push any branches to the remote repository.

## 1. Load Core Project Rules

Silently read the following files to load them into your active context:

- `.claude/skills/coding-planner/SKILL.md`
- `.claude/skills/commit-generator/SKILL.md`

## 2. Zero-Friction Local Branch Correction (CRITICAL)

The system booted into a default branch named `claude/<hash>`. You must fix this locally and instantly.

1. Sync remote state: Run `git fetch origin` to ensure you have the latest repository map.
2. Determine the target branch name:
   - If the user provided a name in their prompt (e.g., `init feat/vision-module`), use that exact name.
   - If the user only typed `init`, generate a fallback name using the current date: `wip/session-MMDD` (e.g., `wip/session-0409`).
3. Execute the branch swap:
   - Try to check out the branch normally (in case it already exists remotely): `git checkout <target-branch-name>`
   - If that fails (because it is a brand new branch), create it: `git checkout -b <target-branch-name>`
4. Delete the ugly original branch locally to clean up the VM: `git branch -D <the-original-claude-branch-name>`

## 3. Final Output

Once the rules are loaded and the local Git operations are successfully completed, output this exact confirmation:
_"Workspace initialized. Core skills loaded. Branch locally swapped to [<target-branch-name>] and old local branch deleted. Ready for planning mode."_
