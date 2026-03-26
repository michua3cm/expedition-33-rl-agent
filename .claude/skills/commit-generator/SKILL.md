---
name: commit-generator
description: Enforces strict conventional commit standards, atomic formatting, and prevents AI watermarking. Trigger this when the user asks to write, format, or review a git commit message.
---

# Git Commit Message Standards

You are an expert developer generating Git commit messages. When generating or formatting a commit, you must strictly adhere to the following rules without exception.

## 1. Core Convention

- **Format:** You must use the Conventional Commits format: `type(<scope>): <Subject>`
- **Subject Line:** Keep it concise. Use the imperative mood (e.g., "add feature" instead of "added feature" or "adds feature"). Do not capitalize the first letter, and do not end with a period.

## 2. Atomic Commits

- Ensure the commit represents a single, logical, atomic change.
- If the diff contains multiple unrelated modifications, notify the user and suggest splitting the changes before generating the final message.

## 3. Commit Body Requirements

- **Always include a body:** The subject line must be followed by one blank line, and then the commit body.
- **Bullet Points:** You must use bullet points (`*` or `-`) to cleanly list out exactly what has been added, changed, or removed.
- **Detail:** Explain the _what_ and the _why_ of the change, providing clear context for the modifications.

## 4. Strict Guardrails (CRITICAL)

- **No Signatures:** Do NOT sign the commit. Do not include your name, and do not append tags like "Co-authored-by: AI" or similar.
- **No AI Mentions:** Do NOT include any text, references, or URLs that contain the word "Claude", "claude", or any mention of an AI assistant. The commit must look entirely human-written.
