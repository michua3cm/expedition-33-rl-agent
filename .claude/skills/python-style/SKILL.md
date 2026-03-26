---
name: python-style
description: Enforces strict Python type hinting, asynchronous FastAPI routing, and Pydantic validation. Trigger this when generating, reviewing, or refactoring Python or FastAPI code.
---

# Python & FastAPI Style Guide

You are an expert backend developer. When writing or reviewing Python and FastAPI code, you must strictly adhere to the following standards.

## 1. Strict Type Hinting

- **Universal Typing:** Every function signature must have explicit type hints for all arguments and the return type.
- **Modern Syntax:** Use the built-in modern typing syntax (e.g., `list[str]`, `str | None` instead of `List[str]`, `Optional[str]`).

## 2. FastAPI & Pydantic

- **Data Validation:** All request payloads and response bodies must be defined using Pydantic `BaseModel` classes. Do not parse raw dictionaries.
- **Dependency Injection:** Use FastAPI's `Depends()` for database sessions, authentication checks, and shared logic.

## 3. Asynchronous Execution (CRITICAL)

- **I/O Bound Tasks:** Any route performing network requests, database queries, or file I/O must be defined with `async def`.
- **Blocking Code Warning:** You must actively scan for and flag synchronous, blocking calls (like standard `requests.get` or synchronous database drivers) inside an `async def` function. Suggest asynchronous alternatives (e.g., `httpx`, `asyncpg`).
