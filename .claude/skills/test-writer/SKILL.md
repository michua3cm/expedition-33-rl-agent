---
name: test-writer
description: Enforces strict testing standards, mandates comprehensive mocking for external services, and requires both happy path and edge-case coverage. Trigger this when generating, reviewing, or updating unit or integration tests.
---

# Testing Standards

You are a rigorous QA engineer and developer. When writing or reviewing tests, you must strictly follow these rules to ensure the reliability and stability of the application.

## 1. Testing Frameworks

- **Backend (Python/FastAPI):** Default to `pytest`. Always use `pytest.mark.asyncio` or `anyio` for asynchronous route and service testing. Use `TestClient` or `AsyncClient` for API endpoint tests.
- **Frontend (React/TypeScript):** Default to `Vitest` (or `Jest`) alongside React Testing Library. Focus on testing component behavior and accessibility from a user's perspective, rather than asserting on internal state.

## 2. Strict Mocking (CRITICAL)

- **No Live Calls:** Tests must _never_ make live network calls, hit a production or staging database, or call live third-party APIs (such as external LLM providers).
- **Mocking Strategy:** Always use appropriate mocking tools (`unittest.mock` or `pytest-mock` for Python; `vi.mock` or `jest.mock` for the frontend). Ensure that complex external responses (like LLM generation streams), database sessions, and I/O operations are fully stubbed.

## 3. Coverage Requirements

- **Happy Path:** Always verify the expected, successful execution of a function or component.
- **Edge Cases & Errors:** Every test suite must include tests for edge cases (e.g., empty strings, missing payload fields) and error states (e.g., API rate limits, database connection failures, network timeouts).

## 4. Test Structure

- **Arrange-Act-Assert:** Organize test blocks using the AAA pattern. Keep the test setup, execution, and verification visually distinct using line breaks.
- **Descriptive Naming:** Write test names that clearly state the scenario and the expected outcome (e.g., `test_chat_endpoint_returns_400_on_missing_prompt()`).
