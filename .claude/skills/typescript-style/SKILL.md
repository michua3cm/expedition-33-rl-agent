---
name: typescript-style
description: Enforces strict TypeScript rules, React functional component patterns, and prevents the use of the `any` type. Trigger this when generating, reviewing, or refactoring TypeScript or React code.
---

# TypeScript & React Style Guide

You are an expert frontend developer. When writing or reviewing TypeScript and React code, you must strictly follow these rules to maintain a robust and scalable architecture.

## 1. Strict Typing (CRITICAL)

- **No `any`:** You are strictly forbidden from using the `any` type. If a type is truly unknown, use `unknown` and perform type narrowing.
- **Interfaces over Types:** Prefer `interface` for object shapes and class contracts. Use `type` only for unions, intersections, or utility types (e.g., `Pick`, `Omit`).
- **Explicit Returns:** Always explicitly type the return value of functions and custom hooks.

## 2. React Patterns

- **Functional Components:** Always use functional components with Hooks. Do not write or suggest class components.
- **Destructuring:** Destructure props directly in the function signature (e.g., `const MyComponent = ({ title, isActive }: Props) => { ... }`).
- **State Management:** Keep local state minimal. For complex state logic, extract it into custom hooks to keep the component body clean.

## 3. Asynchronous Code

- Always use `async/await` instead of `.then()/.catch()` chains.
- Wrap asynchronous operations inside `try/catch` blocks for proper error handling.
