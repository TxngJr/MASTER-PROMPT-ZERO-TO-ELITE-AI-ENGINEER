# MCP Concepts, Tool Security, and Failure Recovery

## MCP mental model

Model Context Protocol (MCP) is a protocol concept for exposing structured tools/resources/context to an AI application through a defined interface. The important engineering idea is **capability discovery plus typed boundaries**, not unrestricted remote execution.

~~~text
host/application
↕
MCP client
↕ protocol
MCP server
├─ tools
├─ resources
└─ prompts/context capabilities
~~~

Concrete implementations can differ; always follow the versioned protocol/runtime documentation you actually deploy.

## Security boundary

Treat every model-generated tool request and every external tool result as untrusted.

Required controls:

- explicit allowlist of available tools;
- typed argument validation;
- least-privilege credentials;
- per-tool timeout;
- response-size limits;
- maximum tool calls / loop steps;
- filesystem/network sandbox where appropriate;
- redaction of secrets from model-visible context and logs;
- audit trail of requested tool, validated arguments, result status, and policy decision.

## Prompt injection

Retrieved pages/files/tool results may contain text instructing the model to ignore policy or call privileged tools. Data is not authority.

Keep separate:

~~~text
system/developer policy
tool permission policy
user request
untrusted retrieved/tool data
~~~

Never elevate instructions merely because they came from a retrieved source.

## Failure recovery state machine

~~~text
READY
 → CALLING
 → SUCCESS
 → READY

CALLING
 → RETRYABLE_ERROR
 → bounded retry/backoff
 → READY or FAILED

CALLING
 → NON_RETRYABLE_ERROR
 → FAILED
~~~

A retry must be idempotent or carry an idempotency key when side effects are possible.

## Recovery checklist

- classify retryable vs permanent errors;
- cap retries;
- preserve enough state to avoid repeating completed side effects;
- cancel downstream work on timeout;
- surface a clear partial-failure result;
- require re-authorization for a materially different action;
- log the recovery path for audit/debugging.
