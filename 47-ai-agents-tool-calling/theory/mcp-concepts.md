# Model Context Protocol (MCP) Concepts

MCP is a protocol pattern for exposing external tools, resources and prompts to an AI application through a structured client/server interface.

## Roles
- Host: the AI application coordinating the interaction.
- Client: the protocol participant connected to one server.
- Server: exposes capabilities such as tools or resources.
- Tool: an operation with a declared input schema and structured result.
- Resource: readable contextual data identified by a URI-like reference.

## Core engineering ideas
1. Capability discovery: the client learns what a server exposes.
2. Typed schemas: tool inputs should be validated before execution.
3. Explicit boundaries: model text is not automatically executable authority.
4. Least privilege: expose only the operations and data needed for the task.
5. Result validation: tool output can be malformed, stale or incomplete.
6. State and idempotency: repeated actions must have well-defined semantics.
7. Timeouts and retries: external systems can fail independently.
8. Observability: record tool name, request identity, latency, result status and relevant versioning without leaking secrets.

## Agent integration
A reliable loop is:
request -> choose capability -> validate arguments -> execute -> validate result -> update state -> decide next action.

Do not confuse MCP itself with an autonomous-agent algorithm. It is an interoperability layer; planning, memory, policy and evaluation are separate concerns.

## Lab
Design a tiny calculator/resource server interface on paper. Specify tool name, JSON-like input schema, return schema, errors, idempotency expectations and permission boundary. Then compare it with the chapter's existing internal tool abstraction.
