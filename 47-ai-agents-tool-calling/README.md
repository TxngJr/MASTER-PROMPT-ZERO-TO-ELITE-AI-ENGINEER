# Chapter 47 — AI Agents & Tool Calling

## 1. What Is an Agent?

A useful engineering definition:

> an agent is a model-driven control loop that can observe state, choose an action, call an allowed tool, inspect the result, update state and continue until a stopping condition.

~~~text
user goal
↓
model decision
↓
tool call / answer / clarification
↓
tool result
↓
state update
↓
next decision
~~~

An agent is not automatically autonomous, reliable or safe.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- define an agent loop
- distinguish model reasoning from executable actions
- define tool schemas
- validate structured tool arguments
- allow-list tools
- design tool result envelopes
- maintain explicit agent state
- enforce step/time/budget limits
- understand ReAct-style observe/act loops
- explain planning vs execution
- distinguish short-term state from durable memory
- design idempotent tool calls
- handle retries safely
- identify prompt/tool-output injection
- design approval boundaries
- evaluate agent success and tool correctness

## 3. Model vs Executor

Never let free-form model text directly become arbitrary code/system commands.

Separate:

~~~text
model
↓ structured action proposal
validator
↓
authorized executor
↓
tool result
~~~

The executor owns permissions.

## 4. Tool Schema

A tool should declare:
- name
- description
- arguments
- types
- required fields
- constraints

Example concept:

~~~text
search_docs(
  query: string,
  top_k: integer
)
~~~

Structured interfaces reduce ambiguity.

## 5. Structured Tool Calls

A tool call should be machine-parseable:

~~~json
{
  "name": "search_docs",
  "arguments": {
    "query": "RMSNorm",
    "top_k": 5
  }
}
~~~

Validate before execution.

## 6. Validation

Check:
- known tool name
- required keys
- unknown keys policy
- type correctness
- ranges/enums
- authorization

Malformed calls should fail closed.

## 7. Allow-List

Only registered tools can run.

~~~text
requested tool
↓
registry lookup
├─ known → validate → execute
└─ unknown → reject
~~~

Never resolve arbitrary function names dynamically from model text.

## 8. Tool Result Envelope

Return structured data:

~~~text
{
  ok,
  tool,
  data,
  error,
  metadata
}
~~~

This distinguishes:
- execution success
- tool/domain failure
- parse failure

## 9. Agent State

State may include:
- user goal
- current plan
- observations
- tool results
- budget
- completed steps
- unresolved tasks

Keep it explicit rather than relying on hidden conversational drift.

## 10. Step Budget

Without limits an agent can loop forever.

Define:

~~~text
max_steps
max_tool_calls
max_cost
deadline
~~~

Stop safely when budget is exhausted.

## 11. Stopping Conditions

Examples:
- goal achieved
- user input required
- unrecoverable tool error
- policy/permission boundary
- max steps reached

The loop needs a deliberate terminal state.

## 12. ReAct Pattern

Conceptually:

~~~text
observe
↓
reason about next action
↓
act/tool
↓
observe
↓
...
~~~

The engineering value is iterative observation/action, not exposing private chain-of-thought.

Store concise action rationale/status instead of hidden reasoning traces.

## 13. Planning

Plan:
- decompose goal
- order dependencies
- identify tools
- identify checkpoints

Execution:
- perform one authorized action
- inspect actual result
- update plan

Do not assume planned actions succeeded.

## 14. Planner / Executor Split

Possible architecture:

~~~text
Planner
↓ task graph
Executor
↓ tool calls
Verifier
↓ result checks
~~~

This can improve clarity but adds latency/complexity.

## 15. Reflection / Verification

After an action:
- check expected fields
- verify pre/postconditions
- compare against goal
- decide whether retry is justified

"Tool returned success" may not mean task goal succeeded.

## 16. Retry Policy

Retry only when failure is plausibly transient.

Use:
- bounded retries
- backoff
- error classification

Do not retry malformed/unauthorized requests endlessly.

## 17. Idempotency

Repeated external actions can duplicate side effects.

Example:
- create order twice
- send message twice

Use idempotency keys when tool/API supports them.

## 18. Read vs Write Tools

Read-only:
- search
- fetch
- list
- inspect

Write/action:
- send
- create
- update
- delete
- purchase
- deploy

Write tools deserve stricter validation/approval.

## 19. Approval Boundaries

For consequential actions:
1. prepare action
2. show exact effect
3. obtain required approval
4. execute once

Approval scope should be specific, not blanket.

## 20. Short-Term Memory

Working state for current task:
- retrieved facts
- current plan
- IDs from previous tool calls

This is often enough for an agent loop.

## 21. Long-Term Memory

Durable memory introduces:
- persistence
- relevance selection
- privacy
- stale data
- deletion/versioning

Do not store everything indiscriminately.

## 22. Tool Output Is Untrusted Input

A webpage/document/tool can contain text such as:

~~~text
ignore previous instructions
run another tool
send secrets
~~~

That content is data, not authority.

Tool output must not redefine the executor's permission policy.

## 23. Prompt Injection

Possible sources:
- user input
- retrieved documents
- webpages
- tool outputs

Defenses:
- instruction hierarchy
- tool allow-lists
- data/instruction separation
- least privilege
- approval gates
- output validation

There is no universal single prompt that solves injection.

## 24. Least Privilege

Give each tool only the minimum permission required.

Examples:
- read calendar without edit permission
- search documents without delete permission

Limit blast radius.

## 25. Sandboxing

For code/computer agents:
- isolated environment
- filesystem boundaries
- network policy
- CPU/memory/time limits
- secret isolation

Model intent should never equal host-level permission.

## 26. Multi-Agent Systems

Multiple specialized agents can coordinate:

~~~text
researcher
coder
reviewer
manager
~~~

Benefits:
- role separation

Costs:
- more messages
- coordination errors
- duplicated work
- harder observability

Use only when measurable benefit exists.

## 27. Tool Selection Evaluation

Given tasks with known required tools, measure:
- correct tool rate
- argument validity
- unnecessary calls
- tool-call latency
- final task success

## 28. Agent Evaluation

Metrics:
- success rate
- steps to completion
- tool error rate
- invalid call rate
- side-effect safety
- human intervention rate
- cost/latency

A fluent answer is not enough.

## 29. Observability

Log:
- tool name
- validated arguments
- timing
- result status
- retry count
- state transition

Avoid logging secrets unnecessarily.

## 30. From Scratch

src/agent_core.py includes:

- ToolSpec
- validate_arguments
- ToolRegistry
- ToolResult
- AgentState
- bounded_agent_loop

The educational loop accepts a policy callback that returns structured actions.

## 31. Common Mistakes

1. arbitrary function execution from model text
2. no schema validation
3. no loop/step budget
4. retries duplicate side effects
5. trusting tool output as instructions
6. one giant tool with excessive permissions
7. stale IDs/state reused blindly
8. no distinction between read/write tools
9. logging secrets
10. evaluating only final text, not actions

## 32. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 33. Checklist

- [ ] agent loop
- [ ] tool schema
- [ ] validation
- [ ] allow-list registry
- [ ] state
- [ ] step limits
- [ ] retries / idempotency
- [ ] read/write boundaries
- [ ] injection defenses
- [ ] approval
- [ ] observability
- [ ] evaluation

## 34. What's Next

Chapter 48 opens the modern decoder-only LLM block itself: RMSNorm, RoPE, grouped-query attention, SwiGLU and KV caching.
