---
name: experiment-graph-development
description: Edit an EchoMind Project's mutable development Graph from conversation, including nodes, edges, requirements and node configuration.
---

# Experiment Graph Development

Use this Skill only after a Project has been created. The Project owns one mutable
Working Graph during development; a conversation edit is a direct Graph update,
not a new plan revision, Graph version or execution.

Before editing, call `get_project_graph` with the confirmed `project_id`. If the
target node or edge is ambiguous, ask one focused question instead of guessing.
After editing, report the actual returned node, edge or requirement changes.

Use `apply_graph_patch` with one atomic patch and a stable `request_id` for retries.

- `add_node` adds a complete node with a unique ID.
- `update_node` changes selected fields on an existing node.
- `remove_node` removes the node and its connected edges.
- `add_edge` and `remove_edge` change Graph dependencies.
- `update_requirements` replaces the complete requirement list.

Keep node IDs stable when only changing a title, executor or configuration.
Deleting and recreating a node should be used only when the user explicitly wants
a different node identity. Do not invent executor names or configuration values.
The server validates node types, edge references and cycles.

Do not create NodeRuns, start executions, publish a Graph or introduce version
confirmation while using this Skill. Those actions belong to later development
stages or explicit user actions.
