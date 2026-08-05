# Interface scope and disclosure boundary

This document records what the reviewer-facing package exposes and what remains private.

| Component | Public release | Notes |
|---|---:|---|
| Request and response schemas | Yes | Shape and metadata contract only |
| Model lifecycle methods | Yes | Load, domain selection, and prediction signatures |
| Dataset and evaluator protocols | Yes | Caller-facing method signatures only |
| Input schema example | Yes | Uses placeholders and no private paths |
| Private core adapter | Stub only | Raises `NotImplementedError` by design |
| Network architecture | No | Omitted to prevent reconstruction |
| Loss and uncertainty computation | No | Omitted to protect the unpublished method |
| Training and validation loops | No | Omitted |
| Dataset and annotations | No | Not distributed |
| Checkpoints and weights | No | Not distributed |
| Raw predictions and experiment logs | No | Not distributed |
| Exact internal configuration | No | Only a redacted schema is shown |
| Paper figures | Selected images | Documentation only; no generation code |

The public package is intentionally insufficient to reproduce the reported results. It is designed to show how a caller would communicate with the private core, not how that core is implemented.

