# AGENTS.md

This document defines the rules that coding agents must follow when collaborating in this repository. The rules can be extended over time; currently there are two hard requirements.

## Rules

1. **Use conda environments by default**
   - All Python-related operations (dependency install, script execution, training, evaluation, export, etc.) must be done inside a conda environment by default.
   - Example: `conda activate <env>` followed by `python` / `pip`, or use `conda run -n <env> python ...` / `conda run -n <env> pip ...` directly.
   - Do not switch to venv / virtualenv / bare system Python; conda takes priority.

2. **Never write local environment information into documents**
   - No documentation that may appear later (README, architecture docs, tutorials, script comments, etc.) may contain anything tied to this machine's environment, including but not limited to:
     - Local absolute paths (e.g., `/Users/...`, `/home/...`)
     - Local conda environment names
     - Local hardware / platform details (GPU model, RK3588, macOS/Linux, etc.)
     - Personal configuration and credentials on this machine
   - Documents must only describe generic, portable operations (e.g., let the reader supply their own `conda activate` env name) or use placeholders.

## Notes

- This file itself is a repo constraint file, not subject to the "documents must not contain local environment info" rule; actual env names can be maintained here.
- If rules change or new ones are added, append them here. Keep it concise and executable.
