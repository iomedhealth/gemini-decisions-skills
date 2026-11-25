# Gemini Decisions Agent Skills

Agent skills for building with the [Gemini Decisions API](https://aistudio.google.com/docs/decisions-api) (`gemini-decision-1`): calibrated yes/no (`noul`), single-select (`choice`), and rubric (`score`) decisions over text, images, audio, video, and PDFs with zero output tokens.

Adapted from [typesafe-ai/skills](https://github.com/typesafe-ai/skills) (MIT) for Google's Gemini Decisions API.

You can [read `SKILL.md` on GitHub](https://github.com/iomedhealth/gemini-decisions-skills/blob/main/skills/gemini-decisions/SKILL.md) or fetch its [raw Markdown](https://raw.githubusercontent.com/iomedhealth/gemini-decisions-skills/main/skills/gemini-decisions/SKILL.md).

## Install

### Using [skills.sh](https://skills.sh) (OpenCode, Claude Code, Cursor, Codex, and more)

```sh
npx skills add iomedhealth/gemini-decisions-skills --skill gemini-decisions
```

Select your agent when prompted. Installation is project-local by default; add `-g` to install globally.

### Claude Code plugin

```sh
claude plugin marketplace add iomedhealth/gemini-decisions-skills
claude plugin install gemini-decisions@gemini-decisions-skills
```

### Cursor

```text
/add-plugin iomedhealth/gemini-decisions-skills
```

### OpenAI Codex

```sh
codex plugin marketplace add https://github.com/iomedhealth/gemini-decisions-skills
codex plugin add gemini-decisions@gemini-decisions-skills
```

### Antigravity

```sh
agy plugin install https://github.com/iomedhealth/gemini-decisions-skills
```

## Use

Ask your agent, for example:

> Use the Gemini Decisions API to route incoming support tickets by department, score churn risk, and escalate uncertain decisions to human review.

> Triage uploaded invoice PDFs with `gemini-decision-1`: check whether a PO number is present and classify the expense category in a single request.

In Claude Code, you can explicitly invoke the plugin skill with `/gemini-decisions:gemini-decisions`.

| Skill | Purpose |
| :--- | :--- |
| [`gemini-decisions`](skills/gemini-decisions/SKILL.md) | Design `gemini-decision-1` integrations: truth-table architecture, multimodal `input` design, atomic `noul`/`choice`/`score` questions, calibration thresholds, live API probing, and links to official AI Studio docs |
| [`references/patterns.md`](skills/gemini-decisions/references/patterns.md) | Runnable Python and TypeScript recipes for companion `noul` scope gates, speculative fan-out, pre-parsed value selection, RAG verification cascades, composite scoring, and multimodal PDF/audio/video triage |

## License

[MIT](LICENSE). Originally forked from [typesafe-ai/skills](https://github.com/typesafe-ai/skills).
