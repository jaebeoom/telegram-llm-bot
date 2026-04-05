from __future__ import annotations

import re
from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def normalize_model_name(model_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    return re.sub(r"-{2,}", "-", normalized)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text.strip()

    closing_index = text.find("\n---\n", 4)
    if closing_index == -1:
        return {}, text.strip()

    raw_meta = text[4:closing_index]
    body = text[closing_index + 5 :].strip()
    metadata: dict[str, str] = {}

    for line in raw_meta.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower()] = value.strip()

    return metadata, body


def _load_markdown(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    return _parse_frontmatter(text)


def _match_score(model_name: str, metadata: dict[str, str], path: Path) -> int:
    normalized_model = normalize_model_name(model_name)
    tokens = []

    alias_text = metadata.get("match", "")
    if alias_text:
        tokens.extend(part.strip() for part in alias_text.split(",") if part.strip())

    prompt_id = metadata.get("id")
    if prompt_id:
        tokens.append(prompt_id)

    tokens.append(path.stem)

    best = 0
    for token in tokens:
        normalized_token = normalize_model_name(token)
        if not normalized_token:
            continue
        if normalized_model == normalized_token:
            best = max(best, 10_000 + len(normalized_token))
        elif normalized_token in normalized_model:
            best = max(best, len(normalized_token))
    return best


def load_prompt_profile(model_name: str, prompts_dir: Path | None = None) -> str:
    prompts_root = prompts_dir or PROMPTS_DIR
    sections: list[str] = []

    base_path = prompts_root / "base.md"
    if base_path.exists():
        _, base_body = _load_markdown(base_path)
        if base_body:
            sections.append(base_body)

    model_dir = prompts_root / "models"
    if not model_dir.exists():
        return "\n\n".join(sections).strip()

    best_score = 0
    best_body = ""

    for path in sorted(model_dir.glob("*.md")):
        metadata, body = _load_markdown(path)
        if not body:
            continue
        score = _match_score(model_name, metadata, path)
        if score > best_score:
            best_score = score
            best_body = body

    if best_body:
        sections.append(best_body)

    return "\n\n".join(section.strip() for section in sections if section.strip()).strip()


def render_prompt_profile(
    model_name: str,
    variables: dict[str, str] | None = None,
    prompts_dir: Path | None = None,
) -> str:
    prompt = load_prompt_profile(model_name, prompts_dir=prompts_dir)
    if not variables:
        return prompt
    rendered = prompt
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered
