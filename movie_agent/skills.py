"""
Agent Skills 协议层 —— 三级渐进式披露（progressive disclosure）。

Skill 是「方法论」而非「知识」：知识由 RAG 提供，Skill 规定该走哪几步、
每步调哪个工具、输出成什么结构。

之所以要分级，是因为把所有 Skill 正文常驻在 system prompt 里，token 成本
随 Skill 数量线性增长，而任一轮对话通常只用得上一个：

    L1  索引     启动时扫描 frontmatter，只把 name + description 注入 prompt（每个约 40 token）
    L2  正文     模型判断需要时调 ``load_skill("<name>")`` 拉全文
    L3  引用文件 正文里可以指向 ``<name>/references/xxx.md``，同一个工具按需再拉

frontmatter 用极小的手写解析器处理（``key: value`` + 缩进续行），
不为三个字段引入 pyyaml 依赖。
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import tool

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

_MAX_BODY_CHARS = 12_000


@dataclass(frozen=True)
class SkillMeta:
    """一个 Skill 的 L1 索引条目。"""

    name: str
    description: str
    path: Path


_cache: list[SkillMeta] | None = None


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """拆出 ``---`` 包裹的 frontmatter 与正文。

    只支持 ``key: value`` 与缩进续行两种形态 —— 够 Skill 元数据用，
    也避免把整个 YAML 语法面暴露成隐性契约。
    """
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return {}, text

    meta: dict[str, str] = {}
    key: str | None = None
    for raw in lines[1:end]:
        if not raw.strip():
            continue
        if raw[0].isspace() and key:  # 缩进续行，拼到上一个 key 上
            meta[key] = f"{meta[key]} {raw.strip()}".strip()
            continue
        if ":" not in raw:
            continue
        key, _, value = raw.partition(":")
        key = key.strip()
        meta[key] = value.strip()

    return meta, "\n".join(lines[end + 1:]).strip()


def list_skills(refresh: bool = False) -> list[SkillMeta]:
    """扫描 ``skills/*/SKILL.md``，返回 L1 索引（进程内缓存）。"""
    global _cache
    if _cache is not None and not refresh:
        return _cache

    found: list[SkillMeta] = []
    if _SKILLS_DIR.is_dir():
        for skill_md in sorted(_SKILLS_DIR.glob("*/SKILL.md")):
            meta, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            name = meta.get("name") or skill_md.parent.name
            description = meta.get("description", "").strip()
            if not description:
                print(f"[skills] {skill_md.parent.name}: missing description — skipped")
                continue
            found.append(SkillMeta(name=name, description=description, path=skill_md))

    _cache = found
    return found


def skill_index_prompt() -> str:
    """L1：注入 system prompt 的 Skill 索引。没有 Skill 时返回空串。"""
    skills = list_skills()
    if not skills:
        return ""

    lines = [
        "--- Available Skills ---",
        "Each skill is a playbook for a kind of request. When one matches what the "
        "user is asking for, call `load_skill(\"<name>\")` FIRST to read its "
        "procedure, then follow it. Do not guess a skill's contents.",
        "",
    ]
    lines += [f"- {s.name}: {s.description}" for s in skills]
    return "\n".join(lines)


def _resolve(reference: str) -> Path:
    """把 ``load_skill`` 的入参解析成 ``skills/`` 内的真实文件。

    接受 ``"<name>"``（取该 Skill 的 SKILL.md）和
    ``"<name>/references/x.md"``（取引用文件）两种形态。
    解析后强制校验仍位于 ``skills/`` 之下 —— 否则 ``../../.env`` 之类的入参
    会让这个工具变成任意文件读取。
    """
    reference = reference.strip().strip("/")
    if not reference:
        raise ValueError("skill name is empty")

    candidate = (_SKILLS_DIR / reference).resolve()
    root = _SKILLS_DIR.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"'{reference}' resolves outside the skills directory")

    if candidate.is_dir():
        candidate = candidate / "SKILL.md"
    return candidate


@tool
def load_skill(skill_name: str) -> str:
    """Load the full playbook for a skill listed under "Available Skills".

    Call this before acting on a request that matches a skill's description. The
    playbook tells you which tools to call, in what order, and how to structure
    the answer.

    Args:
        skill_name: The skill's name (e.g. "film-analysis"), or a path to one of its
            reference files (e.g. "film-analysis/references/noir.md") when the
            playbook points you at one.

    Returns:
        The skill's instructions as markdown, with the frontmatter stripped.
    """
    try:
        path = _resolve(skill_name)
    except ValueError as exc:
        return f"Cannot load skill: {exc}"

    if not path.is_file():
        available = ", ".join(s.name for s in list_skills()) or "(none)"
        return f"No such skill or reference: '{skill_name}'. Available skills: {available}"

    text = path.read_text(encoding="utf-8")
    _, body = parse_frontmatter(text)
    body = body or text
    if len(body) > _MAX_BODY_CHARS:
        body = body[:_MAX_BODY_CHARS] + "\n\n[truncated]"
    return body
