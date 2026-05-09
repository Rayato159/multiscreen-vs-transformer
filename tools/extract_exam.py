from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PROBLEM_RE = re.compile(r"^\d{1,3}(?:\.\d+)?$")
ANSWER_STOP_RE = re.compile(r"^\*$")


@dataclass(frozen=True)
class Example:
    source: str
    problem_id: str
    split: str
    question: str
    answer_code: str
    answer_full: str


def load_pypdf() -> type:
    repo_root = Path(__file__).resolve().parents[1]
    vendored = repo_root / ".python-packages"
    if vendored.exists():
        sys.path.insert(0, str(vendored))
    from pypdf import PdfReader

    return PdfReader


def clean_text(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def answer_key_tail(page_text: str) -> str | None:
    text = clean_text(page_text)
    marker = text.rfind("Problem Set")
    if marker < 0:
        return None
    return text[marker + len("Problem Set") :].strip()


def parse_answer_key(page_text: str) -> dict[str, str]:
    tail = answer_key_tail(page_text)
    if not tail:
        return {}

    tokens = tail.split()
    if len(tokens) >= 2 and tokens[0] == "1" and tokens[1] == "1":
        tokens = tokens[1:]

    answers: dict[str, str] = {}
    idx = 0
    while idx + 1 < len(tokens):
        problem = tokens[idx].strip()
        if ANSWER_STOP_RE.match(problem):
            break
        if not PROBLEM_RE.match(problem):
            idx += 1
            continue

        idx += 1
        parts = [tokens[idx].strip()]
        idx += 1
        while idx + 1 < len(tokens) and tokens[idx] == "หรือ":
            parts.extend([tokens[idx], tokens[idx + 1]])
            idx += 2
        answers[problem] = " ".join(parts)

    return answers


def find_answer_page(reader) -> tuple[int, dict[str, str]]:
    best_idx = -1
    best_answers: dict[str, str] = {}
    for idx, page in enumerate(reader.pages):
        answers = parse_answer_key(page.extract_text() or "")
        if len(answers) > len(best_answers):
            best_idx = idx
            best_answers = answers
    if not best_answers:
        raise ValueError("cannot find answer key page")
    return best_idx, best_answers


def top_level(problem_id: str) -> int:
    return int(problem_id.split(".", 1)[0])


def candidate_positions(text: str, problem: int) -> list[int]:
    pattern = re.compile(rf"(?:ข้อที\S*\s*{problem}\b|(?<![\d.]){problem}[\.．]\s+)")
    return [match.start() for match in pattern.finditer(text)]


def choose_problem_starts(text: str, problems: list[int]) -> dict[int, int]:
    candidates = {problem: candidate_positions(text, problem) for problem in problems}
    missing = [problem for problem, values in candidates.items() if not values]
    if missing:
        raise ValueError(f"cannot locate problem starts: {missing[:12]}")

    # Dynamic programming beats greedy here because answer choices are also numbered.
    # Short segments are usually choices, not full questions.
    dp: list[dict[int, tuple[float, int | None]]] = []
    first: dict[int, tuple[float, int | None]] = {}
    for pos in candidates[problems[0]]:
        first[pos] = (start_penalty(text, pos), None)
    dp.append(first)

    for idx in range(1, len(problems)):
        current: dict[int, tuple[float, int | None]] = {}
        for pos in candidates[problems[idx]]:
            best_cost = math.inf
            best_prev = None
            for prev_pos, (prev_cost, _) in dp[idx - 1].items():
                if pos <= prev_pos:
                    continue
                cost = prev_cost + segment_cost(pos - prev_pos) + start_penalty(text, pos)
                if cost < best_cost:
                    best_cost = cost
                    best_prev = prev_pos
            if best_prev is not None:
                current[pos] = (best_cost, best_prev)
        if not current:
            raise ValueError(f"cannot build increasing starts at problem {problems[idx]}")
        dp.append(current)

    last_problem = problems[-1]
    end = len(text)
    best_last = min(
        dp[-1].items(),
        key=lambda item: item[1][0] + segment_cost(end - item[0]),
    )[0]

    selected: dict[int, int] = {last_problem: best_last}
    current_pos = best_last
    for idx in range(len(problems) - 1, 0, -1):
        prev_pos = dp[idx][current_pos][1]
        assert prev_pos is not None
        selected[problems[idx - 1]] = prev_pos
        current_pos = prev_pos
    return selected


def start_penalty(text: str, pos: int) -> float:
    # Prefer starts after whitespace/page headers. Penalize candidates buried inside text.
    prev = text[max(0, pos - 3) : pos]
    if pos == 0 or prev.strip() == "":
        return 0.0
    return 30.0


def segment_cost(length: int) -> float:
    if length < 80:
        return 20_000.0 + (80 - length) * 10.0
    if length > 4_500:
        return 2_000.0 + (length - 4_500) * 0.2
    target = 650.0
    return abs(math.log(length / target)) * 100.0


def extract_question_texts(reader, answer_page_idx: int, answers: dict[str, str]) -> dict[str, str]:
    exam_text = clean_text(" ".join(reader.pages[idx].extract_text() or "" for idx in range(answer_page_idx)))
    top_problems = sorted({top_level(problem_id) for problem_id in answers})
    starts = choose_problem_starts(exam_text, top_problems)

    ordered = sorted(starts.items())
    question_by_top: dict[int, str] = {}
    for idx, (problem, start) in enumerate(ordered):
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(exam_text)
        question_by_top[problem] = exam_text[start:end].strip()

    questions: dict[str, str] = {}
    for problem_id in answers:
        parent = top_level(problem_id)
        question = question_by_top[parent]
        if "." in problem_id:
            question = f"{question}\nSubproblem: {problem_id}"
        questions[problem_id] = question
    return questions


def answer_full_text(problem_id: str, question: str, answer_code: str) -> str:
    normalized = answer_code.strip()
    if not normalized:
        return normalized
    if "ฟรี" in normalized:
        return normalized

    option_map = extract_options(problem_id, question)
    choice_codes = [part.strip() for part in normalized.split(",")]
    if option_map and all(code in option_map for code in choice_codes):
        return " | ".join(option_map[code] for code in choice_codes)
    return normalized


def extract_options(problem_id: str, question: str) -> dict[str, str]:
    parent = re.escape(str(top_level(problem_id)))
    body = re.sub(rf"^(?:ข้อที\S*\s*)?{parent}[\.．]?\s*", "", question, count=1)
    matches = list(re.finditer(r"(?<!\d)([1-5])[\.．]\s+", body))
    if len(matches) < 2:
        return {}

    options: dict[str, str] = {}
    for idx, match in enumerate(matches):
        code = match.group(1)
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        text = body[match.start() : end].strip()
        if len(text) > 260:
            text = text[:260].rstrip() + "..."
        options[code] = text
    return options


def split_for(global_index: int) -> str:
    return "test" if global_index % 5 == 0 else "train"


def extract_pdf(pdf_path: Path, global_start: int) -> tuple[list[Example], list[str]]:
    PdfReader = load_pypdf()
    reader = PdfReader(str(pdf_path))
    answer_page_idx, answers = find_answer_page(reader)
    questions = extract_question_texts(reader, answer_page_idx, answers)

    warnings: list[str] = []
    examples: list[Example] = []
    for local_idx, problem_id in enumerate(sorted(answers, key=lambda key: (top_level(key), key)), start=1):
        question = questions[problem_id]
        answer_code = answers[problem_id]
        full = answer_full_text(problem_id, question, answer_code)
        if full == answer_code and re.fullmatch(r"[1-5](?:,[1-5])*", answer_code):
            warnings.append(f"{pdf_path.name} problem {problem_id}: could not map choice code to option text")
        examples.append(
            Example(
                source=pdf_path.name,
                problem_id=problem_id,
                split=split_for(global_start + local_idx),
                question=question,
                answer_code=answer_code,
                answer_full=full,
            )
        )
    return examples, warnings


def pdf_inputs(pdf: Path | None, pdf_dir: Path | None) -> list[Path]:
    if pdf:
        return [pdf]
    assert pdf_dir is not None
    paths = sorted(pdf_dir.glob("*.pdf"), key=lambda path: (" (1)" in path.name, path.name))
    seen: set[tuple[str, int]] = set()
    unique: list[Path] = []
    for path in paths:
        normalized_name = path.name.replace(" (1)", "")
        key = (normalized_name, path.stat().st_size)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def write_dataset(paths: list[Path], out_path: Path) -> None:
    all_examples: list[Example] = []
    warnings: list[str] = []
    skipped: list[str] = []

    for path in paths:
        try:
            examples, pdf_warnings = extract_pdf(path, len(all_examples))
        except Exception as err:
            skipped.append(f"{path.name}: {err}")
            continue
        all_examples.extend(examples)
        warnings.extend(pdf_warnings)

    if not all_examples:
        raise RuntimeError("no examples extracted")

    lines: list[str] = [
        "# Generated by tools/extract_exam.py",
        f"# Sources: {', '.join(path.name for path in paths)}",
        "# Split rule: global example index % 5 == 0 -> test, all other examples -> train.",
        "# Answer field contains the full selected answer text when the option can be resolved.",
        "",
    ]

    if skipped:
        lines.append("# Skipped PDFs:")
        lines.extend(f"# - {item}" for item in skipped)
        lines.append("")
    if warnings:
        lines.append("# Extraction warnings:")
        lines.extend(f"# - {item}" for item in warnings[:120])
        if len(warnings) > 120:
            lines.append(f"# - ... {len(warnings) - 120} more warnings")
        lines.append("")

    for idx, example in enumerate(all_examples, start=1):
        lines.extend(
            [
                f"### Example {idx:04d}",
                f"Split: {example.split}",
                f"Source: {example.source}",
                f"Problem: {example.problem_id}",
                "Question:",
                example.question,
                "Answer Code:",
                example.answer_code,
                "Answer:",
                example.answer_full,
                "### End",
                "",
            ]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(all_examples)} examples -> {out_path}")
    print(f"skipped {len(skipped)} pdf(s), warnings {len(warnings)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pdf", type=Path, help="Path to one exam PDF.")
    group.add_argument("--pdf-dir", type=Path, default=Path("exam"), help="Directory of exam PDFs.")
    parser.add_argument(
        "--out",
        default="exam/tcas68-all.dataset.txt",
        type=Path,
        help="Output UTF-8 dataset path.",
    )
    args = parser.parse_args()
    write_dataset(pdf_inputs(args.pdf, args.pdf_dir), args.out)


if __name__ == "__main__":
    main()
