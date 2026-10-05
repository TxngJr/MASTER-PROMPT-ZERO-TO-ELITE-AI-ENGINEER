#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
errors = []

chapter_dirs = []
for n in range(1, 81):
    matches = [p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith(f"{n:02d}-")]
    if len(matches) != 1:
        errors.append(f"chapter {n:02d}: expected exactly one directory, found {len(matches)}")
    else:
        chapter_dirs.append(matches[0])

required = [
    "README.md",
    "exercises/README.md",
    "solutions/README.md",
    "mini-project/README.md",
    "references.md",
]
for d in chapter_dirs:
    for rel in required:
        if not (d / rel).exists():
            errors.append(f"{d.name}: missing {rel}")
    if not any((d / "src").glob("*")):
        errors.append(f"{d.name}: src/ has no files")
    if not any((d / "tests").glob("test_*.py")):
        errors.append(f"{d.name}: no pytest file")

workbook = (ROOT / "SUPPLEMENTARY_EXERCISE_WORKBOOK.md").read_text(encoding="utf-8")
for n in range(13, 79):
    d = next(p for p in chapter_dirs if p.name.startswith(f"{n:02d}-"))
    local = (d / "exercises/README.md").read_text(encoding="utf-8")
    nums = [int(x) for x in re.findall(r"(?m)^\s*(\d+)[.)]\s+", local)]
    local_count = len(set(nums))
    section = f"## {d.name} —"
    if local_count < 20 and section not in workbook:
        errors.append(f"{d.name}: fewer than 20 local exercises and no workbook supplement")

for name in [
    "CURRICULUM_COMPLETION_STANDARD.md",
    "CHAPTER_COMPLETION_ADDENDA.md",
    "HARDWARE_LAB_MATRIX.md",
    "SUPPLEMENTARY_EXERCISE_WORKBOOK.md",
]:
    if not (ROOT / name).exists():
        errors.append(f"missing canonical companion: {name}")

topic_checks = {
    "21-activations-losses/theory/contrastive-loss.md": "Contrastive",
    "38-reinforcement-learning/theory/sarsa.md": "SARSA",
    "42-speech-stt-tts/theory/mfcc.md": "MFCC",
    "47-ai-agents-tool-calling/theory/mcp-concepts.md": "MCP",
    "49-tokenizer-from-scratch/theory/wordpiece-unigram.md": "Unigram",
    "54-mixed-precision/theory/precision-formats.md": "FP8",
    "53-gpu-cuda-fundamentals/src/vector_add.cu": "__global__",
}
for rel, needle in topic_checks.items():
    p = ROOT / rel
    if not p.exists() or needle not in p.read_text(encoding="utf-8"):
        errors.append(f"missing topic completion: {rel} / {needle}")

bad = []
for p in ROOT.rglob("*.md"):
    if "cite" in p.read_text(encoding="utf-8"):
        bad.append(str(p.relative_to(ROOT)))
if bad:
    errors.append("unresolved chat citation tokens: " + ", ".join(sorted(bad)))

if errors:
    print("CURRICULUM AUDIT FAILED")
    for e in errors:
        print("-", e)
    sys.exit(1)

print("CURRICULUM AUDIT PASSED")
print(f"chapters={len(chapter_dirs)}")
print("contract=complete against CURRICULUM_COMPLETION_STANDARD.md")
