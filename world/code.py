"""Code tasks: write one Python function, graded by hidden tests in a subprocess."""

from __future__ import annotations

import ast
import json
import os
import random
import re
import resource
import secrets
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from world.tasks import Grade, Task, check_difficulty, new_task_id

TIMEOUT_SEC = 5.0
FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class Problem:
    name: str
    signature: str
    describe: Callable[[dict], str]
    reference: Callable[..., object]  # called as reference(*args, **params)
    make_args: Callable[[random.Random, dict], tuple]
    make_params: Callable[[random.Random], dict] = lambda rng: {}


def _words(rng: random.Random, n: int) -> str:
    pool = ["red", "blue", "green", "red", "sky", "blue", "sea", "red", "tree", "sky"]
    return " ".join(rng.choice(pool) for _ in range(n))


def _rle(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{s[i]}{j - i}")
        i = j
    return "".join(out)


def _balanced(s: str) -> bool:
    pairs, stack = {")": "(", "]": "[", "}": "{"}, []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs and (not stack or stack.pop() != pairs[ch]):
            return False
    return not stack


def _merge(intervals: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _roman(n: int) -> str:
    table = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out = ""
    for value, sym in table:
        while n >= value:
            out, n = out + sym, n - value
    return out


def _roman_to_int(s: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        v = values[ch]
        total += -v if i + 1 < len(s) and values[s[i + 1]] > v else v
    return total


def _lis(nums: list[int]) -> int:
    best = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                best[i] = max(best[i], best[j] + 1)
    return max(best, default=0)


def _top_words(text: str, k: int) -> list[str]:
    counts = Counter(text.split())
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:k]


def _ints(rng: random.Random, n: int, lo: int = -20, hi: int = 20) -> list[int]:
    return [rng.randint(lo, hi) for _ in range(n)]


def _letters(rng: random.Random, n: int, alphabet: str = "abcxyz") -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


PROBLEMS: dict[int, list[Problem]] = {
    1: [
        Problem(
            "add_k",
            "add_k(x)",
            lambda p: f"returns x plus {p['k']}",
            lambda x, k: x + k,
            lambda rng, p: (rng.randint(-50, 50),),
            lambda rng: {"k": rng.randint(2, 30)},
        ),
        Problem(
            "maximum",
            "maximum(a, b)",
            lambda p: "returns the larger of two numbers",
            max,
            lambda rng, p: (rng.randint(-50, 50), rng.randint(-50, 50)),
        ),
        Problem(
            "is_even",
            "is_even(n)",
            lambda p: "returns True if the integer n is even, otherwise False",
            lambda n: n % 2 == 0,
            lambda rng, p: (rng.randint(-99, 99),),
        ),
    ],
    2: [
        Problem(
            "reverse_string",
            "reverse_string(s)",
            lambda p: "returns the string s reversed",
            lambda s: s[::-1],
            lambda rng, p: (_letters(rng, rng.randint(0, 8)),),
        ),
        Problem(
            "count_vowels",
            "count_vowels(s)",
            lambda p: "returns how many characters of s are vowels (a, e, i, o, u; "
            "lowercase or uppercase)",
            lambda s: sum(ch in "aeiouAEIOU" for ch in s),
            lambda rng, p: (_letters(rng, rng.randint(0, 10), "aeiouAEbcdXYZ"),),
        ),
        Problem(
            "sum_even",
            "sum_even(nums)",
            lambda p: "returns the sum of the even numbers in the list nums",
            lambda nums: sum(n for n in nums if n % 2 == 0),
            lambda rng, p: (_ints(rng, rng.randint(0, 7)),),
        ),
    ],
    3: [
        Problem(
            "fizzbuzz",
            "fizzbuzz(n)",
            lambda p: "returns a list of strings for 1..n where multiples of 3 are "
            "'Fizz', multiples of 5 are 'Buzz', multiples of both are 'FizzBuzz', "
            "and other numbers are the number as a string",
            lambda n: [
                (
                    "FizzBuzz"
                    if i % 15 == 0
                    else "Fizz" if i % 3 == 0 else "Buzz" if i % 5 == 0 else str(i)
                )
                for i in range(1, n + 1)
            ],
            lambda rng, p: (rng.randint(0, 20),),
        ),
        Problem(
            "second_largest",
            "second_largest(nums)",
            lambda p: "returns the second largest distinct value in nums, or None "
            "if there are fewer than two distinct values",
            lambda nums: sorted(set(nums))[-2] if len(set(nums)) > 1 else None,
            lambda rng, p: (_ints(rng, rng.randint(0, 6), -5, 5),),
        ),
        Problem(
            "is_palindrome",
            "is_palindrome(s)",
            lambda p: "returns True if s reads the same forwards and backwards "
            "after lowercasing and ignoring every non-alphanumeric character",
            lambda s: (t := [c.lower() for c in s if c.isalnum()]) == t[::-1],
            lambda rng, p: (
                rng.choice(
                    [
                        "A man, a plan, a canal: Panama",
                        "No lemon, no melon",
                        "abca",
                        "",
                        "Was it a car?",
                        "ab ba",
                        "Hello",
                    ]
                ),
            ),
        ),
    ],
    4: [
        Problem(
            "run_length",
            "run_length(s)",
            lambda p: "returns the run-length encoding of s, writing each run as "
            "the character followed by its count, e.g. 'aaabcc' -> 'a3b1c2'",
            _rle,
            lambda rng, p: (_letters(rng, rng.randint(0, 10), "aab"),),
        ),
        Problem(
            "balanced",
            "balanced(s)",
            lambda p: "returns True if the brackets (), [] and {} in s are "
            "balanced and properly nested, ignoring other characters",
            _balanced,
            lambda rng, p: (_letters(rng, rng.randint(0, 8), "()[]{}x"),),
        ),
        Problem(
            "merge_intervals",
            "merge_intervals(intervals)",
            lambda p: "takes a list of [start, end] lists and returns the merged "
            "overlapping intervals as a list of [start, end] lists sorted by start; "
            "intervals that touch (end == next start) are merged",
            _merge,
            lambda rng, p: (
                [
                    [s, s + rng.randint(0, 4)]
                    for s in (rng.randint(0, 15) for _ in range(rng.randint(0, 5)))
                ],
            ),
        ),
    ],
    5: [
        Problem(
            "roman_to_int",
            "roman_to_int(s)",
            lambda p: "converts a Roman numeral string (using I, V, X, L, C, D, M "
            "with subtractive pairs like IV and CM) to an integer",
            _roman_to_int,
            lambda rng, p: (_roman(rng.randint(1, 3999)),),
        ),
        Problem(
            "lis_length",
            "lis_length(nums)",
            lambda p: "returns the length of the longest strictly increasing "
            "subsequence of nums (0 for an empty list)",
            _lis,
            lambda rng, p: (_ints(rng, rng.randint(0, 9), 0, 9),),
        ),
        Problem(
            "top_words",
            "top_words(text, k)",
            lambda p: "returns the k most frequent space-separated words in text, "
            "most frequent first, breaking ties alphabetically",
            _top_words,
            lambda rng, p: (_words(rng, rng.randint(0, 12)), rng.randint(1, 3)),
        ),
    ],
}


KEEP_NODES = (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.Assign)


def extract_code(answer: str) -> str:
    """Fenced code (or the whole reply), keeping only definitions and imports.

    Top-level demo calls and self-written asserts are dropped so only the hidden
    tests decide the grade. Unparseable code is returned as-is and fails there.
    """
    blocks = FENCE_RE.findall(answer)
    code = "\n\n".join(blocks) if blocks else answer
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    tree.body = [node for node in tree.body if isinstance(node, KEEP_NODES)]
    return ast.unparse(tree)


HARNESS = """
import json, sys
NONCE = {nonce!r}
CASES = json.loads({cases!r})
namespace = {{}}
try:
    exec(compile({code!r}, "<solution>", "exec"), namespace)
    fn = namespace[{name!r}]
except BaseException as exc:
    print(NONCE + json.dumps({{"passed": 0, "error": type(exc).__name__}}))
    sys.exit(0)
passed = 0
for args, expected in CASES:
    try:
        result = fn(*args)
        if isinstance(result, tuple):
            result = list(result)
        passed += json.loads(json.dumps(result)) == expected
    except BaseException:
        pass
print(NONCE + json.dumps({{"passed": passed}}))
"""


def _limit_resources() -> None:  # pragma: no cover - runs in the child process
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    os.setsid()


def run_tests(code: str, name: str, cases: list) -> tuple[int, str]:
    """Run ``cases`` against ``code`` in an isolated interpreter; return passes."""
    nonce = secrets.token_hex(8)
    script = HARNESS.format(nonce=nonce, cases=json.dumps(cases), code=code, name=name)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", script],
                cwd=tmp,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SEC,
                preexec_fn=_limit_resources,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return 0, "timeout"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(nonce):
            result = json.loads(line[len(nonce) :])
            return int(result["passed"]), result.get("error", "")
    return 0, "no result"


class CodeFamily:
    """One skill, ``code.func``: implement a function described in the prompt."""

    name = "code"
    skills = ("code.func",)
    max_tokens = 320  # answer length cap for generation

    def __init__(self, n_cases: int = 8):
        self.n_cases = n_cases

    def sample(self, skill: str, difficulty: int, rng: random.Random) -> Task:
        check_difficulty(difficulty)
        if skill not in self.skills:
            raise ValueError(f"unknown code skill {skill}")
        problem = rng.choice(PROBLEMS[difficulty])
        params = problem.make_params(rng)
        cases, seen = [], set()
        for _ in range(self.n_cases * 4):
            args = problem.make_args(rng, params)
            key = json.dumps(args)
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                [list(args), json.loads(json.dumps(problem.reference(*args, **params)))]
            )
            if len(cases) == self.n_cases:
                break
        example_args, example_out = cases[0]
        shown = ", ".join(repr(a) for a in example_args)
        prompt = (
            f"Write a Python function `{problem.signature}` that "
            f"{problem.describe(params)}. Example: {problem.name}({shown}) "
            f"returns {example_out!r}. Reply with only the code."
        )
        hidden = {"name": problem.name, "cases": cases}
        return Task(
            new_task_id(rng, skill, difficulty),
            self.name,
            skill,
            difficulty,
            prompt,
            hidden,
        )

    def grade(self, task: Task, answer: str) -> Grade:
        cases = task.hidden["cases"]
        passed, error = run_tests(extract_code(answer), task.hidden["name"], cases)
        ok = passed == len(cases)
        feedback = f"{passed}/{len(cases)} tests passed" + (
            f" ({error})" if error else ""
        )
        return Grade(ok, passed / len(cases), feedback)
