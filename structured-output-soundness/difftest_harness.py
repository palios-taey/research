#!/usr/bin/env python3
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

import llguidance
import xgrammar as xgr
from jsonschema import Draft202012Validator, FormatChecker
from outlines_core import json_schema as outlines_json_schema


@dataclass(frozen=True)
class Case:
    name: str
    schema: dict[str, Any]
    candidates: tuple[str, ...]


class ByteTokenizer:
    eos_token_id = 256
    bos_token_id = None
    tokens = [bytes([i]) for i in range(256)] + [b"<eos>"]
    special_token_ids = [256]

    def __call__(self, value: str | bytes) -> list[int]:
        if isinstance(value, str):
            value = value.encode("utf-8")
        return list(value)


CASES = (
    Case("number_multipleOf", {"type": "number", "multipleOf": 2}, ("2", "3", "4.0", "5.0")),
    Case("integer_minimum", {"type": "integer", "minimum": 5}, ("5", "4", "-10")),
    Case("integer_maximum", {"type": "integer", "maximum": 5}, ("5", "6", "100")),
    Case("number_exclusive_min", {"type": "number", "exclusiveMinimum": 0}, ("0", "0.1", "-1")),
    Case("number_exp_no_sign", {"type": "number"}, ("1e2", "1e+2", "-2e-6")),
    Case("string_minLength", {"type": "string", "minLength": 2}, ('"a"', '"ab"', '""')),
    Case("string_maxLength", {"type": "string", "maxLength": 2}, ('"ab"', '"abc"')),
    Case("string_pattern", {"type": "string", "pattern": "^[A-Z]+$"}, ('"ABC"', '"abc"', '"A1"')),
    Case("string_unicode_escape", {"type": "string"}, ('"\\u00e9"', '"é"')),
    Case("array_minItems", {"type": "array", "items": {"type": "integer"}, "minItems": 2}, ("[1]", "[1,2]", "[]")),
    Case("array_maxItems", {"type": "array", "items": {"type": "integer"}, "maxItems": 2}, ("[1,2]", "[1,2,3]")),
    Case(
        "array_tuple_prefix",
        {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}], "items": False},
        ('["x",1]', '["x"]', '["x",1,2]', '[1,"x"]'),
    ),
    Case(
        "object_required",
        {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
        ("{}", '{"a":1}', '{"a":"x"}'),
    ),
    Case(
        "object_additional_false",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": False,
        },
        ('{"a":1}', '{"a":1,"b":2}', '{"b":2,"a":1}'),
    ),
    Case(
        "object_additional_schema",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": {"type": "string"},
        },
        ('{"a":1,"b":"x"}', '{"a":1,"b":2}'),
    ),
    Case(
        "object_patternProperties",
        {"type": "object", "patternProperties": {"^x_": {"type": "integer"}}, "additionalProperties": False},
        ('{"x_a":1}', '{"x_a":"bad"}', '{"y":1}'),
    ),
    Case("anyOf_string_or_int", {"anyOf": [{"type": "string"}, {"type": "integer"}]}, ('"x"', "1", "1.2", "true")),
    Case(
        "oneOf_number_multiple",
        {"oneOf": [{"type": "integer"}, {"type": "number", "multipleOf": 2}]},
        ("3", "4", "1.5"),
    ),
    Case("const_string", {"const": "ok"}, ('"ok"', '"no"')),
    Case("enum_mixed", {"enum": ["red", 1, False, None]}, ('"red"', '"blue"', "1", "false", "null", "2")),
    Case("format_email", {"type": "string", "format": "email"}, ('"a@b.com"', '"not-email"')),
)


def ground_truth(schema: dict[str, Any], candidate: str) -> bool:
    try:
        instance = json.loads(candidate)
    except json.JSONDecodeError:
        return False
    return Draft202012Validator(schema, format_checker=FormatChecker()).is_valid(instance)


class Engines:
    def __init__(self) -> None:
        self._ll_tokenizer = llguidance.LLTokenizer(
            llguidance.TokenizerWrapper(ByteTokenizer()),
            n_vocab=257,
            eos_token=256,
        )
        xg_vocab = [bytes([i]) for i in range(256)] + [b"<eos>"]
        xg_tokenizer = xgr.TokenizerInfo(xg_vocab, vocab_size=257, stop_token_ids=[256])
        self._xg_compiler = xgr.GrammarCompiler(xg_tokenizer)

    def llguidance(self, schema: dict[str, Any], candidate: str) -> bool:
        grammar = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema, separators=(",", ":")))
        matcher = llguidance.LLMatcher(self._ll_tokenizer, grammar)
        tokens = self._ll_tokenizer.tokenize_str(candidate)
        consumed = matcher.try_consume_tokens(tokens)
        return consumed == len(tokens) and matcher.is_accepting()

    def xgrammar(self, schema: dict[str, Any], candidate: str) -> bool:
        compiled = self._xg_compiler.compile_json_schema(schema)
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        return bool(matcher.accept_string(candidate) and matcher.is_completed())

    def outlines(self, schema: dict[str, Any], candidate: str) -> bool:
        pattern = outlines_json_schema.build_regex_from_schema(json.dumps(schema, separators=(",", ":")))
        return re.fullmatch(pattern, candidate) is not None


def classify(engine_accepts: bool, ground_accepts: bool) -> str:
    if engine_accepts == ground_accepts:
        return "ok"
    if engine_accepts:
        return "false_accept"
    return "false_reject"


def main() -> None:
    engines = Engines()
    engine_fns: tuple[tuple[str, Callable[[dict[str, Any], str], bool]], ...] = (
        ("llguidance", engines.llguidance),
        ("xgrammar", engines.xgrammar),
        ("outlines", engines.outlines),
    )
    divergences: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for case in CASES:
        for candidate in case.candidates:
            ground_accepts = ground_truth(case.schema, candidate)
            for engine, fn in engine_fns:
                try:
                    engine_accepts = fn(case.schema, candidate)
                except Exception as exc:
                    errors.append(
                        {
                            "engine": engine,
                            "case": case.name,
                            "schema": case.schema,
                            "candidate": candidate,
                            "ground_truth": ground_accepts,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                result = classify(engine_accepts, ground_accepts)
                if result != "ok":
                    divergences.append(
                        {
                            "engine": engine,
                            "case": case.name,
                            "schema": case.schema,
                            "candidate": candidate,
                            "ground_truth": ground_accepts,
                            "engine_accepts": engine_accepts,
                            "classification": result,
                        }
                    )

    print(
        json.dumps(
            {
                "divergences": divergences,
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
