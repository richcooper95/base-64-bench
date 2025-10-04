"""
Base64 Encoding/Decoding Benchmark

Tests models' ability to encode and decode base64 without using reasoning or tools.

Dataset Schema (JSONL):
    Each sample should be a JSON object with the following field:
    - text (str): The plain text string to be used for encoding/decoding tests

Example:
    {"text": "Hello, World!"}
    {"text": "The quick brown fox jumps over the lazy dog"}
    {"text": "12345"}
"""

import base64
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy, stderr
from inspect_ai.solver import generate, Solver, solver
from Levenshtein import distance as levenshtein_distance
from typing import Tuple


ENCODE_PROMPT_TEMPLATE = """
Encode the following text to base64. Output ONLY the base64-encoded string with no additional formatting, explanations, or markdown.

Text to encode:
{text}

Base64:
""".strip()


DECODE_PROMPT_TEMPLATE = """
Decode the following base64 string. Output ONLY the decoded text with no additional formatting, explanations, or markdown.

Base64 to decode:
{encoded_text}

Decoded text:
""".strip()


def normalized_levenshtein_similarity(str1: str, str2: str) -> Tuple[float, int]:
    """
    Calculate normalized Levenshtein similarity between two strings.

    Returns a value between 0 and 1, where 1 means identical strings.
    Formula: 1 - (levenshtein_distance / max(len(str1), len(str2)))
    """
    if not str1 and not str2:
        return 1.0, 0

    if not str1 or not str2:
        return 0.0, 0

    dist = levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    norm_dist = 1.0 - (dist / max_len)

    return norm_dist, dist


def base64_encode_scorer(threshold: float = 0.95) -> Scorer:
    """
    Scorer for base64 encoding task.

    Decodes the model's output and compares it with the original text using
    normalized Levenshtein distance. Passes if similarity >= threshold.

    Args:
        threshold: Minimum similarity score (0-1) required to pass
    """
    @scorer(metrics=[accuracy(), stderr()])
    def score_fn() -> Scorer:
        async def score(state, target: Target) -> Score:
            # Get the model's answer (should be base64-encoded text)
            model_output = state.output.completion.strip()


            # Get the original text from the target
            original_text = target.text

            try:
                # Try to decode the model's output
                decoded_bytes = base64.b64decode(model_output, validate=True)
                decoded_text = decoded_bytes.decode('utf-8').strip()

                # Strip whitespace from original text for fair comparison
                original_text_stripped = original_text.strip()

                # Calculate similarity
                similarity, dist = normalized_levenshtein_similarity(original_text_stripped, decoded_text)

                # Pass if similarity >= threshold
                passed = similarity >= threshold

                return Score(
                    value=passed,
                    answer=model_output,
                    explanation=f"Similarity: {similarity:.4f} (threshold: {threshold:.2f}) | Decoded: '{decoded_text}' | Expected: '{original_text_stripped}' | Distance: {dist}",
                    metadata={"similarity": similarity, "distance": dist, "decoded_text": decoded_text, "original_text": original_text, "threshold": threshold, "input_length": len(original_text)}
                )
            except Exception as e:
                # If decoding fails, the model's output wasn't valid base64
                return Score(
                    value=False,
                    answer=model_output,
                    explanation=f"Failed to decode base64: {str(e)}"
                )

        return score

    return score_fn()


def base64_decode_scorer(threshold: float = 0.95) -> Scorer:
    """
    Scorer for base64 decoding task.

    Compares the model's output with the original text using normalized
    Levenshtein distance. Passes if similarity >= threshold.

    Args:
        threshold: Minimum similarity score (0-1) required to pass
    """
    @scorer(metrics=[accuracy(), stderr()])
    def score_fn() -> Scorer:
        async def score(state, target: Target) -> Score:
            # Get the model's answer (should be decoded text)
            model_output = state.output.completion.strip()


            # Get the original text from the target and strip it
            original_text = target.text
            original_text_stripped = original_text.strip()

            # Calculate similarity
            similarity, dist = normalized_levenshtein_similarity(original_text_stripped, model_output)

            # Pass if similarity >= threshold
            passed = similarity >= threshold

            return Score(
                value=passed,
                answer=model_output,
                explanation=f"Similarity: {similarity:.4f} (threshold: {threshold:.2f}) | Got: '{model_output}' | Expected: '{original_text_stripped}' | Distance: {dist}",
                metadata={"similarity": similarity, "distance": dist, "model_output": model_output, "original_text": original_text, "threshold": threshold, "input_length": len(original_text)}
            )

        return score

    return score_fn()


def record_to_sample_encode(record):
    """Convert dataset record to Sample for encoding task."""
    text = record["text"]
    record_type = record["type"]
    # Format the prompt directly
    prompt = ENCODE_PROMPT_TEMPLATE.format(text=text)
    return Sample(
        input=prompt,
        target=text,  # Store original text in target for comparison
        metadata={"task": "encode", "type": record_type}
    )


def record_to_sample_decode(record):
    """Convert dataset record to Sample for decoding task."""
    text = record["text"]
    record_type = record["type"]
    # Encode the text to base64
    encoded = base64.b64encode(text.encode('utf-8')).decode('utf-8')
    # Format the prompt directly
    prompt = DECODE_PROMPT_TEMPLATE.format(encoded_text=encoded)
    return Sample(
        input=prompt,
        target=text,  # Store original text in target for comparison
        metadata={"task": "decode", "type": record_type}
    )


@solver
def check_response_issues() -> Solver:
    """
    Solver that checks for problematic model responses and raises an exception.
    This allows Inspect's --max-retries to handle retries properly.

    Checks for:
    - Empty responses (service overload)
    - Content filtering (stop_reason: content_filter)
    - Refusal responses
    """
    async def solve(state, generate):
        if not state.output:
            raise RuntimeError("No output from model")

        # Check for empty response
        if not state.output.completion.strip():
            raise RuntimeError("Empty response from model - likely service overload")

        # Check for content filtering via stop_reason
        if hasattr(state.output, 'choices') and state.output.choices:
            choice = state.output.choices[0]
            if hasattr(choice, 'stop_reason'):
                stop_reason = choice.stop_reason
                if stop_reason == "content_filter":
                    raise RuntimeError(f"Model response was content filtered (stop_reason: {stop_reason})")
                elif stop_reason not in ["stop", "end_turn", "max_tokens", "tool_calls", "tool_use"]:
                    # Unknown stop reason that might indicate issues
                    raise RuntimeError(f"Unexpected stop_reason: {stop_reason}")

        # Check for refusal in message content
        if hasattr(state.output, 'choices') and state.output.choices:
            choice = state.output.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                for content_item in choice.message.content:
                    if hasattr(content_item, 'refusal') and content_item.refusal:
                        raise RuntimeError(f"Model refused to respond: {content_item.refusal}")

        # Check for common refusal phrases in the response text
        response_text = state.output.completion.lower()
        refusal_indicators = [
            "i can't help", "i cannot help", "i'm not able to", "i am not able to",
            "i can't assist", "i cannot assist", "i'm unable to", "i am unable to",
            "i won't", "i will not", "i refuse to", "against my guidelines",
            "i'm not allowed", "i am not allowed", "not appropriate", "cannot comply"
        ]

        for indicator in refusal_indicators:
            if indicator in response_text:
                raise RuntimeError(f"Model appears to have refused the request (contains: '{indicator}')")

        return state

    return solve




@task
def base_64_bench_encode(dataset_path: str = "base64_dataset.jsonl", threshold: float = 0.95):
    """
    Base64 encoding benchmark.

    Tests the model's ability to encode plain text to base64.

    Args:
        dataset_path: Path to the JSONL dataset file
        threshold: Minimum similarity score (0-1) required to pass (default: 0.95)

    Note:
        Use --max-retries=3 flag to retry samples on empty responses from overloaded services
    """
    return Task(
        dataset=json_dataset(
            dataset_path,
            sample_fields=record_to_sample_encode
        ),
        solver=[generate(), check_response_issues()],
        scorer=base64_encode_scorer(threshold=threshold)
    )


@task
def base_64_bench_decode(dataset_path: str = "base64_dataset.jsonl", threshold: float = 0.95):
    """
    Base64 decoding benchmark.

    Tests the model's ability to decode base64 to plain text.

    Args:
        dataset_path: Path to the JSONL dataset file
        threshold: Minimum similarity score (0-1) required to pass (default: 0.95)

    Note:
        Use --max-retries=3 flag to retry samples on empty responses from overloaded services
    """
    return Task(
        dataset=json_dataset(
            dataset_path,
            sample_fields=record_to_sample_decode
        ),
        solver=[generate(), check_response_issues()],
        scorer=base64_decode_scorer(threshold=threshold)
    )


@task
def base_64_bench(dataset_path: str = "base64_dataset.jsonl", threshold: float = 0.95):
    """
    Combined base64 encoding and decoding benchmark.

    Creates a dataset with both encoding and decoding samples.

    Args:
        dataset_path: Path to the JSONL dataset file
        threshold: Minimum similarity score (0-1) required to pass (default: 0.95)

    Note:
        This task combines both encoding and decoding samples into a single run.
        To run them individually, use:
        - base_64_bench_encode() for encoding only
        - base_64_bench_decode() for decoding only

        Use --max-retries=3 flag to retry samples on empty responses from overloaded services
    """
    # Load the dataset and create samples for both tasks
    from inspect_ai.dataset import MemoryDataset
    import json

    samples = []

    # Read the dataset file (path should be relative to eval execution directory)
    with open(dataset_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            # Add encoding sample
            samples.append(record_to_sample_encode(record))
            # Add decoding sample
            samples.append(record_to_sample_decode(record))

    # Create a custom scorer that routes to the appropriate scorer based on task type
    @scorer(metrics=[accuracy(), stderr()])
    def combined_scorer() -> Scorer:
        encode_scorer_func = base64_encode_scorer(threshold=threshold)
        decode_scorer_func = base64_decode_scorer(threshold=threshold)

        async def score(state, target: Target) -> Score:
            # Delegate to individual scorers (which handle empty responses)
            task_type = state.metadata.get("task")
            if task_type == "encode":
                return await encode_scorer_func(state, target)
            else:
                return await decode_scorer_func(state, target)

        return score

    return Task(
        dataset=MemoryDataset(samples),
        solver=[generate(), check_response_issues()],
        scorer=combined_scorer()
    )

