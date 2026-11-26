"""Check that interrupted SSE attempts share one verified output prefix."""

import io
import json
import unittest

from serving_replay import CompletionReplay, ReplayMismatch, UpstreamInterrupted, sse


DONE = b"data: [DONE]\n\n"


def frame(ids, text, request_id="first", *, prompt=None, finish=None):
    choice = {"index": 0, "token_ids": ids, "text": text, "finish_reason": finish}
    if prompt is not None:
        choice["prompt_token_ids"] = prompt
    return sse({"id": request_id, "created": 10, "model": "test", "choices": [choice]})


def completion(return_token_ids=True):
    return CompletionReplay({
        "model": "test", "prompt": [7], "stream": True, "temperature": 0,
        "return_token_ids": return_token_ids,
    })


class CompletionReplayTest(unittest.TestCase):
    def test_splits_at_the_visible_prefix_even_when_chunk_boundaries_change(self):
        replay = completion()
        out = []
        interrupted = frame([1, 2], "Aβ", prompt=[7]) + frame([3], "C", finish="length")
        with self.assertRaises(UpstreamInterrupted):
            replay.drain_attempt(io.BytesIO(interrupted), out.append)
        self.assertEqual(len(out), 1)
        self.assertEqual(replay.tokens, [1, 2])
        replacement = (
            frame([1], "AβC", "second", prompt=[7])
            + frame([2, 3], "", "second")
            + frame([4], "D", "second", finish="length")
            + sse({"id": "second", "model": "test", "choices": [], "usage": {"completion_tokens": 4}})
            + DONE
        )
        replay.drain_attempt(io.BytesIO(replacement), out.append)
        events = b"".join(out).split(b"\n\n")
        bodies = [json.loads(line[6:]) for line in events if line and line != DONE.strip()]
        self.assertEqual({body["id"] for body in bodies}, {"first"})
        choices = [choice for body in bodies for choice in body["choices"]]
        self.assertEqual([token for choice in choices for token in choice["token_ids"]], [1, 2, 3, 4])
        self.assertEqual("".join(choice["text"] for choice in choices), "AβCD")
        self.assertEqual(sum("prompt_token_ids" in choice for choice in choices), 1)
        self.assertEqual(sum(choice["finish_reason"] is not None for choice in choices), 1)
        self.assertEqual(b"".join(out).count(DONE), 1)

    def test_discards_the_unverified_suffix_on_token_text_or_prompt_divergence(self):
        for replacement in (
            frame([99, 2], "AB", prompt=[7]),
            frame([1, 2], "XB", prompt=[7]),
            frame([1, 2], "AB", prompt=[8]),
        ):
            with self.subTest(replacement=replacement):
                replay = completion()
                out = []
                with self.assertRaises(UpstreamInterrupted):
                    replay.drain_attempt(io.BytesIO(frame([1], "A", prompt=[7])), out.append)
                with self.assertRaises(ReplayMismatch):
                    replay.drain_attempt(io.BytesIO(replacement + DONE), out.append)
                self.assertEqual(out, [frame([1], "A", prompt=[7])])

    def test_collects_ids_upstream_without_adding_them_to_the_client_stream(self):
        replay = completion(return_token_ids=False)
        self.assertIs(json.loads(replay.payload)["return_token_ids"], True)
        out = []
        replay.drain_attempt(io.BytesIO(frame([1], "A", prompt=[7], finish="length") + DONE), out.append)
        self.assertNotIn(b"token_ids", b"".join(out))


if __name__ == "__main__":
    unittest.main()
