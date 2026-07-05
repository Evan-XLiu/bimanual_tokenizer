# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from praxis.policies.core.vlm.hf_chatprompt import (
    ACTION_SENTINEL,
    USER_BODY_SENTINEL,
    render_chat_texts,
)


class _TemplateProcessor:
    def apply_chat_template(
        self, message, *, tokenize: bool, add_generation_prompt: bool
    ):
        assert tokenize is False
        parts: list[str] = []
        for msg in message:
            parts.append(f"{msg['role']}:")
            for item in msg["content"]:
                if item["type"] == "image":
                    parts.append("<image>")
                else:
                    parts.append(item["text"])
        if add_generation_prompt:
            parts.append("assistant:")
        return "|".join(parts)


def test_render_chat_texts_omits_optional_state_text():
    rendered = render_chat_texts(
        _TemplateProcessor(),
        num_images=2,
        instructions=["open drawer"],
        action_token_strings=None,
        bsz=1,
        state_texts=None,
    )

    assert len(rendered) == 1
    assert "Task: open drawer." in rendered[0]
    assert "State:" not in rendered[0]
    assert USER_BODY_SENTINEL not in rendered[0]


def test_render_chat_texts_includes_state_text_when_provided():
    rendered = render_chat_texts(
        _TemplateProcessor(),
        num_images=1,
        instructions=["open drawer"],
        action_token_strings=["<|action_0|>"],
        bsz=1,
        state_texts=["State: 1 2 3"],
    )

    assert len(rendered) == 1
    assert "Task: open drawer." in rendered[0]
    assert "State: 1 2 3." in rendered[0]
    assert "<|action_0|>" in rendered[0]
    assert ACTION_SENTINEL not in rendered[0]
    assert USER_BODY_SENTINEL not in rendered[0]
