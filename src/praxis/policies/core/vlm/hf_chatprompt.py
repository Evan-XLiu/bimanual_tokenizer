# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""HF chat-template prompting helpers for VLM policy backbones."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = "Analyze the input images and predict robot actions."
USER_BODY_SENTINEL = "\x00PRAXIS_UB\x00"
ACTION_SENTINEL = "\x00PRAXIS_AT\x00"


def _build_user_body(
    *,
    instruction: str,
    state_text: str | None,
) -> str:
    parts = [f"Task: {instruction}."]
    if state_text:
        parts.append(f"{state_text}.")
    return "\n".join(parts)


def _state_text_at(state_texts: list[str] | None, index: int) -> str | None:
    if state_texts is None or index >= len(state_texts):
        return None
    return state_texts[index] or None


def build_messages_batch(
    *,
    num_images_per_sample: list[int],
    instructions: list[str],
    action_token_strings: list[str] | None,
    state_texts: list[str] | None = None,
) -> list[list[dict]]:
    """Build chat-template messages with image placeholders."""
    bsz = len(num_images_per_sample)
    messages_batch: list[list[dict]] = []
    for i in range(bsz):
        content: list[dict] = []
        content.extend({"type": "image"} for _ in range(num_images_per_sample[i]))
        content.append(
            {
                "type": "text",
                "text": _build_user_body(
                    instruction=instructions[i],
                    state_text=_state_text_at(state_texts, i),
                ),
            }
        )
        msg: list[dict] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {"role": "user", "content": content},
        ]

        if action_token_strings is not None:
            msg.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": action_token_strings[i]}],
                }
            )
        messages_batch.append(msg)
    return messages_batch


def render_chat_texts(
    processor: Any,
    *,
    num_images: int,
    instructions: list[str],
    action_token_strings: list[str] | None,
    bsz: int,
    state_texts: list[str] | None = None,
    include_system: bool = True,
) -> list[str]:
    """Render chat-template strings with optional state text."""
    user_body_texts = [
        _build_user_body(
            instruction=instructions[i],
            state_text=_state_text_at(state_texts, i),
        )
        for i in range(bsz)
    ]

    rep_content: list[dict] = [{"type": "image"} for _ in range(num_images)]
    rep_content.append({"type": "text", "text": USER_BODY_SENTINEL})
    is_training = action_token_strings is not None

    rep_msg: list[dict] = []
    if include_system:
        rep_msg.append(
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        )
    rep_msg.append({"role": "user", "content": rep_content})
    if is_training:
        rep_msg.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ACTION_SENTINEL}],
            }
        )

    template = processor.apply_chat_template(
        rep_msg,
        tokenize=False,
        add_generation_prompt=not is_training,
    )

    if USER_BODY_SENTINEL in template:
        if is_training and ACTION_SENTINEL in template:
            if action_token_strings is None:
                raise RuntimeError(
                    "Training chat-template rendering requires action_token_strings."
                )
            pre, _, rest = template.partition(USER_BODY_SENTINEL)
            mid, _, post = rest.partition(ACTION_SENTINEL)
            return [
                pre + user_body_texts[i] + mid + action_token_strings[i] + post
                for i in range(bsz)
            ]
        if not is_training:
            pre, _, post = template.partition(USER_BODY_SENTINEL)
            return [pre + user_body_texts[i] + post for i in range(bsz)]

    messages_batch = build_messages_batch(
        num_images_per_sample=[num_images] * bsz,
        instructions=instructions,
        action_token_strings=action_token_strings,
        state_texts=state_texts,
    )
    return [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=not is_training,
        )
        for message in messages_batch
    ]
