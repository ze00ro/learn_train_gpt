from typing import Tuple

from transformers import (
    HfArgumentParser
)

from util.config import ModelArguments


def prepare_args() -> Tuple[ModelArguments]:
    parser = HfArgumentParser((ModelArguments,))
    model_args, = parser.parse_args_into_dataclasses()

    return model_args,
