from typing import Optional
from dataclasses import dataclass, field


REPO_ID = "uer/gpt2-chinese-cluecorpussmall"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=REPO_ID,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
