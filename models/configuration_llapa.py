# coding=utf-8
import warnings
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class LlapaConfig(PretrainedConfig):

    model_type = "llapa"
    is_composition = False

    def __init__(
        self,
        protein_config=None,
        text_config=None,
        ignore_index=-100,
        protein_token_index=32000,
        projector_hidden_act="gelu",
        protein_feature_select_strategy="local",
        p_mode="p",
        ROOT_DIR="xxx",
        DATA_DIR="xxx",
        backbone='llama3',
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.protein_token_index = protein_token_index
        self.projector_hidden_act = projector_hidden_act

        if protein_feature_select_strategy not in ["global", "local", "mix"]:
            raise ValueError(
                "protein_feature_select_strategy should be one of 'global', 'local', 'mix'."
                f"Got: {protein_feature_select_strategy}"
            )

        if "vocab_size" in kwargs:
            warnings.warn(
                "The `vocab_size` argument is deprecated and will be removed in v4.42, since it can be inferred from the `text_config`. Passing this argument has no effect",
                FutureWarning,
            )

        self.protein_feature_select_strategy = protein_feature_select_strategy

        if isinstance(protein_config, dict):
            protein_config["model_type"] = (
                protein_config["model_type"] if "model_type" in protein_config else "esm"
            )
            protein_config = CONFIG_MAPPING[protein_config["model_type"]](**protein_config)
        elif protein_config is None:
            protein_config = CONFIG_MAPPING["esm"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )

        self.protein_config = protein_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config
        
        self.p_mode = p_mode
        self.ROOT_DIR = ROOT_DIR
        self.DATA_DIR = DATA_DIR
        self.backbone = backbone
        
        super().__init__(**kwargs)