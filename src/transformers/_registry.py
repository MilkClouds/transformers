# Copyright 2026 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unified model registry for transformers.

Consolidates 51 independent OrderedDicts across 7 files into a single
Namespace + Registry structure. The auto class system delegates lookups
to this registry; the MAPPING_NAMES dicts remain as the data source.

Usage:
    from transformers._registry import REGISTRY, list_model_types, get_model_info, pprint_registry

    # What does "bert" support?
    get_model_info("bert")

    # List all model types
    list_model_types()

    # Pretty-print registry
    pprint_registry()
"""

import importlib

from lazyregistry import ImportString, Namespace, Registry


# ---------------------------------------------------------------------------
# Custom Registry subclass with transformers-specific fallback
# ---------------------------------------------------------------------------


class _TransformersRegistry(Registry):
    """Registry that falls back to the top-level transformers module.

    Some MAPPING_NAMES entries reference classes from OTHER model modules
    (e.g., model_type="aimv2" maps to "CLIPProcessor" which lives in the
    clip module, not aimv2). The current system handles this via
    `getattribute_from_module`'s fallback. We replicate that here.
    """

    def _resolve_single(self, value):
        """Resolve a single ImportString with fallback to top-level transformers."""
        try:
            return value.load()
        except Exception:
            # lazyregistry uses pydantic's ImportString internally, which
            # raises pydantic ValidationError on import failure — so we
            # catch broadly here.
            _, class_name = str(value).rsplit(":", 1)
            transformers_module = importlib.import_module("transformers")
            resolved = getattr(transformers_module, class_name, None)
            if resolved is None:
                raise ValueError(f"Could not resolve '{value}': not found in primary module or top-level transformers")
            return resolved

    def __getitem__(self, key):
        value = self.data[key]
        if isinstance(value, ImportString):
            self.data[key] = self._resolve_single(value)
        elif isinstance(value, tuple) and any(isinstance(v, ImportString) for v in value):
            # Tuple of ImportStrings (e.g., modeling entries with multiple classes)
            self.data[key] = tuple(self._resolve_single(v) if isinstance(v, ImportString) else v for v in value)
        return self.data[key]


class _TransformersNamespace(Namespace):
    """Namespace that auto-creates _TransformersRegistry instances."""

    def __missing__(self, key):
        self.data[key] = _TransformersRegistry(name=key)
        return self.data[key]


# ---------------------------------------------------------------------------
# The unified registry
# ---------------------------------------------------------------------------

REGISTRY = _TransformersNamespace()


# ---------------------------------------------------------------------------
# Module resolution (reuses existing logic)
# ---------------------------------------------------------------------------


def _make_import_string(model_type: str, class_name: str) -> str:
    """Convert (model_type, class_name) to 'module.path:ClassName' format."""
    from .models.auto.configuration_auto import model_type_to_module_name

    module_name = model_type_to_module_name(model_type)
    return f"transformers.models.{module_name}:{class_name}"


# ---------------------------------------------------------------------------
# Data population
# ---------------------------------------------------------------------------

# Mapping from MAPPING_NAMES variable name suffix to registry component name
_MODELING_COMPONENTS = {
    "MODEL_MAPPING_NAMES": "model",
    "MODEL_FOR_PRETRAINING_MAPPING_NAMES": "pretraining",
    "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES": "causal_lm",
    "MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES": "causal_image_modeling",
    "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES": "image_classification",
    "MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES": "zero_shot_image_classification",
    "MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES": "image_segmentation",
    "MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES": "semantic_segmentation",
    "MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES": "instance_segmentation",
    "MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES": "universal_segmentation",
    "MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES": "video_classification",
    "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES": "image_text_to_text",
    "MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES": "multimodal_lm",
    "MODEL_FOR_RETRIEVAL_MAPPING_NAMES": "retrieval",
    "MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES": "visual_question_answering",
    "MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES": "document_question_answering",
    "MODEL_FOR_MASKED_LM_MAPPING_NAMES": "masked_lm",
    "MODEL_FOR_IMAGE_MAPPING_NAMES": "image",
    "MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES": "masked_image_modeling",
    "MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES": "object_detection",
    "MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES": "zero_shot_object_detection",
    "MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES": "depth_estimation",
    "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES": "seq_to_seq_causal_lm",
    "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES": "sequence_classification",
    "MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES": "question_answering",
    "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES": "table_question_answering",
    "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES": "token_classification",
    "MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES": "multiple_choice",
    "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES": "next_sentence_prediction",
    "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES": "audio_classification",
    "MODEL_FOR_CTC_MAPPING_NAMES": "ctc",
    "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES": "speech_seq_to_seq",
    "MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES": "audio_frame_classification",
    "MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES": "audio_xvector",
    "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES": "text_to_spectrogram",
    "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES": "text_to_waveform",
    "MODEL_FOR_BACKBONE_MAPPING_NAMES": "backbone",
    "MODEL_FOR_MASK_GENERATION_MAPPING_NAMES": "mask_generation",
    "MODEL_FOR_KEYPOINT_DETECTION_MAPPING_NAMES": "keypoint_detection",
    "MODEL_FOR_KEYPOINT_MATCHING_MAPPING_NAMES": "keypoint_matching",
    "MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES": "text_encoding",
    "MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING_NAMES": "time_series_classification",
    "MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING_NAMES": "time_series_regression",
    "MODEL_FOR_TIME_SERIES_PREDICTION_MAPPING_NAMES": "time_series_prediction",
    "MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES": "image_to_image",
    "MODEL_FOR_AUDIO_TOKENIZATION_NAMES": "audio_tokenization",
}


def _populate_simple(component_name: str, mapping_names: dict):
    """Populate a registry component from a simple str→str or str→tuple mapping.

    Some modeling mappings have tuple values like ("FunnelModel", "FunnelBaseModel")
    where a single model_type maps to multiple classes. We store these as tuples
    of ImportStrings.
    """
    for model_type, class_name in mapping_names.items():
        if class_name is None:
            continue
        if isinstance(class_name, tuple):
            # Multiple classes for one model_type (e.g., funnel, deit, perceiver)
            import_strings = tuple(ImportString(_make_import_string(model_type, cn)) for cn in class_name)
            # Store as raw tuple in .data to bypass auto_import_strings
            REGISTRY[component_name].data[model_type] = import_strings
        else:
            REGISTRY[component_name][model_type] = _make_import_string(model_type, class_name)


def _populate_tuple(component_name: str, mapping_names: dict):
    """Populate a registry component from a str→tuple mapping (image/video processors).

    For tuple entries like (slow_class, fast_class), we store BOTH as separate
    sub-registries: "{component_name}_slow" and "{component_name}_fast".
    The main component_name maps to the slow class (primary).
    """
    for model_type, class_info in mapping_names.items():
        if class_info is None:
            continue
        if isinstance(class_info, tuple):
            slow_class, fast_class = class_info
            if slow_class is not None:
                REGISTRY[component_name][model_type] = _make_import_string(model_type, slow_class)
            if fast_class is not None:
                REGISTRY[f"{component_name}_fast"][model_type] = _make_import_string(model_type, fast_class)
        elif isinstance(class_info, str):
            # Some entries might be plain strings (e.g., video processor after filtering)
            REGISTRY[component_name][model_type] = _make_import_string(model_type, class_info)


def populate_registry():
    """Populate REGISTRY from all existing MAPPING_NAMES.

    This function is called once at module load time. It reads data from the
    existing OrderedDicts and stores it in the unified registry.
    """
    from .models.auto import configuration_auto, modeling_auto
    from .models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING_NAMES
    from .models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES
    from .models.auto.processing_auto import PROCESSOR_MAPPING_NAMES
    from .models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
    from .models.auto.video_processing_auto import VIDEO_PROCESSOR_MAPPING_NAMES

    # 1. Config
    _populate_simple("config", configuration_auto.CONFIG_MAPPING_NAMES)

    # 2. All modeling components (43 dicts + MODEL_MAPPING_NAMES)
    for var_name, component_name in _MODELING_COMPONENTS.items():
        mapping = getattr(modeling_auto, var_name, None)
        if mapping is not None:
            _populate_simple(component_name, mapping)

    # 3. Tokenizer (str → str|None)
    _populate_simple("tokenizer", TOKENIZER_MAPPING_NAMES)

    # 4. Processor (str → str)
    _populate_simple("processor", PROCESSOR_MAPPING_NAMES)

    # 5. Image processor (str → tuple(slow, fast))
    _populate_tuple("image_processor", IMAGE_PROCESSOR_MAPPING_NAMES)

    # 6. Feature extractor (str → str)
    _populate_simple("feature_extractor", FEATURE_EXTRACTOR_MAPPING_NAMES)

    # 7. Video processor (str → str|None, after filtering)
    _populate_simple("video_processor", VIDEO_PROCESSOR_MAPPING_NAMES)


# ---------------------------------------------------------------------------
# Introspection API
# ---------------------------------------------------------------------------


def list_model_types() -> list[str]:
    """Return a sorted list of all registered model types."""
    all_types = set()
    for registry in REGISTRY.values():
        all_types.update(registry.data.keys())
    return sorted(all_types)


def get_model_info(model_type: str) -> dict[str, str]:
    """Return all registered components for a given model type.

    Returns a dict mapping component_name → import_string (not resolved).
    This does NOT trigger any imports.
    """
    info = {}
    for component_name, registry in REGISTRY.items():
        if model_type in registry.data:
            raw_value = registry.data[model_type]
            info[component_name] = str(raw_value)
    return info


def get_component_model_types(component: str) -> list[str]:
    """Return all model types registered for a given component."""
    if component not in REGISTRY:
        return []
    return sorted(REGISTRY[component].data.keys())


def pprint_registry(model_type: str | None = None):
    """Pretty-print the registry.

    If model_type is given, print only that model's info.
    Otherwise, print summary statistics.
    """
    if model_type is not None:
        info = get_model_info(model_type)
        if not info:
            print(f"Model type '{model_type}' not found in registry.")
            return
        print(f"=== {model_type} ===")
        for component, import_str in sorted(info.items()):
            print(f"  {component}: {import_str}")
        return

    # Summary mode
    all_types = list_model_types()
    print("Unified Registry Summary")
    print(f"  Model types: {len(all_types)}")
    print(f"  Components:  {len(REGISTRY)}")
    print(f"  Total entries: {sum(len(r.data) for r in REGISTRY.values())}")
    print()
    print("Components:")
    for name, registry in sorted(REGISTRY.items()):
        print(f"  {name}: {len(registry.data)} model types")


# ---------------------------------------------------------------------------
# Class resolution by name (replaces 5 duplicated *_class_from_name funcs)
# ---------------------------------------------------------------------------


def class_from_name(component: str, class_name: str):
    """Resolve a class name to its actual class via the registry.

    Searches the given component registry for an entry whose class name
    matches. This replaces the 5 near-identical ``*_class_from_name()``
    functions scattered across the auto modules.

    Args:
        component: Registry component to search (e.g., "processor", "feature_extractor").
        class_name: The class name string to look up (e.g., "CLIPProcessor").

    Returns:
        The resolved class, or None if not found.
    """
    from lazyregistry import ImportString

    if component not in REGISTRY:
        return None

    registry = REGISTRY[component]
    for model_type, value in registry.data.items():
        raw = value
        if isinstance(raw, ImportString):
            # Check the class name portion of the import string (after ":")
            _, name = str(raw).rsplit(":", 1)
            if name == class_name:
                try:
                    return registry[model_type]  # triggers lazy load
                except Exception:
                    continue
        elif isinstance(raw, tuple):
            for i, v in enumerate(raw):
                if isinstance(v, ImportString):
                    _, name = str(v).rsplit(":", 1)
                    if name == class_name:
                        try:
                            resolved = registry[model_type]
                            return resolved[i] if isinstance(resolved, tuple) else resolved
                        except Exception:
                            continue
                elif hasattr(v, "__name__") and v.__name__ == class_name:
                    return v
        elif hasattr(raw, "__name__") and raw.__name__ == class_name:
            return raw
    return None


# ---------------------------------------------------------------------------
# Initialize on import
# ---------------------------------------------------------------------------


def _initialize():
    """Initialize the registry. Called once on first import."""
    if not REGISTRY.data:
        populate_registry()


_initialize()
