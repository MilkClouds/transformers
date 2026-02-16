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
Equivalence tests for the unified registry prototype.

Verifies that for every (model_type, class_name) in every MAPPING_NAMES dict,
the new REGISTRY resolves to the same class as the old system.
"""

import importlib

import pytest

from transformers._registry import (
    _MODELING_COMPONENTS,
    REGISTRY,
    get_component_model_types,
    get_model_info,
    list_model_types,
)
from transformers.models.auto.auto_factory import getattribute_from_module
from transformers.models.auto.configuration_auto import (
    CONFIG_MAPPING_NAMES,
    model_type_to_module_name,
)
from transformers.models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING_NAMES
from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES
from transformers.models.auto.processing_auto import PROCESSOR_MAPPING_NAMES
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from transformers.models.auto.video_processing_auto import VIDEO_PROCESSOR_MAPPING_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_old(model_type: str, class_name: str):
    """Resolve a class using the old system (importlib + getattribute_from_module)."""
    module_name = model_type_to_module_name(model_type)
    module = importlib.import_module(f".{module_name}", "transformers.models")
    return getattribute_from_module(module, class_name)


# ---------------------------------------------------------------------------
# Config equivalence
# ---------------------------------------------------------------------------


class TestConfigEquivalence:
    @pytest.mark.parametrize("model_type", list(CONFIG_MAPPING_NAMES.keys()))
    def test_config_resolution(self, model_type):
        class_name = CONFIG_MAPPING_NAMES[model_type]
        old_cls = _resolve_old(model_type, class_name)
        new_cls = REGISTRY["config"][model_type]
        assert old_cls is new_cls, f"Config mismatch for {model_type}: old={old_cls}, new={new_cls}"


# ---------------------------------------------------------------------------
# Modeling components equivalence (45 dicts)
# ---------------------------------------------------------------------------


class TestModelingEquivalence:
    @pytest.fixture(params=list(_MODELING_COMPONENTS.items()), ids=lambda x: x[1])
    def component(self, request):
        return request.param  # (var_name, component_name)

    def test_modeling_resolution(self, component):
        var_name, component_name = component
        modeling_auto = importlib.import_module("transformers.models.auto.modeling_auto")
        mapping = getattr(modeling_auto, var_name, None)
        if mapping is None:
            pytest.skip(f"{var_name} not found in modeling_auto")

        errors = []
        for model_type, class_name in mapping.items():
            if class_name is None:
                continue
            try:
                old_cls = _resolve_old(model_type, class_name)
            except (ValueError, ImportError, ModuleNotFoundError):
                # Old system can't resolve it either — skip
                continue

            if model_type not in REGISTRY[component_name].data:
                errors.append(f"  {model_type}: missing from new registry")
                continue

            new_cls = REGISTRY[component_name][model_type]
            # For tuple entries, compare element-by-element with `is`
            if isinstance(old_cls, tuple) and isinstance(new_cls, tuple):
                if len(old_cls) != len(new_cls) or not all(a is b for a, b in zip(old_cls, new_cls)):
                    errors.append(f"  {model_type}: old={old_cls}, new={new_cls}")
            elif old_cls is not new_cls:
                errors.append(f"  {model_type}: old={old_cls}, new={new_cls}")

        assert not errors, f"Mismatches in {var_name} ({component_name}):\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# Simple mapping equivalence (tokenizer, processor, feature_extractor, video)
# ---------------------------------------------------------------------------

_SIMPLE_MAPPINGS = {
    "tokenizer": TOKENIZER_MAPPING_NAMES,
    "processor": PROCESSOR_MAPPING_NAMES,
    "feature_extractor": FEATURE_EXTRACTOR_MAPPING_NAMES,
    "video_processor": VIDEO_PROCESSOR_MAPPING_NAMES,
}


class TestSimpleMappingEquivalence:
    @pytest.fixture(params=list(_SIMPLE_MAPPINGS.items()), ids=lambda x: x[0])
    def mapping_info(self, request):
        return request.param  # (component_name, mapping_dict)

    def test_simple_mapping_resolution(self, mapping_info):
        component_name, mapping = mapping_info
        errors = []
        for model_type, class_name in mapping.items():
            if class_name is None:
                continue
            try:
                old_cls = _resolve_old(model_type, class_name)
            except (ValueError, ImportError, ModuleNotFoundError):
                continue

            if model_type not in REGISTRY[component_name].data:
                errors.append(f"  {model_type}: missing from new registry")
                continue

            new_cls = REGISTRY[component_name][model_type]
            if old_cls is not new_cls:
                errors.append(f"  {model_type}: old={old_cls.__name__}, new={new_cls.__name__}")


# ---------------------------------------------------------------------------
# Image processor tuple equivalence
# ---------------------------------------------------------------------------


class TestImageProcessorEquivalence:
    def test_image_processor_slow(self):
        errors = []
        for model_type, class_info in IMAGE_PROCESSOR_MAPPING_NAMES.items():
            if class_info is None:
                continue
            if isinstance(class_info, tuple):
                slow_class, _ = class_info
            else:
                slow_class = class_info

            if slow_class is None:
                continue

            try:
                old_cls = _resolve_old(model_type, slow_class)
            except (ValueError, ImportError, ModuleNotFoundError):
                continue

            if model_type not in REGISTRY["image_processor"].data:
                errors.append(f"  {model_type}: missing from new registry (slow)")
                continue

            new_cls = REGISTRY["image_processor"][model_type]
            if old_cls is not new_cls:
                errors.append(f"  {model_type}: old={old_cls.__name__}, new={new_cls.__name__}")

        assert not errors, "Mismatches in image_processor (slow):\n" + "\n".join(errors)

    def test_image_processor_fast(self):
        errors = []
        for model_type, class_info in IMAGE_PROCESSOR_MAPPING_NAMES.items():
            if class_info is None or not isinstance(class_info, tuple):
                continue
            _, fast_class = class_info
            if fast_class is None:
                continue

            try:
                old_cls = _resolve_old(model_type, fast_class)
            except (ValueError, ImportError, ModuleNotFoundError):
                continue

            if model_type not in REGISTRY["image_processor_fast"].data:
                errors.append(f"  {model_type}: missing from new registry (fast)")
                continue

            new_cls = REGISTRY["image_processor_fast"][model_type]
            if old_cls is not new_cls:
                errors.append(f"  {model_type}: old={old_cls.__name__}, new={new_cls.__name__}")

        assert not errors, "Mismatches in image_processor_fast:\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# Introspection API tests
# ---------------------------------------------------------------------------


class TestIntrospectionAPI:
    def test_list_model_types_nonempty(self):
        types = list_model_types()
        assert len(types) > 400, f"Expected 400+ model types, got {len(types)}"

    def test_list_model_types_sorted(self):
        types = list_model_types()
        assert types == sorted(types)

    def test_list_model_types_contains_known(self):
        types = list_model_types()
        for mt in ["bert", "gpt2", "llama", "t5", "whisper"]:
            assert mt in types, f"{mt} not in list_model_types()"

    def test_get_model_info_bert(self):
        info = get_model_info("bert")
        assert "config" in info
        assert "model" in info
        assert "masked_lm" in info
        assert "tokenizer" in info
        # bert should have many components
        assert len(info) >= 10

    def test_get_model_info_nonexistent(self):
        info = get_model_info("nonexistent_model_xyz")
        assert info == {}

    def test_get_model_info_returns_strings(self):
        """get_model_info should always return strings (import strings or class reprs)."""
        # Use a model type unlikely to have been resolved by other tests
        info = get_model_info("autoformer")
        assert len(info) > 0, "autoformer should have at least one component"
        for component, value in info.items():
            assert isinstance(value, str), f"Expected str for {component}, got {type(value)}"

    def test_get_component_model_types(self):
        types = get_component_model_types("config")
        assert len(types) > 400
        assert "bert" in types

    def test_get_component_model_types_nonexistent(self):
        types = get_component_model_types("nonexistent_component")
        assert types == []

    def test_registry_completeness(self):
        """Every model_type in CONFIG_MAPPING_NAMES should be in the registry."""
        registry_types = set(REGISTRY["config"].data.keys())
        config_types = set(CONFIG_MAPPING_NAMES.keys())
        missing = config_types - registry_types
        assert not missing, f"Missing from registry: {missing}"

    def test_registry_no_extra_config(self):
        """Registry config should not have types not in CONFIG_MAPPING_NAMES."""
        registry_types = set(REGISTRY["config"].data.keys())
        config_types = set(CONFIG_MAPPING_NAMES.keys())
        extra = registry_types - config_types
        assert not extra, f"Extra in registry: {extra}"
