# Proposal: Unifying the Auto Class Registry in Transformers

*2026-02-16*

---

## 1. Summary

Transformers' Auto class system manages **505 model types** across **51 independent `OrderedDict`s** spread over **7 files**. Each model type (e.g., `"bert"`) has its capabilities scattered — config in one dict, tokenizer in another, each of 40+ task-specific model mappings in yet another. There is no single place that answers "what does model X support?"

This document proposes **unifying these 51 fragmented registries into a single `Namespace` + `Registry` structure**, using the pattern from [`lazyregistry`](https://github.com/milkclouds/lazyregistry) (vendored, not added as an external dependency). The public API (`AutoClass.register()`, `from_pretrained()`) remains unchanged. The result is:

- **Single source of truth** per model type — all capabilities queryable from one place
- **~500 lines of duplicated machinery eliminated** — `_LazyAutoMapping`, `_LazyConfigMapping`, 5× duplicated `class_from_name()`, 40 boilerplate AutoModel definitions
- **Introspection for free** — `pprint_registry()`, `list_model_types()`, `get_model_info()` as natural dict operations
- **Unified storage** — built-in and third-party models in the same structure, eliminating the `_extra_content` dual-storage problem

---

## 2. Current State

### The scale of fragmentation

The auto class system currently consists of:

| Metric | Count |
|--------|-------|
| Unique model types | 505 |
| Separate `MAPPING_NAMES` OrderedDicts | 51 |
| Total data entries across all dicts | 3,045 |
| Files containing registry data | 7 |
| `_LazyAutoMapping` instances | 46 |
| AutoModel class definitions (boilerplate) | 40 |
| Duplicated `*_class_from_name()` functions | 5 (144 lines) |

A single model type like `"bert"` appears in **13 separate dicts** across multiple files. To answer "what does bert support?", you must grep 7 files.

### How registration works today

The public API is well-designed — registering Config, Model, and Tokenizer separately is correct because they are fundamentally different components with different lifecycles:

```python
AutoConfig.register("my-model", MyConfig)
AutoModel.register(MyConfig, MyModel)
AutoModelForCausalLM.register(MyConfig, MyModelForCausalLM)
AutoTokenizer.register(MyConfig, slow_tokenizer_class=MyTokenizer)
```

Internally, each `register()` call stores into a **separate** `_LazyAutoMapping._extra_content` dict. There is no connection between these registrations.

### What works well

The **public API is fine** and should not change. Per-component registration is the right design.

### What doesn't work well

The problems are in the **internal implementation**:

**1. 51 independent registries with no unified view.** Each Auto class maintains its own `_LazyAutoMapping` with its own `_extra_content`. There is no way to ask "what components are registered for model type X?" without manually checking every mapping.

**2. No introspection.** There is no API to list registered models, inspect registration state, or verify completeness.

```python
# None of these exist today:
AutoModel.list_registered()
transformers.pprint_registry()
transformers.get_model_info("bert")  # → all capabilities in one view
```

**3. Dual storage (`_MAPPING_NAMES` vs `_extra_content`).** Built-in models use string-based lazy loading via `CONFIG_MAPPING_NAMES` / `MODEL_MAPPING_NAMES`. Third-party models are stored as direct class references in `_extra_content`. This duality caused the bug fixed in **PR #41865** and forces 5 separate `*_class_from_name()` functions (144 lines of near-identical code) to search both storage paths.

**4. `_extra_content` is a de facto internal API.** Despite being private, `_extra_content` is directly accessed from 93 lines of test code (`del mapping._extra_content[key]` for cleanup), `processing_utils.py`, `pipelines/base.py`, and all 5 `*_class_from_name()` functions. There is no `unregister()` method.

**5. No cross-cutting queries.** "Which models support causal LM and have a tokenizer?" requires manually joining two OrderedDicts. "Which models have an image processor but no video processor?" requires checking 3 files.

---

## 3. Prior Art

### 3.1 The `"module:Class"` String Pattern

Multiple libraries use `"module:Class"` strings for lazy-loading registries. The core idea is simple: store a string like `"transformers.models.bert:BertModel"` instead of importing the class, then resolve it via `importlib` on first access.

Gymnasium (the standard RL environment library) demonstrates this pattern at scale:

```python
# Registration stores a string — no import happens
register(id="CartPole-v1", entry_point="gymnasium.envs.classic_control:CartPoleEnv")

# Import happens only when make() is called
env = gym.make("CartPole-v1")

# Introspection comes naturally
gym.pprint_registry()
```

This is the same pattern transformers already uses internally with `CONFIG_MAPPING_NAMES` (storing `"bert" → "BertConfig"` strings). The difference is that transformers wraps this in a custom `_LazyAutoMapping` class with bolted-on `_extra_content`, while the `"module:Class"` pattern gives both lazy loading and introspection from a single dict.

### 3.2 lazyregistry

[`lazyregistry`](https://github.com/milkclouds/lazyregistry) packages this pattern into ~100 lines providing two constructs:

**Registry** — a lazy-loading dict:

```python
from lazyregistry import Registry

registry = Registry(name="models")
registry["bert"] = "transformers:BertModel"     # string → lazy import
registry["custom"] = MyCustomModel               # direct object → immediate

bert = registry["bert"]  # import happens here, on first access
list(registry.keys())    # introspection — it's just a dict
```

**Namespace** — multi-registry container with per-component isolation:

```python
from lazyregistry import NAMESPACE

NAMESPACE["models"]["bert"] = "transformers:BertModel"
NAMESPACE["tokenizers"]["bert"] = "transformers:BertTokenizer"
# same key "bert" in different registries — no conflict
```

Transformers has 7+ component types (Config, Model×40 task variants, Tokenizer, Processor, etc.) sharing the same `model_type` key space. A `Namespace` provides per-component isolation while enabling unified queries across all registries — something 51 independent `OrderedDict`s cannot do.

**Note on dependency**: `lazyregistry` source code (~100 lines) would be **vendored** directly into transformers, not added as an external dependency. This eliminates any dependency risk while retaining the implementation.

---

## 4. Proposed Design

### Design Principles

1. **Keep the existing per-component `register()` API unchanged.** Config, Model, and Tokenizer are fundamentally different components — separate registration calls are the right design.
2. **Replace the internal mapping with `Namespace` + `Registry`.** Swap `_LazyAutoMapping` + `_extra_content` for a dict-based registry that supports lazy `"module:Class"` strings.
3. **Vendor, don't depend.** The `lazyregistry` source code (~100 lines) is copied into `transformers/_vendor/lazyregistry.py`. No external dependency is added.
4. **Expose introspection via the dict API.** Because `Registry` is a dict, `keys()`, `len()`, `in`, and `pprint_registry()` come naturally.

### 4.1 Internal Structure: Introducing `Namespace` + `Registry`

```python
# transformers/_registry.py (new file)
from transformers._vendor.lazyregistry import Namespace  # vendored, not external

REGISTRY = Namespace()

# Each Auto class references its own registry:
# REGISTRY["configs"]     — model_type → Config class or "module:Class" string
# REGISTRY["models"]      — model_type → Model class or string
# REGISTRY["tokenizers"]  — model_type → Tokenizer class or string
# REGISTRY["processors"]  — model_type → Processor class or string
# REGISTRY["pipelines"]   — task_name  → Pipeline class or string
```

### 4.2 Built-in Model Registration via Lazy Strings

Currently, built-in models are registered via `CONFIG_MAPPING_NAMES` / `MODEL_MAPPING_NAMES` dicts + the `_LazyAutoMapping` wrapper. Under the new design, these become direct `Registry` entries with `"module:Class"` lazy strings:

```python
# transformers/models/auto/configuration_auto.py (modified)
from transformers._registry import REGISTRY

REGISTRY["configs"].update({
    "bert": "transformers.models.bert.configuration_bert:BertConfig",
    "gpt2": "transformers.models.gpt2.configuration_gpt2:GPT2Config",
    "llama": "transformers.models.llama.configuration_llama:LlamaConfig",
    # ... 500+ models
})
```

```python
# transformers/models/auto/modeling_auto.py (modified)
REGISTRY["models"].update({
    "bert": "transformers.models.bert.modeling_bert:BertModel",
    "gpt2": "transformers.models.gpt2.modeling_gpt2:GPT2Model",
    # ...
})

REGISTRY["causal_lm"].update({
    "bert": "transformers.models.bert.modeling_bert:BertLMHeadModel",
    "gpt2": "transformers.models.gpt2.modeling_gpt2:GPT2LMHeadModel",
    # ...
})
```

This eliminates `_LazyAutoMapping`, `_LazyConfigMapping`, and the dual `CONFIG_MAPPING_NAMES` + `_extra_content` structure. Built-in and third-party models live in the same `Registry` dict.

### 4.3 `AutoClass.register()` — Same Public API, New Internals

```python
# transformers/models/auto/auto_factory.py (modified)
class _BaseAutoModelClass:
    @classmethod
    def register(cls, config_class, model_class, exist_ok=False):
        # Before: self._model_mapping._extra_content[config_class] = model_class
        # After:  direct Registry insertion
        model_type = getattr(config_class, "model_type", config_class.__name__)
        registry_name = _auto_class_to_registry_name(cls.__name__)
        REGISTRY[registry_name][model_type] = model_class
```

From the caller's perspective, nothing changes:

```python
AutoConfig.register("my-llm", MyLLMConfig)           # same as today
AutoModel.register(MyLLMConfig, MyLLMModel)            # same as today
AutoModelForCausalLM.register(MyLLMConfig, MyLLMForCausalLM)  # same as today
```

### 4.4 Key Type Transition: Config Class → `model_type` String

The current system uses the **config class** as the lookup key:

```python
# Current: _LazyAutoMapping.__getitem__
type(config) in self._model_mapping  # config CLASS is the key
```

The proposed system uses the **`model_type` string** as the key:

```python
# Proposed: Registry lookup
config.model_type in REGISTRY["models"]  # model_type STRING is the key
```

This is a real change in dispatch semantics, not a trivial swap. Why it is safe:

1. **The current system already resolves through `model_type` internally.** `_LazyAutoMapping` stores `config_name → model_name` strings in `_config_mapping` / `_model_mapping`, and resolves them via `config_class.__name__` → string lookup. The config class is just an indirection layer.
2. **`config.model_type` is the canonical identifier.** Every `PretrainedConfig` has a `model_type` attribute. The `from_pretrained()` path already extracts `model_type` from `config.json` to find the config class.
3. **Edge cases are bounded.** The only case where `model_type` differs from what the class-based lookup would give is if two different config classes share the same `model_type` — which is already a bug in the current system.
4. **Exhaustively testable.** For all 505 model types: `current_dispatch(config) == new_dispatch(config)`. This is a finite, enumerable check.

### 4.5 Introspection API

Because `Registry` is a dict, introspection is a natural consequence rather than a separate feature:

```python
# transformers/_registry.py (additions)
def pprint_registry(component: str | None = None):
    """Print all registered models. Inspired by gymnasium.pprint_registry()."""
    if component:
        registries = {component: REGISTRY[component]}
    else:
        registries = {name: REGISTRY[name] for name in REGISTRY.keys()}

    for name, reg in registries.items():
        print(f"\n=== {name} ({len(reg)} entries) ===")
        for key in sorted(reg.keys()):
            value = reg._raw_get(key)  # inspect without triggering lazy import
            origin = "builtin" if isinstance(value, str) and value.startswith("transformers.") else "third-party"
            print(f"  {key}: {value} [{origin}]")

def list_model_types() -> list[str]:
    """Return all registered model_type strings."""
    return sorted(REGISTRY["configs"].keys())

def get_model_info(model_type: str) -> dict:
    """Return registration info for a specific model_type across all components."""
    return {
        component: REGISTRY[component]._raw_get(model_type)
        for component in REGISTRY.keys()
        if model_type in REGISTRY[component]
    }
```

Usage:

```python
import transformers

transformers.pprint_registry("configs")
# === configs (523 entries) ===
#   bert: transformers.models.bert.configuration_bert:BertConfig [builtin]
#   my-llm: <class 'my_custom_model.config.MyLLMConfig'> [third-party]
#   ...

transformers.get_model_info("my-llm")
# {'configs': <class 'MyLLMConfig'>,
#  'models': <class 'MyLLMModel'>,
#  'causal_lm': <class 'MyLLMForCausalLM'>}
```

### 4.6 Architecture Diagram

**Current (AS-IS)**:

```
[User Code / Third-party __init__.py]
    │
    ├── AutoConfig.register("type", Config)              ─┐
    ├── AutoModel.register(Config, Model)                  │  separate calls
    ├── AutoModelForCausalLM.register(Config, Model)       │  (in-memory only)
    ├── AutoTokenizer.register(Config, Tokenizer)          │
    └── PIPELINE_REGISTRY.register_pipeline(...)         ─┘
                    │
                    ▼
        _LazyAutoMapping._extra_content (dict, in-memory, no introspection)
                    │
                    ▼
        AutoModel.from_pretrained(name)
            → _LazyAutoMapping lookup
            → Model class
```

**Proposed (TO-BE)**:

```
[import my_custom_model]                             ┌───────────────────────┐
        │                                            │  Public API unchanged │
        ▼                                            │                       │
  my_custom_model/__init__.py                        │  AutoConfig.register  │
    ├── AutoConfig.register("my-llm", MyConfig)      │  AutoModel.register   │
    ├── AutoModel.register(MyConfig, MyModel)        │  AutoTokenizer.register│
    └── ...                                          └───────────────────────┘
        │
        ▼
  REGISTRY (lazyregistry.Namespace)
    ├── ["configs"]     {"bert": "...:BertConfig",     "my-llm": MyLLMConfig}
    ├── ["models"]      {"bert": "...:BertModel",      "my-llm": MyLLMModel}
    ├── ["causal_lm"]   {"bert": "...:BertForCausalLM","my-llm": MyLLMForCausalLM}
    ├── ["tokenizers"]  {"bert": "...:BertTokenizer",  "my-llm": MyLLMTokenizer}
    └── ["pipelines"]   {"text-generation": ...,       "my-task": ...}
        │
        │  ← built-in: "module:Class" strings (lazy import on access)
        │  ← third-party: direct objects or strings (both supported)
        │
        ├── pprint_registry()        ← Introspection
        ├── list_model_types()
        ├── get_model_info("my-llm")
        │
        ▼
  AutoModel.from_pretrained(name)
    → config = AutoConfig.from_pretrained(name)
    → model_type = config.model_type
    → REGISTRY["models"][model_type]   ← lazy import trigger
    → Model class
```

---

## 5. Implementation Strategy

### 5.1 Phased Migration (Not Big-Bang)

The migration is structured in 3 phases. At each phase boundary, the system is fully functional and all existing tests pass.

**Phase 1 — Add alongside.** Introduce `REGISTRY` as a new module. Populate it with the same data as the existing `MAPPING_NAMES` dicts. At this point, both systems coexist and neither depends on the other.

**Phase 2 — Delegate.** Modify `_LazyAutoMapping` and `_LazyConfigMapping` to delegate their lookups to `REGISTRY` internally, while keeping the same external interface. All existing code that accesses these mappings continues to work — it just hits `REGISTRY` underneath.

**Phase 3 — Remove old.** Once all tests pass with the delegation layer, remove `_LazyAutoMapping`, `_LazyConfigMapping`, `_LazyLoadAllMappings`, and the 5 duplicated `*_class_from_name()` functions. The `MAPPING_NAMES` OrderedDicts become `REGISTRY[component].update(...)` calls.

Each phase is a self-contained PR with passing tests.

### 5.2 Why This is Low-Risk

A registry is a **mapping**: key → value. Correctness verification is:

```python
# For every model_type in the current system:
for model_type in all_model_types:
    assert current_dispatch(model_type) == new_dispatch(model_type)
```

This is **exhaustively testable** — 505 model types × N registries = a finite, enumerable set. Unlike refactoring complex business logic, there are no hidden state interactions or ordering dependencies. If the mapping test passes for all keys, the refactor is correct.

### 5.3 Quantified Impact

| Deleted | Lines | What |
|---------|-------|------|
| `_LazyAutoMapping` + `_LazyConfigMapping` + `_LazyLoadAllMappings` | ~200 | Custom lazy-loading wrappers |
| `*_class_from_name()` × 5 files | 144 | Near-identical function duplicated 5 times |
| 40 AutoModel boilerplate definitions | ~200 | 35 are pure 4-5 line boilerplate |
| 46 `_LazyAutoMapping()` instantiation lines | ~51 | Per-mapping instance creation |
| `_extra_content` direct access patterns | ~50 | Test cleanup, processing_utils, pipelines/base |
| **Total deleted** | **~645** | |

| Added | Lines | What |
|-------|-------|------|
| `_vendor/lazyregistry.py` | ~100 | Vendored registry implementation |
| `_registry.py` | ~70 | Namespace setup + introspection API |
| **Total added** | **~170** | |

**Net: approximately +170 -645 (~475 lines reduced).** The 3,045 data entries (model type → class name mappings) remain — they are data, not logic, and do not shrink regardless of implementation.

---

## 6. Future Improvement: `entry_points` Auto-Discovery

The design in Section 4 relies on `import my_package` to trigger third-party registration — the same pattern Gymnasium uses successfully. This is explicit, has zero startup cost, and already works today.

However, if there is future demand for fully automatic registration (no `import` needed after `pip install`), Python `entry_points` can be layered on top of the Registry foundation:

```toml
# Third-party package: pyproject.toml
[project.entry-points."transformers.plugins"]
my-llm = "my_custom_model:register"
```

```python
# transformers/__init__.py (hypothetical addition)
import importlib.metadata

def _discover_plugins():
    for ep in importlib.metadata.entry_points(group="transformers.plugins"):
        try:
            register_fn = ep.load()
            register_fn()
        except Exception as e:
            logger.warning(f"Failed to load plugin '{ep.name}': {e}")

_discover_plugins()  # runs at import time
```

**Why this is not part of the core proposal:**

- **Startup cost.** `importlib.metadata.entry_points()` scans all installed packages' metadata. In large environments with hundreds of packages, this adds measurable latency to `import transformers`.
- **Gymnasium proves it's unnecessary.** Gymnasium has a thriving ecosystem of third-party environments (gymnasium-robotics, ale-py, etc.) without using entry_points at all.
- **`import my_package` is sufficient and Pythonic.** One explicit import line is not a meaningful burden. "Explicit is better than implicit."
- **Config metadata can provide on-demand import.** If `config.json` includes a `library_name` field, `from_pretrained()` could auto-import the package when the `model_type` is not found in the registry — this is essentially Gymnasium's `"module:ID"` pattern applied on demand, with zero startup cost.

The `Namespace` + `Registry` foundation makes entry_points trivial to add later if needed — it's just calling `REGISTRY[component][key] = value` from a different trigger point.

---

## 7. Summary

### What changes

| | Current | Proposed |
|---|---|---|
| **Registry structure** | 51 independent `OrderedDict`s across 7 files | 1 `Namespace` with per-component `Registry` dicts |
| **Internal implementation** | `_LazyAutoMapping` + `_extra_content` dual storage | Single dict per component (built-in and third-party unified) |
| **Lazy loading** | Built-in models only | Built-in and third-party (via `"module:Class"` strings) |
| **Introspection** | Not available | `pprint_registry()`, `list_model_types()`, `get_model_info()` |
| **External dependencies** | None added | None added (lazyregistry is vendored) |
| **Net code impact** | — | ~475 lines reduced (+170 -645) |

### What doesn't change

| | |
|---|---|
| **Public registration API** | `AutoConfig.register()`, `AutoModel.register()`, etc. — identical |
| **Third-party registration flow** | `import my_package` triggers registration — same as today |
| **`from_pretrained()` behavior** | Unchanged for both built-in and third-party models |
| **Data** | 3,045 model type → class name entries remain (data, not logic) |

### What this enables that the current architecture cannot

| Capability | Example |
|------------|---------|
| Single source of truth per model | `get_model_info("bert")` → all 13 capabilities in one call |
| Cross-cutting queries | "Which models have both causal_lm and tokenizer?" |
| Atomic registration verification | "Is my-llm fully registered?" (config + model + tokenizer) |
| Clean `unregister()` | Replace 93 lines of `del _extra_content[key]` in tests |
| Eliminate `_extra_content` as de facto API | No more `pipelines/base.py` reaching into `_model_mapping._extra_content.values()` |

### User-facing workflow (unchanged)

```python
# Third-party developer (same as today)
AutoConfig.register("my-llm", MyLLMConfig)
AutoModel.register(MyLLMConfig, MyLLMModel)
AutoModelForCausalLM.register(MyLLMConfig, MyLLMForCausalLM)

# End user (same as today)
import my_custom_model
model = AutoModelForCausalLM.from_pretrained("user/my-llm")

# Introspection (new)
transformers.get_model_info("my-llm")
# {'configs': <class 'MyLLMConfig'>, 'models': <class 'MyLLMModel'>, 'causal_lm': <class 'MyLLMForCausalLM'>}
```

---

*End of Proposal*