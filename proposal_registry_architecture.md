# Proposal: Improving Third-Party Model Registration in Transformers

*2026-02-16*

---

## 1. Summary

Transformers supports third-party model registration through several pathways, and the public API (`AutoClass.register()`) is well-designed. However, the **internal registry implementation** has structural limitations — no introspection, no lazy loading for third-party models, and a fragmented internal mapping structure (`_LazyAutoMapping` + `_extra_content`).

This document proposes replacing the internal registry with a `Namespace` + `Registry` pattern (as implemented by [`lazyregistry`](https://github.com/milkclouds/lazyregistry)), while **keeping the existing public API unchanged**. This gives us introspection and unified lazy loading for free, with zero breaking changes.

---

## 2. Current State

### How third-party registration works today

There are five pathways for registering third-party models. The most relevant for this proposal are **A** and **D**:

**Pathway A — `AutoClass.register()` (runtime, in-memory)**:

```python
AutoConfig.register("my-model", MyConfig)
AutoModel.register(MyConfig, MyModel)
AutoModelForCausalLM.register(MyConfig, MyModelForCausalLM)
AutoTokenizer.register(MyConfig, slow_tokenizer_class=MyTokenizer)
```

Internally, these store into `_LazyAutoMapping._extra_content`, a plain dict that lives only in process memory.

**Pathway D — Third-party library pattern** (most common for installable packages):

```python
# my_models_lib/__init__.py
from transformers import AutoConfig, AutoModel
AutoConfig.register("my-model", MyConfig)
AutoModel.register(MyConfig, MyModel)

# User code
import my_models_lib  # triggers registration
model = AutoModel.from_pretrained("path/to/my-model")
```

Other pathways: **B** (`trust_remote_code` — Hub dynamic code loading), **C** (`register_for_auto_class()` — Hub push preparation), **E** (`PIPELINE_REGISTRY.register_pipeline()`).

### Components that can be registered

| Component | Auto Class | Registration |
|-----------|-----------|-------------|
| Config | `AutoConfig` | `AutoConfig.register(model_type, ConfigClass)` |
| Model | `AutoModel`, `AutoModelForCausalLM`, ... (20+) | `AutoModelForX.register(ConfigClass, ModelClass)` |
| Tokenizer | `AutoTokenizer` | `AutoTokenizer.register(ConfigClass, slow_tokenizer_class=...)` |
| Processor | `AutoProcessor` | `AutoProcessor.register(ConfigClass, ProcessorClass)` |
| ImageProcessor | `AutoImageProcessor` | `AutoImageProcessor.register(ConfigClass, IPClass)` |
| FeatureExtractor | `AutoFeatureExtractor` | `AutoFeatureExtractor.register(ConfigClass, FEClass)` |
| Pipeline | `PIPELINE_REGISTRY` | `PIPELINE_REGISTRY.register_pipeline(task, ...)` |

### What works well

The **public API is fine**. Registering Config, Model, and Tokenizer separately is the right design — they are fundamentally different components with different lifecycles. A user might register Config + Model but reuse an existing Tokenizer, or register multiple task-specific models for one Config. Explicit per-component registration is clear and Pythonic.

### What doesn't work well

The problems are in the **internal implementation**, not the public API:

**1. No introspection.** There is no way to list registered third-party models, inspect their registration state, or verify completeness.

```python
# None of these exist today:
AutoModel.list_registered()
transformers.pprint_registry()
```

**2. Fragmented internal structure.** Each of the 20+ Auto classes maintains its own `_LazyAutoMapping` with a separate `_extra_content` dict. There is no unified view of what's registered.

**3. Lazy loading only for built-in models.** `_LazyAutoMapping` provides lazy loading via string-based lookups in `CONFIG_MAPPING_NAMES` / `MODEL_MAPPING_NAMES`, but third-party models registered via `register()` are stored as direct class references in `_extra_content` — they must be imported at registration time.

---

## 3. Prior Art

### 3.1 Gymnasium

Gymnasium (the standard RL environment library) solves an analogous registration problem — third-party environments need to be registered and used through a unified API — with a remarkably simple design.

**Core architecture** (from `gymnasium/envs/registration.py`):

```python
registry: dict[str, EnvSpec] = {}  # global registry — just a dict

def register(id: str, entry_point: str | Callable | None = None, ...):
    """entry_point is a "module:ClassName" string — no import at registration time."""
    registry[id] = EnvSpec(id=id, entry_point=entry_point, ...)

def make(id: str, **kwargs) -> Env:
    """Import happens at call time — lazy loading is built in."""
    spec = registry[id]
    env_creator = load(spec.entry_point)  # importlib resolves the string
    return env_creator(**kwargs)

def pprint_registry():    # introspection
def spec(id: str):         # individual spec lookup
```

**Built-in environments use the same pattern**:

```python
# gymnasium/envs/__init__.py
register(id="CartPole-v1", entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv")
register(id="LunarLander-v3", entry_point="gymnasium.envs.box2d.lunar_lander:LunarLander")
```

**Third-party registration**:

```python
# Option 1: register in package __init__.py
import gymnasium as gym
gym.register(id="my_envs/MyEnv-v0", entry_point="my_envs.envs:MyEnv")

# Option 2: pass module path directly to make()
env = gym.make("my_envs:my_envs/MyEnv-v0")  # detects ":" → auto-imports module
```

**Key insight**: Gymnasium does **not** use `entry_points` for auto-discovery. The `"module:Class"` string pattern provides lazy loading, and `pprint_registry()` provides introspection. Third-party packages register via `import my_envs` (same as transformers Pathway D).

### 3.2 lazyregistry

[`lazyregistry`](https://github.com/milkclouds/lazyregistry) generalizes the Gymnasium pattern into a reusable library. Its key constructs:

**Registry** — a lazy-loading dict:

```python
from lazyregistry import Registry

registry = Registry(name="models")
registry["bert"] = "transformers:BertModel"     # string → lazy import
registry["custom"] = MyCustomModel               # direct object → immediate

bert = registry["bert"]  # import happens here, on first access
list(registry.keys())    # introspection for free — it's just a dict
```

**Namespace** — isolated multi-registry container:

```python
from lazyregistry import NAMESPACE

NAMESPACE["models"]["bert"] = "transformers:BertModel"
NAMESPACE["tokenizers"]["bert"] = "transformers:BertTokenizer"
# same key "bert" coexists across models/tokenizers without conflict
```

**AutoRegistry** — automatic class dispatch via `type_key` in config:

```python
from lazyregistry.pretrained import AutoRegistry, PretrainedMixin

class AutoModel(AutoRegistry):
    registry = NAMESPACE["models"]
    config_class = ModelConfig
    type_key = "model_type"  # dispatches on config.model_type

@AutoModel.register_module("bert")
class BertModel(PretrainedMixin[ModelConfig]): ...

# bulk registration with lazy import strings
AutoModel.registry.update({
    "roberta": "transformers:RobertaModel",
    "t5": "transformers:T5Model",
})
```

### 3.3 Comparison

| | Transformers (current) | Gymnasium | lazyregistry |
|---|---|---|---|
| **Registry structure** | `_LazyAutoMapping` × 20+ Auto classes | single `registry` dict | `Registry` dict × N (isolated via Namespace) |
| **Registration** | `AutoConfig.register()`, `AutoModel.register()`, ... separately | `register()` — one function | `registry["key"] = "module:Class"` |
| **Lazy loading** | Built-in only (`_LazyAutoMapping`) | `"module:Class"` string | `"module:object"` → `ImportString` auto-conversion |
| **Introspection** | ❌ None | `pprint_registry()`, `spec()` | `keys()`, `len()`, `in` (dict API) |
| **Auto-discovery** | ❌ None | ❌ None | ❌ None |
| **type_key dispatch** | `model_type` → hardcoded mapping | N/A | `AutoRegistry.type_key` |
| **Component types** | 7+ (Config, Model×N, Tokenizer, ...) | 1 (Env) | Unlimited (Namespace isolation) |

**Why Namespace fits transformers better than Gymnasium's single-registry approach**: Gymnasium registers only one component type (Env), so a single dict suffices. Transformers registers Config, Model (×20+ task variants), Tokenizer, Processor, etc. — fundamentally different component types that share the same `model_type` key. The `Namespace` pattern provides per-component isolation while allowing unified introspection across all registries.

---

## 4. Proposed Design

### Design Principles

1. **Keep the existing per-component `register()` API unchanged.** Config, Model, and Tokenizer are fundamentally different components — separate registration calls are the right design.
2. **Replace the internal mapping with `Namespace` + `Registry`.** Swap `_LazyAutoMapping` + `_extra_content` for a dict-based registry that supports lazy `"module:Class"` strings.
3. **Expose introspection via the dict API.** Because `Registry` is a dict, `keys()`, `len()`, `in`, and `pprint_registry()` come naturally.

### 4.1 Internal Structure: Introducing `Namespace` + `Registry`

```python
# transformers/_registry.py (new file)
from lazyregistry import Namespace

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

### 4.4 Introspection API

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

### 4.5 Architecture Diagram

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

## 5. Implementation Priority

| Priority | Change | Impact | Difficulty | Compatibility |
|----------|--------|--------|------------|---------------|
| **🥇 1** | Replace internals with `Namespace` + `Registry` | ★★★★★ | ★★★☆☆ | 100% backward-compatible (API identical) |
| **🥈 2** | Expose introspection API | ★★★★☆ | ★★☆☆☆ | Pure addition |

**#1 is the foundation.** Once `_LazyAutoMapping` + `_extra_content` is replaced with `Registry`, #2 (introspection) comes for free as dict API, and future improvements (see Section 6) can be layered on top naturally.

---

## 6. Future Improvement: `entry_points` Auto-Discovery

The design in Section 4 relies on `import my_package` (Pathway D) to trigger third-party registration — the same pattern Gymnasium uses successfully. This is explicit, has zero startup cost, and already works today.

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
| **Internal registry** | `_LazyAutoMapping` + `_extra_content` per Auto class | `Namespace` + `Registry` (single dict-based structure) |
| **Lazy loading** | Built-in models only | Built-in and third-party (via `"module:Class"` strings) |
| **Introspection** | Not available | `pprint_registry()`, `list_model_types()`, `get_model_info()` |

### What doesn't change

| | |
|---|---|
| **Public registration API** | `AutoConfig.register()`, `AutoModel.register()`, etc. — identical |
| **Third-party registration flow** | `import my_package` triggers registration — same as today |
| **`from_pretrained()` behavior** | Unchanged for both built-in and third-party models |

### Third-party library developer workflow

```python
# my_custom_model/__init__.py
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from .config import MyLLMConfig
from .model import MyLLMModel, MyLLMForCausalLM

AutoConfig.register("my-llm", MyLLMConfig)
AutoModel.register(MyLLMConfig, MyLLMModel)
AutoModelForCausalLM.register(MyLLMConfig, MyLLMForCausalLM)
```

No changes needed from today's Pathway D — it just works.

### End-user workflow

```python
import my_custom_model  # one line — triggers registration
from transformers import AutoModelForCausalLM

# introspection (new)
import transformers
transformers.pprint_registry("configs")
# === configs (524 entries) ===
#   ...
#   my-llm: <class 'my_custom_model.config.MyLLMConfig'> [third-party]

# usage (unchanged)
model = AutoModelForCausalLM.from_pretrained("user/my-llm")
```

---

*End of Proposal*