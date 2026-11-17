# ONNX Studio worker - loads scikit-learn models saved with joblib/pickle, runs
# inference on them and converts them to ONNX. Invoked as:
#   python onnxstudio_worker.py <command> <request.json> <response.json>
#   python onnxstudio_worker.py --daemon
# Commands: inspect, predict, convert.
# The response file always contains {"ok": true, "result": ...} or
# {"ok": false, "error": {"code", "message", "trace"}}.

import ast
import glob
import inspect
import json
import math
import os
import sys
import traceback
import warnings

WORKER_VERSION = 3
MAX_OUTPUT_ELEMENTS = 200000
MAX_PREVIEW = 64                 # values previewed per fitted attribute
MAX_COMPONENTS = 200
MAX_COMPONENT_DEPTH = 8
MAX_COMPONENT_CHILDREN = 50
MAX_FITTED_ATTRIBUTES = 60
MAX_EMBEDDED_METADATA = 1000000  # characters stored in the ONNX file
METADATA_KEY = "onnxstudio.sklearn.info"
CHILD_ATTRIBUTES = ("best_estimator_", "estimator_", "base_estimator_", "final_estimator_", "regressor_", "classifier_")
INFERENCE_METHODS = ("predict", "predict_proba", "predict_log_proba", "decision_function", "transform")
NUMERIC_TRANSFORMERS = (
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    "Normalizer",
    "QuantileTransformer",
    "PowerTransformer",
    "PolynomialFeatures",
    "KBinsDiscretizer",
    "PCA",
)
TEXT_VECTORIZERS = ("CountVectorizer", "TfidfVectorizer", "HashingVectorizer")
CATEGORY_ENCODERS = ("OneHotEncoder", "OrdinalEncoder", "TargetEncoder")
_MISSING_TEXT_MARKERS = {"", "nan", "na", "null", "none"}

_MODEL_CACHE_MAX = 8
_MODEL_CACHE = {}
_SIGNATURE_CACHE = {}
_INSTALLED_VERSIONS = None
_SKL2ONNX_PATCHED = False
_PROTECTED_MAIN_NAMES = None


class WorkerError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------- JSON helpers

def to_py(value):
    """Converts numpy / pandas / scipy values to JSON compatible python values."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if np is not None:
        if isinstance(value, np.generic):
            return to_py(value.item())
        if isinstance(value, np.ndarray):
            return to_py(value.tolist())
        if isinstance(value, np.dtype):
            return str(value)
    if isinstance(value, (list, tuple, set)):
        return [to_py(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_py(v) for k, v in value.items()}
    return str(value)


def write_response(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)


# --------------------------------------------------------------- model loading

def _cache_put_model(abs_path, mtime_ns, size, model, load_warnings):
    if abs_path in _MODEL_CACHE:
        del _MODEL_CACHE[abs_path]
    elif len(_MODEL_CACHE) >= _MODEL_CACHE_MAX:
        del _MODEL_CACHE[next(iter(_MODEL_CACHE))]
    _MODEL_CACHE[abs_path] = (mtime_ns, size, model, list(load_warnings))


def _preload_sibling_functions(abs_path):
    """Load top-level undecorated helper functions from sibling/parent .py files into __main__ for custom FunctionTransformer pickles."""
    global _PROTECTED_MAIN_NAMES
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    if _PROTECTED_MAIN_NAMES is None:
        _PROTECTED_MAIN_NAMES = frozenset(dir(main_mod))

    parent_dir = os.path.dirname(abs_path)
    grandparent_dir = os.path.dirname(parent_dir) if parent_dir else ""
    search_dirs = []
    for directory in (parent_dir, grandparent_dir):
        if directory and os.path.isdir(directory) and directory not in search_dirs:
            search_dirs.append(directory)
    if not search_dirs:
        return

    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        pd = None

    files_checked = 0
    for directory in search_dirs:
        for py_path in sorted(glob.glob(os.path.join(directory, "*.py"))):
            if files_checked >= 10:
                return
            try:
                if os.path.getsize(py_path) >= 256 * 1024:
                    continue
                files_checked += 1
                with open(py_path, "r", encoding="utf-8", errors="ignore") as handle:
                    source = handle.read()
                tree = ast.parse(source, filename=py_path)
            except Exception:  # noqa: BLE001
                continue

            ns = {"np": np, "pd": pd, "numpy": np, "pandas": pd, "math": math}
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or node.decorator_list:
                    continue
                if node.name in _PROTECTED_MAIN_NAMES:
                    continue
                try:
                    node.returns = None
                    for arg in list(node.args.args) + list(node.args.posonlyargs) + list(node.args.kwonlyargs):
                        arg.annotation = None
                    if node.args.vararg is not None:
                        node.args.vararg.annotation = None
                    if node.args.kwarg is not None:
                        node.args.kwarg.annotation = None
                    mod = ast.Module(body=[node], type_ignores=[])
                    ast.fix_missing_locations(mod)
                    exec(compile(mod, py_path, "exec"), ns)
                    if node.name in ns:
                        setattr(main_mod, node.name, ns[node.name])
                except Exception:  # noqa: BLE001
                    pass


def load_model(path, collected_warnings):
    if not os.path.isfile(path):
        raise WorkerError("file_not_found", "The model file does not exist: " + path)

    abs_path = os.path.abspath(path)
    st = os.stat(abs_path)
    cached = _MODEL_CACHE.get(abs_path)
    if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        del _MODEL_CACHE[abs_path]
        _MODEL_CACHE[abs_path] = cached
        collected_warnings.extend(cached[3])
        return cached[2]

    _preload_sibling_functions(abs_path)

    try:
        import joblib
    except ImportError:
        joblib = None
    import pickle

    extension = os.path.splitext(path)[1].lower()
    loaders = []
    if joblib is not None:
        loaders.append(("joblib", joblib.load))
    loaders.append(("pickle", lambda p: _pickle_load(pickle, p)))
    if extension not in (".joblib", ".jbl", ".z", ".gz", ".bz2", ".xz", ".lzma") and len(loaders) > 1:
        loaders.reverse()

    last_error = None
    for name, loader in loaders:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = loader(path)
                load_warnings = ["{}: {}".format(item.category.__name__, item.message) for item in caught]
            model = _harmonize_loaded_model(model, load_warnings)
            _cache_put_model(abs_path, st.st_mtime_ns, st.st_size, model, load_warnings)
            collected_warnings.extend(load_warnings)
            return model
        except ModuleNotFoundError as error:
            # The model needs a package that is not installed in this interpreter.
            raise WorkerError("missing_module", "The model needs the Python package '{}', which is not installed in the selected runtime.".format(error.name))
        except Exception as error:  # noqa: BLE001
            last_error = error
    if last_error is not None and _is_cython_binary_mismatch(last_error):
        raise WorkerError(
            "load_failed",
            "Incompatibilite binaire Cython entre la version scikit-learn d'entrainement et ce runtime ({}: {}).".format(
                type(last_error).__name__, last_error
            ),
        )
    raise WorkerError("load_failed", "The file could not be loaded as a joblib or pickle model: {}: {}".format(type(last_error).__name__, last_error))


def _pickle_load(pickle, path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _iter_all_estimators(model, seen=None):
    """Yield every nested estimator/transformer inside pipelines, ensembles, and meta-estimators."""
    if seen is None:
        seen = set()
    if model is None or isinstance(model, (str, bytes, int, float, bool)):
        return
    obj_id = id(model)
    if obj_id in seen:
        return
    seen.add(obj_id)
    yield model

    for _, step in steps_of(model):
        yield from _iter_all_estimators(step, seen)

    for attr_name in ("transformers_", "transformers", "transformer_list"):
        items = getattr(model, attr_name, None)
        if isinstance(items, (list, tuple)):
            for item in items:
                if isinstance(item, (list, tuple)) and len(item) > 1:
                    yield from _iter_all_estimators(item[1], seen)

    for attr_name in ("estimators_", "estimators"):
        items = getattr(model, attr_name, None)
        if isinstance(items, (list, tuple)):
            for item in items:
                est = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item
                yield from _iter_all_estimators(est, seen)

    for attr_name in CHILD_ATTRIBUTES:
        child = getattr(model, attr_name, None)
        if child is not None:
            yield from _iter_all_estimators(child, seen)

    calibrated = getattr(model, "calibrated_classifiers_", None)
    if isinstance(calibrated, (list, tuple)):
        for cal in calibrated:
            for sub_attr in ("estimator", "base_estimator"):
                sub = getattr(cal, sub_attr, None)
                if sub is not None:
                    yield from _iter_all_estimators(sub, seen)


def _resolve_column_selector(cols, feature_names, n_features):
    """Normalize ColumnTransformer selectors to plain Python str, int, list[str], or list[int]."""
    if cols is None or callable(cols):
        return cols
    if hasattr(cols, "ndim") and getattr(cols, "ndim", None) == 0:
        cols = cols.item()
    if isinstance(cols, str):
        return str(cols)

    try:
        import numpy as np
        int_types = (int, np.integer)
        bool_types = (bool, np.bool_)
        str_types = (str, np.str_)
    except ImportError:  # pragma: no cover
        int_types = (int,)
        bool_types = (bool,)
        str_types = (str,)

    if isinstance(cols, bool_types):
        return cols
    if isinstance(cols, int_types):
        idx = int(cols)
        if idx < 0 and n_features is not None:
            idx += n_features
        if feature_names is not None and 0 <= idx < len(feature_names):
            return str(feature_names[idx])
        return idx
    if isinstance(cols, slice):
        total = n_features if n_features is not None else (len(feature_names) if feature_names is not None else None)
        if total is not None:
            indices = list(range(*cols.indices(total)))
        elif cols.stop is not None:
            start = 0 if cols.start is None else int(cols.start)
            step = 1 if cols.step is None else int(cols.step)
            indices = list(range(start, int(cols.stop), step))
        else:
            return cols
        if feature_names is not None:
            return [str(feature_names[i]) for i in indices if 0 <= i < len(feature_names)]
        return indices

    if hasattr(cols, "__iter__"):
        seq = list(cols)
        if not seq:
            return []
        if all(isinstance(x, bool_types) for x in seq):
            indices = [i for i, flag in enumerate(seq) if bool(flag)]
            if feature_names is not None:
                return [str(feature_names[i]) for i in indices if 0 <= i < len(feature_names)]
            return indices
        if all(isinstance(x, int_types) and not isinstance(x, bool_types) for x in seq):
            indices = []
            for x in seq:
                idx = int(x)
                if idx < 0 and n_features is not None:
                    idx += n_features
                indices.append(idx)
            if feature_names is not None:
                return [str(feature_names[i]) for i in indices if 0 <= i < len(feature_names)]
            return indices
        if all(isinstance(x, str_types) for x in seq):
            return [str(x) for x in seq]
        resolved = []
        for x in seq:
            if isinstance(x, str_types):
                resolved.append(str(x))
            elif isinstance(x, int_types) and not isinstance(x, bool_types):
                idx = int(x)
                if idx < 0 and n_features is not None:
                    idx += n_features
                if feature_names is not None and 0 <= idx < len(feature_names):
                    resolved.append(str(feature_names[idx]))
                else:
                    resolved.append(idx)
            else:
                resolved.append(x)
        return resolved

    return cols


def _get_signature_defaults(cls):
    cached = _SIGNATURE_CACHE.get(cls)
    if cached is not None:
        return cached
    params = []
    try:
        sig = inspect.signature(cls.__init__)
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default is inspect.Parameter.empty:
                continue
            params.append((param_name, param.default))
    except (ValueError, TypeError):
        pass
    _SIGNATURE_CACHE[cls] = params
    return params


def _same_selector(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(type(x) is type(y) and x == y for x, y in zip(a, b))
    return a == b


def _harmonize_loaded_model(model, collected_warnings):
    """Best-effort compatibility patching for cross-version scikit-learn models."""
    patched = []

    for est in _iter_all_estimators(model):
        cls = type(est)
        cls_name = cls.__name__
        if not hasattr(est, "__dict__"):
            continue

        for param_name, default_value in _get_signature_defaults(cls):
            if not hasattr(est, param_name):
                try:
                    setattr(est, param_name, default_value)
                    patched.append("{}.{}".format(cls_name, param_name))
                except Exception:  # noqa: BLE001
                    pass

        if cls_name == "LogisticRegression" and not hasattr(est, "multi_class"):
            try:
                est.multi_class = "auto"
                patched.append("LogisticRegression.multi_class='auto'")
            except Exception:  # noqa: BLE001
                pass

        if cls_name == "OneHotEncoder":
            if hasattr(est, "sparse_output") and not hasattr(est, "sparse"):
                try:
                    est.sparse = est.sparse_output
                    patched.append("OneHotEncoder.sparse")
                except Exception:  # noqa: BLE001
                    pass
            elif hasattr(est, "sparse") and not hasattr(est, "sparse_output"):
                try:
                    est.sparse_output = est.sparse
                    patched.append("OneHotEncoder.sparse_output")
                except Exception:  # noqa: BLE001
                    pass

        if cls_name == "SimpleImputer" and not hasattr(est, "_fill_dtype"):
            stats = getattr(est, "statistics_", None)
            if stats is not None and hasattr(stats, "dtype"):
                try:
                    est._fill_dtype = stats.dtype
                    patched.append("SimpleImputer._fill_dtype")
                except Exception:  # noqa: BLE001
                    pass

        if any(
            k in cls_name
            for k in ("Tree", "Forest", "GradientBoosting", "ExtraTrees", "HistGradientBoosting")
        ) and not hasattr(est, "monotonic_cst"):
            try:
                est.monotonic_cst = None
                patched.append("{}.monotonic_cst=None".format(cls_name))
            except Exception:  # noqa: BLE001
                pass

        if cls_name == "ColumnTransformer" and isinstance(getattr(est, "transformers_", None), list):
            ct_names_raw = getattr(est, "feature_names_in_", None)
            if ct_names_raw is None and first_estimator(model) is est:
                ct_names_raw = getattr(model, "feature_names_in_", None)
            ct_names = [str(n) for n in ct_names_raw] if ct_names_raw is not None else None
            ct_n_raw = getattr(est, "n_features_in_", None)
            ct_n = int(ct_n_raw) if ct_n_raw is not None else (len(ct_names) if ct_names is not None else None)
            normalized = []
            changed = False
            for entry in est.transformers_:
                if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    t_name, t_trans, t_cols = entry[0], entry[1], entry[2]
                    if t_trans != "drop" and t_cols is not None:
                        resolved = _resolve_column_selector(t_cols, ct_names, ct_n)
                        if not _same_selector(resolved, t_cols):
                            changed = True
                        normalized.append((t_name, t_trans, resolved, *entry[3:]))
                        continue
                normalized.append(entry)
            if changed:
                est.transformers_ = normalized

    if patched:
        unique_patched = list(dict.fromkeys(patched))
        preview = ", ".join(unique_patched[:8])
        suffix = " (+{} autres)".format(len(unique_patched) - 8) if len(unique_patched) > 8 else ""
        collected_warnings.append("Compatibilite auto : attribut(s) complete(s) : {}{}".format(preview, suffix))

    return model


def _is_cython_binary_mismatch(error):
    msg = str(error).lower()
    return (
        "incompatible dtype" in msg
        or "node array from the pickle" in msg
        or "buffer dtype mismatch" in msg
    )


# --------------------------------------------------------------- introspection

def steps_of(model):
    steps = getattr(model, "steps", None)
    if isinstance(steps, list):
        return [(name, estimator) for name, estimator in steps if estimator not in (None, "passthrough")]
    return []


def first_estimator(model):
    steps = steps_of(model)
    return steps[0][1] if steps else model


def last_estimator(model):
    steps = steps_of(model)
    return last_estimator(steps[-1][1]) if steps else model


def iter_estimators(model):
    yield model
    for _, step in steps_of(model):
        yield from iter_estimators(step)
    transformers = getattr(model, "transformers", None)
    if isinstance(transformers, list):
        for item in transformers:
            if len(item) > 1 and hasattr(item[1], "fit"):
                yield from iter_estimators(item[1])


def _ordered_input_candidates(model):
    ordered = []
    seen = set()

    def add(est):
        if est is not None and id(est) not in seen:
            seen.add(id(est))
            ordered.append(est)

    add(model)
    add(first_estimator(model))
    all_ests = list(_iter_all_estimators(model))
    for est in all_ests:
        if type(est).__name__ == "ColumnTransformer":
            add(est)
    for est in all_ests:
        add(est)
    return ordered


def get_feature_names_in(model):
    for est in _ordered_input_candidates(model):
        names = getattr(est, "feature_names_in_", None)
        if names is not None:
            return [str(n) for n in names]
    return None


def get_n_features_in(model, feature_names=None):
    for est in _ordered_input_candidates(model):
        n = getattr(est, "n_features_in_", None)
        if n is not None:
            return int(n)
    if feature_names is not None:
        return len(feature_names)
    return None


def has_column_transformer(model):
    return any(type(e).__name__ == "ColumnTransformer" for e in _iter_all_estimators(model))


def is_text_model(model):
    try:
        from sklearn.feature_extraction import text as sk_text
        text_types = (sk_text.CountVectorizer, sk_text.TfidfVectorizer, sk_text.HashingVectorizer)
        return isinstance(first_estimator(model), text_types)
    except Exception:  # noqa: BLE001
        return False


def sklearn_version_of(model):
    try:
        return model.__getstate__().get("_sklearn_version")
    except Exception:  # noqa: BLE001
        return None


def installed_versions():
    global _INSTALLED_VERSIONS
    if _INSTALLED_VERSIONS is not None:
        return dict(_INSTALLED_VERSIONS)
    versions = {}
    try:
        from importlib import metadata
        for name in ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "skl2onnx", "onnx", "onnxruntime"):
            try:
                versions[name] = metadata.version(name)
            except Exception:  # noqa: BLE001
                versions[name] = None
    except Exception:  # noqa: BLE001
        pass
    _INSTALLED_VERSIONS = versions
    return dict(_INSTALLED_VERSIONS)


def short_repr(value, limit=120):
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def is_classifier(estimator, classes):
    try:
        from sklearn.base import is_classifier as sk_is_classifier
        if sk_is_classifier(estimator):
            return True
    except Exception:  # noqa: BLE001
        pass
    return classes is not None and hasattr(estimator, "predict_proba")


def is_estimator(value):
    return hasattr(value, "get_params") and hasattr(value, "fit")


def describe_param(value):
    """Readable, bounded text of a hyper-parameter (nested estimators are shown by name)."""
    if is_estimator(value):
        return type(value).__name__
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, tuple) and v and isinstance(v[0], str) for v in value):
        return "[" + ", ".join(str(v[0]) for v in value[:20]) + ("..." if len(value) > 20 else "") + "]"
    return short_repr(value, 200)


def parameters_of(estimator):
    try:
        return {str(k): describe_param(v) for k, v in list(estimator.get_params(deep=False).items())[:100]}
    except Exception:  # noqa: BLE001
        return {}


def estimator_kind(estimator):
    name = type(estimator).__name__
    if getattr(estimator, "steps", None) is not None:
        return "Pipeline"
    if name in ("ColumnTransformer", "FeatureUnion"):
        return name
    try:
        from sklearn.base import is_classifier as sk_is_classifier, is_regressor as sk_is_regressor
        if sk_is_classifier(estimator):
            return "Classifier"
        if sk_is_regressor(estimator):
            return "Regressor"
    except Exception:  # noqa: BLE001
        pass
    if hasattr(estimator, "transform"):
        return "Transformer"
    return "Estimator"


def child_estimators(model):
    """Nested estimators as (name, estimator, note): pipeline steps, column transformers, ensembles, search wrappers."""
    children = []
    steps = getattr(model, "steps", None)
    transformers = getattr(model, "transformers_", None) or getattr(model, "transformers", None)
    union = getattr(model, "transformer_list", None)
    estimators = getattr(model, "estimators", None)
    if isinstance(steps, list):
        children = [(item[0], item[1], "") for item in steps if len(item) > 1]
    elif isinstance(transformers, list):
        children = [(item[0], item[1], "columns: " + short_repr(item[2], 2000)) for item in transformers if len(item) > 2]
    elif isinstance(union, list):
        children = [(item[0], item[1], "") for item in union if len(item) > 1]
    elif isinstance(estimators, list) and estimators and all(isinstance(item, tuple) and len(item) == 2 for item in estimators):
        children = [(item[0], item[1], "") for item in estimators]
    known = {id(c[1]) for c in children}
    for attribute in CHILD_ATTRIBUTES:
        try:
            value = getattr(model, attribute, None)
        except Exception:  # noqa: BLE001
            value = None
        if value is not None and is_estimator(value) and id(value) not in known:
            children.append((attribute, value, ""))
            known.add(id(value))
    return children


def scalar_text(value):
    if isinstance(value, float):
        value = float(value)
        return repr(value) if math.isfinite(value) else str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(int(value))
    return short_repr(value, 200) if not isinstance(value, str) else (value if len(value) <= 200 else value[:197] + "...")


def summarize_attribute(name, value, preview):
    """One learned (fitted) attribute: scalars as text, arrays/lists/dicts as a bounded preview of their values."""
    import numpy as np

    item = {"name": name, "kind": "scalar", "dtype": None, "shape": None, "count": 1, "values": [], "summary": ""}
    try:
        if hasattr(value, "to_numpy") and not hasattr(value, "tocoo"):
            value = value.to_numpy()
        if value is None or isinstance(value, (bool, int, float, str)):
            item["summary"] = scalar_text(value)
        elif isinstance(value, np.generic):
            item["summary"] = scalar_text(value.item())
            item["dtype"] = str(value.dtype)
        elif hasattr(value, "tocoo"):
            item.update(kind="sparse", shape=[int(d) for d in value.shape], count=int(value.nnz), dtype=str(value.dtype))
            item["summary"] = "sparse {} matrix {}, {} stored values".format(value.dtype, tuple(value.shape), value.nnz)
        elif isinstance(value, np.ndarray) or (isinstance(value, (list, tuple)) and len(value) <= 100000 and all(
                isinstance(v, (bool, int, float, str, np.generic)) or v is None for v in value[:200])):
            array = value if isinstance(value, np.ndarray) else np.asarray(value)
            item.update(kind="array", dtype=str(array.dtype), shape=[int(d) for d in array.shape], count=int(array.size))
            item["summary"] = "{} {}".format(array.dtype, tuple(array.shape))
            head = array.flat[:preview]
            item["values"] = [(v if isinstance(v, str) else short_repr(v, 80)) if array.dtype == object else v for v in to_py(head)] if preview > 0 else []
        elif isinstance(value, dict):
            item.update(kind="dict", count=len(value))
            item["summary"] = "{} entries".format(len(value))
            item["values"] = ["{}: {}".format(k, short_repr(v, 80)) for k, v in list(value.items())[:preview]]
        elif isinstance(value, (list, tuple)):
            kinds = {}
            for entry in value:
                kinds[type(entry).__name__] = kinds.get(type(entry).__name__, 0) + 1
            item.update(kind="list", count=len(value))
            item["summary"] = "{} x ".format(len(value)) + ", ".join("{} {}".format(n, k) for k, n in list(kinds.items())[:5])
        elif is_estimator(value):
            item.update(kind="object")
            item["summary"] = type(value).__name__
        else:
            item.update(kind="object")
            item["summary"] = "{}: {}".format(type(value).__name__, short_repr(value, 160))
    except Exception as error:  # noqa: BLE001
        item.update(kind="object", values=[])
        item["summary"] = "unavailable ({}: {})".format(type(error).__name__, error)
    return item


def fitted_attributes(estimator, skip, preview):
    try:
        attributes = vars(estimator)
    except TypeError:
        attributes = {}
    entries = []
    for key, value in attributes.items():
        if not key.endswith("_") or key.startswith("_") or id(value) in skip:
            continue
        if key == "tree_":
            for stat in ("node_count", "max_depth", "n_leaves", "n_features", "n_outputs"):
                if hasattr(value, stat):
                    entries.append({"name": "tree_." + stat, "kind": "scalar", "dtype": None, "shape": None, "count": 1,
                                    "values": [], "summary": str(getattr(value, stat))})
            continue
        entries.append(summarize_attribute(key, value, preview))
    if "feature_importances_" not in attributes and hasattr(type(estimator), "feature_importances_"):
        try:
            entries.append(summarize_attribute("feature_importances_", estimator.feature_importances_, preview))
        except Exception:  # noqa: BLE001
            pass
    return entries[:MAX_FITTED_ATTRIBUTES]


def build_component(name, estimator, note, preview, state, depth=0):
    state["count"] += 1
    node = {"name": name, "className": type(estimator).__name__, "module": type(estimator).__module__,
            "kind": "Passthrough" if isinstance(estimator, str) else estimator_kind(estimator),
            "description": note, "parameters": {}, "fitted": [], "children": []}
    if not is_estimator(estimator):
        node["className"] = str(estimator) if isinstance(estimator, str) else type(estimator).__name__
        return node
    node["parameters"] = parameters_of(estimator)
    children = child_estimators(estimator)
    node["fitted"] = fitted_attributes(estimator, {id(c[1]) for c in children}, preview)
    if depth < MAX_COMPONENT_DEPTH:
        for child_name, child, child_note in children[:MAX_COMPONENT_CHILDREN]:
            if state["count"] >= MAX_COMPONENTS:
                break
            node["children"].append(build_component(str(child_name), child, child_note, preview, state, depth + 1))
    return node


def _is_missing_scalar(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        import numpy as np
        if isinstance(value, np.floating) and np.isnan(value):
            return True
    except Exception:  # noqa: BLE001
        pass
    return type(value).__name__ == "NAType"


def _is_string_array(arr):
    if arr is None:
        return False
    dtype = getattr(arr, "dtype", None)
    if dtype is None:
        return False
    if dtype.kind in ("U", "S", "T"):
        return True
    if "str" in str(dtype).lower():
        return True
    if dtype == object:
        return any(isinstance(v, str) for v in arr.flat)
    return False


def _column_indices_for_selector(cols, feature_names, col_names, n_features):
    if cols is None or callable(cols):
        return []
    name_to_idx = {name: i for i, name in enumerate(col_names)}
    resolved = _resolve_column_selector(cols, feature_names, n_features)
    if isinstance(resolved, str):
        return [name_to_idx[resolved]] if resolved in name_to_idx else []
    if isinstance(resolved, int) and not isinstance(resolved, bool):
        return [resolved] if 0 <= resolved < n_features else []
    if hasattr(resolved, "__iter__"):
        indices = []
        for item in resolved:
            if isinstance(item, str):
                if item in name_to_idx:
                    indices.append(name_to_idx[item])
            elif isinstance(item, int) and not isinstance(item, bool):
                if 0 <= item < n_features:
                    indices.append(item)
        return indices
    return []


def _apply_transformer_to_schema(est, col_indices, schema):
    import numpy as np

    if est is None or isinstance(est, (str, bytes)) or not col_indices:
        return
    steps = steps_of(est)
    if steps:
        for _, step in steps:
            _apply_transformer_to_schema(step, col_indices, schema)
        return
    transformer_list = getattr(est, "transformer_list", None)
    if isinstance(transformer_list, list):
        for item in transformer_list:
            if isinstance(item, (list, tuple)) and len(item) > 1:
                _apply_transformer_to_schema(item[1], col_indices, schema)
        return

    cls_name = type(est).__name__
    if cls_name in TEXT_VECTORIZERS:
        for idx in col_indices:
            schema[idx]["type"] = "string"
        return

    if cls_name == "SimpleImputer":
        stats = getattr(est, "statistics_", None)
        if stats is not None:
            stats_arr = np.asarray(stats)
            if _is_string_array(stats_arr):
                for k, idx in enumerate(col_indices):
                    schema[idx]["type"] = "string"
                    schema[idx]["impute_nan"] = True
                    if k < stats_arr.size:
                        val = stats_arr.flat[k]
                        if isinstance(val, str) and val != "" and val not in schema[idx]["categories"]:
                            schema[idx]["categories"].append(val)
            elif stats_arr.dtype.kind == "f":
                for idx in col_indices:
                    schema[idx]["type"] = "float"
            elif stats_arr.dtype.kind in ("i", "u"):
                for idx in col_indices:
                    if schema[idx]["type"] is None:
                        schema[idx]["type"] = "int64"
        return

    if cls_name in CATEGORY_ENCODERS:
        cats_list = getattr(est, "categories_", None)
        if isinstance(cats_list, (list, tuple)):
            for k, idx in enumerate(col_indices):
                if k >= len(cats_list):
                    break
                cats_arr = np.asarray(cats_list[k])
                if _is_string_array(cats_arr):
                    schema[idx]["type"] = "string"
                    clean = [str(x) for x in cats_arr.flat if not _is_missing_scalar(x)]
                    if clean:
                        schema[idx]["categories"] = clean
                    continue
                non_missing = [x for x in cats_arr.flat if not _is_missing_scalar(x)]
                is_int_cats = cats_arr.dtype.kind in ("i", "u", "b") or (
                    cats_arr.dtype == object
                    and bool(non_missing)
                    and all(isinstance(x, (int, bool, np.integer, np.bool_)) for x in non_missing)
                )
                if is_int_cats:
                    if schema[idx]["type"] != "float":
                        schema[idx]["type"] = "int64"
                    clean = [int(x) for x in non_missing]
                    if clean:
                        schema[idx]["categories"] = clean
                elif cats_arr.dtype.kind == "f":
                    if schema[idx]["type"] is None:
                        schema[idx]["type"] = "float"
        return

    if cls_name in NUMERIC_TRANSFORMERS:
        for idx in col_indices:
            if schema[idx]["type"] is None:
                schema[idx]["type"] = "float"


def infer_column_schema(model):
    """Infer per-feature name, type ('float'|'int64'|'string'), categories, and nan imputation flag."""
    feature_names = get_feature_names_in(model)
    n_features = get_n_features_in(model, feature_names)
    if n_features is None or n_features <= 0:
        return []

    col_names = (
        feature_names
        if (feature_names is not None and len(feature_names) == n_features)
        else ["f{}".format(i) for i in range(n_features)]
    )
    schema = [
        {"name": col_names[i], "type": None, "categories": [], "impute_nan": False}
        for i in range(n_features)
    ]

    cts = []
    if type(model).__name__ == "ColumnTransformer":
        cts = [model]
    elif type(first_estimator(model)).__name__ == "ColumnTransformer":
        cts = [first_estimator(model)]
    else:
        cts = [est for est in _iter_all_estimators(model) if type(est).__name__ == "ColumnTransformer"]

    if cts:
        for ct in cts:
            transformers = getattr(ct, "transformers_", None) or getattr(ct, "transformers", None) or []
            for entry in transformers:
                if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                    continue
                _, trans, cols = entry[0], entry[1], entry[2]
                if trans == "drop" or cols is None:
                    continue
                indices = _column_indices_for_selector(cols, feature_names, col_names, n_features)
                _apply_transformer_to_schema(trans, indices, schema)
    else:
        target = model
        for attr in ("best_estimator_", "regressor_", "classifier_", "estimator_"):
            sub = getattr(target, attr, None)
            if sub is not None and is_estimator(sub):
                target = sub
                break
        if target is model and getattr(model, "calibrated_classifiers_", None):
            cal0 = model.calibrated_classifiers_[0]
            sub = getattr(cal0, "estimator", None) or getattr(cal0, "base_estimator", None)
            if sub is not None:
                target = sub
        _apply_transformer_to_schema(target, list(range(n_features)), schema)

    for entry in schema:
        if entry["type"] is None:
            entry["type"] = "float"

    return schema


def inspect_model(model, collected_warnings, preview=MAX_PREVIEW):
    import sklearn

    feature_names = get_feature_names_in(model)
    n_features = get_n_features_in(model, feature_names)
    col_schema = infer_column_schema(model)
    classes = getattr(last_estimator(model), "classes_", None)

    try:
        params = {k: short_repr(v) for k, v in list(model.get_params(deep=False).items())[:40]}
    except Exception:  # noqa: BLE001
        params = {}

    steps = [{"name": name, "className": type(est).__name__} for name, est in steps_of(model)]

    return {
        "components": build_component(type(model).__name__, model, "", preview, {"count": 0}),
        "className": type(model).__name__,
        "module": type(model).__module__,
        "isPipeline": bool(steps),
        "steps": steps,
        "finalEstimator": type(last_estimator(model)).__name__,
        "isClassifier": is_classifier(last_estimator(model), classes),
        "isTextModel": is_text_model(model),
        "nFeaturesIn": int(n_features) if n_features is not None else None,
        "featureNamesIn": feature_names,
        "featureTypesIn": [c["type"] for c in col_schema] if col_schema else None,
        "classes": to_py(classes)[:200] if classes is not None else None,
        "methods": [m for m in INFERENCE_METHODS if callable(getattr(model, m, None))],
        "parameters": params,
        "trainedWithSklearn": sklearn_version_of(model),
        "runtimeSklearn": sklearn.__version__,
        "warnings": collected_warnings,
        "packages": installed_versions(),
    }


# ------------------------------------------------------------------- inference

def parse_column(values):
    """Strings -> floats when every value is numeric (empty = NaN), otherwise kept as text."""
    parsed = []
    for raw in values:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            parsed.append(float(raw))
            continue
        text = "" if raw is None else str(raw).strip()
        if text == "" or text.lower() in ("nan", "na", "null", "none"):
            parsed.append(float("nan"))
            continue
        try:
            parsed.append(float(text))
        except ValueError:
            return [("" if v is None else str(v)) for v in values]
    return parsed


def _parse_string_column(values, impute_nan):
    parsed = []
    for raw in values:
        if raw is None:
            parsed.append(float("nan") if impute_nan else "")
            continue
        if isinstance(raw, float) and math.isnan(raw):
            parsed.append(float("nan") if impute_nan else "")
            continue
        if isinstance(raw, int) and not isinstance(raw, bool):
            parsed.append(str(raw))
            continue
        text = str(raw)
        if text.strip().lower() in _MISSING_TEXT_MARKERS:
            parsed.append(float("nan") if impute_nan else "")
        else:
            parsed.append(text)
    return parsed


def build_input(model, columns, rows):
    import numpy as np

    if not rows:
        raise WorkerError("invalid_input", "No input rows were provided.")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise WorkerError("invalid_input", "All input rows must have the same number of values.")

    if is_text_model(model):
        if width != 1:
            raise WorkerError("invalid_input", "This model takes raw text: provide exactly one value per row.")
        return [("" if row[0] is None else str(row[0])) for row in rows]

    col_schema = infer_column_schema(model)
    names = get_feature_names_in(model)
    if names is not None:
        if columns and set(names).issubset(set(columns)):
            order = [columns.index(n) for n in names]
        elif len(names) == width:
            order = list(range(width))
        else:
            raise WorkerError("invalid_input", "The model expects {} features ({}), but {} were provided.".format(len(names), ", ".join(names), width))
        try:
            import pandas as pd
        except ImportError:
            raise WorkerError("missing_module", "The model was trained on named columns and needs the Python package 'pandas'.")
        schema_by_name = {c["name"]: c for c in col_schema}
        df_cols = {}
        for k, col_name in enumerate(names):
            raw_col = [row[order[k]] for row in rows]
            col_info = schema_by_name.get(col_name)
            if col_info is not None and col_info.get("type") == "string":
                col_vals = _parse_string_column(raw_col, bool(col_info.get("impute_nan")))
                df_cols[col_name] = pd.Series(col_vals, dtype=object)
            else:
                df_cols[col_name] = parse_column(raw_col)
        return pd.DataFrame(df_cols, columns=names)

    cells = []
    for i in range(width):
        raw_col = [row[i] for row in rows]
        if col_schema and i < len(col_schema) and col_schema[i].get("type") == "string":
            cells.append(_parse_string_column(raw_col, bool(col_schema[i].get("impute_nan"))))
        else:
            cells.append(parse_column(raw_col))

    if all(all(isinstance(v, float) for v in column) for column in cells):
        return np.array(cells, dtype=float).T
    return np.array(cells, dtype=object).T


def to_dense(value):
    try:
        from scipy import sparse
        if sparse.issparse(value):
            if value.shape[0] * value.shape[1] > MAX_OUTPUT_ELEMENTS:
                raise WorkerError("prediction_failed", "The sparse result is too large to display ({} x {}).".format(*value.shape))
            return value.toarray()
    except ImportError:
        pass
    return value


def output_of(name, value):
    import numpy as np

    value = to_dense(value)
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    if isinstance(value, (list, tuple)):
        # e.g. multi-output predict_proba returns one array per output
        parts = [np.asarray(to_dense(v)) for v in value]
        try:
            array = np.stack(parts, axis=1)
        except ValueError:
            array = np.asarray(parts, dtype=object)
    else:
        array = np.asarray(value)

    truncated = False
    if array.size > MAX_OUTPUT_ELEMENTS and array.ndim > 0:
        per_row = max(1, array.size // max(1, array.shape[0]))
        array = array[: max(1, MAX_OUTPUT_ELEMENTS // per_row)]
        truncated = True
    return {
        "name": name,
        "dtype": str(array.dtype),
        "shape": [int(d) for d in array.shape],
        "values": to_py(array),
        "truncated": truncated,
    }


def predict(model, request, collected_warnings):
    requested = (request.get("method") or "auto").lower()
    available = [m for m in INFERENCE_METHODS if callable(getattr(model, m, None))]
    if not available:
        raise WorkerError("unsupported_model", "The loaded object ({}) exposes none of: {}.".format(type(model).__name__, ", ".join(INFERENCE_METHODS)))

    if requested == "auto":
        methods = ["predict" if "predict" in available else available[0]]
    elif requested == "all":
        methods = available
    elif requested in available:
        methods = [requested]
    else:
        raise WorkerError("invalid_input", "The model does not support '{}'. Available: {}.".format(requested, ", ".join(available)))

    data = build_input(model, request.get("columns"), request.get("rows") or [])
    outputs = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for method in methods:
            outputs.append(output_of(method, getattr(model, method)(data)))
        for item in caught:
            collected_warnings.append("{}: {}".format(item.category.__name__, item.message))

    classes = getattr(last_estimator(model), "classes_", None)
    return {
        "outputs": outputs,
        "classes": to_py(classes)[:200] if classes is not None else None,
        "rowCount": len(request.get("rows") or []),
        "warnings": collected_warnings,
    }


# ------------------------------------------------------------------ conversion

def parse_input_overrides(specs):
    from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, StringTensorType

    types = {"float": FloatTensorType, "float32": FloatTensorType, "double": FloatTensorType,
             "int": Int64TensorType, "int64": Int64TensorType, "string": StringTensorType, "str": StringTensorType}
    result = []
    for spec in specs:
        parts = [p.strip() for p in str(spec).split(":")]
        if len(parts) < 2 or parts[1].lower() not in types:
            raise WorkerError("invalid_input", "Invalid input type '{}'. Use name:type[:width] with type in {}.".format(spec, ", ".join(sorted(types))))
        width = 1
        if len(parts) > 2 and parts[2]:
            try:
                width = int(parts[2])
            except ValueError:
                raise WorkerError("invalid_input", "Invalid width in '{}'.".format(spec))
        result.append((parts[0], types[parts[1].lower()]([None, width])))
    return result


def infer_initial_types(model, overrides):
    from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, StringTensorType

    if overrides:
        return parse_input_overrides(overrides)
    if is_text_model(model):
        return [("text", StringTensorType([None, 1]))]
    col_schema = infer_column_schema(model)
    if col_schema and (has_column_transformer(model) or any(c["type"] != "float" for c in col_schema)):
        type_map = {
            "float": FloatTensorType,
            "int64": Int64TensorType,
            "string": StringTensorType,
        }
        return [
            (col["name"], type_map.get(col["type"], FloatTensorType)([None, 1]))
            for col in col_schema
        ]
    n_features = get_n_features_in(model, get_feature_names_in(model))
    if n_features is not None:
        return [("float_input", FloatTensorType([None, int(n_features)]))]
    raise WorkerError("invalid_input", "The number of input features cannot be inferred from this model. Specify the input types (name:type[:width]).")


def describe_value_info(infos):
    result = []
    for info in infos:
        tensor = info.type.tensor_type
        dims = []
        for d in tensor.shape.dim:
            dims.append(int(d.dim_value) if d.HasField("dim_value") else None)
        result.append({"name": info.name, "elementType": int(tensor.elem_type), "shape": dims})
    return result


def validate_conversion(model, onnx_path, initial_types, collected_warnings):
    if is_text_model(model):
        return {"performed": False, "passed": None, "detail": "Validation skipped (text input)."}

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return {"performed": False, "passed": None, "detail": "onnxruntime is not installed in the selected runtime: conversion was not validated."}

    try:
        rng = np.random.default_rng(0)
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        session_inputs = session.get_inputs()
        session_input_names = [inp.name for inp in session_inputs]
        session_input_set = set(session_input_names)
        samples = 8 if os.path.getsize(onnx_path) > 10 * 1024 * 1024 else 32
        feeds = {}
        frames = {}
        col_schema = infer_column_schema(model)
        schema_by_name = {c["name"]: c for c in col_schema}

        def _resolve_feed_name(raw_name, index):
            if raw_name in session_input_set:
                return raw_name
            if len(session_input_names) == len(initial_types):
                return session_input_names[index]
            sanitized = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in raw_name)
            if sanitized in session_input_set:
                return sanitized
            return raw_name

        for idx, (name, tensor_type) in enumerate(initial_types):
            width = tensor_type.shape[1] if len(tensor_type.shape) > 1 and tensor_type.shape[1] else 1
            kind = type(tensor_type).__name__
            schema_entry = schema_by_name.get(name)
            if schema_entry is None and len(initial_types) == len(col_schema):
                schema_entry = col_schema[idx]
            feed_name = _resolve_feed_name(name, idx)

            if kind == "StringTensorType":
                cats = (schema_entry or {}).get("categories") or []
                if not cats:
                    return {"performed": False, "passed": None, "detail": "Validation skipped (text input)."}
                data = rng.choice(cats, size=(samples, width)).astype(object)
                if feed_name in session_input_set:
                    feeds[feed_name] = data
                frames[name] = data
            elif kind == "Int64TensorType":
                cats = (schema_entry or {}).get("categories") or []
                int_cats = []
                for c in cats:
                    try:
                        int_cats.append(int(c))
                    except (TypeError, ValueError):
                        int_cats = []
                        break
                if int_cats:
                    data = rng.choice(int_cats, size=(samples, width)).astype(np.int64)
                else:
                    data = np.round(rng.standard_normal((samples, width)) * 3).astype(np.int64)
                if feed_name in session_input_set:
                    feeds[feed_name] = data
                frames[name] = data
            else:
                data = rng.standard_normal((samples, width))
                data32 = data.astype(np.float32)
                if feed_name in session_input_set:
                    feeds[feed_name] = data32
                frames[name] = data32.astype(np.float64)

        if len(frames) == 1:
            sample = next(iter(frames.values()))
            names = get_feature_names_in(model)
            if names is not None and sample.shape[1] == len(names):
                import pandas as pd
                df_cols = {}
                for i, col_name in enumerate(names):
                    col_arr = sample[:, i]
                    if col_arr.dtype == object or col_arr.dtype.kind in ("U", "S", "T"):
                        df_cols[col_name] = pd.Series(col_arr, dtype=object)
                    else:
                        df_cols[col_name] = col_arr
                sample = pd.DataFrame(df_cols, columns=names)
        else:
            import pandas as pd
            if all(v.shape[1] == 1 for v in frames.values()):
                names = get_feature_names_in(model)
                ordered_cols = names if (names is not None and set(names) == set(frames.keys())) else list(frames.keys())
                df_cols = {}
                for col_name in ordered_cols:
                    values = frames[col_name]
                    if values.dtype == object or values.dtype.kind in ("U", "S", "T"):
                        df_cols[col_name] = pd.Series(values[:, 0], dtype=object)
                    else:
                        df_cols[col_name] = values[:, 0]
                sample = pd.DataFrame(df_cols, columns=ordered_cols)
            else:
                stacked = np.hstack([frames[k] for k in frames])
                names = get_feature_names_in(model)
                if names is not None and stacked.shape[1] == len(names):
                    sample = pd.DataFrame(stacked, columns=names)
                else:
                    sample = stacked

        method = "predict" if callable(getattr(model, "predict", None)) else "transform"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = np.asarray(to_dense(getattr(model, method)(sample)))
        actual = np.asarray(session.run(None, feeds)[0])
        if expected.ndim == 1 and actual.ndim == 2 and actual.shape[1] == 1:
            actual = actual.ravel()

        if expected.shape != actual.shape:
            return {"performed": True, "passed": False, "detail": "Output shape differs: scikit-learn {} vs ONNX {}.".format(expected.shape, actual.shape)}
        if expected.dtype.kind in "fc":
            matches = np.isclose(expected, actual, rtol=1e-3, atol=1e-3)
        else:
            matches = expected.astype(str) == actual.astype(str)
        agreement = float(np.mean(matches))
        return {"performed": True, "passed": agreement >= 0.99, "agreement": agreement,
                "detail": "{:.1%} of the outputs on {} random samples match scikit-learn.".format(agreement, samples)}
    except Exception as error:  # noqa: BLE001
        return {"performed": True, "passed": None, "detail": "Validation could not complete: {}: {}".format(type(error).__name__, error)}


def embed_source_metadata(onnx_model, model):
    """Keeps what scikit-learn knew about the model in the ONNX file (metadata_props) so ONNX Studio can show it later."""
    try:
        payload = None
        for preview in (32, 0, None):
            info = inspect_model(model, [], preview=preview or 0)
            info.pop("warnings", None)
            if preview is None:
                info.pop("components", None)
            text = json.dumps(to_py(info), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            if len(text) <= MAX_EMBEDDED_METADATA:
                payload = text
                break
        if payload is None:
            return
        for key, value in (("onnxstudio.source", "scikit-learn"),
                           ("onnxstudio.sklearn.class", type(model).__name__),
                           ("onnxstudio.sklearn.version", sklearn_version_of(model) or ""),
                           (METADATA_KEY, payload)):
            entry = onnx_model.metadata_props.add()
            entry.key = key
            entry.value = value
        if not onnx_model.doc_string:
            onnx_model.doc_string = "scikit-learn {} converted by ONNX Studio".format(type(model).__name__)
    except Exception:  # noqa: BLE001 - the metadata is a bonus, never a reason to fail the conversion
        pass


_MAX_SUBTREE_NODES = 768
_MAX_OPERATOR_TREE_NODES = 20000


def _fast_add_tree_to_attribute_pairs(
    attr_pairs,
    is_classifier,
    tree,
    tree_id,
    tree_weight,
    weight_id_bias,
    leaf_weights_are_counts,
    adjust_threshold_for_sklearn=False,
    dtype=None,
):
    """Fast tree attribute builder that splits oversized trees into strict binary subtrees to prevent quadratic ORT init."""
    import numpy as np
    import skl2onnx.common.tree_ensemble as te

    n = tree.node_count
    cl_list = tree.children_left.tolist()
    cr_list = tree.children_right.tolist()
    feat_list = tree.feature.tolist()
    thresh = tree.threshold.astype(np.float64)
    if adjust_threshold_for_sklearn:
        if dtype == np.float32:
            fy = thresh.astype(np.float32)
            mask = fy > thresh
            if np.any(mask):
                fy_m = fy[mask]
                eps = np.maximum(np.abs(fy_m), np.finfo(np.float32).eps) * 10
                fy[mask] = np.nextafter(fy_m, fy_m - eps, dtype=np.float32)
            thresh = fy.astype(np.float64)
        elif dtype == np.float64:
            for u in range(n):
                if cl_list[u] > u or cr_list[u] > u:
                    thresh[u] = te.sklearn_threshold(thresh[u], dtype, "BRANCH_LEQ")
    thresh_list = thresh.tolist()

    flat_vals = tree.value.reshape(n, -1)
    if leaf_weights_are_counts:
        row_sums = flat_vals.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        scaled_vals = flat_vals * (float(tree_weight) / row_sums)
    else:
        scaled_vals = flat_vals * float(tree_weight)
    if is_classifier and scaled_vals.shape[1] == 2:
        scaled_vals = scaled_vals[:, 1:2]
    single_w = scaled_vals.shape[1] == 1
    w_list = scaled_vals[:, 0].tolist() if single_w else scaled_vals.tolist()

    nodes_treeids = attr_pairs["nodes_treeids"]
    nodes_nodeids = attr_pairs["nodes_nodeids"]
    nodes_featureids = attr_pairs["nodes_featureids"]
    nodes_modes = attr_pairs["nodes_modes"]
    nodes_values = attr_pairs["nodes_values"]
    nodes_truenodeids = attr_pairs["nodes_truenodeids"]
    nodes_falsenodeids = attr_pairs["nodes_falsenodeids"]
    nodes_missing = attr_pairs["nodes_missing_value_tracks_true"]
    nodes_hitrates = attr_pairs["nodes_hitrates"]

    t_tree = attr_pairs["class_treeids"] if is_classifier else attr_pairs["target_treeids"]
    t_node = attr_pairs["class_nodeids"] if is_classifier else attr_pairs["target_nodeids"]
    t_id = attr_pairs["class_ids"] if is_classifier else attr_pairs["target_ids"]
    t_w = attr_pairs["class_weights"] if is_classifier else attr_pairs["target_weights"]

    base_tid = (nodes_treeids[-1] + 1) if nodes_treeids else int(tree_id)

    def emit_node_list(tid, ordered_nodes, nid_offset, child_map):
        for i, old_u in enumerate(ordered_nodes):
            l = cl_list[old_u]
            r = cr_list[old_u]
            if l > old_u or r > old_u:
                nodes_featureids.append(feat_list[old_u])
                nodes_modes.append("BRANCH_LEQ")
                nodes_values.append(thresh_list[old_u])
                nodes_truenodeids.append(child_map[l] if child_map is not None else l)
                nodes_falsenodeids.append(child_map[r] if child_map is not None else r)
            else:
                nodes_featureids.append(0)
                nodes_modes.append("LEAF")
                nodes_values.append(0.0)
                nodes_truenodeids.append(0)
                nodes_falsenodeids.append(0)
                nid = nid_offset + i
                if single_w:
                    t_tree.append(tid)
                    t_node.append(nid)
                    t_id.append(weight_id_bias)
                    t_w.append(w_list[old_u])
                else:
                    for k, w in enumerate(w_list[old_u]):
                        t_tree.append(tid)
                        t_node.append(nid)
                        t_id.append(k + weight_id_bias)
                        t_w.append(w)

    if n <= _MAX_SUBTREE_NODES:
        tid = base_tid
        nodes_treeids.extend([tid] * n)
        nodes_nodeids.extend(range(n))
        nodes_missing.extend([0] * n)
        nodes_hitrates.extend([1.0] * n)
        emit_node_list(tid, range(n), 0, None)
        return

    sizes = [1] * n
    for idx in range(n - 1, -1, -1):
        l = cl_list[idx]
        r = cr_list[idx]
        if l > idx or r > idx:
            sizes[idx] = 1 + sizes[l] + sizes[r]

    subtrees = []
    stack = [(0, [])]
    while stack:
        u, path = stack.pop()
        l = cl_list[u]
        r = cr_list[u]
        if sizes[u] <= _MAX_SUBTREE_NODES or (l <= u and r <= u):
            subtrees.append((u, path))
        else:
            stack.append((r, path + [(u, False)]))
            stack.append((l, path + [(u, True)]))

    old_to_new = [0] * n
    for sub_idx, (root_u, path) in enumerate(subtrees):
        tid = base_tid + sub_idx
        p_len = len(path)
        sub_offset = 2 * p_len

        sub_nodes = []
        queue = [root_u]
        while queue:
            curr = queue.pop()
            old_to_new[curr] = sub_offset + len(sub_nodes)
            sub_nodes.append(curr)
            l = cl_list[curr]
            r = cr_list[curr]
            if l > curr or r > curr:
                queue.append(r)
                queue.append(l)

        total_sub = sub_offset + len(sub_nodes)
        nodes_treeids.extend([tid] * total_sub)
        nodes_nodeids.extend(range(total_sub))
        nodes_missing.extend([0] * total_sub)
        nodes_hitrates.extend([1.0] * total_sub)

        for idx, (anc_u, went_left) in enumerate(path):
            dummy_id = p_len + idx
            next_id = (idx + 1) if (idx + 1 < p_len) else sub_offset
            nodes_featureids.append(feat_list[anc_u])
            nodes_modes.append("BRANCH_LEQ")
            nodes_values.append(thresh_list[anc_u])
            nodes_truenodeids.append(next_id if went_left else dummy_id)
            nodes_falsenodeids.append(dummy_id if went_left else next_id)

        if p_len:
            nodes_featureids.extend([0] * p_len)
            nodes_modes.extend(["LEAF"] * p_len)
            nodes_values.extend([0.0] * p_len)
            nodes_truenodeids.extend([0] * p_len)
            nodes_falsenodeids.extend([0] * p_len)
            t_tree.extend([tid] * p_len)
            t_node.extend(range(p_len, 2 * p_len))
            t_id.extend([weight_id_bias] * p_len)
            t_w.extend([0.0] * p_len)

        emit_node_list(tid, sub_nodes, sub_offset, old_to_new)


def _emit_chunked_regressor_trees(
    scope,
    operator,
    container,
    op_type,
    op_domain,
    op_version,
    trees,
    tree_weight,
    n_targets,
    base_values,
    dtype,
):
    """Emit chunked TreeEnsembleRegressor operators summed together so each operator stays under _MAX_OPERATOR_TREE_NODES."""
    import numpy as np
    from skl2onnx.common._apply_operation import apply_cast
    from skl2onnx.common.data_types import BooleanTensorType, Int64TensorType
    from skl2onnx.common.tree_ensemble import get_default_tree_regressor_attribute_pairs
    from skl2onnx.proto import onnx_proto

    input_name = operator.input_full_names
    if type(operator.inputs[0].type) in (BooleanTensorType, Int64TensorType):
        cast_input_name = scope.get_unique_variable_name("cast_input")
        apply_cast(scope, operator.input_full_names, cast_input_name, container, to=onnx_proto.TensorProto.FLOAT)
        input_name = cast_input_name

    chunks = []
    cur_chunk = []
    cur_nodes = 0
    for tree in trees:
        if cur_chunk and cur_nodes + tree.node_count > _MAX_OPERATOR_TREE_NODES:
            chunks.append(cur_chunk)
            cur_chunk = [tree]
            cur_nodes = tree.node_count
        else:
            cur_chunk.append(tree)
            cur_nodes += tree.node_count
    if cur_chunk:
        chunks.append(cur_chunk)

    chunk_outputs = []
    for c_idx, chunk_trees in enumerate(chunks):
        attrs = get_default_tree_regressor_attribute_pairs()
        attrs["name"] = scope.get_unique_operator_name("{}_{}".format(op_type, c_idx))
        attrs["n_targets"] = n_targets
        if base_values is not None and c_idx == 0:
            attrs["base_values"] = [float(x) for x in base_values]
        for local_i, tree in enumerate(chunk_trees):
            _fast_add_tree_to_attribute_pairs(attrs, False, tree, local_i, tree_weight, 0, False, True, dtype=dtype)
        for key in ("nodes_values", "class_weights", "target_weights", "nodes_hitrates", "base_values"):
            if key in attrs:
                attrs[key] = np.array(attrs[key], dtype=dtype).ravel()
        out_var = operator.outputs[0].full_name if len(chunks) == 1 else scope.get_unique_variable_name("tree_part_{}".format(c_idx))
        container.add_node(op_type, input_name, out_var, op_domain=op_domain, op_version=op_version, **attrs)
        chunk_outputs.append(out_var)

    if len(chunk_outputs) > 1:
        container.add_node("Sum", chunk_outputs, operator.outputs[0].full_name, name=scope.get_unique_operator_name("Sum"))


def _register_tree_ensemble_patches():
    """Patch skl2onnx tree builders and large regressor ensembles to avoid quadratic ONNX Runtime init times."""
    import numpy as np
    import skl2onnx.common.tree_ensemble as te
    import skl2onnx.operator_converters.decision_tree as dt
    import skl2onnx.operator_converters.gradient_boosting as gb
    import skl2onnx.operator_converters.random_forest as rf
    from skl2onnx.common._registration import register_converter
    from skl2onnx.common.data_types import guess_numpy_type

    orig_add_node = te.add_node
    if not getattr(orig_add_node, "_onnxstudio_patched", False):
        def _safe_add_node(*args, **kwargs):
            if "nodes_missing_value_tracks_true" in kwargs:
                val = kwargs["nodes_missing_value_tracks_true"]
                if isinstance(val, (bool, np.bool_)):
                    kwargs["nodes_missing_value_tracks_true"] = int(val)
            elif len(args) >= 16:
                val = args[15]
                if isinstance(val, (bool, np.bool_)):
                    args = list(args)
                    args[15] = int(val)
                    args = tuple(args)
            return orig_add_node(*args, **kwargs)

        _safe_add_node._onnxstudio_patched = True
        te.add_node = _safe_add_node

    for mod in (te, rf, gb, dt):
        if hasattr(mod, "add_tree_to_attribute_pairs"):
            mod.add_tree_to_attribute_pairs = _fast_add_tree_to_attribute_pairs

    orig_rf_reg = rf.convert_sklearn_random_forest_regressor_converter

    def _chunked_rf_regressor(scope, operator, container, op_type="TreeEnsembleRegressor", op_domain="ai.onnx.ml", op_version=1):
        op = operator.raw_operator
        if not hasattr(op, "estimators_"):
            return orig_rf_reg(scope, operator, container, op_type=op_type, op_domain=op_domain, op_version=op_version)
        options = scope.get_options(op, dict(decision_path=False, decision_leaf=False))
        if options.get("decision_path", False) or options.get("decision_leaf", False):
            return orig_rf_reg(scope, operator, container, op_type=op_type, op_domain=op_domain, op_version=op_version)
        trees = [est.tree_ for est in op.estimators_]
        total_nodes = sum(t.node_count for t in trees)
        if total_nodes <= _MAX_OPERATOR_TREE_NODES or len(trees) <= 1:
            return orig_rf_reg(scope, operator, container, op_type=op_type, op_domain=op_domain, op_version=op_version)

        dtype = guess_numpy_type(operator.inputs[0].type)
        if dtype != np.float64:
            dtype = np.float32
        _emit_chunked_regressor_trees(
            scope, operator, container, op_type, op_domain, op_version,
            trees, 1.0 / len(trees), int(op.n_outputs_), None, dtype,
        )

    register_converter(
        "SklearnRandomForestRegressor",
        _chunked_rf_regressor,
        overwrite=True,
        options={"decision_path": [True, False], "decision_leaf": [True, False]},
    )
    register_converter(
        "SklearnExtraTreesRegressor",
        _chunked_rf_regressor,
        overwrite=True,
        options={"decision_path": [True, False], "decision_leaf": [True, False]},
    )

    orig_gb_reg = gb.convert_sklearn_gradient_boosting_regressor

    def _chunked_gb_regressor(scope, operator, container, op_type="TreeEnsembleRegressor", op_domain="ai.onnx.ml", op_version=1):
        op = operator.raw_operator
        n_est = op.n_estimators_ if hasattr(op, "n_estimators_") else op.n_estimators
        if not hasattr(op, "estimators_") or n_est <= 1:
            return orig_gb_reg(scope, operator, container, op_type=op_type, op_domain=op_domain, op_version=op_version)
        trees = [op.estimators_[i][0].tree_ for i in range(n_est)]
        total_nodes = sum(t.node_count for t in trees)
        if total_nodes <= _MAX_OPERATOR_TREE_NODES:
            return orig_gb_reg(scope, operator, container, op_type=op_type, op_domain=op_domain, op_version=op_version)

        if op.init == "zero":
            loss = op._loss if hasattr(op, "_loss") else op.loss_
            cst = np.zeros(loss.K) if hasattr(loss, "K") else np.zeros(1)
        elif op.init is None:
            if hasattr(op.init_, "constant_"):
                c_arr = op.init_.constant_
                cst = [float(c_arr)] if len(c_arr.shape) == 0 else [float(x) for x in c_arr.ravel().tolist()]
            elif op.loss == "ls":
                cst = [op.init_.mean]
            else:
                cst = [op.init_.quantile]
        else:
            return orig_gb_reg(scope, operator, container, op_type=op_type, op_domain=op_domain, op_version=op_version)

        dtype = guess_numpy_type(operator.inputs[0].type)
        if dtype != np.float64:
            dtype = np.float32
        _emit_chunked_regressor_trees(
            scope, operator, container, op_type, op_domain, op_version,
            trees, op.learning_rate, 1, cst, dtype,
        )

    register_converter("SklearnGradientBoostingRegressor", _chunked_gb_regressor, overwrite=True)


def _register_imputer_indicator_patch():
    """Extend skl2onnx's SimpleImputer/Imputer converter and shape calculator to support add_indicator=True."""
    import numpy as np
    import skl2onnx.operator_converters.imputer_op as imp_op
    import skl2onnx.shape_calculators.imputer as imp_sc
    from skl2onnx.common._apply_operation import apply_concat
    from skl2onnx.common._registration import register_converter, register_shape_calculator
    from skl2onnx.common.data_types import Int64TensorType, StringTensorType
    from skl2onnx.operator_converters.common import concatenate_variables
    from skl2onnx.proto import onnx_proto

    orig_imp_sc = imp_sc.calculate_sklearn_imputer_output_shapes
    orig_imp_conv = imp_op.convert_sklearn_imputer

    def _imputer_indicator_shape_calc(operator):
        orig_imp_sc(operator)
        op = operator.raw_operator
        ind = getattr(op, "indicator_", None) if getattr(op, "add_indicator", False) else None
        if ind is not None and getattr(ind, "features_", None) is not None:
            out_shape = operator.outputs[0].type.shape
            if out_shape and out_shape[1] is not None:
                operator.outputs[0].type.shape = [out_shape[0], out_shape[1] + int(len(ind.features_))]

    def _imputer_indicator_converter(scope, operator, container):
        op = operator.raw_operator
        ind = getattr(op, "indicator_", None) if getattr(op, "add_indicator", False) else None
        if (
            ind is None
            or getattr(ind, "features_", None) is None
            or len(ind.features_) == 0
            or isinstance(operator.inputs[0].type, StringTensorType)
        ):
            return orig_imp_conv(scope, operator, container)

        op_type = "Imputer"
        attrs = {"name": scope.get_unique_operator_name(op_type)}
        use_int = isinstance(operator.inputs[0].type, Int64TensorType)
        if use_int:
            attrs["imputed_value_int64s"] = op.statistics_.astype(np.int64)
        else:
            attrs["imputed_value_floats"] = op.statistics_.astype(np.float32)

        is_nan_mv = isinstance(op.missing_values, str) and op.missing_values == "NaN"
        if is_nan_mv:
            attrs["replaced_value_float"] = np.nan
        elif isinstance(op.missing_values, (float, int, np.number)):
            is_nan_mv = bool(np.isnan(op.missing_values))
            if use_int:
                attrs["replaced_value_int64"] = int(np.array([op.missing_values]).astype(np.int64)[0])
            else:
                attrs["replaced_value_float"] = float(op.missing_values)
        else:
            return orig_imp_conv(scope, operator, container)

        concatenated_feature = concatenate_variables(scope, operator.inputs, container)
        imputed_out = scope.get_unique_variable_name("imputed_out")
        container.add_node(op_type, concatenated_feature, imputed_out, op_domain="ai.onnx.ml", **attrs)

        ind_feats = np.asarray(ind.features_, dtype=np.int64)
        ind_idx_name = scope.get_unique_variable_name("ind_indices")
        container.add_initializer(ind_idx_name, onnx_proto.TensorProto.INT64, [len(ind_feats)], ind_feats)
        extracted_name = scope.get_unique_variable_name("ind_extracted")
        container.add_node("ArrayFeatureExtractor", [concatenated_feature, ind_idx_name], [extracted_name], op_domain="ai.onnx.ml")

        cond_name = scope.get_unique_variable_name("ind_cond")
        if is_nan_mv:
            container.add_node("IsNaN", [extracted_name], [cond_name])
        else:
            mv_name = scope.get_unique_variable_name("ind_mv")
            mv_dtype = onnx_proto.TensorProto.INT64 if use_int else onnx_proto.TensorProto.FLOAT
            mv_val = [attrs["replaced_value_int64"]] if use_int else [attrs["replaced_value_float"]]
            container.add_initializer(mv_name, mv_dtype, [1], mv_val)
            container.add_node("Equal", [extracted_name, mv_name], [cond_name])

        cast_to = onnx_proto.TensorProto.INT64 if use_int else onnx_proto.TensorProto.FLOAT
        cast_ind_name = scope.get_unique_variable_name("ind_cast")
        container.add_node("Cast", [cond_name], [cast_ind_name], to=cast_to)
        apply_concat(scope, [imputed_out, cast_ind_name], operator.outputs[0].full_name, container, axis=1)

    for alias in ("SklearnImputer", "SklearnSimpleImputer"):
        register_shape_calculator(alias, _imputer_indicator_shape_calc, overwrite=True)
        register_converter(alias, _imputer_indicator_converter, overwrite=True)


def _register_ohe_infrequent_patch():
    """Allow skl2onnx's OneHotEncoder converter to handle single-input encoders with infrequent categories."""
    import skl2onnx.operator_converters.one_hot_encoder as ohe_conv
    from skl2onnx.common._registration import register_converter

    orig_ohe = ohe_conv.convert_sklearn_one_hot_encoder
    try:
        with open(ohe_conv.__file__, "r", encoding="utf-8") as handle:
            ohe_src = handle.read()
    except OSError:
        return
    if "if len(operator.inputs) > 1:" not in ohe_src:
        return

    body_src = ohe_src.rsplit("register_converter(", 1)[0]
    ohe_ns = dict(ohe_conv.__dict__)
    exec(body_src.replace("if len(operator.inputs) > 1:", "if len(operator.inputs) >= 1:", 1), ohe_ns)
    infreq_ohe = ohe_ns["convert_sklearn_one_hot_encoder"]

    def _patched_ohe(scope, operator, container):
        ohe_op = operator.raw_operator
        if (
            len(operator.inputs) == 1
            and getattr(ohe_op, "_infrequent_enabled", False)
            and operator.inputs[0].type.shape
            and isinstance(operator.inputs[0].type.shape[1], int)
        ):
            return infreq_ohe(scope, operator, container)
        return orig_ohe(scope, operator, container)

    register_converter("SklearnOneHotEncoder", _patched_ohe, overwrite=True)


def _register_sym_function_transformer_patch():
    """Trace simple pandas column arithmetic/replace FunctionTransformer steps into native ONNX ops."""
    import numpy as np
    from sklearn.preprocessing import FunctionTransformer
    import skl2onnx._parse as sk_parse
    from skl2onnx.common._registration import register_converter, register_shape_calculator
    from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, StringTensorType
    from skl2onnx.proto import onnx_proto

    class _SymSeries:
        def __init__(self, df, var):
            self.df = df
            self.var = var

        def _ensure_float(self, var):
            if isinstance(var.type, FloatTensorType):
                return var
            return self.df._emit("CastFloat", [var], FloatTensorType(var.type.shape))

        def _binop(self, other, op_type, rev=False):
            if isinstance(other, _SymSeries):
                lhs_var, rhs_var = (other.var, self.var) if rev else (self.var, other.var)
                need_float = (
                    op_type == "Div"
                    or isinstance(lhs_var.type, FloatTensorType)
                    or isinstance(rhs_var.type, FloatTensorType)
                )
                if need_float:
                    lhs_var = self._ensure_float(lhs_var)
                    rhs_var = self._ensure_float(rhs_var)
                out_type = FloatTensorType(lhs_var.type.shape) if need_float else lhs_var.type
                return _SymSeries(self.df, self.df._emit(op_type, [lhs_var, rhs_var], out_type))

            need_float = (
                op_type == "Div"
                or isinstance(self.var.type, FloatTensorType)
                or (isinstance(other, float) and not other.is_integer())
            )
            base_var = self._ensure_float(self.var) if need_float else self.var
            const_type = FloatTensorType(base_var.type.shape) if need_float else base_var.type
            const_val = float(other) if need_float or isinstance(base_var.type, FloatTensorType) else int(other)
            c_var = self.df._emit("ConstScalar", [], const_type, value=const_val)
            lhs_var, rhs_var = (c_var, base_var) if rev else (base_var, c_var)
            return _SymSeries(self.df, self.df._emit(op_type, [lhs_var, rhs_var], const_type))

        def __add__(self, other):
            return self._binop(other, "Add")

        def __radd__(self, other):
            return self._binop(other, "Add", rev=True)

        def __sub__(self, other):
            return self._binop(other, "Sub")

        def __rsub__(self, other):
            return self._binop(other, "Sub", rev=True)

        def __mul__(self, other):
            return self._binop(other, "Mul")

        def __rmul__(self, other):
            return self._binop(other, "Mul", rev=True)

        def __truediv__(self, other):
            return self._binop(other, "Div")

        def __rtruediv__(self, other):
            return self._binop(other, "Div", rev=True)

        def abs(self):
            return _SymSeries(self.df, self.df._emit("Abs", [self.var], self.var.type))

        def replace(self, to_replace):
            cur = self.var
            if isinstance(to_replace, dict):
                for key, val in to_replace.items():
                    cur = self.df._emit("ReplaceScalar", [cur], cur.type, k=key, v=val)
            return _SymSeries(self.df, cur)

        def astype(self, dtype):
            if dtype in (str, "str", "O", object):
                if isinstance(self.var.type, StringTensorType):
                    return self
                return _SymSeries(self.df, self.df._emit("CastString", [self.var], StringTensorType(self.var.type.shape)))
            if dtype in (float, np.float32, np.float64, "float", "float32", "float64"):
                if isinstance(self.var.type, FloatTensorType):
                    return self
                return _SymSeries(self.df, self.df._emit("CastFloat", [self.var], FloatTensorType(self.var.type.shape)))
            if dtype in (int, np.int64, "int", "int64"):
                if isinstance(self.var.type, Int64TensorType):
                    return self
                return _SymSeries(self.df, self.df._emit("CastInt64", [self.var], Int64TensorType(self.var.type.shape)))
            return self

    class _SymDF:
        def __init__(self, scope, inputs):
            self.scope = scope
            self.cols = {v.raw_name: _SymSeries(self, v) for v in inputs}

        def copy(self):
            cloned = _SymDF.__new__(_SymDF)
            cloned.scope = self.scope
            cloned.cols = dict(self.cols)
            return cloned

        def __getitem__(self, key):
            if isinstance(key, (list, tuple)):
                cloned = _SymDF.__new__(_SymDF)
                cloned.scope = self.scope
                cloned.cols = {k: self.cols[k] for k in key}
                return cloned
            return self.cols[key]

        def __setitem__(self, key, val):
            if isinstance(val, _SymSeries):
                self.cols[key] = val

        def _emit(self, kind, in_vars, out_type, var_name="sym_tmp", **kwargs):
            op = self.scope.declare_local_operator("SklearnSymExpr")
            op.raw_operator = (kind, kwargs)
            op.inputs.extend(in_vars)
            out = self.scope.declare_local_variable(var_name, out_type.__class__(list(out_type.shape)))
            op.outputs.append(out)
            return out

    def _sym_shape_calc(operator):
        pass

    def _sym_converter(scope, operator, container):
        kind, kwargs = operator.raw_operator
        out_name = operator.outputs[0].full_name
        out_type = operator.outputs[0].type
        out_is_int = isinstance(out_type, Int64TensorType)
        out_is_str = isinstance(out_type, StringTensorType)
        if kind == "ConstScalar":
            if out_is_int:
                container.add_initializer(out_name, onnx_proto.TensorProto.INT64, [1, 1], [int(kwargs["value"])])
            elif out_is_str:
                container.add_initializer(out_name, onnx_proto.TensorProto.STRING, [1, 1], [str(kwargs["value"])])
            else:
                container.add_initializer(out_name, onnx_proto.TensorProto.FLOAT, [1, 1], [float(kwargs["value"])])
        elif kind == "Identity":
            container.add_node("Identity", [operator.inputs[0].full_name], [out_name], name=scope.get_unique_operator_name("Identity"))
        elif kind in ("Add", "Sub", "Mul", "Div", "Abs"):
            container.add_node(kind, [v.full_name for v in operator.inputs], [out_name], name=scope.get_unique_operator_name(kind))
        elif kind == "CastFloat":
            container.add_node("Cast", [operator.inputs[0].full_name], [out_name], to=onnx_proto.TensorProto.FLOAT)
        elif kind == "CastInt64":
            container.add_node("Cast", [operator.inputs[0].full_name], [out_name], to=onnx_proto.TensorProto.INT64)
        elif kind == "CastString":
            container.add_node("Cast", [operator.inputs[0].full_name], [out_name], to=onnx_proto.TensorProto.STRING)
        elif kind == "ReplaceScalar":
            inp_name = operator.inputs[0].full_name
            k_name = scope.get_unique_variable_name("rep_k")
            v_name = scope.get_unique_variable_name("rep_v")
            eq_name = scope.get_unique_variable_name("rep_eq")
            if out_is_int:
                container.add_initializer(k_name, onnx_proto.TensorProto.INT64, [1, 1], [int(kwargs["k"])])
                container.add_initializer(v_name, onnx_proto.TensorProto.INT64, [1, 1], [int(kwargs["v"])])
            elif out_is_str:
                container.add_initializer(k_name, onnx_proto.TensorProto.STRING, [1, 1], [str(kwargs["k"])])
                container.add_initializer(v_name, onnx_proto.TensorProto.STRING, [1, 1], [str(kwargs["v"])])
            else:
                container.add_initializer(k_name, onnx_proto.TensorProto.FLOAT, [1, 1], [float(kwargs["k"])])
                container.add_initializer(v_name, onnx_proto.TensorProto.FLOAT, [1, 1], [float(kwargs["v"])])
            container.add_node("Equal", [inp_name, k_name], [eq_name])
            container.add_node("Where", [eq_name, v_name, inp_name], [out_name])

    register_shape_calculator("SklearnSymExpr", _sym_shape_calc, overwrite=True)
    register_converter("SklearnSymExpr", _sym_converter, overwrite=True)

    orig_ft_parser = sk_parse.sklearn_parsers_map.get(FunctionTransformer)

    def _parse_func_transformer(scope, model, inputs, custom_parsers=None):
        if getattr(model, "func", None) is not None and len(inputs) >= 1:
            try:
                df = _SymDF(scope, inputs)
                res = model.func(df)
                if isinstance(res, _SymDF):
                    out_vars = []
                    for col_name, series in res.cols.items():
                        v = series.var
                        if v.raw_name != col_name:
                            v = df._emit("Identity", [v], v.type, var_name=col_name)
                        out_vars.append(v)
                    return out_vars
            except Exception:  # noqa: BLE001 - fall back to skl2onnx's default FunctionTransformer handling
                pass
        if orig_ft_parser is not None:
            return orig_ft_parser(scope, model, inputs, custom_parsers=custom_parsers)
        return sk_parse._parse_sklearn_simple_model(scope, model, inputs, custom_parsers=custom_parsers)

    sk_parse.sklearn_parsers_map[FunctionTransformer] = _parse_func_transformer


def _patch_skl2onnx():
    global _SKL2ONNX_PATCHED
    if _SKL2ONNX_PATCHED:
        return
    for patch_fn in (
        _register_tree_ensemble_patches,
        _register_imputer_indicator_patch,
        _register_ohe_infrequent_patch,
        _register_sym_function_transformer_patch,
    ):
        try:
            patch_fn()
        except Exception:  # noqa: BLE001
            pass
    _SKL2ONNX_PATCHED = True


def _tensor_width(tensor_type):
    shape = getattr(tensor_type, "shape", None)
    if shape and len(shape) > 1 and isinstance(shape[1], int) and shape[1] > 0:
        return int(shape[1])
    return 1


def _can_merge_numeric_inputs(initial_types):
    for _, tensor_type in initial_types:
        if type(tensor_type).__name__ not in ("FloatTensorType", "Int64TensorType", "DoubleTensorType"):
            return False
        shape = getattr(tensor_type, "shape", None)
        if not shape or len(shape) != 2 or not isinstance(shape[1], int) or shape[1] <= 0:
            return False
    return True


def _unique_tensor_name(prefix, existing_names):
    candidate = prefix
    suffix = 0
    while candidate in existing_names:
        suffix += 1
        candidate = "{}_{}".format(prefix, suffix)
    existing_names.add(candidate)
    return candidate


def _restore_named_numeric_inputs(onnx_model, initial_types, onnx):
    existing_names = {name for name, _ in initial_types}
    existing_names.add("__merged_input__")
    for node in onnx_model.graph.node:
        existing_names.update(node.output)
    for init in onnx_model.graph.initializer:
        existing_names.add(init.name)
    for vi in onnx_model.graph.value_info:
        existing_names.add(vi.name)

    new_inputs = []
    concat_inputs = []
    prepend_nodes = []
    for idx, (name, tensor_type) in enumerate(initial_types):
        kind = type(tensor_type).__name__
        if kind == "Int64TensorType":
            elem_type = onnx.TensorProto.INT64
        elif kind == "DoubleTensorType":
            elem_type = onnx.TensorProto.DOUBLE
        else:
            elem_type = onnx.TensorProto.FLOAT
        dims = [int(d) if isinstance(d, int) else None for d in tensor_type.shape]
        new_inputs.append(onnx.helper.make_tensor_value_info(name, elem_type, dims))

        if kind in ("Int64TensorType", "DoubleTensorType"):
            cast_out = _unique_tensor_name("__cast_{}_{}__".format(idx, name), existing_names)
            prepend_nodes.append(
                onnx.helper.make_node("Cast", inputs=[name], outputs=[cast_out], to=onnx.TensorProto.FLOAT)
            )
            concat_inputs.append(cast_out)
        else:
            concat_inputs.append(name)

    prepend_nodes.append(
        onnx.helper.make_node("Concat", inputs=concat_inputs, outputs=["__merged_input__"], axis=1)
    )
    for offset, node in enumerate(prepend_nodes):
        onnx_model.graph.node.insert(offset, node)
    del onnx_model.graph.input[:]
    onnx_model.graph.input.extend(new_inputs)


def _convert_with_zipmap_fallback(convert_sklearn, model, initial_types, opset, options):
    try:
        return convert_sklearn(model, initial_types=initial_types, target_opset=opset, options=options)
    except (NameError, KeyError, ValueError) as error:
        if options is None or "zipmap" not in str(error).lower():
            raise
        return convert_sklearn(model, initial_types=initial_types, target_opset=opset)


def convert(model, request, collected_warnings):
    try:
        import numpy as np
        import onnx
        import skl2onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as error:
        raise WorkerError("missing_module", "The Python package '{}' is required to convert models to ONNX.".format(error.name))

    try:
        from skl2onnx.common.utils import InvalidInputLengthException
    except ImportError:
        InvalidInputLengthException = ()

    _patch_skl2onnx()

    output_path = request.get("outputPath")
    if not output_path:
        raise WorkerError("invalid_input", "No output path was provided.")

    requested_opset = int(request.get("targetOpset") or 18)
    opset = min(requested_opset, int(onnx.defs.onnx_opset_version()), int(getattr(skl2onnx, "__max_supported_opset__", requested_opset)))

    initial_types = infer_initial_types(model, request.get("inputTypes"))

    options = None
    final = last_estimator(model)
    if request.get("zipmap") is False and hasattr(final, "predict_proba") and getattr(final, "classes_", None) is not None:
        options = {id(final): {"zipmap": False}}

    patched_imputers = []
    for est in _iter_all_estimators(model):
        if type(est).__name__ == "SimpleImputer":
            stats = getattr(est, "statistics_", None)
            if stats is not None and _is_string_array(np.asarray(stats)):
                orig_mv = getattr(est, "missing_values", None)
                if not isinstance(orig_mv, (str, np.str_)):
                    patched_imputers.append((est, orig_mv))
                    est.missing_values = ""

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                onnx_model = _convert_with_zipmap_fallback(convert_sklearn, model, initial_types, opset, options)
            except InvalidInputLengthException:
                if len(initial_types) <= 1 or not _can_merge_numeric_inputs(initial_types):
                    raise
                total_width = sum(_tensor_width(t) for _, t in initial_types)
                merged_types = [("__merged_input__", FloatTensorType([None, total_width]))]
                onnx_model = _convert_with_zipmap_fallback(convert_sklearn, model, merged_types, opset, options)
                _restore_named_numeric_inputs(onnx_model, initial_types, onnx)
            for item in caught:
                collected_warnings.append("{}: {}".format(item.category.__name__, item.message))
    except WorkerError:
        raise
    except Exception as error:  # noqa: BLE001
        name = type(error).__name__
        if name in ("MissingShapeCalculator", "MissingConverter", "NotImplementedError"):
            raise WorkerError("unsupported_model", "skl2onnx cannot convert '{}' yet: {}".format(type(model).__name__, error))
        raise WorkerError("convert_failed", "Conversion failed: {}: {}".format(name, error))
    finally:
        for est, orig_mv in patched_imputers:
            est.missing_values = orig_mv

    embed_source_metadata(onnx_model, model)

    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    temporary = output_path + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(onnx_model.SerializeToString())
    os.replace(temporary, output_path)

    validation = {"performed": False, "passed": None, "detail": "Validation disabled."}
    if request.get("validate", True):
        validation = validate_conversion(model, output_path, initial_types, collected_warnings)

    return {
        "outputPath": output_path,
        "targetOpset": opset,
        "requestedOpset": requested_opset,
        "inputs": describe_value_info(onnx_model.graph.input),
        "outputs": describe_value_info(onnx_model.graph.output),
        "size": os.path.getsize(output_path),
        "validation": validation,
        "warnings": collected_warnings,
        "skl2onnx": skl2onnx.__version__,
    }


# ------------------------------------------------------------------------ main

def handle_single_request(command, request_path, response_path):
    try:
        with open(request_path, "r", encoding="utf-8-sig") as handle:
            request = json.load(handle)
        collected = []
        model = load_model(request["modelPath"], collected)
        if command == "inspect":
            result = inspect_model(model, collected)
        elif command == "predict":
            result = predict(model, request, collected)
        elif command == "convert":
            result = convert(model, request, collected)
        else:
            raise WorkerError("invalid_input", "Unknown command '{}'.".format(command))
        write_response(response_path, {"ok": True, "result": to_py(result), "workerVersion": WORKER_VERSION})
        return 0
    except WorkerError as error:
        write_response(response_path, {"ok": False, "error": {"code": error.code, "message": error.message, "trace": traceback.format_exc()}})
        return 0
    except Exception as error:  # noqa: BLE001
        code = "predict_failed" if command == "predict" else "worker_failed"
        write_response(response_path, {"ok": False, "error": {"code": code, "message": "{}: {}".format(type(error).__name__, error), "trace": traceback.format_exc()}})
        return 0


def run_daemon():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for mod_name in ("numpy", "joblib", "sklearn", "pandas"):
            try:
                __import__(mod_name)
            except Exception:  # noqa: BLE001
                pass

    print("__ONNXSTUDIO_READY__", file=sys.__stdout__, flush=True)

    while True:
        raw_line = sys.stdin.readline()
        if raw_line == "":
            break
        line = raw_line.strip()
        if not line:
            continue
        if line in ("exit", "quit"):
            break
        try:
            msg = json.loads(line)
            command = msg.get("command")
            if command in ("exit", "quit"):
                break
            if command == "ping":
                continue
            request_path = msg.get("requestPath")
            response_path = msg.get("responsePath")
            handle_single_request(command, request_path, response_path)
        except Exception as error:  # noqa: BLE001
            sys.stderr.write("Daemon error: {}: {}\n".format(type(error).__name__, error))
            sys.stderr.flush()
        finally:
            print("__ONNXSTUDIO_DONE__", file=sys.__stdout__, flush=True)

    return 0


def main(argv):
    if len(argv) == 2 and argv[1] == "--daemon":
        return run_daemon()
    if len(argv) != 4:
        sys.stderr.write("usage: onnxstudio_worker.py <command> <request.json> <response.json>\n")
        return 2
    command, request_path, response_path = argv[1:4]
    return handle_single_request(command, request_path, response_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
