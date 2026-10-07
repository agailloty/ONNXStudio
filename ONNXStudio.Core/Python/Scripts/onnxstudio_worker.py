# ONNX Studio worker - loads scikit-learn models saved with joblib/pickle, runs
# inference on them and converts them to ONNX. Invoked as:
#   python onnxstudio_worker.py <command> <request.json> <response.json>
# Commands: inspect, predict, convert.
# The response file always contains {"ok": true, "result": ...} or
# {"ok": false, "error": {"code", "message", "trace"}}. Nothing is read from stdin.

import json
import math
import os
import sys
import traceback
import warnings

WORKER_VERSION = 1
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

def load_model(path, collected_warnings):
    if not os.path.isfile(path):
        raise WorkerError("file_not_found", "The model file does not exist: " + path)

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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name, loader in loaders:
            try:
                model = loader(path)
                for item in caught:
                    collected_warnings.append("{}: {}".format(item.category.__name__, item.message))
                return model
            except ModuleNotFoundError as error:
                # The model needs a package that is not installed in this interpreter.
                raise WorkerError("missing_module", "The model needs the Python package '{}', which is not installed in the selected runtime.".format(error.name))
            except Exception as error:  # noqa: BLE001
                last_error = error
    raise WorkerError("load_failed", "The file could not be loaded as a joblib or pickle model: {}: {}".format(type(last_error).__name__, last_error))


def _pickle_load(pickle, path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


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


def has_column_transformer(model):
    return any(type(e).__name__ == "ColumnTransformer" for e in iter_estimators(model))


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
    return versions


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
        children = [(item[0], item[1], "columns: " + short_repr(item[2], 160)) for item in transformers if len(item) > 2]
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


def inspect_model(model, collected_warnings, preview=MAX_PREVIEW):
    import sklearn

    feature_names = getattr(model, "feature_names_in_", None)
    n_features = getattr(model, "n_features_in_", None)
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
        "featureNamesIn": [str(n) for n in feature_names] if feature_names is not None else None,
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


def build_input(model, columns, rows):
    import numpy as np

    if not rows:
        raise WorkerError("invalid_input", "No input rows were provided.")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise WorkerError("invalid_input", "All input rows must have the same number of values.")

    cells = [parse_column([row[i] for row in rows]) for i in range(width)]

    if is_text_model(model):
        if width != 1:
            raise WorkerError("invalid_input", "This model takes raw text: provide exactly one value per row.")
        return [str(v) for v in [row[0] for row in rows]]

    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        names = [str(n) for n in names]
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
        return pd.DataFrame({names[k]: cells[order[k]] for k in range(len(names))}, columns=names)

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
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType

    if overrides:
        return parse_input_overrides(overrides)
    if is_text_model(model):
        return [("text", StringTensorType([None, 1]))]
    names = getattr(model, "feature_names_in_", None)
    if names is not None and has_column_transformer(model):
        return [(str(n), FloatTensorType([None, 1])) for n in names]
    n_features = getattr(model, "n_features_in_", None)
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
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return {"performed": False, "passed": None, "detail": "onnxruntime is not installed in the selected runtime: conversion was not validated."}

    try:
        rng = np.random.default_rng(0)
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        samples = 32
        feeds = {}
        frames = {}
        for name, tensor_type in initial_types:
            width = tensor_type.shape[1] if len(tensor_type.shape) > 1 and tensor_type.shape[1] else 1
            kind = type(tensor_type).__name__
            if kind == "StringTensorType":
                return {"performed": False, "passed": None, "detail": "Validation skipped (text input)."}
            data = rng.standard_normal((samples, width))
            if kind == "Int64TensorType":
                data = np.round(data * 3).astype(np.int64)
                feeds[name] = data
                frames[name] = data
            else:
                feeds[name] = data.astype(np.float32)
                frames[name] = feeds[name].astype(np.float64)

        if len(frames) == 1:
            sample = next(iter(frames.values()))
            names = getattr(model, "feature_names_in_", None)
            if names is not None:
                import pandas as pd
                sample = pd.DataFrame(sample, columns=[str(n) for n in names])
        else:
            import pandas as pd
            sample = pd.DataFrame({name: values[:, 0] for name, values in frames.items()})
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


def convert(model, request, collected_warnings):
    try:
        import onnx
        import skl2onnx
        from skl2onnx import convert_sklearn
    except ImportError as error:
        raise WorkerError("missing_module", "The Python package '{}' is required to convert models to ONNX.".format(error.name))

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

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=opset, options=options)
            except (NameError, KeyError, ValueError) as error:
                if options is None or "zipmap" not in str(error).lower():
                    raise
                onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=opset)
            for item in caught:
                collected_warnings.append("{}: {}".format(item.category.__name__, item.message))
    except WorkerError:
        raise
    except Exception as error:  # noqa: BLE001
        name = type(error).__name__
        if name in ("MissingShapeCalculator", "MissingConverter", "NotImplementedError"):
            raise WorkerError("unsupported_model", "skl2onnx cannot convert '{}' yet: {}".format(type(model).__name__, error))
        raise WorkerError("convert_failed", "Conversion failed: {}: {}".format(name, error))

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

def main(argv):
    if len(argv) != 4:
        sys.stderr.write("usage: onnxstudio_worker.py <command> <request.json> <response.json>\n")
        return 2
    command, request_path, response_path = argv[1:4]

    try:
        with open(request_path, "r", encoding="utf-8") as handle:
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
        code = "predict_failed" if sys.argv[1:2] == ["predict"] else "worker_failed"
        write_response(response_path, {"ok": False, "error": {"code": code, "message": "{}: {}".format(type(error).__name__, error), "trace": traceback.format_exc()}})
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
