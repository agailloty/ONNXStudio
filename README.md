# ONNX Studio

Cross-platform desktop application for data scientists: load, inspect, run and
serve **ONNX models** — built as a modular monolith on **Avalonia**,
**ONNX Runtime** and **ASP.NET Core**.

## Features

| Capability | Description |
|------------|-------------|
| Load models (US-001) | File picker, drag-and-drop, CLI (`ONNXStudioUI --model path.onnx`); validation (extension, size, signature, opset <= 18) with actionable error messages |
| Inspect (US-002) | Computation graph with category colors, live search and filter, node details (attributes, inputs/outputs, dependencies), statistics, initializer list |
| Dynamic forms (US-003) | Form generated from the model schema: number, vector, image (real decode + resize to CHW float32) and text fields |
| Local inference (US-004) | Real ONNX Runtime execution with input validation, LRU session cache, typed outputs (scalars, vectors, top classes) |
| REST API (US-005) | Embedded Kestrel server (`/models`, `/models/{id}`, `/models/{id}/schema`, `/models/{id}/predict`, `/health`) with CORS and JSON schema/cURL previews |
| API sandbox (US-006) | Real HTTP requests against the embedded server, status/timing, history with re-run |
| Python models | Open scikit-learn models saved with joblib / pickle (`.joblib`, `.pkl`, `.pickle`...), inspect them (class, pipeline steps, features, classes), run inference (`predict`, `predict_proba`, `decision_function`, `transform`) on typed rows, and convert them to ONNX with skl2onnx (opset capped to what the installed packages support, optional validation against scikit-learn). The converted model opens like any ONNX model (graph, inference, API), and the scikit-learn metadata (pipeline steps, hyper-parameters, learned attributes such as `coef_` or `classes_`) is stored in the ONNX file and browsable in **Inspector > Structure & Parameters**. Needs a Python runtime, see [Python support](#python-support) |
| UX | VS Code style workbench (activity bar, model explorer side bar, editor tabs, status bar), Dark+/Light+ themes, toasts, model screens cached per model. Shortcuts: Ctrl+O open, Ctrl+B side bar, Ctrl+W close tab, Ctrl+, settings |

## Solution layout (modular monolith)

```
ONNXStudio.slnx
├── ONNXStudio.Core/            # Domain + services (no UI, no HTTP)
│   ├── Models/                 # OnnxModel, graph, tensors, errors, Result<T,E>
│   ├── Services/               # ModelLoader, ModelRegistry, InferenceService,
│   │                           # InferenceSessionManager (LRU), FormGeneration,
│   │                           # GraphAnalysis
│   ├── Python/                 # Python runtime discovery / installation, worker
│   │                           # script (joblib / pickle inference, skl2onnx)
│   └── Utilities/              # OnnxProtoParser (dependency-free protobuf)
├── ONNXStudio.Api/             # ASP.NET Core Minimal APIs (library)
│   ├── Endpoints/              # /health, /models..., /predict
│   ├── Payloads/               # JSON payload parsing -> validated tensors
│   └── ApiServerHost.cs        # Embedded Kestrel lifecycle
├── ONNXStudioUI/               # Avalonia executable (UI module, hosts the API)
└── tests/ONNXStudio.Core.Tests # unit + integration tests (real .onnx fixtures)
```

Communication between modules is direct .NET method calls through interfaces,
wired by `Microsoft.Extensions.DependencyInjection` (single process, no IPC).

## Build and run

```bash
dotnet build ONNXStudio.slnx
dotnet run --project ONNXStudioUI            # GUI, opens maximized
dotnet run --project ONNXStudioUI -- --model path/to/model.onnx
```

## Python support

ONNX Studio embeds **no** Python and no Python library. joblib / pickle models run in
a separate Python process (`onnxstudio_worker.py`, written to
`LocalApplicationData/ONNXStudio/python/worker`), so you choose where Python comes
from in **Settings > Python runtime**:

| Option | What happens |
|--------|--------------|
| Install the ONNX Studio runtime | Downloads a standalone CPython 3.12 ([python-build-standalone](https://github.com/astral-sh/python-build-standalone), SHA-256 verified) into `LocalApplicationData/ONNXStudio/python/runtime`, then `pip install`s numpy, scipy, pandas, scikit-learn, joblib, skl2onnx, onnx and onnxruntime (binary wheels only). Nothing is installed system-wide and the user site-packages are ignored. Needs an internet connection once. |
| Use a Python already on the machine | Interpreters found through the `py` launcher, `PATH`, `VIRTUAL_ENV` and `CONDA_PREFIX` are listed with the packages they contain. Pick one; ONNX Studio never installs packages into it. |
| Point to your own Python | Browse to an interpreter or to a virtual / conda environment folder. |

With the automatic choice the managed runtime is used when installed, otherwise the
first system interpreter that has scikit-learn, numpy and joblib. Inference needs
`numpy`, `scikit-learn` and `joblib`; conversion also needs `skl2onnx` and `onnx`
(`onnxruntime` enables the post-conversion check). Models that need other packages
(xgboost, lightgbm...) work as long as the selected interpreter has them.

**Security:** unpickling executes code contained in the file. Nothing runs until you
tick *I trust this file* on the model screen; only open files you created or trust.

Typical flow: open a `.joblib` / `.pkl` (Ctrl+O, drag-and-drop or CLI), choose or
install Python if needed, tick the trust box, and select *Load model*. The model
then appears in the same explorer and **Inspector** as ONNX models, without
conversion. **Graph** shows pipeline steps and transformer branches;
**Structure & Parameters** exposes nested estimators, hyper-parameters and learned
attributes, with selection linked to the graph. **Inference**, **API** and
**Sandbox** also use the shared screens and the model's input/output schemas.
The Python import tab retains the optional *Convert to ONNX* action. Converted
models target opset 18 at most, which is what the studio can open.

The shared screens depend on `IModel` (metadata, schemas, computation graph and
lazy `StructureNode` trees). `OnnxModel` and `SklearnModel` supply those views;
`IInferenceBackend` handles execution with ONNX Runtime or the Python worker.
Another format, such as ML.NET, can implement these contracts and register its
backend without duplicating the analysis screens. ML.NET support is not yet
implemented.

## Release builds

Push a version tag (for example `v1.0.0`) to build, test and package the application
for Windows x64, Linux x64 and macOS Apple Silicon. Successful builds create a
GitHub Release with portable archives, installers and SHA-256 checksums.
See [Build and release](packaging/README.md) for tag conventions, manual builds
and platform details.

## Tests

```bash
dotnet test tests/ONNXStudio.Core.Tests
```

The Python integration tests (real download and installation of Python, scikit-learn
and skl2onnx, several hundred MB) are skipped unless `ONNXSTUDIO_PYTHON_INTEGRATION=1`
is set; set `ONNXSTUDIO_TEST_PYTHON_DIR` to reuse an installation between runs.

The suite covers model loading (valid/invalid files, opset rejection), graph
extraction (attributes, dynamic shapes, tensor-flow edges), real inference on
generated ONNX fixtures (add, linear regression, convnet, dynamic batch),
session LRU eviction, form generation, and full-stack API integration tests
(real Kestrel + real inference over HTTP).

## API example

```bash
curl -X POST http://localhost:5000/models/{id}/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"a": [1, 2], "b": [3, 4]}}'
```

```json
{ "modelId": "…", "executionTimeMs": 12, "outputs": { "y": [4, 6] },
  "outputShapes": { "y": [2] } }
```

## Notes and V1 limitations

- Theme and API port are persisted under LocalApplicationData/ONNXStudio/settings.json.
- String tensors, exact integer inputs, dynamic shapes and rank-zero scalars are executable.
- The predict payload uses the `{"inputs": {...}}` named-tensor format; custom
  JSON field renaming (mapping) is not supported server-side yet.
- The API starts after model loading and stops when the last model is unloaded.
  `/openapi.json` describes the currently loaded models; the sandbox can test all exposed endpoints.
- See [UI architecture review](ONNXStudioUI/ARCHITECTURE_REVIEW.md) for verified paths,
  test commands and remaining specification gaps.
- AOT: compiled bindings and a reflection-free ViewLocator keep the UI on the
  AOT-friendly path (full NativeAOT publishing not yet validated).

Test fixtures are real ONNX models generated by `tests/fixtures/generate_fixtures.py`
(python-onnx) and committed so the .NET test suite does not require Python.
