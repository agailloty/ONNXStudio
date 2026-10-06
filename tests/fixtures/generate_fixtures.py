"""Generate small real ONNX models used as test fixtures.

Run:  python generate_fixtures.py
Outputs are written next to this script (committed to the repository so the
.NET test suite does not require Python).
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = os.path.dirname(os.path.abspath(__file__))


def _tensor(name, dims, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, dims)


def build_add_model():
    # y = a + b, element-wise on [2]
    node = helper.make_node("Add", ["a", "b"], ["y"], name="add_node")
    graph = helper.make_graph(
        [node], "add_graph",
        [_tensor("a", [2]), _tensor("b", [2])],
        [_tensor("y", [2])],
    )
    model = helper.make_model(graph, producer_name="onnx-fixtures", opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8
    return model


def build_linreg_model():
    # y = X @ W + C  (a tiny linear regression, like a mini California Housing)
    w = numpy_helper.from_array(np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32), "W")
    c = numpy_helper.from_array(np.array([0.5], dtype=np.float32), "C")
    node = helper.make_node("Gemm", ["X", "W", "C"], ["y"], name="gemm_node", alpha=1.0, beta=1.0)
    graph = helper.make_graph(
        [node], "linreg_graph",
        [_tensor("X", [1, 4])],
        [_tensor("y", [1, 1])],
        [w, c],
    )
    model = helper.make_model(graph, producer_name="onnx-fixtures", opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 9
    return model


def build_convnet_model():
    # Small classifier: Conv -> Relu -> GlobalAveragePool -> Gemm
    conv_w = numpy_helper.from_array(
        np.random.default_rng(42).normal(0, 0.1, (8, 3, 3, 3)).astype(np.float32), "conv_w")
    conv_b = numpy_helper.from_array(np.zeros(8, dtype=np.float32), "conv_b")
    fc_w = numpy_helper.from_array(
        np.random.default_rng(43).normal(0, 0.1, (8, 10)).astype(np.float32), "fc_w")
    fc_b = numpy_helper.from_array(np.zeros(10, dtype=np.float32), "fc_b")

    nodes = [
        helper.make_node("Conv", ["X", "conv_w", "conv_b"], ["conv_out"],
                         name="conv1", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv_out"], ["relu_out"], name="relu1"),
        helper.make_node("GlobalAveragePool", ["relu_out"], ["pool_out"], name="pool1"),
        # squeeze spatial dims to [1, 8]: Reshape with constant shape
        helper.make_node("Reshape", ["pool_out", "reshape_shape"], ["flat_out"], name="reshape1"),
        helper.make_node("Gemm", ["flat_out", "fc_w", "fc_b"], ["Y"], name="fc1"),
    ]
    shape_const = numpy_helper.from_array(np.array([1, 8], dtype=np.int64), "reshape_shape")

    graph = helper.make_graph(
        nodes, "convnet_graph",
        [_tensor("X", [1, 3, 32, 32])],
        [_tensor("Y", [1, 10])],
        [conv_w, conv_b, fc_w, fc_b, shape_const],
    )
    model = helper.make_model(graph, producer_name="onnx-fixtures", opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


def build_dynamic_model():
    # Dynamic batch dimension: X [batch, 4] -> Gemm -> Y [batch, 2]
    w = numpy_helper.from_array(np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32), "W")
    node = helper.make_node("Gemm", ["X", "W"], ["Y"], name="dyn_gemm")
    graph = helper.make_graph(
        [node], "dynamic_graph",
        [_tensor("X", ["batch", 4])],
        [_tensor("Y", ["batch", 2])],
        [w],
    )
    model = helper.make_model(graph, producer_name="onnx-fixtures", opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8
    return model


def build_opset19_model():
    # Same as add but declared with opset 19 (unsupported by the studio)
    node = helper.make_node("Add", ["a", "b"], ["y"], name="add_node")
    graph = helper.make_graph([node], "add_graph", [_tensor("a", [2])], [_tensor("y", [2])])
    model = helper.make_model(graph, producer_name="onnx-fixtures", opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 8
    return model


def build_empty_model():
    # Graph with no inputs and no nodes: rejected as InvalidModel
    graph = helper.make_graph([], "empty_graph", [], [])
    model = helper.make_model(graph, producer_name="onnx-fixtures", opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8
    return model


def main():
    models = {
        "add.onnx": build_add_model(),
        "linreg.onnx": build_linreg_model(),
        "convnet.onnx": build_convnet_model(),
        "dynamic.onnx": build_dynamic_model(),
        "empty.onnx": build_empty_model(),
    }
    for filename, model in models.items():
        onnx.checker.check_model(model)
        path = os.path.join(HERE, filename)
        onnx.save(model, path)
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")

    # opset19 intentionally unchecked (opset not supported by the installed runtime)
    path = os.path.join(HERE, "opset19.onnx")
    onnx.save(build_opset19_model(), path)
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
