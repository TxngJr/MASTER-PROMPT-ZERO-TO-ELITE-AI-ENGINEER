import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper


def test_onnxruntime_cpu_linear_model() -> None:
    weight = np.array(
        [
            [1.0, 2.0],
            [-1.0, 0.5],
            [0.25, -0.75],
        ],
        dtype=np.float32,
    )

    input_info = helper.make_tensor_value_info(
        "x",
        TensorProto.FLOAT,
        [None, 3],
    )
    output_info = helper.make_tensor_value_info(
        "y",
        TensorProto.FLOAT,
        [None, 2],
    )

    weight_initializer = numpy_helper.from_array(
        weight,
        name="W",
    )

    node = helper.make_node(
        "MatMul",
        ["x", "W"],
        ["y"],
    )

    graph = helper.make_graph(
        [node],
        "batch26_tiny_edge_model",
        [input_info],
        [output_info],
        [weight_initializer],
    )

    model = helper.make_model(
        graph,
        producer_name="batch26",
        opset_imports=[
            helper.make_operatorsetid("", 13)
        ],
    )
    # Keep the educational model compatible with a broad ORT range.
    model.ir_version = 8
    onnx.checker.check_model(model)

    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )

    x = np.array(
        [
            [2.0, 1.0, -1.0],
            [0.5, -2.0, 3.0],
        ],
        dtype=np.float32,
    )

    (output,) = session.run(
        ["y"],
        {"x": x},
    )

    np.testing.assert_allclose(
        output,
        x @ weight,
        rtol=1e-6,
        atol=1e-6,
    )

    assert session.get_providers()[0] == "CPUExecutionProvider"
