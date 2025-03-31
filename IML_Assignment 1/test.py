import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

_LAOD_SUCCESS = True
try:
    from pa1 import prepare_dataset, kernel_function, train, classify
except (ImportError, SyntaxError):
    _LAOD_SUCCESS = False
    try:
        import pa1
    except (ImportError, SyntaxError):
        print(
            "[*] Cannot import `pa1`. "
            "Make sure `pa1.py` in the same directory as `test.py` "
            "and check for any syntax errors."
        )
        traceback.print_exc()
    else:
        _preset = [
            "prepare_dataset",
            "kernel_function",
            "train",
            "classify",
        ]
        _found = [fn for fn in _preset if fn in dir(pa1)]
        print(
            f"[*] Some functions are missing. Expected `{'`, `'.join(_preset)}`, "
            f"but got only `{'`, `'.join(_found)}`"
        )


@dataclass
class TestResult:
    dataset: Optional[Tuple[np.ndarray, np.ndarray]] = None
    models: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    metric: Optional[Dict[str, float]] = None

    state: Literal[
        "IMPORT_ERROR",
        "FAILED_STEP1",
        "FAILED_STEP2",
        "FAILED_STEP3",
        "FAILED_STEP4",
        "EXCEPTION",
        "SUCCESS",
    ] = "SUCCESS"
    _traceback: str = ""

    def __repr__(self) -> str:
        _d = "..."
        if self.dataset is None:
            _d = "None"
        _m = "..."
        if self.models is None:
            _m = "None"

        return (
            f"TestResult(dataset={_d}, models={_m}, "
            f"metric={self.metric}, state={self.state}, _traceback={self._traceback})"
        )


def test(
    dataset_path: Path,
    evalset_path: Path,
    num_training_steps: int,
    learning_rate: float,
    sigma: float = 1.0,
) -> TestResult:
    result = TestResult()
    if not _LAOD_SUCCESS:
        print(
            "[*] The module named `pa1` is not imported. "
            "Make sure to import the `pa1` correctly first."
        )
        result.state = "IMPORT_ERROR"
        return result

    try:
        ## 1. Prepare the dataset
        dataset = prepare_dataset(  # pyright: ignore [reportPossiblyUnboundVariable]
            dataset_path
        )
        result.dataset = dataset
        if dataset is None or not isinstance(dataset, tuple):
            print(
                "[*] Nothing returned. Please check the return value of `prepare_dataset`."
            )
            result.state = "FAILED_STEP1"
            return result

        images, labels = dataset
        if images.ndim != 3:
            print(
                f"[*] Invalid shape for the image array. "
                f"Expected a 3-dimensional array, but got a {images.ndim}-dimensional array."
            )
            result.state = "FAILED_STEP1"
            return result
        if images.dtype != np.float32:
            print(
                f"[*] Invalid data type for the image array. "
                f"Expected a np.float32 type array, but got {images.dtype}"
            )
            result.state = "FAILED_STEP1"
            return result
        if labels.ndim != 1:
            print(
                f"[*] Invalid shape for the label array. "
                f"Expected a 1-dimensional array, but got a {labels.ndim}-dimensional array."
            )
            result.state = "FAILED_STEP1"
            return result
        if labels.dtype != np.int64:
            print(
                f"[*] Invalid data type for the label array. "
                f"Expected a np.long type array, but got {labels.dtype}"
            )
            result.state = "FAILED_STEP1"
            return result

        if images.shape[0] != labels.shape[0]:
            print(
                f"[*] The sizes of both images and labels should be same, "
                f"but the image array has size {images.shape[0]}, while the label array has size {labels.shape[0]}"
            )
            result.state = "FAILED_STEP1"
            return result
        if images.max() > 1.0 or images.min() < 0.0:
            print(
                f"[*] The scale of the images is expected to be in the range [0.0, 1.0], "
                f"but the actual range is [{images.min():.2f}, {images.max():.2f}]"
            )
            result.state = "FAILED_STEP1"
            return result

        print("[*] Successfully load the dataset.")

        ## 2. Write the kernel function
        _test_value = (
            kernel_function(  # pyright: ignore [reportPossiblyUnboundVariable]
                a=np.random.randn(10),
                b=np.random.randn(10),
                sigma=sigma,
            )
        )
        if _test_value is None or not isinstance(_test_value, float):
            print(
                "[*] Invalid value returned. Please check the return value of `kernel_function`"
            )
            result.state = "FAILED_STEP2"
            return result

        print("[*] Kernel function passed the unit test.")

        ## 3. Update the parameters
        parameters = train(  # pyright: ignore [reportPossiblyUnboundVariable]
            dataset,
            num_training_steps,
            learning_rate,
            sigma,
        )
        if parameters is None or not isinstance(parameters, np.ndarray):
            print(
                f"[*] Invalid value returned. Please check the return value of `train`"
            )
            result.state = "FAILED_STEP3"
            return result

        if parameters.ndim != 1 or parameters.shape[0] != images.shape[0]:
            print(
                f"[*] Invalid shape for the parameters. "
                f"Expected a 1-dimensional array of size {images.shape[0]}, "
                f"but got an array with shape {parameters.shape}."
            )
            result.state = "FAILED_STEP3"
            return result

        print("[*] Training done.")

        ## 4. Classify the label.
        model = (parameters, images)
        label = classify(  # pyright: ignore [reportPossiblyUnboundVariable]
            model, images[0]
        )
        if label is None or not isinstance(label, int):
            print(
                "[*] Invalid return value. Please check the return value of `classify`"
            )
            result.state = "FAILED_STEP4"
            return result

        ## 6. Evaluate the result.
        def _accuracy(_dataset: tuple[np.ndarray, np.ndarray]) -> float:
            matches = [
                classify(model, img)  # pyright: ignore [reportPossiblyUnboundVariable]
                == int(label)
                for img, label in zip(*_dataset)
            ]
            return np.mean(matches).item()

        train_accuracy = _accuracy(dataset)
        print(f"[*] Training accuracy: {train_accuracy:.2f}")
        result.metric = {"training-accuracy": train_accuracy}

        evalset = prepare_dataset(  # pyright: ignore [reportPossiblyUnboundVariable]
            evalset_path
        )
        eval_accuracy = _accuracy(evalset)
        print(f"[*] Evaluation accuracy: {eval_accuracy:.2f}")
        result.metric["evaluation-accuracy"] = eval_accuracy

    except Exception:
        tback = traceback.format_exc()
        print(f"[*] SOMETHING WENT WRONG;\n{tback}")

        result.state = "EXCEPTION"
        result._traceback = tback

    return result


if __name__ == "__main__":

    def _console():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset-path", default="trainset.csv")
        parser.add_argument("--evalset-path", default="evalset.csv")
        parser.add_argument("--num-training-steps", type=int, default=1000)
        parser.add_argument("--learning-rate", type=float, default=0.001)
        parser.add_argument("--sigma", type=float, default=1.0)
        args = parser.parse_args()

        response = test(
            Path(args.dataset_path),
            Path(args.evalset_path),
            args.num_training_steps,
            args.learning_rate,
            sigma=args.sigma,
        )
        print(f"[*] Response;\n{response}")

    _console()
