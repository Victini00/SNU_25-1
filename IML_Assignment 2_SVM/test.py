import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

_LOAD_SUCCESS = True
try:
    from pa2 import prepare_dataset, compute_kernel_matrix, train_binary_svm, train_multi_class_svm, classify
except (ImportError, SyntaxError):
    _LOAD_SUCCESS = False
    try:
        import pa2
    except (ImportError, SyntaxError):
        print(
            "[*] Cannot import `pa2`. "
            "Make sure `pa2.py` in the same directory as `test_pa2.py` "
            "and check for any syntax errors."
        )
        traceback.print_exc()
    else:
        _preset = [
            "prepare_dataset",
            "compute_kernel_matrix",
            "train_binary_svm",
            "train_multi_class_svm",
            "classify",
        ]
        _found = [fn for fn in _preset if fn in dir(pa2)]
        print(
            f"[*] Some functions are missing. Expected `{'`, `'.join(_preset)}`, "
            f"but got only `{'`, `'.join(_found)}`"
        )


@dataclass
class TestResult:
    dataset: Optional[Tuple[np.ndarray, np.ndarray]] = None
    models: Optional[List[Tuple[np.ndarray, float]]] = None
    metric: Optional[Dict[str, float]] = None

    state: Literal[
        "IMPORT_ERROR",
        "FAILED_STEP1",
        "FAILED_STEP2",
        "FAILED_STEP3",
        "FAILED_STEP4",
        "FAILED_STEP5",
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
    num_iterations: int,
    C: float = 1.0,
    sigma: float = 1.0,
) -> TestResult:
    result = TestResult()
    if not _LOAD_SUCCESS:
        print(
            "[*] The module named `pa2` is not imported. "
            "Make sure to import the `pa2` correctly first."
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
        # 코드 수정
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

        print("[*] Successfully loaded the dataset.")

        ## 2. Test kernel matrix computation
        b, h, w = images.shape
        X = images[:2].reshape(2, h * w)  # Use just 2 images for testing
        K = compute_kernel_matrix(  # pyright: ignore [reportPossiblyUnboundVariable]
            X, sigma=sigma
        )
        
        if K is None or not isinstance(K, np.ndarray):
            print(
                "[*] Invalid value returned. Please check the return value of `compute_kernel_matrix`"
            )
            result.state = "FAILED_STEP2"
            return result
            
        if K.shape != (2, 2):
            print(
                f"[*] Invalid shape for the kernel matrix. "
                f"Expected (2, 2), but got {K.shape}"
            )
            result.state = "FAILED_STEP2"
            return result

        print("[*] Kernel matrix computation passed the test.")

        ## 3. Test binary SVM training
        alphas_bias = train_binary_svm(  # pyright: ignore [reportPossiblyUnboundVariable]
            (images[:10], labels[:10]),  # Use subset for testing
            target_class=0,
            num_iterations=10,
            C=C,
            sigma=sigma,
        )
        
        if alphas_bias is None or not isinstance(alphas_bias, tuple) or len(alphas_bias) != 2:
            print(
                "[*] Invalid value returned. Please check the return value of `train_binary_svm`. "
                "It should return a tuple of (alphas, bias)."
            )
            result.state = "FAILED_STEP3"
            return result
        
        alphas, bias = alphas_bias
        
        if not isinstance(alphas, np.ndarray) or not isinstance(bias, (float, np.float32, np.float64)):
            print(
                "[*] Invalid types returned. `train_binary_svm` should return (np.ndarray, float)."
            )
            result.state = "FAILED_STEP3"
            return result
            
        if alphas.shape != (10,):
            print(
                f"[*] Invalid shape for alphas. Expected (10,), but got {alphas.shape}"
            )
            result.state = "FAILED_STEP3"
            return result

        print("[*] Binary SVM training passed the test.")

        ## 4. Train multi-class SVM
        models = train_multi_class_svm(  # pyright: ignore [reportPossiblyUnboundVariable]
            dataset,
            num_iterations,
            C=C,
            sigma=sigma,
        )
        result.models = models
        
        if models is None or not isinstance(models, list):
            print(
                "[*] Invalid value returned. Please check the return value of `train_multi_class_svm`"
            )
            result.state = "FAILED_STEP4"
            return result
            
        num_classes = len(np.unique(labels))
        if len(models) != num_classes:
            print(
                f"[*] Invalid number of models. Expected {num_classes}, but got {len(models)}"
            )
            result.state = "FAILED_STEP4"
            return result
            
        for i, model in enumerate(models):
            if not isinstance(model, tuple) or len(model) != 2:
                print(
                    f"[*] Invalid model format for model {i}. Expected a tuple of (alphas, bias)."
                )
                result.state = "FAILED_STEP4"
                return result
                
            model_alphas, model_bias = model
            
            if not isinstance(model_alphas, np.ndarray) or not isinstance(model_bias, (float, np.float32, np.float64)):
                print(
                    f"[*] Invalid types for model {i}. Expected (np.ndarray, float)."
                )
                result.state = "FAILED_STEP4"
                return result
                
            if model_alphas.shape != (images.shape[0],):
                print(
                    f"[*] Invalid shape for model {i} alphas. "
                    f"Expected ({images.shape[0]},), but got {model_alphas.shape}"
                )
                result.state = "FAILED_STEP4"
                return result

        print("[*] Multi-class SVM training done.")

        ## 5. Test classification
        pred_label = classify(  # pyright: ignore [reportPossiblyUnboundVariable]
            models,
            images,
            labels,
            images[0],
            sigma=sigma
        )
        
        if pred_label is None or not isinstance(pred_label, int):
            print(
                "[*] Invalid return value. Please check the return value of `classify`"
            )
            result.state = "FAILED_STEP5"
            return result
            
        if not (0 <= pred_label < num_classes):
            print(
                f"[*] Invalid predicted label. Expected a value between 0 and {num_classes-1}, "
                f"but got {pred_label}"
            )
            result.state = "FAILED_STEP5"
            return result

        ## 6. Evaluate the result
        def _accuracy(_dataset: tuple[np.ndarray, np.ndarray]) -> float:
            _images, _labels = _dataset
            matches = 0
            # total = min(100, len(_images))  # Only use first 100 samples for speed
            total = len(_images)
            
            for i in range(total):
                pred = classify(models, images, labels, _images[i], sigma)  # pyright: ignore [reportPossiblyUnboundVariable]
                if pred == _labels[i]:
                    matches += 1
                    
            return matches / total

        train_accuracy = _accuracy((images, labels))
        print(f"[*] Training accuracy: {train_accuracy:.2f}")
        result.metric = {"training-accuracy": train_accuracy}

        evalset = prepare_dataset(  # pyright: ignore [reportPossiblyUnboundVariable]
            evalset_path
        )

        eval_accuracy = _accuracy(evalset)
        print(f"[*] Evaluation accuracy: {eval_accuracy:.2f}")
        result.metric["evaluation-accuracy"] = eval_accuracy

        print("[*] Testing completed successfully.")

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
        parser.add_argument("--num-iterations", type=int, default=1000)
        parser.add_argument("--C", type=float, default=0.3)
        parser.add_argument("--sigma", type=float, default=5)
        args = parser.parse_args()

        response = test(
            Path(args.dataset_path),
            Path(args.evalset_path),
            args.num_iterations,
            C=args.C,
            sigma=args.sigma,
        )
        print(f"[*] Response;\n{response}")

    _console()