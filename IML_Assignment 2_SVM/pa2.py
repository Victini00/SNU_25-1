from pathlib import Path
from typing import Tuple, List

import numpy as np

# C = 10
# sigma = 0.5

# python==3.12
# numpy==2.2.3
# 가상환경을 통해 위 조건 하에서 실행하였지만, Classum에서 언급된 문제와 동일한
# Invalid data type for the label array. Expected a np.long type array, but got int64
# 오류가 발생하여, test.py를 수정하여 사용하였습니다.

# Classum No.27 참고, test.py의 121번째 줄 
# if labels.dtype != np.int64: 로 수정하여 사용

############################################################

## 1. Prepare the dataset
def prepare_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare the dataset of handwritten digit images.
    Args:
        path: a path to the dataset.
    Returns:
        tuple of read images and corresponding labels,
            [np.float32; [B, H, W]] and [np.long; [B]]
    """
    with path.open() as f:
        csv = f.read()

    _header, *bodies = csv.strip().split("\n")
    labels = np.array(
        [int(label) for line in bodies for label, *_ in (line.split(","),)]
    )
    imgs = np.array(
        [
            np.array([int(pixel) for pixel in img]).reshape(16, 16)
            for line in bodies
            for _, *img in (line.split(","),)
        ]
    )
    return (imgs / 255.0).astype(np.float32), labels


## 2. Compute the kernel matrix
def compute_kernel_matrix(X: np.ndarray, Y: np.ndarray = None, sigma: float = 1.0) -> np.ndarray:
    """Compute the kernel matrix between X and Y.
    If Y is None, compute the kernel matrix between X and itself.
    
    Args:
        X: [np.float32; [N, H*W]], flattened images.
        Y: [np.float32; [M, H*W]], flattened images. Default is None.
        sigma: a bandwidth of the gaussian kernel.
    
    Returns:
        [np.float32; [N, M]], the kernel matrix.
    """
    if Y is None:
        Y = X
    
    # More efficient computation of squared Euclidean distances
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
    
    dist_square = X_norm - 2 * np.dot(X, Y.T) + Y_norm

    # Apply Gaussian kernel

    return np.exp(-dist_square / (2 * (sigma ** 2)))
    # Implement the Gaussian kernel calculation using the squared distances
    


## 3. Train a binary SVM classifier using Sequential Minimal Optimization (SMO)
def train_binary_svm(
    dataset: Tuple[np.ndarray, np.ndarray],
    target_class: int,
    num_iterations: int,
    C: float = 1.0,
    sigma: float = 1.0,
    tol: float = 1e-3,
    eps: float = 1e-3
) -> Tuple[np.ndarray, float]:
    """Train a binary SVM classifier for one-vs-rest classification using SMO-like algorithm.
    
    Args:
        dataset: tuple of read images and corresponding labels,
            [np.float32; [B, H, W]] and [np.long, [B]].
        target_class: the target class for binary classification (one-vs-rest).
        num_iterations: maximum number of iterations.
        C: regularization parameter.
        sigma: a bandwidth of the gaussian kernel.
        tol: tolerance for KKT conditions.
        eps: numerical stability constant.
    
    Returns:
        Tuple of [np.float32; [B]] and float, the trained alphas and bias.
    """
    imgs, labels = dataset
    
    # Reshape images to 2D array [B, H*W]
    b, h, w = imgs.shape
    X = imgs.reshape(b, h * w)
    
    # Convert labels to binary: 1 for target_class, -1 for others
    y = np.where(labels == target_class, 1, -1).astype(np.float32)
    
    # Compute kernel matrix
    K = compute_kernel_matrix(X, sigma=sigma)
    
    # Initialize alphas and bias
    alphas = np.zeros(b, dtype=np.float32)
    bias = 0.0
    
    # For storing errors
    errors = np.zeros(b, dtype=np.float32)
    
    # Initialize errors
    # 사용하지 않았음음
    for i in range(b):
        errors[i] = np.sum(alphas * y * K[i]) + bias - y[i]
    
    # Main training loop - simplified SMO algorithm
    # TODO: Implement the SMO algorithm
    # 1. Loop through iterations
    # 2. Check if example violates KKT conditions
    # 3. Select second example
    # 4. Update alphas and bias
    # 5. Update error cache
    
    for _ in range(num_iterations):
        Early_Stop_Checker = True
        for i in range(b):
            Expected_i = np.sum(alphas * y * K[i]) + bias
            Real_i = y[i]
            g_i = Expected_i * Real_i - 1

            # check KKT condition
            # 1. ai = C / 2. ai = 0
            if (g_i < - tol and alphas[i] < C - eps) or (g_i > tol and alphas[i] > eps):

                # Select second example
                j = np.random.choice([i for i in range(b)]) 
                while i == j:   
                    j = np.random.choice([i for i in range(b)]) 

                if y[i] == y[j]: # 1 or -1로 동일
                    Low = max(0, alphas[i] + alphas[j] - C)
                    High = min(C, alphas[i] + alphas[j])
                else: # j가 기준
                    Low = max(0, alphas[j] - alphas[i])
                    High = min(C, alphas[j] - alphas[i] + C)

                if Low == High: continue # 의미없는 수치면 제외외

                # Update alphas and bias
                # 이차식의 극점: 꼭짓점
                upper = ((y[j] / y[i])-1 -(K[i, i] * y[j] * (y[i] * alphas[i] + y[j] * alphas[j])) / (y[i] ** 2) + (y[i] * y[j] * K[i, j] * (y[i] * alphas[i] + y[j] * alphas[j])) / y[i])
                lower = (-(K[i, i] * (y[j] ** 2)) / (y[i] ** 2) - K[j, j] + (2 * y[i] * y[j] * K[i, j]) / y[i])
                
                new_alpha_j = np.clip(Low, upper/lower, High)

                # 변화량이 작으면 갱신 X
                if abs(new_alpha_j - alphas[j]) < eps: continue

                new_alpha_i = alphas[i] + y[i] * y[j] * (alphas[j] - new_alpha_j)

                # bias도 갱신해야 한다.
                bi = -1 * Expected_i - (y[i] * (new_alpha_i - alphas[i]) * K[i, i]) - (y[j] * (new_alpha_j - alphas[j]) * K[i, j]) + bias

                Expected_j = np.sum(alphas * y * K[j]) + bias - y[j]
                bj = -1* Expected_j - (y[i] * (new_alpha_i - alphas[i]) * K[i, j]) - (y[j] * (new_alpha_j - alphas[j]) * K[j, j]) + bias

                # bias 갱신
                # 갱신할 때, 둘 다 support vector가 아닌 경우, 평균값을 취함
                # # handwritten problem
                if (0 < new_alpha_i) and (new_alpha_i < C): bias = bi
                elif (0 < new_alpha_j) and (new_alpha_j < C): bias = bj
                else: bias = (bi + bj) / 2

                alphas[i] = new_alpha_i
                alphas[j] = new_alpha_j

                Early_Stop_Checker = False

        # Early stopping
        if Early_Stop_Checker == True: break

    # Return the model parameters
    return alphas, bias


## 4. Train a multi-class SVM classifier
def train_multi_class_svm(
    dataset: Tuple[np.ndarray, np.ndarray],
    num_iterations: int,
    C: float = 1.0,
    sigma: float = 1.0,
) -> List[Tuple[np.ndarray, float]]:
    """Train a multi-class SVM classifier using one-vs-rest approach.
    
    Args:
        dataset: tuple of read images and corresponding labels,
            [np.float32; [B, H, W]] and [np.long, [B]].
        num_iterations: the number of the updates.
        C: regularization parameter.
        sigma: a bandwidth of the gaussian kernel.
    
    Returns:
        List of Tuples containing [np.float32; [B]] and float, the trained alphas and bias for each class.
    """
    imgs, labels = dataset
    
    # Get unique classes
    classes = np.unique(labels)
    # num_classes = len(classes)
    
    # TODO: Implement one-vs-rest approach for multi-class classification
    # 1. Train a binary SVM for each class
    # 2. Return list of models

    class_model = []

    for target_class in classes:
        alphas, bias = train_binary_svm((imgs, labels), target_class = target_class, num_iterations = 1000, C = C, sigma = sigma) 
        class_model.append((alphas, bias))

    return class_model


## 5. Classify the given image
def classify(
    models: List[Tuple[np.ndarray, float]], 
    train_images: np.ndarray, 
    train_labels: np.ndarray, 
    test_image: np.ndarray,
    sigma: float = 1.0,
) -> int:
    """Classify the given image with the trained kernel SVM model.
    
    Args:
        models: List of Tuples containing trained alphas and bias for each class.
        train_images: [np.float32; [B, H, W]], the training images.
        train_labels: [np.long; [B]], the training labels.
        test_image: [np.float32; [H, W]], the target image to classify.
        sigma: a bandwidth of the gaussian kernel.
    
    Returns:
        an estimated label of the image (0-9).
    """
    # Reshape images
    b, h, w = train_images.shape
    X_train = train_images.reshape(b, h * w)
    # X_test = test_image.reshape(h * w)
    X_test = test_image.reshape(1, h * w)
    
    # TODO: Implement the classification logic
    # 1. Compute kernel values between test_image and all training images
    # 2. Compute decision values for each class
    # 3. Return the class with highest decision value
    
    K = compute_kernel_matrix(X_train, X_test, sigma = sigma)

    index = 0
    mem = 0

    for i in range(len(models)):
        alphas, bias = models[i]
        y_i = np.array([1 if label == i else -1 for label in train_labels])

        Expected_i = np.sum(alphas * y_i * K.flatten()) + bias # flatten
        if mem <= Expected_i:
            index = i
            mem = Expected_i

    return index


