from pathlib import Path
from typing import Tuple

import numpy as np

##########################################################################
# classum 내용대로, test.py의 np.long 부분을 np.int64로 수정하였습니다.
##########################################################################

## 1. Prepare the dataset
def prepare_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:

    """Prepare the dataset of handwritten digit images.
    Args:
        path: a path to the dataset.
    Returns:
        tuple of read images and corresponding labels,
            [np.float32; [B, H, W]] and [np.long; [B]]
    """
    
    # NaN 값에 대한 예외 처리 가능 ~ genfromtxt
    data = np.genfromtxt(Path(path), delimiter=',',skip_header=1)
    
    Labels = data[:,0].astype(np.int64)
    pixel_value = data[:,1:]

    # 크기 조작 및 타입 변경
    Images = pixel_value.reshape(Labels.shape[0],16,16).astype(np.float32)/255.0

    return Images, Labels

## 2. Write the kernel function
def kernel_function(a: np.ndarray, b: np.ndarray, sigma: float = 1.0) -> float:
    """Compute the kernel of the given vectors.
    Args:
        a, b: [np.float32; [H, W]], two gray-scale images, in range[0, 1].
    Returns:
        single scalar value.
    """
    euc_dis = np.sum((a-b)**2)

    kernel_value = np.exp(-euc_dis / (2 * (sigma**2)))
    
    return kernel_value

## 3. Update the parameters
def train(
    dataset: Tuple[np.ndarray, np.ndarray], # 1의 리턴값
    num_training_steps: int, 
    learning_rate: float,
    sigma: float = 1.0,
) -> np.ndarray:
    """Update the parameters for the given dataset.
    Args:
        dataset: tuple of read images and corresponding labels,
            [np.float32; [B, H, W]] and [np.long, [B]].
        num_training_steps: the number of the updates.
        learning_rate: the scalar value, alpha (learning rate or step-size of a single update).
        sigma: a bandwidth of the gaussian kernel, denoted by `sigma`.
    Returns:
        [np.float32; [B]], the trained parameters.
    """
    
    kernel = [[0 for _ in range(dataset[0].shape[0])] for _ in range(dataset[0].shape[0])]

    # 커널 값 계산
    for i in range(dataset[0].shape[0]):
        for j in range(dataset[0].shape[0]):
            kernel[i][j] = kernel_function(dataset[0][i], dataset[0][j], sigma)

    # 0, 1 -> -1, 1
    # 4. 에서의 매핑을 여기서 먼저 진행했습니다.
    
    mapping_label = dataset[1].copy()
    for i in range(mapping_label.shape[0]):
        if mapping_label[i] == 0: mapping_label[i] = -1
    

    init_beta = np.zeros((dataset[0].shape[0]))
    beta = np.zeros((dataset[0].shape[0]))

    for _ in range(num_training_steps):
        for i in range(dataset[0].shape[0]):

            kernel_sum = 0

            for j in range(dataset[0].shape[0]):
                kernel_sum += (beta[j] * kernel[i][j])

            beta[i] = init_beta[i] + learning_rate * (mapping_label[i] - kernel_sum)

        init_beta = beta.copy()
        beta = np.zeros((dataset[0].shape[0]))

    # return init_beta
    return init_beta
    

## 4. Estimate the labels of the given images.
def classify(model: Tuple[np.ndarray, np.ndarray], given: np.ndarray) -> int:
    """Classify the given image with the trained kernel model.
    Args:
        model: the trained model parameters from `train` and the corresponding training images.
        given: [np.float32; [H, W]], the target image to classify, gray-scale in range[0, 1].
    Returns:
        an estimated label of the image.
    """
    result = 0

    for i, beta in enumerate(model[0]):
        result += (beta * kernel_function(model[1][i], given))
    
    if result < 0:
        return 0
    else:
        return 1
    
