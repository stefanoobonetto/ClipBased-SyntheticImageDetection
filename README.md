# ClipBased-SyntheticImageDetection


Before using the code, download the weights, Luma videos and Real_ones videos from [here](https://drive.google.com/drive/folders/1iJ1gjAz05vGPXHrMafqye_qK2Br-cfI3?usp=drive_link).
 
### Script 
In order to install all the requirements run:

```
python3 -r requirements.txt
```

The test can be executed as follows:

```
python3 main.py
```

The code will ask the user for which set of videos doing tests (Luma, LattE, CogVideoX).

ALl results are storedd in the `results` folder.

For each video in the folder, `main.py` outputed a file called `frames_results_<model>_<video_name>.csv`, where are stored LLR values for each frame of the processed video, notice that if LLR > 0

# Fusion Functions

The `fusion_functions` dictionary defines several methods for combining logits or probabilities along a specified axis. Below is an explanation of each function:

## 1. Mean Logit
```python
'mean_logit': lambda x, axis: np.mean(x, axis)
```
- **Description**: Computes the mean of the logits along the specified axis.
- **Input**:
    - x: Logits (array-like).
    - axis: The axis along which the mean is computed.
- **Output**: The average value of the logits along the specified axis.

## 2. Max Logit
```python
'max_logit': lambda x, axis: np.max(x, axis)
```

- **Description**: Computes the maximum of the logits along the specified axis.
- **Input**:
    - x: Logits (array-like).
    - axis: The axis along which the maximum is computed.
- **Output**: The maximum value of the logits along the specified axis.

## 3. Median Logit
```python
'median_logit': lambda x, axis: np.median(x, axis)
```

- **Description**: Computes the median of the logits along the specified axis.
- **Input**:
    - x: Logits (array-like).
    - axis: The axis along which the median is computed.
- **Output**: The median value of the logits along the specified axis.
  
## 4. LSE Logit (LogSumExp)
```python
'lse_logit': lambda x, axis: logsumexp(x, axis)
```
- **Description**: Computes the LogSumExp of the logits along the specified axis.
LogSumExp is defined as $log(sum(exp(x)))$, which is often used for numerical stability in softmax-like operations.
- **Input**:
    - x: Logits (array-like).
    - axis: The axis along which the LogSumExp is computed.
- **Output**: The LogSumExp value of the logits along the specified axis.

## 5. Mean Probability
```python
'mean_prob': lambda x, axis: softminusinv(logsumexp(log_expit(x), axis) - np.log(x.shape[axis]))
```
- **Description**: Computes the mean of probabilities derived from logits.
Logits are first converted to probabilities using the logistic sigmoid function (log_expit).
The mean probability is computed using the softminusinv function, which inverts the softmin operation.
- **Input**:
    - x: Logits (array-like).
    - axis: The axis along which the mean probability is computed.
- **Output**: The mean probability value along the specified axis.

## 6. Soft-OR Probability
```python
'soft_or_prob': lambda x, axis: -softminusinv(np.sum(log_expit(-x), axis))
```
- **Description**: Computes a "soft OR" operation on probabilities derived from logits.
Logits are first negated and converted to probabilities using the logistic sigmoid function (log_expit).
A sum is applied to the probabilities, and the result is processed using softminusinv to invert the softmin operation.
- **Input**:
    - x: Logits (array-like).
    - axis: The axis along which the soft OR operation is computed.
- **Output**: A single value representing the "soft OR" of probabilities along the specified axis.
