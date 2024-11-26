# ClipBased-SyntheticImageDetection


Before using the code, download the weights, Luma videos and Real_ones videos from [here](https://drive.google.com/drive/folders/1iJ1gjAz05vGPXHrMafqye_qK2Br-cfI3?usp=drive_link).
 
### Script - `main.py`
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

For each video in the folder, `main.py` outputed a file called `frames_results_<model>_<video_name>.csv`, where are stored LLR scores for each frame of the processed video, notice that if LLR > 0, the single frame is detected as synthetic.

Then the results are resume in a table (`results_<model>.txt`) like the following, where for each feature dectetor method and for each fusion method is shown the output ad the confidence level:

```text
file_path
| Model                  | Output         |   Confidence |
|:-----------------------|:---------------|-------------:|
| clipdet_latent10k      | Synthetic/Real |    [0, 1]    |
| clipdet_latent10k_plus | Synthetic/Real |    [0, 1]    |
| Corvi2023              | Synthetic/Real |    [0, 1]    |
| fusion[mean_logit]     | Synthetic/Real |    [0, 1]    |
| fusion[max_logit]      | Synthetic/Real |    [0, 1]    |
| fusion[median_logit]   | Synthetic/Real |    [0, 1]    |
| fusion[lse_logit]      | Synthetic/Real |    [0, 1]    |
| fusion[mean_prob]      | Synthetic/Real |    [0, 1]    |
| fusion[soft_or_prob]   | Synthetic/Real |    [0, 1]    |
```
We compute the `Output` value based on the majority of frames predictions. The `Confidence` is given by the percentage of frames voting the majority-class. 

### Script - `final_analysis.py`

The `final_analysis.py` script takes as input the single table for each video analyzed `results_<model>.txt` and convert them to a recap csv (`final_results_<model>.csv`) with the average for the given `model`.

Finally also a plot is enerated based on those data and saved as `confidence_comparison_plot_<model>.png`

### Script - `results/plot_histo.py`

For a general overview.

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
