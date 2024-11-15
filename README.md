# ClipBased-SyntheticImageDetection


Before using the code, download the weights, Luma videos and Real_ones videos from [here](https://drive.google.com/drive/folders/1iJ1gjAz05vGPXHrMafqye_qK2Br-cfI3?usp=drive_link).

The `main.py` script requires as input a CSV file with the list of images to analyze.
The input CSV file must have a 'filename' column with the path to the images.
The code outputs a CSV file with the LLR score for each image.
If LLR>0, the image is detected as synthetic.

The `compute_metrics.py` script can be used to evaluate metrics.
In this case, the input CSV file must also include the 'typ' column with a value equal to 'real' for real images.


### Script 
In order to install all the requirements run:


```
python3 -r requirements.txt
```


The test can be executed as follows:

```
python3 main.py
```
