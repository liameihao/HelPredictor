# HelPredictor

### Introduction



### Quick Start

1. The HelPredictor tools package is written in Python. It is recommended to use [conda](https://www.anaconda.com/download/) to manage python packages.

   ```
   conda env create -f environment.yml
   ```

   Or please make sure all the following packages are installed in their Python environment: numpy(1.19.2), pandas(1.1.3), scikit-learn(0.23.2), statsmodels(0.12.0), 

2. Input data. Please convert the data to CSV format (Reference format: train.csv).

3. Run HelPredictor.

   ```
   python HelPredictor.py -i train.csv
   ```



### Usage

#### Command line

```bash
usage: HelPredictor.py [-h] -i INPUT_TRAIN [--method {cv2,pca,fscore}] [--start START] [--end END] [--step STEP] [--njobs NJOBS] [--classifier {svm,rf,gnb,lr}]
                       [-o OUTPUT]

Creates DeepCpG data for training and testing.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_TRAIN, --input_train INPUT_TRAIN
                        Input train csv (default: None)
  --method {cv2,pca,fscore}, -m {cv2,pca,fscore}
                        Select a feature selection method: cv2, pca, F-score (default: fscore)
  --start START         Feature Number start (default: 10)
  --end END             Feature Number end (default: None)
  --step STEP           Feature Number step (default: 10)
  --njobs NJOBS         Number of jobs to run in parallel (default: 4)
  --classifier {svm,rf,gnb,lr}, -c {svm,rf,gnb,lr}
                        Select a machine learning method: lr (Logical Regression), svm (Support Vector Machine), rf (Random Forest), gnb (Gaussian Naive Bayes)
                        (default: svm)
  -o OUTPUT, --output OUTPUT
                        Output directory (default: None)

```

#### Example

```bash
# F-score + SVM(Support Vector Machine)
python HelPredictor.py --method fscore --classifier svm -i train.csv
# CV2 + RF(Random Forest)
python HelPredictor.py --method cv2 --classifier rf -i train.csv
# PCA + lr(Logical Regression)
python HelPredictor.py --method pca --classifier lr -i train.csv
```



### Citation



**F-score**

Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011. 

**CV2**

Brennecke, P., S. Anders, J. K. Kim, A. A. Kołodziejczyk, X. Zhang, V. Proserpio, B. Baying, V. Benes, S. A. Teichmann and J. C. Marioni (2013). "Accounting for technical noise in single-cell RNA-seq experiments." Nature methods **10**(11): 1093-1095.

