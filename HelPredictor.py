from lpf_classifier.lpf_GNB import GNB_overall
from lpf_classifier.lpf_SVM import SVM_overall
from lpf_classifier.lpf_RF import RF_overall
from lpf_classifier.lpf_LR import LR_overall
from lpf_selection.lpf_pca import pca_selection
from lpf_selection.Fscore import fscore_main
from lpf_selection.cv2 import cal_cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse
import sys
import os


class App:
    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Creates DeepCpG data for training and testing.')

        # I/O
        p.add_argument(
            "-i", '--input_train',
            required=True,
            help='Input train csv'
        )
        p.add_argument(
            '--method', '-m',
            choices=['cv2', 'pca', 'fscore'],
            default='fscore',
            help='Select a feature selection method: cv2, pca, F-score'
        )
        p.add_argument(
            "--start",
            default=10,
            help="Feature Number start"
        )
        p.add_argument(
            "--end",
            default=None,
            help="Feature Number end"
        )
        p.add_argument(
            "--step",
            default=10,
            help="Feature Number step"
        )
        p.add_argument(
            "--njobs",
            default=4,
            help="Number of jobs to run in parallel"
        )
        p.add_argument(
            '--classifier', '-c',
            choices=['svm', 'rf', 'gnb', 'lr'],
            default="svm",
            help='Select a machine learning method: '
                 'lr (Logical Regression), '
                 'svm (Support Vector Machine), '
                 'rf (Random Forest), '
                 'gnb (Gaussian Naive Bayes)'
        )
        p.add_argument(
            '-o', "--output",
            default=None,
            help='Output directory'
        )
        return p

    def feature_selection(self, data, method, filename):
        if method == "pca":
            print("Feature Selection Method: PCA")
            train_data = pca_selection(data, filename)
        elif method == "fscore":
            print("Feature Selection Method: F-score")
            train_data = fscore_main(data, filename)
        else:
            print("Feature Selection Method: CV2")
            train_data = cal_cv2(data, filename)
        return train_data

    def main(self, opts):
        filename = os.path.split(opts.input_train)[1].split(".")[0]
        data = pd.read_csv(opts.input_train, header=0)
        if opts.output:
            os.chdir(opts.output)
        else:
            pass
        # Feature Selection
        train_data = self.feature_selection(data, opts.method, filename)
        #
        X_train = train_data.iloc[:, 1:].values
        y_train = train_data['Label'].values
        # Split arrays into random train and test subsets
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
        # Max feature number
        if not opts.end:
            opts.end = X_train.shape[1] + 1
        print("Feature Start: {}".format(opts.start))
        print("Feature End: {}".format(opts.end - 1))
        print("Feature Step: {}".format(opts.step))
        if opts.classifier == "svm":
            print("Classifier: Support Vector Machine (svm)")
            print("Training...")
            SVM_overall(X_train, y_train, X_test, y_test,
                        int(opts.start), int(opts.end), int(opts.step),
                        opts.method, int(opts.njobs))
        elif opts.classifier == "lr":
            print("Classifier: Logical Regression (lr)")
            print("Training...")
            LR_overall(X_train, y_train, X_test, y_test,
                       int(opts.start), int(opts.end), int(opts.step),
                       opts.method, int(opts.njobs))
        elif opts.classifier == "rf":
            print("Classifier: Random Forest (rf)")
            print("Training...")
            RF_overall(X_train, y_train, X_test, y_test,
                       int(opts.start), int(opts.end), int(opts.step),
                       opts.method, int(opts.njobs))
        else:
            print("Classifier: Gaussian Naive Bayes (gnb)")
            print("Training...")
            GNB_overall(X_train, y_train, X_test, y_test,
                        int(opts.start), int(opts.end), int(opts.step),
                        opts.method, int(opts.njobs))
        print("Done")


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
