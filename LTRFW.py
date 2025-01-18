from scipy.optimize import differential_evolution
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import math
import random
# from SMOTE_backup import Smote
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import time
import csv
import warnings
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from deap import base, creator, tools, algorithms
import random
from functools import partial
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from scipy.stats import spearmanr
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from scipy.stats import pointbiserialr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io import arff
from sklearn.feature_selection import SelectKBest, mutual_info_classif

np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')
classifier = "nb"

skf = StratifiedKFold(n_splits=6)
print("test1")
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef

def objective_function(de_average_mcc, de_average_auc, de_average_fmeasure,
                       de_average_precision, de_average_recall, de_average_accuracy):
    weight_ranges = np.linspace(-1, 1, 5)
    best_weighted_sum = -np.inf
    best_weights = None

    for w1 in weight_ranges:
        for w2 in weight_ranges:
            for w3 in weight_ranges:
                for w4 in weight_ranges:
                    for w5 in weight_ranges:
                        for w6 in weight_ranges:
                            weights = [w1, w2, w3, w4, w5, w6]

                            weighted_sum = (weights[0] * de_average_mcc + 
                                            weights[1] * de_average_auc + 
                                            weights[2] * de_average_fmeasure + 
                                            weights[3] * de_average_precision + 
                                            weights[4] * de_average_recall + 
                                            weights[5] * de_average_accuracy)

                            if weighted_sum > best_weighted_sum:
                                best_weighted_sum = weighted_sum
                                best_weights = weights

    return best_weighted_sum


def fit(bound, *param):
    de_dataset = []
    for de_i in param:
        de_dataset.append(de_i)

    de_dataset = np.array(de_dataset)
    de_x = de_dataset
    de_y = de_dataset[:, -1]

    de_total_mcc = 0
    de_total_auc = 0
    de_total_fmeasure = 0
    de_total_precision = 0
    de_total_recall = 0
    de_total_accuracy = 0
    for train, test in skf.split(de_x, de_y):
        de_train_x = de_x[train]
        de_train_y = de_y[train]

        for sub_train, sub_test in skf.split(de_train_x, de_train_y):
            de_sub_test_x = de_train_x[sub_test]
            de_sub_test_y = de_sub_test_x[:, -1]
            de_sub_test_x = de_sub_test_x[:, 0:-1]

            de_sub_train_x = de_train_x[sub_train]
            de_sub_defect_x = de_sub_train_x[de_sub_train_x[:, -1] > 0]
            # de_sub_defect_x = de_sub_defect_x[:, 0:-1]
            de_sub_clean_x = de_sub_train_x[de_sub_train_x[:, -1] == 0]
            # print(de_sub_clean_x)
            # de_sub_clean_x = de_sub_clean_x[:, 0:-1]
            de_need_number = len(de_sub_clean_x) - len(de_sub_defect_x)
            pearson_scores = [pearsonr(de_sub_clean_x[:, i], de_sub_clean_x[:, -1])[0] for i in range(de_sub_clean_x.shape[1] - 1)]

            pearson_scores = np.array(pearson_scores)
            # pearson_scores = 1 - (pearson_scores + 1) / 2
            min_score = np.min(pearson_scores)
            max_score = np.max(pearson_scores)
            normalized_scores = (pearson_scores - min_score) / (max_score - min_score)
            pearson_scores = 1 - np.abs(normalized_scores)
            pearson_scores = np.nan_to_num(pearson_scores, nan=1e-8)
            bound = bound * pearson_scores

            de_sub_clean_x[:, 0:-1] = de_sub_clean_x[:, 0:-1] * bound

            de_total_sum = np.sum(de_sub_clean_x[:, 0:-1], axis=1)
            de_sub_clean_x = np.c_[de_sub_clean_x, de_total_sum]
            de_sub_clean_x = de_sub_clean_x[np.argsort(-de_sub_clean_x[:, -1])]
            de_sub_clean_x = np.delete(de_sub_clean_x, [a for a in range(de_need_number)], axis=0)
            de_sub_clean_x = de_sub_clean_x[:, 0:-1]
            de_sub_clean_x[:, 0:-1] = de_sub_clean_x[:, 0:-1] / bound

            de_sub_train_x = np.r_[de_sub_clean_x, de_sub_defect_x]
            de_sub_train_y = de_sub_train_x[:, -1]
            de_sub_train_x = de_sub_train_x[:, 0:-1]
            de_clf = classifier_for_selection[classifier]
            de_clf.fit(de_sub_train_x, de_sub_train_y)
            de_predict_result = de_clf.predict(de_sub_test_x)
            de_mcc = matthews_corrcoef(de_sub_test_y, de_predict_result)
            de_auc = roc_auc_score(de_sub_test_y, de_predict_result)
            de_fmeasure = f1_score(de_sub_test_y, de_predict_result)
            de_precision = precision_score(de_sub_test_y, de_predict_result)
            de_recall = recall_score(de_sub_test_y, de_predict_result)
            de_accuracy = accuracy_score(de_sub_test_y, de_predict_result)

            de_total_mcc += de_mcc
            de_total_auc += de_auc
            de_total_fmeasure += de_fmeasure
            de_total_precision += de_precision
            de_total_recall += de_recall
            de_total_accuracy += de_accuracy

    de_average_mcc = de_total_mcc / 25
    de_average_auc = de_total_auc / 25
    de_average_fmeasure = de_total_fmeasure / 25
    de_average_precision = de_total_precision / 25
    de_average_recall = de_total_recall / 25
    de_average_accuracy = de_total_accuracy / 25

    weighted_sum = objective_function(de_average_mcc, de_average_auc, de_average_fmeasure,
                                        de_average_precision, de_average_recall, de_average_accuracy)
                    
    return -weighted_sum

for iteration in range(10):
    print("test2")
    single = open('.\\results\\original\\'+classifier+'radius based oversampling '+str(iteration)+'.csv', 'w',
                  newline='')
    single_writer = csv.writer(single)
    single_writer.writerow(["inputfile", "mcc", "auc", "balance", "fmeasure", "precision", "pd", "pf"])
    print("test3")
    for inputfile in os.listdir("dataset"):
        print("inputfile:", inputfile)
        start_time = time.asctime(time.localtime(time.time()))
        print("start time:", start_time)
        if inputfile == "ant-1.3.csv" or inputfile == "jedit-3.2.csv" or inputfile == "log4j-1.0.csv" or inputfile == "xalan-2.4.csv" or inputfile == "camel-1.0.csv":
            # continue
            print("test4")
            dataset = pd.read_csv("dataset/" + inputfile)
            dataset = dataset.drop(columns="name")
            dataset = dataset.drop(columns="version")
            dataset = dataset.drop(columns="name.1")
            total_number = len(dataset)
            defect_ratio = len(dataset[dataset["bug"] > 0]) / total_number
            if defect_ratio > 0.45:
                print(inputfile, " defect ratio larger than 0.45")
                continue
            for z in range(total_number):
                if dataset.loc[z, "bug"] > 0:
                    dataset.loc[z, "bug"] = 1
            bound = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                 (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                 (-1, 1), (-1, 1), (-1, 1), (-1, 1)]

        if inputfile == "ar1.csv" or inputfile == "ar3.csv" or inputfile == "ar4.csv" or inputfile == "ar5.csv" or inputfile == "ar6.csv":
            # continue
            print("test4-ar")
            dataset = pd.read_csv("dataset/" + inputfile)
            # dataset = dataset.drop(columns="name")
            # dataset = dataset.drop(columns="version")
            # dataset = dataset.drop(columns="name.1")
            total_number = len(dataset)
            defect_ratio = len(dataset[dataset["defects"] == 'true']) / total_number
            if defect_ratio > 0.45:
                print(inputfile, " defect ratio larger than 0.45")
                continue
            dataset["defects"] = dataset["defects"].replace({"true": 1, "false": 0})

            dataset["defects"] = dataset["defects"].astype(int)

            bound = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]

        if inputfile == "EQ.csv" or inputfile == "JDT.csv" or inputfile == "ML.csv" or inputfile == "PDE.csv":
            # continue
            print("test4-jdt")
            dataset = pd.read_csv("dataset/" + inputfile)
            # dataset = dataset.drop(columns="name")
            # dataset = dataset.drop(columns="version")
            # dataset = dataset.drop(columns="name.1")
            total_number = len(dataset)
            defect_ratio = len(dataset[dataset["class"] == 'buggy']) / total_number
            if defect_ratio > 0.45:
                print(inputfile, " defect ratio larger than 0.45")
                continue
            dataset["class"] = dataset["class"].map({"clean": 0, "buggy": 1})
            dataset["class"] = dataset["class"].astype(int)
            bound = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]

        if inputfile == "CM1.arff" or inputfile == "MW1.arff" or inputfile == "PC1.arff" or inputfile == "PC3.arff" or inputfile == "PC4.arff":
            # continue
            print("test4-arff")
            # dataset = pd.read_csv("dataset/" + inputfile)
            data, meta = arff.loadarff("dataset/" + inputfile)
            dataset = pd.DataFrame(data)
            
            dataset['Defective'] = dataset['Defective'].map({b'Y': 1, b'N': 0})
            total_number = len(dataset)
            defect_ratio = len(dataset[dataset['Defective'] == 1]) / total_number

            bound = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]

        if inputfile == "Zxing.arff"  or inputfile == "Safe.arff":
            # continue
            print("test4-arff")
            # dataset = pd.read_csv("dataset/" + inputfile)
            data, meta = arff.loadarff("dataset/" + inputfile)
            dataset = pd.DataFrame(data)
            dataset['isDefective'] = dataset['isDefective'].str.decode('utf-8')
            dataset['isDefective'] = dataset['isDefective'].map({'clean': 0, 'buggy': 1})
            total_number = len(dataset)
            defect_ratio = len(dataset[dataset['isDefective'] == 1]) / total_number

            bound = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                    (-1, 1), (-1, 1)]

        cols = list(dataset.columns)
        for col in cols:
            column_max = dataset[col].max()
            column_min = dataset[col].min()
            dataset[col] = (dataset[col] - column_min) / (column_max - column_min)

        dataset = np.array(dataset)


        classifier_for_selection = {"knn": neighbors.KNeighborsClassifier(), "svm": svm.SVC(),
                                    "rf": RandomForestClassifier(random_state=0),
                                    "dt": tree.DecisionTreeClassifier(random_state=0),
                                    "lr": LogisticRegression(random_state=0), "nb": GaussianNB()}

        optimal_result = differential_evolution(fit, bound, args=dataset, popsize=10, maxiter=20, mutation=0.3,
                                                recombination=0.9, disp=True)

        optimal_weight = optimal_result.x
        y = dataset[:, -1]
        x = dataset

        total_auc = 0
        total_balance = 0
        total_fmeasure = 0
        total_precision = 0
        total_recall = 0
        total_pf = 0
        total_mcc = 0

        for train, test in skf.split(x, y):
            test_x = x[test]
            test_y = test_x[:, -1]
            test_x = test_x[:, 0:-1]

            train_x = x[train]

            defect_x = train_x[train_x[:, -1] > 0]
            clean_x = train_x[train_x[:, -1] == 0]
            need_number = len(clean_x) - len(defect_x)

            clean_x[:, 0:-1] = optimal_weight * clean_x[:, 0:-1]
            total_sum = np.sum(clean_x[:, 0:-1], axis=1)
            clean_x = np.c_[clean_x, total_sum]
            clean_x = clean_x[np.argsort(-clean_x[:, -1])]
            clean_x = np.delete(clean_x, [a for a in range(need_number)], axis=0)
            clean_x = clean_x[:, 0:-1]
            clean_x[:, 0:-1] = clean_x[:, 0:-1] / optimal_weight
            train_x = np.r_[clean_x, defect_x]
            train_y = train_x[:, -1]
            train_x = train_x[:, 0:-1]
            clf = classifier_for_selection[classifier]
            clf.fit(train_x, train_y)
            predict_result = clf.predict(test_x)

            auc = roc_auc_score(test_y, predict_result)
            total_auc = total_auc + auc

            mcc = matthews_corrcoef(test_y, predict_result)
            total_mcc = total_mcc + mcc

            fmeasure = f1_score(test_y, predict_result)
            total_fmeasure = total_fmeasure + fmeasure

            true_negative, false_positive, false_negative, true_positive = confusion_matrix(test_y,
                                                                                            predict_result).ravel()

            recall = recall_score(test_y, predict_result)
            total_recall = total_recall + recall

            pf = false_positive / (true_negative + false_positive)
            total_pf = total_pf + pf

            balance = 1 - (((0 - pf) ** 2 + (1 - recall) ** 2) / 2) ** 0.5
            total_balance = total_balance + balance

            precision = precision_score(test_y, predict_result)
            total_precision = total_precision + precision

        average_auc = total_auc / 5
        average_balance = total_balance / 5
        average_fmeasure = total_fmeasure / 5
        average_precision = total_precision / 5
        average_recall = total_recall / 5
        average_pf = total_pf / 5
        average_mcc = total_mcc / 5
        single_writer.writerow(
            [inputfile, average_mcc, average_auc, average_balance, average_fmeasure, average_precision, average_recall, average_pf])
        print("final auc: ", average_auc)
        print("end time: ", time.asctime(time.localtime(time.time())))
        print("--------------------------------------")