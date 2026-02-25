#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import json
import os
import time
import xlrd
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from scipy.stats import skew, kurtosis
import sys
import csv
import copy
from tqdm import tqdm  # Explicitly import tqdm for the progress bars
import random
import shutil

# --- THE REQUIRED ET-BERT DEPENDENCIES ---
from data_process import dataset_generation
from data_process import data_preprocess
from data_process import open_dataset_deal

# --- YOUR CUSTOM FOLDER SETUP ---
_category = 24  # Adjusted to your 24 benign/malware files
dataset_dir = "./"  # The path to save the final TSV datasets (saves in current folder)

# Points to your local Data folder and an Output folder for the numpy arrays
pcap_path, dataset_save_path, samples, features, dataset_level = "./Data/", "./Output/", [5000], ["payload"], "packet"


def dataset_extract(model):
    X_dataset = {}
    Y_dataset = {}

    try:
        if os.listdir(dataset_save_path + "dataset\\"):
            print("Reading dataset from %s ..." % (dataset_save_path + "dataset\\"))

            x_payload_train, x_payload_test, x_payload_valid, \
                y_train, y_test, y_valid = \
                np.load(dataset_save_path + "dataset\\x_datagram_train.npy", allow_pickle=True), np.load(
                    dataset_save_path + "dataset\\x_datagram_test.npy", allow_pickle=True), np.load(
                    dataset_save_path + "dataset\\x_datagram_valid.npy", allow_pickle=True), \
                    np.load(dataset_save_path + "dataset\\y_train.npy", allow_pickle=True), np.load(
                    dataset_save_path + "dataset\\y_test.npy", allow_pickle=True), np.load(
                    dataset_save_path + "dataset\\y_valid.npy", allow_pickle=True)

            X_dataset, Y_dataset = models_deal(model, X_dataset, Y_dataset,
                                               x_payload_train, x_payload_test,
                                               x_payload_valid,
                                               y_train, y_test, y_valid)

            return X_dataset, Y_dataset
    except Exception as e:
        print(e)
        print("Dataset directory %s not exist.\nBegin to obtain new dataset." % (dataset_save_path + "dataset\\"))

    # NOTE: This step calls an external script. It may take a moment before the progress bars appear!
    print("Generating dataset arrays from PCAP files (This may take a few minutes)...")
    X, Y = dataset_generation.generation(pcap_path, samples, features, splitcap=False,
                                         dataset_save_path=dataset_save_path, dataset_level=dataset_level)

    dataset_statistic = [0] * _category

    X_payload = []
    Y_all = []

    # Added tqdm to label extraction
    for app_label in tqdm(Y, desc="Flattening labels"):
        for label in app_label:
            Y_all.append(int(label))

    for label_id in range(_category):
        for label in Y_all:
            if label == label_id:
                dataset_statistic[label_id] += 1

    print("\ncategory flow")
    for index in range(len(dataset_statistic)):
        print("%s\t%d" % (index, dataset_statistic[index]))
    print("all\t%d" % (sum(dataset_statistic)))

    for i in range(len(features)):
        if features[i] == "payload":
            # Added tqdm to payload extraction
            for index_label in tqdm(range(len(X[0])), desc="Extracting payloads from arrays"):
                for index_sample in range(len(X[0][index_label])):
                    X_payload.append(X[0][index_label][index_sample])

    print("\nSplitting data into Train/Test/Validation sets...")
    split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=41)
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

    x_payload = np.array(X_payload)
    dataset_label = np.array(Y_all)

    x_payload_train = []
    y_train = []

    x_payload_valid = []
    y_valid = []

    x_payload_test = []
    y_test = []

    for train_index, test_index in split_1.split(x_payload, dataset_label):
        x_payload_train, y_train = x_payload[train_index], dataset_label[train_index]
        x_payload_test, y_test = x_payload[test_index], dataset_label[test_index]
    for test_index, valid_index in split_2.split(x_payload_test, y_test):
        x_payload_valid, y_valid = x_payload_test[valid_index], y_test[valid_index]
        x_payload_test, y_test = x_payload_test[test_index], y_test[test_index]

    if not os.path.exists(dataset_save_path + "dataset\\"):
        os.mkdir(dataset_save_path + "dataset\\")

    print("Saving numpy arrays to Output folder...")
    output_x_payload_train = os.path.join(dataset_save_path + "dataset\\", 'x_datagram_train.npy')
    output_x_payload_test = os.path.join(dataset_save_path + "dataset\\", 'x_datagram_test.npy')
    output_x_payload_valid = os.path.join(dataset_save_path + "dataset\\", 'x_datagram_valid.npy')

    output_y_train = os.path.join(dataset_save_path + "dataset\\", 'y_train.npy')
    output_y_test = os.path.join(dataset_save_path + "dataset\\", 'y_test.npy')
    output_y_valid = os.path.join(dataset_save_path + "dataset\\", 'y_valid.npy')

    np.save(output_x_payload_train, x_payload_train)
    np.save(output_x_payload_test, x_payload_test)
    np.save(output_x_payload_valid, x_payload_valid)

    np.save(output_y_train, y_train)
    np.save(output_y_test, y_test)
    np.save(output_y_valid, y_valid)

    X_dataset, Y_dataset = models_deal(model, X_dataset, Y_dataset,
                                       x_payload_train, x_payload_test, x_payload_valid,
                                       y_train, y_test, y_valid)

    return X_dataset, Y_dataset


def models_deal(model, X_dataset, Y_dataset, x_payload_train, x_payload_test, x_payload_valid, y_train, y_test,
                y_valid):
    for index in range(len(model)):
        print("\nBegin to model %s dealing..." % model[index])
        x_train_dataset = []
        x_test_dataset = []
        x_valid_dataset = []

        if model[index] == "pre-train":
            save_dir = dataset_dir
            write_dataset_tsv(x_payload_train, y_train, save_dir, "train")
            write_dataset_tsv(x_payload_test, y_test, save_dir, "test")
            write_dataset_tsv(x_payload_valid, y_valid, save_dir, "valid")
            print("Finish generating pre-train's datagram dataset.\nPlease check in %s" % save_dir)
            unlabel_data(dataset_dir + "test_dataset.tsv")

        X_dataset[model[index]] = {"train": [], "valid": [], "test": []}
        Y_dataset[model[index]] = {"train": [], "valid": [], "test": []}

        X_dataset[model[index]]["train"], Y_dataset[model[index]]["train"] = x_train_dataset, y_train
        X_dataset[model[index]]["valid"], Y_dataset[model[index]]["valid"] = x_valid_dataset, y_valid
        X_dataset[model[index]]["test"], Y_dataset[model[index]]["test"] = x_test_dataset, y_test

    return X_dataset, Y_dataset


def write_dataset_tsv(data, label, file_dir, type):
    dataset_file = [["label", "text_a"]]
    # Added tqdm to the TSV file writing process
    for index in tqdm(range(len(label)), desc=f"Writing {type}.tsv"):
        dataset_file.append([label[index], data[index]])

    with open(file_dir + type + "_dataset.tsv", 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerows(dataset_file)
    return 0


def unlabel_data(label_data):
    nolabel_data = ""
    with open(label_data, newline='') as f:
        # Convert csv reader to a list so tqdm knows the total length
        data = list(csv.reader(f, delimiter='\t'))

        # Added tqdm to the unlabeling process
        for row in tqdm(data, desc="Generating nolabel_test_dataset.tsv"):
            if len(row) > 1:
                nolabel_data += row[1] + '\n'

    nolabel_file = label_data.replace("test_dataset", "nolabel_test_dataset")
    with open(nolabel_file, 'w', newline='') as f:
        f.write(nolabel_data)
    return 0


def cut_byte(obj, sec):
    result = [obj[i:i + sec] for i in range(0, len(obj), sec)]
    remanent_count = len(result[0]) % 2
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i + sec + remanent_count] for i in range(0, len(obj), sec + remanent_count)]
    return result


def pickle_save_data(path_file, data):
    with open(path_file, "wb") as f:
        pickle.dump(data, f)
    return 0


def count_label_number(samples):
    new_samples = samples * _category

    if 'splitcap' not in pcap_path:
        dataset_length, labels = open_dataset_deal.statistic_dataset_sample_count(pcap_path + 'splitcap\\')
    else:
        dataset_length, labels = open_dataset_deal.statistic_dataset_sample_count(pcap_path)

    for index in range(len(dataset_length)):
        if dataset_length[index] < samples[0]:
            print("label %s has less sample's number than defined samples %d" % (labels[index], samples[0]))
            new_samples[index] = dataset_length[index]
    return new_samples


if __name__ == '__main__':
    open_dataset_not_pcap = 0

    if open_dataset_not_pcap:
        for p, d, f in os.walk(pcap_path):
            for file in f:
                target_file = file.replace('.', '_new.')
                open_dataset_deal.file_2_pcap(p + "\\" + file, p + "\\" + target_file)
                if '_new.pcap' not in file:
                    os.remove(p + "\\" + file)

    file2dir = 0
    if file2dir:
        open_dataset_deal.dataset_file2dir(pcap_path)

    splitcap_finish = 0
    # --- DELETE THE OLD splitcap_finish BLOCK AND REPLACE WITH THIS ---

    print("\nCounting available files in your folders...")
    dynamic_samples = []

    # Sort the folders alphabetically to match ET-BERT's internal logic
    subfolders = sorted([f for f in os.listdir(pcap_path) if os.path.isdir(os.path.join(pcap_path, f))])

    for folder in subfolders:
        folder_path = os.path.join(pcap_path, folder)
        file_count = len(os.listdir(folder_path))

        # Request 5000 files, but safely cap it to the actual file count if it's smaller
        safe_sample_size = min(5000, file_count)
        dynamic_samples.append(safe_sample_size)
        print(f"Folder '{folder}' has {file_count} files -> Using {safe_sample_size} samples.")

    # Override the global samples variable with our safe list
    samples = dynamic_samples

    train_model = ["pre-train"]
    ml_experiment = 0

    dataset_extract(train_model)
