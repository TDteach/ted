import os
import argparse
import numpy as np
from numpy.random import choice
import pandas as pd
import pickle

from collections import Counter

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Subset
from torchmetrics.functional import pairwise_euclidean_distance

from pyod.models.pca import PCA
from sklearn import metrics
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from umap import UMAP

from classifier_models.resnet import ResNet34
from defense_dataloader import get_dataset

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

# Constants for label types
VT_TEMP_LABEL = "VT"   # Victim with Trigger
NVT_TEMP_LABEL = "NVT" # Non-Victim but with Trigger
NoT_TEMP_LABEL = "NoT" # No Trigger

# Change the label mapping here if needed
label_mapping = {
    "VT": 101,
    "NVT": 102,
    "NoT": 103
}

# Define victim label (if needed to be changed)
VICTIM = 7

# Define sizes for unknown positive and negative samples; change here if needed
UNKNOWN_SIZE_POSITIVE = 400
UNKNOWN_SIZE_NEGATIVE = 200


def setopt():
    # Set environment variables
    os.environ['WANDB_NOTEBOOK_NAME'] = 'TED.ipynb'
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Initialize argparse Namespace
    opt = argparse.Namespace()
    opt.dataset = "cifar10"
    # opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = "cuda" # NOTE: Using CPU if GPU is not having enough memory
    opt.batch_size = 100
    opt.data_root = "../data/"
    opt.target = 1
    # opt.attack_mode = "SSDT"
    opt.attack_mode = "NB"

    # Set input dimensions and channels based on dataset
    if opt.dataset in ["cifar10", "gtsrb"]:
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset in ["imagenet", "pubfig"]:
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3

    # Set class number and defense train size
    opt.class_number = {"cifar10": 10, "gtsrb": 43, "mnist": 10, "imagenet": 100, "pubfig": 83}.get(opt.dataset, 10)
    opt.defense_train_size = {"cifar10": 1000, "gtsrb": 1000, "mnist": 1000, "imagenet": (opt.class_number * 100), "pubfig": (opt.class_number * 100)}.get(opt.dataset, 1000)

    return opt


def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


def fetch_activation(model, device, loader, activations):
    model.eval()
    all_h_label = []
    pred_set = []
    h_batch = {}
    activation_container = {}

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        output = model(images.to(device))
        for key in activations:
            activation_container[key] = []

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        output = model(images.to(device))
        pred_set.append(torch.argmax(output, -1).to(device))

        for key in activations:
            h_batch[key] = activations[key].data.view(images.shape[0], -1)
            for h in h_batch[key]:
                activation_container[key].append(h.to(device))

        for label in labels:
            all_h_label.append(label.to(device))

    for key in activation_container:
        activation_container[key] = torch.stack(activation_container[key])

    all_h_label = torch.stack(all_h_label)
    pred_set = torch.concat(pred_set)

    return all_h_label, activation_container, pred_set

def register_hooker(model, activations):
    hook_handle = []
    # Now, reassign the model's modules to a variable
    net_children = model.modules()

    index = 0
    for _, child in enumerate(net_children):
        if isinstance(child, nn.Conv2d) and child.kernel_size != (1, 1):
            hook_handle.append(child.register_forward_hook(get_activation("Conv2d_"+str(index), activations)))
            index += 1

        if isinstance(child, nn.ReLU):
            hook_handle.append(child.register_forward_hook(get_activation("Relu_"+str(index), activations)))
            index = index + 1

        if isinstance(child, nn.Linear):
            hook_handle.append(child.register_forward_hook(get_activation("Linear_"+str(index), activations)))
            index = index + 1
        # Hook more layers here if needed

    return hook_handle


def gather_activation_into_class(target, h, Test_C):
    h_c_c = [0 for _ in range(Test_C)]
    for c in range(Test_C):
        idxs = (target == c).nonzero(as_tuple=True)[0]
        if len(idxs) == 0:
            continue
        h_c = h[idxs, :]
        h_c_c[c] = h_c
    return h_c_c


def get_dis_sort(item, destinations):
    size = item.size
    item = torch.reshape(item, (1, item.shape[0]))
    new_dis = pairwise_euclidean_distance(item.to("cuda"), destinations.to("cuda"))
    _, indices_individual = torch.sort(new_dis)
    return indices_individual.to("cpu")


def getDefenseRegion(final_prediction, h_defense_activation, processing_label, layer, layer_test_region_individual, Test_C, candidate_):
    r_layer = h_defense_activation
    # initialize the dictionary
    if layer not in layer_test_region_individual:
        layer_test_region_individual[layer] = {}
    layer_test_region_individual[layer][processing_label] = []

    candidate_[layer] = gather_activation_into_class(final_prediction,
                                                    h_defense_activation, Test_C)
   
    if np.ndim(candidate_[layer][processing_label]) == 0:  # Check for 0-d array
        print("No sample in this class")
    else:
        for index, item in enumerate(candidate_[layer][processing_label]):
            ranking_array = get_dis_sort(item, r_layer)[0]
            ranking_array = ranking_array[1:]
            r_ = [final_prediction[i] for i in ranking_array]
            if processing_label in r_:
                itemindex = r_.index(processing_label)
                layer_test_region_individual[layer][processing_label].append(itemindex)

    return layer_test_region_individual


def getLayerRegionDistance(new_prediction, new_activation, new_temp_label,
                           h_defense_prediction, h_defense_activation,
                           layer, layer_test_region_individual, Test_C, candidate_):
    r_layer = h_defense_activation
    labels = torch.unique(new_prediction)
    candidate_ = gather_activation_into_class(new_prediction, new_activation, Test_C)

    if layer not in layer_test_region_individual:
        layer_test_region_individual[layer] = {}
    layer_test_region_individual[layer][new_temp_label] = []

    for processing_label in labels:
        for index, item in enumerate(candidate_[processing_label]):
            ranking_array = get_dis_sort(item, r_layer)[0]
            r_ = [h_defense_prediction[i] for i in ranking_array]
            if processing_label in r_:
                itemindex = r_.index(processing_label)
                layer_test_region_individual[layer][new_temp_label].append(itemindex)

    return layer_test_region_individual



def aggregate_by_all_layers(output_label, topological_representation):
    inputs_container = []
    
    first_key = list(topological_representation.keys())[0]
    labels_container = np.repeat(output_label, len(topological_representation[first_key][output_label]))
    for l in topological_representation.keys():
        temp = []
        for j in range(len(topological_representation[l][output_label])):
            temp.append(topological_representation[l][output_label][j])
        if temp:
            inputs_container.append(np.array(temp))

    return np.array(inputs_container).T, np.array(labels_container)



def detect(model, topological_representation):
    inputs_all_benign = []
    labels_all_benign = []

    inputs_all_unknown = []
    labels_all_unknown = []

    first_key = list(topological_representation.keys())[0]
    class_name = list(topological_representation[first_key])

    for inx in class_name:

        inputs, labels = aggregate_by_all_layers(inx, topological_representation)

        if inx != VT_TEMP_LABEL and inx != NVT_TEMP_LABEL and inx != NoT_TEMP_LABEL:
            inputs_all_benign.append(np.array(inputs))
            labels_all_benign.append(np.array(labels))
        else:
            inputs_all_unknown.append(np.array(inputs))
            labels_all_unknown.append(np.array(labels))

    inputs_all_benign = np.concatenate(inputs_all_benign)
    labels_all_benign = np.concatenate(labels_all_benign)

    inputs_all_unknown = np.concatenate(inputs_all_unknown)
    labels_all_unknown = np.concatenate(labels_all_unknown)

    pca_t = sklearn_PCA(n_components=2)
    pca_fit = pca_t.fit(inputs_all_benign)

    benign_trajectories = pca_fit.transform(inputs_all_benign)
    trajectories = pca_fit.transform(np.concatenate((inputs_all_unknown, inputs_all_benign), axis=0))

    df_classes = pd.DataFrame(np.concatenate((labels_all_unknown, labels_all_benign), axis=0))

    fig_ = px.scatter(
        trajectories, x=0, y=1, color=df_classes[0].astype(str), labels={'color': 'digit'},
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )

    fig_.show()

    pca = PCA(contamination=0.01, n_components='mle')
    pca.fit(inputs_all_benign)

    y_train_pred = pca.labels_
    y_train_scores = pca.decision_scores_
    y_train_scores = pca.decision_function(inputs_all_benign)
    y_train_pred = pca.predict(inputs_all_benign)

    y_test_scores = pca.decision_function(inputs_all_unknown)
    y_test_pred = pca.predict(inputs_all_unknown)
    prediction_mask = (y_test_pred == 1)
    prediction_labels = labels_all_unknown[prediction_mask]
    label_counts = Counter(prediction_labels)

    for label, count in label_counts.items():
        print(f'Label {label}: {count}')

    fpr, tpr, thresholds = metrics.roc_curve((labels_all_unknown == VT_TEMP_LABEL).astype(int), y_test_scores, pos_label=1)
    print("AUC:", metrics.auc(fpr, tpr))

    tn, fp, fn, tp = confusion_matrix((labels_all_unknown == VT_TEMP_LABEL).astype(int), y_test_pred).ravel()
    print("TPR:", tp / (tp + fn))
    print("True Positives (TP):", tp)
    print("False Positives (FP):", fp)
    print("True Negatives (TN):", tn)
    print("False Negatives (FN):", fn)


# Function to create backdoor inputs
def create_bd(trigger_mask, trigger_pattern, inputs):
    # bd_inputs = inputs + (patterns - inputs) * masks_output
    bd_inputs = inputs + (trigger_pattern - inputs) * trigger_mask
    return bd_inputs

# Function to create targets
def create_targets(targets, opt, label):
    new_targets = torch.ones_like(targets) * label
    return new_targets.to(opt.device)

# Custom dataset class
class CustomDataset(data.Dataset):
    def __init__(self, data, labels):
        super(CustomDataset, self).__init__()
        self.images = data
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        return img, label


def build_loader(opt, model):
    # Define global constant
    DEFENSE_TRAIN_SIZE = opt.defense_train_size

    # Set up dataset loaders
    testset = get_dataset(opt, train=True)

    # Indices of the whole dataset
    indices = np.arange(len(testset))

    # Split indices into benign_unknown_indices and defense_subset_indices
    benign_unknown_indices, defense_subset_indices = train_test_split(
        indices, test_size=0.1, random_state=42)

    # Create subsets for benign_unknown and defense
    benign_unknown_subset = Subset(testset, benign_unknown_indices)
    defense_subset = Subset(testset, defense_subset_indices)

    # DataLoader for benign_unknown_subset
    benign_unknown_loader = data.DataLoader(
        benign_unknown_subset, 
        batch_size=opt.batch_size, 
        num_workers=0, 
        shuffle=True)

    # DataLoader for defense_subset
    defense_loader = data.DataLoader(
        defense_subset, 
        batch_size=opt.batch_size, 
        num_workers=0, 
        shuffle=True)

    # Create defense dataset for TED training with Defense Size
    h_benign_preds = []
    h_benign_ori_labels = []

    # Predict labels using the model and collect predictions and original labels
    with torch.no_grad():
        for inputs, labels in defense_loader:
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            h_benign_preds.extend(preds.cpu().numpy())
            h_benign_ori_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    h_benign_preds = np.array(h_benign_preds)
    h_benign_ori_labels = np.array(h_benign_ori_labels)

    # Create a mask for correctly predicted (benign) samples
    benign_mask = h_benign_ori_labels == h_benign_preds

    print('on defense loader ACC', np.sum(benign_mask)/len(benign_mask) * 100, '%')

    # Select indices of benign samples
    benign_indices = defense_subset_indices[benign_mask]

    # If the number of benign samples exceeds DEFENSE_TRAIN_SIZE, randomly select DEFENSE_TRAIN_SIZE samples
    if len(benign_indices) > DEFENSE_TRAIN_SIZE:
        benign_indices = np.random.choice(benign_indices, DEFENSE_TRAIN_SIZE, replace=False)

    # Create a new defense subset and DataLoader
    defense_subset = Subset(testset, benign_indices)
    defense_loader = data.DataLoader(defense_subset, batch_size=opt.batch_size, num_workers=0, shuffle=True)

    # Initialize counters for the different types of samples
    vt_count = nvt_count = NoT_count = 0

    # Initialize lists to store samples for different types
    temp_bd_inputs_set = []    # Store inputs for VT samples
    temp_bd_labels_set = []    # Store labels for VT samples
    temp_bd_pred_set = []      # Store predictions for VT samples

    temp_cleanT_inputs_set = []  # Store inputs for NVT samples
    temp_cleanT_labels_set = []  # Store labels for NVT samples
    temp_cleanT_pred_set = []    # Store predictions for NVT samples

    trigger_path = 'epoch_99.pth'
    trigger_dict = torch.load(trigger_path, map_location=opt.device)
    trigger_mask = trigger_dict['mask']
    trigger_pattern = trigger_dict['trigger']

    with torch.no_grad():
        # Main loop for generating VT and NVT sets
        while vt_count < UNKNOWN_SIZE_POSITIVE or nvt_count < UNKNOWN_SIZE_NEGATIVE:
            for batch_idx, (inputs, labels) in enumerate(benign_unknown_loader):
                inputs, labels = inputs.to(opt.device), labels.to(opt.device)
                inputs_triggered = create_bd(trigger_mask, trigger_pattern, inputs)
                preds_bd = torch.argmax(model(inputs_triggered), 1)
                victim_indices = (labels == VICTIM)
                non_victim_indices = (labels != VICTIM)

                # VT samples processing
                if vt_count < UNKNOWN_SIZE_POSITIVE:
                    label_value = label_mapping[VT_TEMP_LABEL]
                    targets_victim_bd = create_targets(labels, opt, label_value)
                    correct_preds_indices = (preds_bd == opt.target)
                    final_indices = victim_indices & correct_preds_indices
                    temp_bd_inputs_set.append(inputs_triggered[final_indices])
                    temp_bd_labels_set.append(targets_victim_bd[final_indices].to('cpu'))
                    temp_bd_pred_set.append(preds_bd[final_indices].to('cpu'))
                    vt_count += final_indices.sum().item()

                # NVT samples processing
                if nvt_count < UNKNOWN_SIZE_NEGATIVE:
                    label_value = label_mapping[NVT_TEMP_LABEL]
                    targets_clean = create_targets(labels, opt, label_value)
                    temp_cleanT_inputs_set.append(inputs_triggered[non_victim_indices])
                    temp_cleanT_labels_set.append(targets_clean[non_victim_indices].to('cpu'))
                    temp_cleanT_pred_set.append(preds_bd[non_victim_indices].to('cpu'))
                    nvt_count += non_victim_indices.sum().item()

    # Concatenate and trim sets to required size
    bd_inputs_set = torch.cat(temp_bd_inputs_set)[:UNKNOWN_SIZE_POSITIVE]
    bd_labels_set = np.hstack(temp_bd_labels_set)[:UNKNOWN_SIZE_POSITIVE]
    bd_pred_set = np.hstack(temp_bd_pred_set)[:UNKNOWN_SIZE_POSITIVE]

    cleanT_inputs_set = torch.cat(temp_cleanT_inputs_set)[:UNKNOWN_SIZE_NEGATIVE]
    cleanT_labels_set = np.hstack(temp_cleanT_labels_set)[:UNKNOWN_SIZE_NEGATIVE]
    cleanT_pred_set = np.hstack(temp_cleanT_pred_set)[:UNKNOWN_SIZE_NEGATIVE]

    # Initialize lists for benign set
    benign_real_labels_set = []
    benign_inputs_set = []
    benign_labels_set = []
    benign_pred_set = []


    with torch.no_grad():
        # Process NoT samples
        for batch_idx, (inputs, labels) in enumerate(benign_unknown_loader):
            _inputs, _labels= inputs.to(opt.device), labels.to(opt.device)
            bs = _inputs.shape[0]
            NoT_count += bs
            label_value = label_mapping[NoT_TEMP_LABEL]
            targets_benign = torch.ones_like(labels) * label_value

            # NoT samples processing
            if NoT_count <= UNKNOWN_SIZE_NEGATIVE:
                benign_real_labels_set.append(_labels.to('cpu'))
                benign_inputs_set.append(_inputs.clone().detach())
                benign_labels_set.append(targets_benign.to('cpu'))
                benign_pred_set.append(torch.argmax(model(_inputs), 1).to('cpu'))
            elif NoT_count > UNKNOWN_SIZE_NEGATIVE:
                break

    # Concatenate benign sets
    benign_inputs_set = torch.concatenate(benign_inputs_set)
    benign_labels_set = np.concatenate(benign_labels_set)
    benign_pred_set = np.concatenate(benign_pred_set)

    # Data loaders for different sets
    # VT Loader
    bd_set = CustomDataset(data=bd_inputs_set, labels=bd_labels_set)
    bd_loader = torch.utils.data.DataLoader(bd_set, batch_size=opt.batch_size, num_workers=0, shuffle=True)
    print("VT set size:", len(bd_loader))
    del bd_inputs_set, bd_labels_set, bd_pred_set

    # NVT Loader
    cleanT_set = CustomDataset(data=cleanT_inputs_set, labels=cleanT_labels_set)
    cleanT_loader = torch.utils.data.DataLoader(cleanT_set, batch_size=opt.batch_size, num_workers=0, shuffle=True)
    print("NVT set size:", len(cleanT_loader))
    del cleanT_inputs_set, cleanT_labels_set, cleanT_pred_set

    # NoT Loader
    benign_set = CustomDataset(data=benign_inputs_set, labels=benign_labels_set)
    benign_loader = torch.utils.data.DataLoader(benign_set, batch_size=opt.batch_size, num_workers=0, shuffle=True)
    print("NoT set size:", len(benign_loader))
    del benign_inputs_set, benign_labels_set, benign_pred_set

    return bd_loader, benign_loader, cleanT_loader, defense_loader



def aggregate_by_all_layers(output_label):
    inputs_container = []
    
    first_key = list(topological_representation.keys())[0]
    labels_container = np.repeat(output_label, len(topological_representation[first_key][output_label]))
    for l in topological_representation.keys():
        temp = []
        for j in range(len(topological_representation[l][output_label])):
            temp.append(topological_representation[l][output_label][j])
        if temp:
            inputs_container.append(np.array(temp))

    return np.array(inputs_container).T, np.array(labels_container)


def detect_representation(topological_representation):
    inputs_all_benign = []
    labels_all_benign = []

    inputs_all_unknown = []
    labels_all_unknown = []

    first_key = list(topological_representation.keys())[0]
    class_name = list(topological_representation[first_key])

    for inx in class_name:

        inputs, labels = aggregate_by_all_layers(output_label=inx)

        if inx != VT_TEMP_LABEL and inx != NVT_TEMP_LABEL and inx != NoT_TEMP_LABEL:
            inputs_all_benign.append(np.array(inputs))
            labels_all_benign.append(np.array(labels))
        else:
            inputs_all_unknown.append(np.array(inputs))
            labels_all_unknown.append(np.array(labels))

    inputs_all_benign = np.concatenate(inputs_all_benign)
    labels_all_benign = np.concatenate(labels_all_benign)

    inputs_all_unknown = np.concatenate(inputs_all_unknown)
    labels_all_unknown = np.concatenate(labels_all_unknown)

    pca_t = sklearn_PCA(n_components=2)
    pca_fit = pca_t.fit(inputs_all_benign)

    benign_trajectories = pca_fit.transform(inputs_all_benign)
    trajectories = pca_fit.transform(np.concatenate((inputs_all_unknown, inputs_all_benign), axis=0))

    df_classes = pd.DataFrame(np.concatenate((labels_all_unknown, labels_all_benign), axis=0))

    fig_ = px.scatter(
        trajectories, x=0, y=1, color=df_classes[0].astype(str), labels={'color': 'digit'},
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )

    fig_.show()

    pca = PCA(contamination=0.01, n_components='mle')
    pca.fit(inputs_all_benign)

    y_train_pred = pca.labels_
    y_train_scores = pca.decision_scores_
    y_train_scores = pca.decision_function(inputs_all_benign)
    y_train_pred = pca.predict(inputs_all_benign)

    y_test_scores = pca.decision_function(inputs_all_unknown)
    y_test_pred = pca.predict(inputs_all_unknown)
    prediction_mask = (y_test_pred == 1)
    prediction_labels = labels_all_unknown[prediction_mask]
    label_counts = Counter(prediction_labels)

    for label, count in label_counts.items():
        print(f'Label {label}: {count}')

    fpr, tpr, thresholds = metrics.roc_curve((labels_all_unknown == VT_TEMP_LABEL).astype(int), y_test_scores, pos_label=1)
    print("AUC:", metrics.auc(fpr, tpr))

    tn, fp, fn, tp = confusion_matrix((labels_all_unknown == VT_TEMP_LABEL).astype(int), y_test_pred).ravel()
    print("TPR:", tp / (tp + fn))
    print("True Positives (TP):", tp)
    print("False Positives (FP):", fp)
    print("True Negatives (TN):", tn)
    print("False Negatives (FN):", fn)

    return inputs_all_benign, labels_all_benign, inputs_all_unknown, labels_all_unknown


def show_fig(inputs_all_unknown, labels_all_unknown):
    inputs = inputs_all_unknown
    labels = labels_all_unknown

    inputs_flatten = inputs.flatten()
    colors_ = []
    index_ = []
    layer_ = []

    class_labels = {'VT': 'VT', 'NVT': 'NVT', 'NoT': 'NoT'}

    for i, input in enumerate(inputs):
        colors_.extend([str(labels[i])] * len(input))
        index_.extend([i] * len(input))
        layer_.extend(range(len(input)))

    df = pd.DataFrame(dict(
        x=layer_,
        y=inputs_flatten,
        z=colors_,
        i=index_
    ))

    df = df.sort_values(by=['z', 'i', 'x'])

    fig = go.Figure()
    line_color_map = {'VT': '#EA6253', 'NoT': '#4668d8', 'NVT': '#fd9300'}

    fill_color_map = {
        'VT': '#EA6253',
        'NoT': '#4668d8',
        'NVT': '#fd9300'
    }

    for label in ['NoT', 'NVT', 'VT']:
        data = df[df['z'] == label]
        fig.add_trace(go.Box(
            x=data['x'],
            y=data['y'],
            name=class_labels[label],
            fillcolor=fill_color_map[label],
            marker=dict(
                color=line_color_map[label],
            ),
            jitter=0.01,
            whiskerwidth=0.5,
            boxpoints='all',
            marker_size=2,
        ))

    y_max = df['y'].max()

    fig.update_layout(
        title="Topology Persistence Diagram",
        xaxis_title="Layer",
        yaxis_title="Nearest Rank of Same Class",
        width=1000,
        height=500,
        font=dict(
            family="Calibri",
            size=20,
            color="black"
        ),

    )

    fig.update_yaxes(range=[0, y_max])
    save_format = "pdf"
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })


    fig.update_layout(
        autosize=False,
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    color="Black",
                    width=2,
                )
            )
        ]
    )

    # Show all integers on x-axis
    fig.update_xaxes(showticklabels=True, showgrid=False, zeroline=False, dtick=1)

    fig.update_yaxes(autorange=True, showticklabels=True,
                    showgrid=False, zeroline=False)
    # Hide x and y axis labels and lines
    fig.update_xaxes(showticklabels=True, zeroline=False, visible=True)
    fig.update_yaxes(showticklabels=True, zeroline=False, visible=True)
    fig.update_layout(showlegend=True, title=None)
    fig.update_layout(legend_title_text='')
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bordercolor="Black",
            borderwidth=0
        ),

        boxmode='group'  # group together boxes of the different traces for each value of x

    )
    save_path = f"{opt.dataset}_k_{opt.target}_{opt.attack_mode}_topology_persistence_diagram.pdf"
    pio.write_image(fig, save_path, format=save_format)

    fig.show()


def main():

    opt = setopt()

    model = ResNet34()
    model_path = './checkpoints/cifar10/ckpt.pth'
    assert os.path.exists(model_path), 'Error: no checkpoint found! '+model_path
    ckpt = torch.load(model_path, map_location=opt.device)
    model.load_state_dict(ckpt['net'])
    best_acc = ckpt['acc']
    start_epoch = ckpt['epoch']
    print(best_acc, 'obtained in', start_epoch, 'training')
    model.to(opt.device)
    model.eval()

    bd_loader, benign_loader, cleanT_loader, defense_loader = build_loader(opt, model)
    
    activations = {}
    hook_handle = register_hooker(model, activations)

    h_bd_ori_labels, h_bd_activations, h_bd_preds = fetch_activation(model, opt.device, bd_loader, activations)
    h_benign_ori_labels, h_benign_activations, h_benign_preds = fetch_activation(model, opt.device, benign_loader, activations)
    h_cleanT_ori_labels, h_cleanT_activations, h_cleanT_preds = fetch_activation(model, opt.device, cleanT_loader, activations)
    h_defense_ori_labels, h_defense_activations, h_defense_preds = fetch_activation(model, opt.device, defense_loader, activations)

    for handle in hook_handle:
        handle.remove()

    class_names = np.unique(h_defense_ori_labels.cpu().numpy())

    candidate_ = {}
    topological_representation = {}
    for index, label in enumerate(class_names):
        for layer in h_defense_activations:
                topological_representation = getDefenseRegion(
                        final_prediction=h_defense_preds,
                        h_defense_activation=h_defense_activations[layer],
                        processing_label=label,
                        layer=layer,
                        layer_test_region_individual=topological_representation,
                        Test_C = opt.class_number+3,
                        candidate_ = candidate_
                )
                topo_rep_array = np.array(topological_representation[layer][label])
                print(f"Topological Representation Label [{label}] & layer [{layer}]: {topo_rep_array}")
                print(f"Mean: {np.mean(topo_rep_array)}\n")

    for layer_ in h_bd_activations:
        topological_representation = getLayerRegionDistance(
                new_prediction=h_bd_preds,
                new_activation=h_bd_activations[layer_],
                new_temp_label=VT_TEMP_LABEL,
                h_defense_prediction=h_defense_preds, 
                h_defense_activation=h_defense_activations[layer_],
                layer=layer_,
                layer_test_region_individual=topological_representation,
                Test_C = opt.class_number+3,
                candidate_ = candidate_
        )
        topo_rep_array_vt = np.array(topological_representation[layer_][VT_TEMP_LABEL])
        print(f"Topological Representation Label [{VT_TEMP_LABEL}] & layer [{layer_}]: {topo_rep_array_vt}")
        print(f"Mean: {np.mean(topo_rep_array_vt)}\n")

    for layer_ in h_benign_activations:
        topological_representation = getLayerRegionDistance(
                new_prediction=h_benign_preds,
                new_activation=h_benign_activations[layer_],
                new_temp_label=NoT_TEMP_LABEL,
                h_defense_prediction=h_defense_preds,
                h_defense_activation=h_defense_activations[layer_],
                layer=layer_,
                layer_test_region_individual=topological_representation,
                Test_C = opt.class_number+3,
                candidate_ = candidate_
        )
        topo_rep_array_not = np.array(topological_representation[layer_][NoT_TEMP_LABEL])
        print(f"Topological Representation Label [{NoT_TEMP_LABEL}] - layer [{layer_}]: {topo_rep_array_not}")
        print(f"Mean: {np.mean(topo_rep_array_not)}\n")

    for layer_ in h_cleanT_activations:
        topological_representation = getLayerRegionDistance(
                new_prediction=h_cleanT_preds,
                new_activation=h_cleanT_activations[layer_],
                new_temp_label=NVT_TEMP_LABEL,
                h_defense_prediction=h_defense_preds,
                h_defense_activation=h_defense_activations[layer_],
                layer=layer_,
                layer_test_region_individual=topological_representation,
                Test_C = opt.class_number+3,
                candidate_ = candidate_
        )
        topo_rep_array_nvt = np.array(topological_representation[layer_][NVT_TEMP_LABEL])
        print(f"Topological Representation [{NVT_TEMP_LABEL}] - layer [{layer_}]: {topo_rep_array_nvt}")
        print(f"Mean: {np.mean(topo_rep_array_nvt)}\n")


    file_name = f"{opt.dataset}_k_{opt.target}_{opt.attack_mode}.pkl"
    file_path = os.path.join(file_name)

    with open(file_path, 'wb') as file:
        pickle.dump(topological_representation, file)

    with open(file_path, 'rb') as file:
        topological_representation = pickle.load(file)

    inputs_all_benign, labels_all_benign, inputs_all_unknown, labels_all_unknown = detect_representation(topological_representation)
    show_fig(inputs_all_unknown, labels_all_unknown)



if __name__=='__main__':
    main()