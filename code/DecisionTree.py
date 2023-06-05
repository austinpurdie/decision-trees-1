import math
import pandas as pd

class DecisionTreeNode:
    def __init__(self, attribute, value, type, parent):
        self.attribute = attribute # which attribute the node is giving instructions for (e.g., outlook, humidity)
        self.value = value # the value of the attribute the node is giving instructions for (e.g., the outlook is //sunny//)
        self.type = type # whether the node is a root, branch, or leaf
        self.parent = parent # the node's parent node
        self.child = [] # a list of nodes that this node points to in the tree
        self.depth = 0 # the depth of the tree is maintained at the level of each node for convenience of access
        self.most_common = None # the most common label at each particular node is recorded

class DecisionTree:
    def __init__(self, max_depth):
        self.root = DecisionTreeNode(None, None, "root", None)
        self.max_depth = max_depth
        self.node_count = 0

    def add_branch(self, parent, attribute, value):
        new_node = DecisionTreeNode(attribute, value, "branch", parent)
        new_node.parent.child.append(new_node)
        new_node.depth = new_node.parent.depth + 1
        self.node_count += 1
        return new_node

    def add_leaf(self, parent, value):
        new_leaf = DecisionTreeNode(None, value, 'leaf', parent)
        new_leaf.parent.child.append(new_leaf)

def label_proportions(labels):
    unique_labels = []
    unique_proportions = []
    for x in labels:
        if x not in unique_labels:
            unique_labels.append(x)
    for y in unique_labels:
        unique_proportions.append(labels.count(y)/len(labels))
    return unique_proportions

def entropy(labels):
    proportions = label_proportions(labels)
    transformed_proportions = [ -x*math.log(x, 2) for x in proportions]
    entropy = sum(transformed_proportions)
    return entropy

def get_unique_values(data):
    unique_values_dict = {}
    for x in list(data.columns):
        values = []
        for y in list(data[x]):
            if y not in values:
                values.append(y)
        unique_values_dict[x] = values
    return unique_values_dict

def select_attribute(data, target, method):
    base_entropy = entropy(list(data[target]))
    information_gain_dict = {}
    total_records = data.shape[0]
    unique_values_dict = get_unique_values(data)
    non_target_features = list(data.columns)
    non_target_features.remove(target)
    for x in non_target_features:
        feature_unique_values = unique_values_dict[x]
        information_gain = base_entropy
        for y in feature_unique_values:
            filtered_data_by_feature = data[data[x] == y]
            filtered_total_records = filtered_data_by_feature.shape[0]
            information_gain -= entropy(list(filtered_data_by_feature[target]))*(filtered_total_records/total_records)
        information_gain_dict[x] = information_gain
    selected_attribute = max(information_gain_dict, key = information_gain_dict.get)
    return [selected_attribute, unique_values_dict[selected_attribute]]

def add_leaf_condition(data, target, node, max_depth):
    
    if len(data.columns) == 1:
        return True
    elif node.depth == max_depth and max_depth > 0:
        return True
    elif len(get_unique_values(data)[target]) == 1:
        return True
    else:
        return False

def most_common(list):
    value_counts = {}
    for y in list:
        if y not in value_counts.keys():
            value_counts[y] = 1
        else:
            value_counts[y] += 1
    return max(value_counts, key = value_counts.get)

def build_tree(tree, data, target, parent):
    next_branch_attribute = select_attribute(data, target, method)
    for i in next_branch_attribute[1]:
        new_node = tree.add_branch(parent, next_branch_attribute[0], i)
        filter_attribute = str(new_node.attribute)
        filter_value = str(new_node.value)
        filtered_data = data[data[filter_attribute] == filter_value]
        filtered_data = filtered_data.loc[:, filtered_data.columns != new_node.attribute]
        new_node.most_common = most_common(list(filtered_data.loc[:, target]))
        if add_leaf_condition(filtered_data, target, new_node, tree.max_depth):
            tree.add_leaf(new_node, new_node.most_common)
        else:
            build_tree(tree, filtered_data, target, new_node)

def get_tree_predictions(tree, data):
    predictions = []
    for i in range(len(data)):
        current_node = tree.root
        unseen_trigger = False
        while current_node.child[0].type != 'leaf':
            current_node_children = current_node.child
            current_attribute = current_node.child[0].attribute
            data_row_value = data.loc[i, current_attribute]

            children_values = []
            for k in current_node_children:
                children_values.append(k.value)
            if data_row_value not in children_values:
                predictions.append(current_node.most_common)
                unseen_trigger = True
                break

            for j in current_node_children:
                if j.value == data_row_value:
                    current_node = j
                    break

        if unseen_trigger == False:
            predictions.append(j.child[0].value)
    return predictions

def get_test_accuracy(data, target, predictions):
    num_of_rows = len(predictions)
    actual = list(data.loc[:, target])
    num_correct = 0
    for i in range(num_of_rows):
        if actual[i] == predictions[i]:
            num_correct += 1
    return 100*num_correct/num_of_rows