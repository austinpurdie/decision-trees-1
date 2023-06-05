import DecisionTree as dt
import pandas as pd

tree = dt.DecisionTree(0)

data = pd.read_csv("../data/votes.csv")

train_data = pd.read_csv("../data/train_data.csv")
test_data = pd.read_csv("../data/test_data.csv")

dt.build_tree(tree, train_data, "votes", tree.root)

train_predictions = dt.get_tree_predictions(tree, train_data)
test_predictions = dt.get_tree_predictions(tree, test_data)

train_accuracy = dt.get_test_accuracy(train_data, "votes", train_predictions)
test_accuracy = dt.get_test_accuracy(test_data, "votes", test_predictions)

print("Train Accuracy: ")
print(train_accuracy)
print("Test Accuracy: ")
print(test_accuracy)

tree_md2 = dt.DecisionTree(2)

dt.build_tree(tree_md2, train_data, "votes", tree_md2.root)

train_predictions = dt.get_tree_predictions(tree_md2, train_data)
test_predictions = dt.get_tree_predictions(tree_md2, test_data)

train_accuracy = dt.get_test_accuracy(train_data, "votes", train_predictions)
test_accuracy = dt.get_test_accuracy(test_data, "votes", test_predictions)

print("Train Accuracy, Max Depth = 2: ")
print(train_accuracy)
print("Test Accuracy, Max Depth = 2: ")
print(test_accuracy)