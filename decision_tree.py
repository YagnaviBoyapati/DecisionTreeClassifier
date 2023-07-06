# decision_tree.py
# Yagnavi Boyapati(yxb220001@utdallas.edu)
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.

# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),


import numpy as np
import matplotlib.pyplot as pt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier,export_graphviz
import graphviz
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    uniq = np.unique(x)
    parti = {}
    for i in uniq:
        parti[i] = []
        for j, v in enumerate(x):
            if v == i:
                parti[i].append(j)
    return parti
    raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """


    uniq_val = partition(y)
    uniq = len(np.unique(y))
    v = np.full(uniq, 0)
    uni = np.unique(y)
    ent = 0
    for i in uniq_val.keys():
        p = (float)(len(uniq_val[i])/len(y))
        ent = ent+(-p*np.log2(p))

    return ent
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # call to partition func
    parti = partition(x)
    keys = list(parti.keys())
    sum = []
    y1 = []
    hyx = 0
    for i in parti.keys():
        val = (float)(len(parti[i])/len(x))
        for k in parti[i]:
            y1.append(y[k])
        ent = entropy(y1)
        hyx = hyx+(val*ent)

    return entropy(y)-hyx
    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.


    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """


    if attribute_value_pairs is None:
        attribute_value_pairs = []
        for i in range(len(x[0])):
            for vals in np.unique(np.array([k[i] for k in x])):
                attribute_value_pairs.append((i, vals))


    y_val, y_num = np.unique(y, return_counts=True)
    #  (1) first termination condition
    if len(y_val) == 1:
        return y_val[0]
    # (2) and (3) termination condition
    if len(attribute_value_pairs) == 0:
        # return most common value of y
        return y_val[np.argmax(y_num)]

    if depth == max_depth:
        return y_val[np.argmax(y_num)]

    mutuInfo = []

    for i, v in attribute_value_pairs:
        mutuInfo.append(mutual_information((x[:, i] == v), y))

    max_val = max(mutuInfo)
    max_ind = mutuInfo.index(max_val)

    max_attr = attribute_value_pairs[max_ind][0]+1
    max_value = attribute_value_pairs[max_ind][1]


    dataPartition = partition(x[:, [max_attr-1]])

    delIndex = np.all(attribute_value_pairs == (max_attr-1, max_value))

    attribute_value_pairs = np.delete(attribute_value_pairs, max_ind, 0)

    dtree = {}
    indexes_if_true = dataPartition[max_value]
    indexes_if_false = [i for i in range(
        len(x[:, 0])) if i not in indexes_if_true]
    try:
        return {(max_attr-1, max_value, False): id3(x[indexes_if_false], y[indexes_if_false], attribute_value_pairs, depth+1, max_depth),
                (max_attr-1, max_value, True): id3(x[indexes_if_true], y[indexes_if_true], attribute_value_pairs, depth+1, max_depth)}
    except:
        return y_val[np.argmax(y_num)]

    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    for node, child in tree.items():
        i = node[0]
        val = node[1]
        tf = node[2]
        if((tf == (x[i] == val))):
            if type(child) is not dict:
                label = child
            else:
                label = predict_example(x, child)

            return label
    # raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    return np.sum(np.absolute(y_true - y_pred))/len(y_true)
    raise Exception('Function not yet implemented!')


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print(
            '+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./data/monks-1.train', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/monks-1.test', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3

    decision_tree = id3(Xtrn, ytrn, max_depth=4)
    visualize(decision_tree)

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    #print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

train_set = ['./data/monks-1.train',
             './data/monks-2.train', './data/monks-3.train']
test_set = ['./data/monks-1.test',
            './data/monks-2.test', './data/monks-3.test']
decision_tree_total = []
for i in range(len(train_set)):
    # Load the training data
    M = np.genfromtxt(train_set[i], missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt(test_set[i], missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    depth = 10
    avg_train_err = []
    avg_test_err = []
    new_dec_trees = []
    depths = []
    for j in range(1, depth+1):

        decision_tree = id3(Xtrn, ytrn, max_depth=j)
        new_dec_trees.append(decision_tree)
        # visualize(decision_tree,j)
        pred_ytrn = [predict_example(x, decision_tree) for x in Xtrn]
        pred_ytst = [predict_example(x, decision_tree) for x in Xtst]
        train_err = compute_error(ytrn, pred_ytrn)
        tst_err = compute_error(ytst, pred_ytst)
        depths.append(j)
        avg_train_err.append(train_err)
        avg_test_err.append(tst_err)
    decision_tree_total.append((new_dec_trees))
    print("Train Error dataset = ", np.average(avg_train_err)*100)
    print("Test Error dataset = ", np.average(avg_test_err)*100)

    # plot tree depth vs err
    pt.subplot()
    pt.title("Monk Dataset:"+str(i+1))
    pt.xlabel("depth of the tree")
    pt.ylabel("err")
    pt.plot(depths,avg_train_err, 'o-',label='Training Dataset',color='b')
    pt.plot(depths,avg_test_err,'o-',label='Testing Dataset', color='r')
    pt.show()

# Load the training data
M = np.genfromtxt('./data/monks-1.train', missing_values=0,
                  skip_header=0, delimiter=',', dtype=int)
ytrn = M[:, 0]
Xtrn = M[:, 1:]

#  Load the test data
M = np.genfromtxt('./data/monks-1.test', missing_values=0,
                  skip_header=0, delimiter=',', dtype=int)
ytst = M[:, 0]
Xtst = M[:, 1:]

# decision tree at depth =1
dtree_depth1 = decision_tree_total[0][0]
visualize(dtree_depth1)

y_pred_1 = [predict_example(x, dtree_depth1) for x in Xtst]
c1 = confusion_matrix(ytst, y_pred_1)
disp = ConfusionMatrixDisplay(confusion_matrix=c1)
disp.plot()

pt.show()

# decision tree at depth=2
dtree_depth2 = decision_tree_total[0][1]
visualize(dtree_depth2)

y_pred_2 = [predict_example(x, dtree_depth2) for x in Xtst]
c2 = confusion_matrix(ytst, y_pred_2)
disp = ConfusionMatrixDisplay(confusion_matrix=c2)
disp.plot()

pt.show()

# Load the training data
M = np.genfromtxt('./data/monks-1.train', missing_values=0,
                  skip_header=0, delimiter=',', dtype=int)
ytrn = M[:, 0]
Xtrn = M[:, 1:]

#  Load the test data
M = np.genfromtxt('./data/monks-1.test', missing_values=0,
                  skip_header=0, delimiter=',', dtype=int)
ytst = M[:, 0]
Xtst = M[:, 1:]

clf=tree.DecisionTreeClassifier()
clf=clf.fit(Xtrn,ytrn)
pred_y=clf.predict(Xtst,clf)

c=confusion_matrix(ytst,pred_y)
disp_c = ConfusionMatrixDisplay(confusion_matrix=c)
disp_c.plot()
pt.show()


data=tree.export_graphviz(clf,out_file=None)
graph=graphviz.Source(data)
graph.view()
graph.render("new DT through Graphviz",format="png")


data=pd.read_csv("./data/tic-tac-toe.data")
for col in data.columns:
    data.drop(data[data[col] == '?'].index, inplace = True)
data = data.reset_index(drop=True)
colname = data.columns
ordinal_encoder = OrdinalEncoder()
data[colname] = ordinal_encoder.fit_transform(data[colname])
X = data.iloc[:,0:-1]
y = data.iloc[:,-1] #no recurrence = 0, recurrence = 1

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=2)


X_test_np=X_test.to_numpy().astype('int32')
Y_test_np=Y_test.to_numpy().astype('int32')
X_train_np=X_train.to_numpy().astype('int32')
Y_train_np=Y_train.to_numpy().astype('int32')
depth = 10
avg_train_err = []
avg_test_err = []
new_dec_trees = []
depths = []
dtree_total_newset=[]
for j in range(1, depth+1):
    decision_tree = id3(X_train_np, Y_train_np, max_depth=j)
    new_dec_trees.append(decision_tree)
    pred_ytrn = [predict_example(x, decision_tree) for x in X_train_np]
    pred_ytst = [predict_example(x, decision_tree) for x in X_test_np]
    train_err = compute_error(Y_train_np, pred_ytrn)
    tst_err = compute_error(Y_test_np, pred_ytst)
    depths.append(j)
    avg_train_err.append(train_err)
    avg_test_err.append(tst_err)
dtree_total_newset.append((new_dec_trees))
print("Train Error dataset = ", np.average(avg_train_err)*100)
print("Test Error dataset = ", np.average(avg_test_err)*100)

# step2
# decision tree at depth =1
dtree_depth1 = dtree_total_newset[0][0]
visualize(dtree_depth1)

y_pred_1_new = [predict_example(x, dtree_depth1) for x in X_test_np]
predict_y_d1_int = [int(i) for i in y_pred_1_new]
c1 = confusion_matrix(Y_test_np, predict_y_d1_int)
print("confusion matrix for depth 1 in tictactoe")
disp = ConfusionMatrixDisplay(confusion_matrix=c1)
disp.plot()

pt.show()

# decision tree at depth=2
dtree_depth2 = dtree_total_newset[0][1]
visualize(dtree_depth2)

y_pred_2_new = [predict_example(x, dtree_depth2) for x in X_test_np]
predict_y_d2_int = [int(ele) for ele in y_pred_2_new]
c2 = confusion_matrix(Y_test_np, predict_y_d2_int)
print("confusion matrix for depth 2 in tictactoe")
disp = ConfusionMatrixDisplay(confusion_matrix=c2)
disp.plot()

pt.show()

# step3

dtree_sklearn=tree.DecisionTreeClassifier()

dtree_sklearn=dtree_sklearn.fit(X_train,Y_train)

y_predicted=dtree_sklearn.predict(X_test,dtree_sklearn)
c=confusion_matrix(Y_test,y_predicted)
disp_c = ConfusionMatrixDisplay(confusion_matrix=c)
disp_c.plot()
pt.show()


data=tree.export_graphviz(dtree_sklearn,out_file=None)
graph=graphviz.Source(data)
graph.view()
graph.render("new DT using sklearn through Graphviz")

