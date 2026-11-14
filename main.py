import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def solve():
    df = pd.read_csv("test-input.csv")

    # Identify target column: 'slab_released' or the last column
    if "slab_released" in df.columns:
        target_col = "slab_released"
    else:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train DecisionTreeClassifier with default hyperparameters and integer random_state
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)

    tree = clf.tree_

    # Find the leaf node predicting class 1 (slab released / slope failed)
    class_1_idx = None
    for idx, c in enumerate(clf.classes_):
        if c == 1:
            class_1_idx = idx
            break
    if class_1_idx is None:
        class_1_idx = 1 if len(clf.classes_) > 1 else 0

    target_leaf = None
    for i in range(tree.node_count):
        if tree.feature[i] < 0:
            if tree.value[i][0].argmax() == class_1_idx:
                target_leaf = i
                break

    # Trace decision path from root to the failure leaf
    if target_leaf is not None:
        parent = {}
        for i in range(tree.node_count):
            l = tree.children_left[i]
            r = tree.children_right[i]
            if l >= 0:
                parent[l] = i
            if r >= 0:
                parent[r] = i

        curr = target_leaf
        path_nodes = []
        while curr in parent:
            p = parent[curr]
            path_nodes.append(p)
            curr = p
        path_nodes.reverse()
        word = "".join(
            [X.columns[tree.feature[node]][0].upper() for node in path_nodes]
        )
    else:
        # Fallback to non-leaf nodes in pre-order traversal
        non_leaf = [
            X.columns[tree.feature[i]][0].upper()
            for i in range(tree.node_count)
            if tree.feature[i] >= 0
        ]
        word = "".join(non_leaf)

    flag = f"SIGMOID_{word}"

    with open("test-output.txt", "w", encoding="utf-8") as f:
        f.write(f"{flag}\n")


if __name__ == "__main__":
    solve()
