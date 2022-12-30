import random
import math
import typing


class MonteCarloTreeSearch:
    def __init__(self, root, max_iter=1000, max_depth=1000, epsilon=0.2):
        self.root = root
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.epsilon = epsilon

    def search(self):
        for i in range(self.max_iter):
            node = self.select(self.root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)

    def select(self, node):
        while not node.is_leaf():
            node = node.select_child(self.epsilon)
        return node

    def simulate(self, node):
        for i in range(self.max_depth):
            if node.is_terminal():
                return node.reward()
            node = node.random_child()
        return node.reward()

    def backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

class Node:
    def __init__(self, name = None, parent=None):
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.name = name


    def is_leaf(self):
        return len(self.children) == 0

    def reward(self):
        return self.state.reward()

    def select_child(self, epsilon):
        return max(self.children, key=lambda c: c.value + epsilon * math.sqrt(math.log(self.visits) / c.visits))

    def random_child(self):
        return random.choice(self.children)

    def edit_distance(self, target):
        current_node = self.name
        target_node = target.name

        if len(current_node) > len(target_node):
            current_node, target_node = target_node, current_node

        distances = range(len(current_node) + 1)
        for i2, c2 in enumerate(target_node):
            distances_ = [i2+1]
            for i1, c1 in enumerate(current_node):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1+1], distances_[-1])))
            distances = distances_
        return distances[-1]


    def update(self, reward):
        self.visits += 1
        self.value += (reward - self.value) / self.visits

    def __repr__(self):
        return f"Node(visits={self.visits}, value={self.value})"

class BinaryTree:
    def __init__(self, root, depth=5):
        self.root = root
        self.depth = depth
        self.create_tree(self.root, depth)

    def create_tree(self, node, depth):
        if depth == 0:
            return
        parent_name = node.name if node.name  != "root" else ""
        node.children = [Node(parent=node, name=parent_name+"L"), Node(parent=node, name=parent_name+"R")]
        for child in node.children:
            self.create_tree(child, depth - 1)

    def plot_tree(self, start_node=None):
        if start_node is None:
            start_node = self.root
        self.plot_node(start_node, 0)

    def plot_node(self, node, level):

        if node!=None:
            if len(node.children) != 0:
                self.plot_node(node.children[1], level+1)
            print(" " * 10 * level + "->"+ str(node.name))
            if len(node.children) != 0:
                self.plot_node(node.children[0], level+1)



if __name__ == "__main__":
    root = Node(name="root")
    tree = BinaryTree(root, depth=4)
    tree.plot_tree()


    #mcts = MonteCarloTreeSearch(root)
    #mcts.search()
    #print(root)