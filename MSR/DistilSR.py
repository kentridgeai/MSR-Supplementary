import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
import itertools
import time
import warnings
from itertools import product 

warnings.filterwarnings("ignore")

operators = {"Pow": {"arity":2, "symmetric": False}, "Mul": {"arity":2, "symmetric": True},
             "Add": {"arity":2, "symmetric": True}, "Sub": {"arity":2, "symmetric": False},
             "Div": {"arity":2, "symmetric": False},}

def Pow(op1, op2):
    if op1>0:
        return np.power(op1, op2)
    else:
        return -np.power(-op1, op2)
Pow = np.vectorize(Pow)

def Mul(op1, op2):
    return op1 * op2

def Add(op1, op2):
    return op1 + op2

def Sub(op1, op2):
    return op1 - op2

def Div(op1, op2):
    return op1 / op2

class Node:
    def __init__(self, symbol="UNFILLED", parent=None):
        self.symbol = symbol
        self.parent = parent
        self.children = []
        self.on_variable_path = False

def kexp_to_tree(kexp, ordered_symbols = True):
    kexp = list(kexp)
    root = Node()
    queue = [root]  # FIFO queue
    seen = [root]  # Tracker
    for symbol in kexp:
        if not queue:
            break
        cur_Node = queue[0]
        queue = queue[1:]
        cur_Node.symbol = symbol
        if symbol in operators:
            no_of_children = operators[symbol]["arity"]
            cur_Node.children = [Node(parent=cur_Node) for i in range(no_of_children)]
            queue.extend(cur_Node.children)
            seen.extend(cur_Node.children)
        elif symbol == "R":
            pass
    if ordered_symbols:
        queue = [root]  # FIFO queue
        all_nodes = [root]
        while queue:
            cur_Node = queue[0]
            queue = queue[1:]
            queue.extend(cur_Node.children)
            all_nodes.extend(cur_Node.children)
        for node in all_nodes:
            if node.symbol!="R" and operators[node.symbol]["symmetric"]:
                node.children = sorted(node.children, key=lambda x:x.symbol)
    return root

def tree_to_exp(node):
    symbol = node.symbol
    if node.symbol in operators:
        return (
            symbol
            + "("
            + "".join([tree_to_exp(child) + "," for child in node.children])[:-1]
            + ")"
        )
    else:
        return symbol

def cost(x, xdata, ydata, lambda_string):  # simply use globally defined x and y
    y_pred = eval(lambda_string)(x, xdata)
    return np.mean(((y_pred - ydata)/ydata)**2) # quadratic cost function

def func(xdata, a, b, c):
    return np.power(xdata[0,:],a)+b/xdata[1,:]

def get_exp_set(k_exp_front_length = 3):
    exhuastive_symbol_set = [i for i in operators]+["R"]
    k_exp_list = [list(i)+["R"]*(k_exp_front_length+1) for i in product(exhuastive_symbol_set,repeat=k_exp_front_length)]
    exp_list = [tree_to_exp(kexp_to_tree(list(i)+["R"]*(k_exp_front_length+1))) for i in product(exhuastive_symbol_set,repeat=k_exp_front_length)]
    exp_set = set(exp_list)
    return exp_set
