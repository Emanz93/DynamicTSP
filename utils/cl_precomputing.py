# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from time import time

def create_candidate_list():
    """Create a candidate list to speed up the computation. Takes the first
    CL_NUM best cities."""
    # top = .CL_NUM + 1
    # .cl = [] # temporary candidate list.
    # for t in range(.N_CITIES): # create a candidate list for each timestamp?
    # cl_cities = []
    t1 = time()

    cl = []
    for t in range(N_CITIES):
        print(t)
        cl.append(alpha_nearness(t))
    cl = np.array(cl)
    t2 = time()
    print("Created candidate list of shape={} in {}".format(cl.shape, t2 - t1))
    return cl


def alpha_nearness(t):
    """Compute the alpha distance for each arcs (c, j).
    Returns the list of all c neighbors distances.
    """
    # compute the one minimum tree for the graph.
    one_tree = compute_minimum_one_tree(t)
    l = np.sum(one_tree) / 2.0

    alpha_lenghts = np.zeros((N_CITIES, N_CITIES))

    # compute the alpha distance for all j neighbors:
    for i in range(N_CITIES):
        for j in range(i, N_CITIES):
            if j == i:
                alpha_lenghts[i, j] = float('Inf')
            else:
                alpha_lenghts[i, j] = alpha_lenghts[j, i] = compute_alpha_len(i, j, one_tree)

    alpha_lenghts = np.abs(alpha_lenghts - l)
    #tmp_cl.append(np.argsort(alpha_nearness(i, t), kind='heapsort')[:(CL_NUM + 1)])

    # compute the alpha_nearness:
    return np.argsort(alpha_lenghts, axis=1, kind='heapsort')[:, :CL_NUM]


def compute_minimum_one_tree(t):
    # get the mst collision matrix without the 0 element.
    w_t = weights[t, 1:, 1:].copy()
    mst = minimum_spanning_tree(w_t)  # Compute the MST over the graph without vertex 1.
    mst_graph = np.array(mst.toarray().astype(float))  # get the collision matrix
    np.shape(mst_graph)
    one_tree = np.insert(mst_graph, 0, np.zeros(N_CITIES - 1), axis=0)  # insert the first row.
    one_tree = np.insert(one_tree, 0, np.zeros(N_CITIES), axis=1)  # insert the first column.

    # respect the simmetry of the collision matrix.
    for i in range(N_CITIES):
        for j in range(N_CITIES):
            if one_tree[i, j] != 0.0:
                one_tree[j, i] = one_tree[i, j]

    # Find the two closest cities to vertex 0, belonging to T.
    closest_cities = np.argpartition(weights[t, 0, 1:], 2)
    closest_city1 = closest_cities[0]
    closest_city2 = closest_cities[1]
    # add the two links.
    one_tree[0, closest_city1] = weights[t, 0, closest_city1]
    one_tree[closest_city1, 0] = one_tree[0, closest_city1]
    one_tree[0, closest_city2] = weights[t, 0, closest_city2]
    one_tree[closest_city2, 0] = one_tree[0, closest_city2]
    return one_tree


def max_in_loop(c, one_tree_cm):
    """Find the loop, isolate it and return the value of the indices of the maximum weight inside the loop."""
    collision_matrix = one_tree_cm.copy()  # use a copy of the tree due to later modifications.
    # create the list of degree value for each vertex.
    degree_list = np.count_nonzero(collision_matrix, axis=1).tolist()

    try:
        while True:  # Search for the cycle in collision_matrix.
            c_i = degree_list.index(1)  # get the index of degree == 1
            c_j = np.nonzero(collision_matrix[c_i])[0][0]  # get its j value.

            # degrease the degree for the removed edges.
            degree_list[c_i] = 0
            degree_list[c_j] -= 1

            # update the collision matrix removing the edge.
            collision_matrix[c_i, c_j] = collision_matrix[c_j, c_i] = 0.0
    except ValueError:
        pass

    # get the cycle indicies
    cycle = np.nonzero(degree_list)[0]

    if c in cycle:  # test if the node belongs to the cycle.
        return np.unravel_index(np.argmax(collision_matrix), (N_CITIES, N_CITIES))
    else:  # else return None.
        return None


def compute_alpha_len(c1, c2, one_t):
    """Compute the value of the alpha length for the arc (c1, c2). """
    one_tree = one_t.copy()  # save a copy of one_tree.
    t = 0  # start from t = 0
    # alpha_len = float('Inf')  # initialize the alpha_len to infinite

    # case 1
    if one_tree[c1, c2] != 0:
        return 0
    elif c1 == 0 or c2 == 0:
        # take the largest of the two arcs incidents to node 0
        x = np.argmax(one_tree[0])
        # replace it with (c1, c2).
        one_tree[0, x] = one_tree[x, 0] = weights[t, c1, c2]
    else:
        one_tree[c1, c2] = one_tree[c2, c1] = weights[t, c1, c2]  # add the link (c1,c2).
        max_arc_idx = max_in_loop(c1, one_tree)
        one_tree[max_arc_idx[0], max_arc_idx[1]] = one_tree[
            max_arc_idx[1], max_arc_idx[0]] = 0  # remove the heavier link in the loop.
    return np.sum(one_tree) / 2.0

loaded = np.load('../instances/iridium_weights_matricies.npz')
weights = loaded['w']
loaded.close()
N_CITIES = np.shape(weights)[1]
CL_NUM = 15

cl = create_candidate_list()

np.savez_compressed("cl.npz", cl)
