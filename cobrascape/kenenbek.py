import os

import numpy as np
import pandas as pd
import cobra
import torch
from tqdm import tqdm, trange
import scipy.optimize as opt

from torch_geometric.data import HeteroData


def convert_to_graph_and_save(COBRA_MODEL, strain_id, save_samples_dir, num_iter, fva_df):
    S = cobra.util.create_stoichiometric_matrix(COBRA_MODEL)
    m = S.shape[0]
    n = S.shape[1]
    c = np.array([-reaction.objective_coefficient for reaction in COBRA_MODEL.reactions])
    b = np.zeros(S.shape[0])

    EdgeIndex = []
    EdgeFeature = []

    # Loop through the matrix to find non-zero entries
    for i in range(m):
        for j in range(n):
            if S[i, j] != 0:
                EdgeIndex.append((i, j))
                EdgeFeature.append(S[i, j])

    # Convert lists to appropriate numpy arrays
    EdgeIndex = np.array(EdgeIndex)
    EdgeFeature = np.array(EdgeFeature)

    bounds = []
    for reaction in COBRA_MODEL.reactions:
        bounds.append([reaction.lower_bound, reaction.upper_bound])

    result = opt.linprog(c, A_eq=S, b_eq=b, bounds=bounds)

    constr_features = b.reshape(m, 1)
    reaction_features = np.hstack((c.reshape(n, 1), bounds))
    data = HeteroData()

    data["reactions"].x = torch.from_numpy(reaction_features)
    data["reactions"].y = torch.from_numpy(fva_df.values)

    data["constraints"].x = torch.from_numpy(constr_features)
    data["constraints", "to", "reactions"].edge_index = torch.from_numpy(EdgeIndex.T)
    data["constraints", "to", "reactions"].edge_attr = torch.from_numpy(EdgeFeature)

    data["reactions", "to", "constraints"].edge_index = torch.from_numpy(EdgeIndex.T).flip(0)
    data["reactions", "to", "constraints"].edge_attr = torch.from_numpy(EdgeFeature)

    data["objective_value"] = torch.tensor(-result.fun)

    # data["S"] = torch.from_numpy(S)
    # data["c"] = torch.from_numpy(c)
    # data["b"] = torch.from_numpy(b)

    save_name = f"{save_samples_dir}/{strain_id}--{num_iter}.pth"

    torch.save(data, save_name)
    if not os.path.exists(f"{save_samples_dir}/S.pth"):
        torch.save(S, f"{save_samples_dir}/S.pth")
        torch.save(c, f"{save_samples_dir}/c.pth")
        torch.save(b, f"{save_samples_dir}/b.pth")













def get_input_of_fva(COBRA_MODEL, strain_id, save_samples_dir, num_iter):
    S = cobra.util.create_stoichiometric_matrix(COBRA_MODEL)
    m = S.shape[0]
    n = S.shape[1]
    circ = np.ones(S.shape[0], dtype=int)
    c = np.array([-reaction.objective_coefficient for reaction in COBRA_MODEL.reactions])
    b = np.zeros(S.shape[0])

    EdgeIndex = []
    EdgeFeature = []

    # Loop through the matrix to find non-zero entries
    for i in range(m):
        for j in range(n):
            if S[i, j] != 0:
                EdgeIndex.append((i, j))
                EdgeFeature.append(S[i, j])

    # Convert lists to appropriate numpy arrays
    EdgeIndex = np.array(EdgeIndex)
    EdgeFeature = np.array(EdgeFeature)

    bounds = []
    for reaction in COBRA_MODEL.reactions:
        bounds.append([reaction.lower_bound, reaction.upper_bound])

    result = opt.linprog(c, A_eq=S, b_eq=b, bounds=bounds)

    path = os.path.join(save_samples_dir, str(strain_id) + "---" + str(num_iter))
    if not os.path.exists(path):
        os.makedirs(path)

    np.savetxt(path + '/ConFeatures.csv', np.hstack((b.reshape(m, 1), circ.reshape(m, 1))), delimiter=',',
               fmt='%10.5f')
    np.savetxt(path + '/EdgeFeatures.csv', EdgeFeature, fmt='%10.5f')
    np.savetxt(path + '/EdgeIndices.csv', EdgeIndex, delimiter=',', fmt='%d')
    np.savetxt(path + '/VarFeatures.csv', np.hstack((c.reshape(n, 1), bounds)), delimiter=',', fmt='%10.5f')
    np.savetxt(path + '/Labels_feas.csv', [result.status], fmt='%d')

    if result.status != 2:  # feasible
        np.savetxt(path + '/Labels_obj.csv', [-result.fun], fmt='%10.5f')
        np.savetxt(path + '/Labels_solu.csv', result.x, fmt='%10.5f')


def clean_flux_samples(model, flux_values):
    fixed_flux_samples = flux_values.copy()
    for index, row in flux_values.iterrows():
        for column_name, flux_value in row.items():
            lb = model.reactions.get_by_id(column_name).lower_bound
            ub = model.reactions.get_by_id(column_name).upper_bound

            if flux_value < lb or flux_value > ub:
                corrected_flux = lb if flux_value < lb else ub
                fixed_flux_samples.at[index, column_name] = corrected_flux
    return fixed_flux_samples