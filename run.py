"""
Runing the proposed Paret Set Learning (PSL) method on 15 test problems.
"""

import numpy as np
import torch
import time
import pickle

from problem import get_problem
from utils import igd, rmse, igd_plus

from lhs import lhs
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform

# new sampler for fairness
from scipy.stats import qmc

from model import ParetoSetModel

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# list of 15 test problems, which are defined in problem.py
# ins_list = ['f1','f2','f3','f4','f5','f6',
#             'vlmop1','vlmop2', 'vlmop3', 'dtlz2',
#             're21', 're23', 're33','re36','re37']
# ins_list = ['mdtlz1_4_1', 'mdtlz1_4_2', 'mdtlz1_4_3', 'mdtlz1_4_4',
#             'mdtlz2_4_1', 'mdtlz2_4_2', 'mdtlz2_4_3', 'mdtlz2_4_4',
#             'mdtlz3_4_1', 'mdtlz3_4_2', 'mdtlz3_4_3', 'mdtlz3_4_4']
# ins_list = ['mdtlz1_4_1', 'mdtlz1_4_2', 'mdtlz1_4_3', 'mdtlz1_4_4',
#             'mdtlz2_4_1', 'mdtlz2_4_2', 'mdtlz2_4_3', 'mdtlz2_4_4',
#             'mdtlz3_4_1', 'mdtlz3_4_2', 'mdtlz3_4_3', 'mdtlz3_4_4']
# ins_list = ['mdtlz1_4_1', 'mdtlz1_4_2', 'mdtlz1_4_3', 'mdtlz1_4_4']
# ins_list = ['ndtlz1_4_1', 'ndtlz1_4_2', 'ndtlz1_4_3', 'ndtlz1_4_4',
#             'ndtlz2_4_1', 'ndtlz2_4_2', 'ndtlz2_4_3', 'ndtlz2_4_4',
#             'ndtlz3_4_1', 'ndtlz3_4_2', 'ndtlz3_4_3', 'ndtlz3_4_4']
# ins_list = ['invdtlz1_4_1', 'invdtlz1_4_2', 'invdtlz1_4_3', 'invdtlz1_4_4',
#             'invdtlz2_4_1', 'invdtlz2_4_2', 'invdtlz2_4_3', 'invdtlz2_4_4',
#             'invdtlz3_4_1', 'invdtlz3_4_2', 'invdtlz3_4_3', 'invdtlz3_4_4']
# ins_list = ['hyper1', 'hyper2', 'hyper3']
# problem list for the ranger problem set
# ins_list = ['hyper_r1', 'hyper_r2', 'hyper_r3']
# ins_list = ['method1_1', 'method1_2',
#             'method2_1', 'method2_2',
#             'method3_1', 'method3_2',
#             'method4_1', 'method4_2']
# ins_list = ['re21_t1', 're21_t2', 're21_t3',
#             're24_t1', 're24_t2', 're24_t3',
#             're25_t1', 're25_t2', 're25_t3']
# ins_list = ['p1t1', 'p2t2',
#             'p2t1', 'p2t2',
#             'p3t1', 'p2t2',
#             'p4t1', 'p2t2',
#             'p5t1', 'p2t2',
#             'p6t1', 'p2t2',
#             'p7t1', 'p2t2']
problem_name_dict = {0: ['mdtlz1_4_1', 'mdtlz1_4_2', 'mdtlz1_4_3', 'mdtlz1_4_4'],
                     1: ['mdtlz2_4_1', 'mdtlz2_4_2', 'mdtlz2_4_3', 'mdtlz2_4_4'],
                     2: ['mdtlz3_4_1', 'mdtlz3_4_2', 'mdtlz3_4_3', 'mdtlz3_4_4'],
                     3: ["P1T1", "P1T2"],
                     4: ["P2T1", "P2T2"],
                     5: ["P3T1", "P3T2"],
                     6: ["P4T1", "P4T2"],
                     7: ["P5T1", "P5T2"],
                     8: ["P6T1", "P6T2"],
                     9: ["P7T1", "P7T2"],
                     10: ['hyper1', 'hyper2', 'hyper3'],
                     11: ['hyper1_dart', 'hyper2_dart', 'hyper3_dart'],
                     12: ['hyper_r1', 'hyper_r2', 'hyper_r3'],
                     13: ['method1_1', 'method1_2'],
                     14: ['method2_1', 'method2_2'],
                     15: ['method3_1', 'method3_2'],
                     16: ['re21_t1', 're21_t2', 're21_t3']}


# time slot to store rmse results
# rmse_list = [25, 50, 75, 99]
# rmse_list = [150, 200, 250, 299]

# number of independent runs
n_run = 10 #20
# number of initialized solutions
n_init = 20
# number of iterations, and batch size per iteration
n_iter = 100
n_sample = 1

# PSL 
# number of learning steps
n_steps = 100
# number of sampled preferences per step
n_pref_update = 1
# coefficient of LCB
coef_lcb = 0.5
# number of sampled candidates on the approxiamte Pareto front
n_candidate = 50
# number of optional local search
n_local = 0
# device
device = 'cuda:0'
# device = 'cpu'
# benchmark or hyper
if_hyper = False
# -----------------------------------------------------------------------------

hv_list = {}

# problem_id = [0, 4, 8]
# problem_id = [0, 2, 4, 6]
# problem_id = [0, 2, 4, 6, 8, 10, 12]
# problem_range = [2, 2, 2, 2, 2, 2, 2]
# problem_range = [4, 4, 4]
# problem_range = [2, 2, 2, 2]
# problem_id = [0, 3, 6]
# problem_range = [3, 3, 3]
# problem_range = [2, 2, 2]

for range_id in problem_name_dict:
    if 3 <= range_id <= 9 or range_id in [12, 13, 15]:
        continue
    else:
        pass

    rmse_list = np.arange(n_iter / 4, n_iter + 1, n_iter / 4) - 1
    if_DTLZ = False
    if_bench = False
    if_hyper = False

    if 0 <= range_id < 3:
        if_DTLZ = True
    elif 3 <= range_id < 10:
        if_bench = True
    elif 10 <= range_id < 17:
        if_hyper = True
    else:
        assert 0 <= range_id < 17

    problem_name_list = problem_name_dict[range_id]
    problem_size = len(problem_name_list)
    print("Start with {}.".format(problem_name_list[0]))
    
    # get problem info
    # We only use hv records for debugging
    hv_all_value = np.zeros([n_run, n_iter])
    # Append all the info into the info list
    problem_list = []
    n_dim_list = []
    n_obj_list = []
    for temp_id in range(problem_size):
        problem = get_problem(problem_name_list[temp_id])
        # print("DEBUG")
        n_dim = problem.n_dim
        n_obj = problem.n_obj
        if if_bench:
            n_dim = 10

        problem_list.append(problem)
        n_dim_list.append(n_dim)
        n_obj_list.append(n_obj)

    # get the temp storage vector
    pareto_tensors_list = []
    pareto_records_list = []
    igd_records_list = []
    rmse_records_list = []
    time_list = []

    # not sure whether ref_point make sense if we use no hv
    # ref_point = problem.nadir_point
    # ref_point = [1.1*x for x in ref_point]
    
    # repeatedly run the algorithm n_run times
    for run_iter in range(n_run):
        # record the starting time
        time_s = time.time()
        
        # TODO: better I/O between torch and np
        # currently, the Pareto Set Model is on torch, and the Gaussian Process Model is on np 

        # We split the single-task evaluation into multi-task settings
        X_list = []
        Y_list = []
        Z_list = []
        for id_id, problem in enumerate(problem_list):
            # initialize n_init solutions
            # x_init = lhs(n_dim_list[id_id], n_init)

            sampler = qmc.LatinHypercube(n_dim_list[id_id])
            x_init = sampler.random(n_init)
            y_init = problem.evaluate(torch.from_numpy(x_init).to(device))

            if isinstance(y_init, torch.Tensor):
                Y = y_init.cpu().numpy()
            else:
                Y = y_init

            X = x_init
            z = torch.zeros(n_obj_list[id_id]).to(device)

            # Store the init results to the list
            X_list.append(X)
            Y_list.append(Y)
            Z_list.append(z)

        # We supply the IGD_records, RMSE_records, and Pareto_records
        # To enable the computation, we prepare the font_list and weight_list
        pareto_tensors = [[] for i in range(problem_size)]
        pareto_records = [torch.zeros(n_iter, 1) for i in range(problem_size)]
        igd_records = [torch.zeros(n_iter, 1) for i in range(problem_size)]
        rmse_records = []
        front_list = []
        weight_list = []

        # only specialized in the if_hyper settings
        # result_multi = []
        # rmse_multi = [[], [], []]
        # front_multi = [[], [], []]
        # weight_multi = [[], [], []]

        # prepare the ground true pareto front and weights for evaluation
        true_result = None
        tmp_path = None
        if if_hyper:
            tmp_path = "{}_{}_truth.pth".format(range_id, problem_list[0].current_name)
            true_result = torch.load(tmp_path)

        # prepare the ground truth PF and weights for evaluation (con't)
        for task_id in range(problem_size):
            front_item = None
            weight_item = None
            if if_DTLZ:
                weight_item, front_item = problem_list[task_id].ref_and_obj()
            if if_hyper:
                # main result storage
                weight_item = true_result["weight"][task_id].float()
                front_item = true_result["front"][task_id][:, :problem_list[task_id].n_obj].float()
            if if_bench:
                front_item, weight_item = problem_list[task_id].pareto_y()
            front_list.append(front_item)
            weight_list.append(weight_item)

        psmodel_list = []
        optimizer_list = []
        train_input_list = []
        train_output_list = []
        # n_iter batch selections 
        for i_iter in range(n_init, n_iter):
            print("Start of iteration {}.".format(i_iter))
            # record the start time for each iteration
            time_s_iter = time.time()
            for task_id in range(problem_size):

                # intitialize the model and optimizer
                psmodel = ParetoSetModel(n_dim_list[task_id], n_obj_list[task_id])
                psmodel.to(device)

                # optimizer
                optimizer = torch.optim.Adam(psmodel.parameters(), lr=1e-3)
                # optimizer_list.append(optimizer)

                # fetch the X and Y
                X = X_list[task_id]
                Y = Y_list[task_id]

                # solution normalization
                transformation = StandardTransform([0, 1])
                transformation.fit(X, Y)
                X_norm, Y_norm = transformation.do(X, Y)
            
                # train GP surrogate model
                surrogate_model = GaussianProcess(n_dim_list[task_id], n_obj_list[task_id], nu=5)
                surrogate_model.fit(X_norm, Y_norm)

                # fetch the Z data
                z = torch.min(torch.cat((Z_list[task_id].reshape(1, n_obj_list[task_id]),
                                         torch.from_numpy(Y_norm).to(device) - 0.1)), axis=0).values.data
                Z_list[task_id] = z
            
                # nondominated X, Y
                nds = NonDominatedSorting()
                idx_nds = nds.do(Y_norm)

                X_nds = X_norm[idx_nds[0]]
                Y_nds = Y_norm[idx_nds[0]]
            
                # t_step Pareto Set Learning with Gaussian Process
                for t_step in range(n_steps):
                    psmodel.train()

                    # sample n_pref_update preferences
                    alpha = np.ones(n_obj_list[task_id])
                    pref = np.random.dirichlet(alpha, n_pref_update)
                    pref_vec = torch.tensor(pref).to(device).float() + 0.0001

                    # get the current coressponding solutions
                    x = psmodel(pref_vec)
                    x_np = x.detach().cpu().numpy()

                    # obtain the value/grad of mean/std for each obj
                    mean = torch.from_numpy(surrogate_model.evaluate(x_np)['F']).to(device)
                    mean_grad = torch.from_numpy(surrogate_model.evaluate(x_np, calc_gradient=True)['dF']).to(device)

                    std = torch.from_numpy(surrogate_model.evaluate(x_np, std=True)['S']).to(device)
                    std_grad = torch.from_numpy(surrogate_model.evaluate(x_np, std=True, calc_gradient=True)['dS']).\
                        to(device)

                    # calculate the value/grad of tch decomposition with LCB
                    value = mean - coef_lcb * std
                    value_grad = mean_grad - coef_lcb * std_grad

                    tch_idx = torch.argmax((1 / pref_vec) * (value - z), axis=1)
                    tch_idx_mat = [torch.arange(len(tch_idx)), tch_idx]
                    tch_grad = (1 / pref_vec)[tch_idx_mat].view(n_pref_update, 1) * \
                        value_grad[tch_idx_mat] + 0.01 * torch.sum(value_grad, axis=1)

                    tch_grad = tch_grad / torch.norm(tch_grad, dim=1)[:, None]

                    # gradient-based pareto set model update
                    optimizer.zero_grad()
                    psmodel(pref_vec).backward(tch_grad)
                    optimizer.step()

                # solutions selection on the learned Pareto set
                psmodel.eval()
            
                # sample n_candidate preferences
                alpha = np.ones(n_obj_list[task_id])
                pref = np.random.dirichlet(alpha, n_candidate)
                pref = torch.tensor(pref).to(device).float() + 0.0001
    
                # generate correponding solutions, get the predicted mean/std
                X_candidate = psmodel(pref).to(torch.float64)
                X_candidate_np = X_candidate.detach().cpu().numpy()
                Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
            
                Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']
                Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
            
                # optional TCH-based local Exploitation
                if n_local > 0:
                    X_candidate_tch = X_candidate_np
                    z_candidate = z.cpu().numpy()
                    pref_np = pref.cpu().numpy()
                    for j in range(n_local):
                        candidate_mean = surrogate_model.evaluate(X_candidate_tch)['F']
                        candidate_mean_grad = \
                            surrogate_model.evaluate(X_candidate_tch, calc_gradient=True)['dF']

                        candidate_std = surrogate_model.evaluate(X_candidate_tch, std=True)['S']
                        candidate_std_grad = \
                            surrogate_model.evaluate(X_candidate_tch, std=True, calc_gradient=True)['dS']

                        candidate_value = candidate_mean - coef_lcb * candidate_std
                        candidate_grad = candidate_mean_grad - coef_lcb * candidate_std_grad

                        candidate_tch_idx = np.argmax((1 / pref_np) * (candidate_value - z_candidate), axis=1)
                        candidate_tch_idx_mat = [np.arange(len(candidate_tch_idx)), list(candidate_tch_idx)]

                        candidate_tch_grad = (1 / pref_np)[np.arange(len(candidate_tch_idx)), list(candidate_tch_idx)].reshape(n_candidate, 1) * \
                        candidate_grad[np.arange(len(candidate_tch_idx)), list(candidate_tch_idx)]
                        candidate_tch_grad += 0.01 * np.sum(candidate_grad, axis=1)

                        X_candidate_tch = X_candidate_tch - 0.01 * candidate_tch_grad
                        X_candidate_tch[X_candidate_tch <= 0] = 0
                        X_candidate_tch[X_candidate_tch >= 1] = 1

                    X_candidate_np = np.vstack([X_candidate_np, X_candidate_tch])

                    Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
                    Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']

                    Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
            
                # greedy batch selection
                best_subset_list = []
                Y_p = Y_nds
                for b in range(n_sample):
                    hv = HV(ref_point=np.max(np.vstack([Y_p, Y_candidate]), axis=0))
                    best_hv_value = 0
                    best_subset = [0]

                    for k in range(len(Y_candidate)):
                        Y_subset = Y_candidate[k]
                        Y_comb = np.vstack([Y_p, Y_subset])
                        hv_value_subset = hv(Y_comb)
                        if hv_value_subset > best_hv_value:
                            best_hv_value = hv_value_subset
                            best_subset = [k]

                    Y_p = np.vstack([Y_p, Y_candidate[best_subset]])
                    best_subset_list.append(best_subset)

                best_subset_list = np.array(best_subset_list).T[0]
            
                # evaluate the selected n_sample solutions
                X_candidate = torch.tensor(X_candidate_np).to(device)
                X_new = X_candidate[best_subset_list]
                Y_new = problem_list[task_id].evaluate(X_new)

                if isinstance(Y_new, list):
                    Y_new = torch.Tensor(Y_new)

                # update the set of evaluated solutions (X,Y)
                X = np.vstack([X, X_new.detach().cpu().numpy()])
                Y = np.vstack([Y, Y_new.detach().cpu().numpy()])

                # Re-do the non-dominated sorting
                nds = NonDominatedSorting()
                idx_nds = nds.do(Y)

                X_nds = X[idx_nds[0]]
                Y_nds = Y[idx_nds[0]]

                # update the X set and Y set to the whole set
                X_list[task_id] = X
                Y_list[task_id] = Y

                # update the stats vector for supervision
                # update pareto set size
                pareto_records[task_id][i_iter, :] = X_nds.shape[0]
                pareto_tensors[task_id].append(Y_list[task_id])
                # update igd value
                igd_records[task_id][i_iter] = igd_plus(front_list[task_id], torch.from_numpy(Y_list[task_id]))
                # DEBUG: Check the pareto frontier distribution so that the IGD data is correct
                # plt.scatter(Y_nds[:, 0], Y_nds[:, 1], alpha=0.4, label="PSL-MOBO")
                # plt.scatter(Y_list[task_id][:, 0], Y_list[task_id][:, 1], alpha=0.4, label="PSL-MOBO"s)
                # plt.scatter(front_list[task_id][:, 0], front_list[task_id][:, 1], alpha=0.02, label="NSGA-III")
                # plt.legend()
                # plt.title("Task {} in iteration {}.".format(task_id + 1, i_iter + 1))
                # plt.show()

                # update rmse value
                if i_iter in rmse_list:
                    # w --> x
                    predict_x = psmodel(weight_list[task_id].to(device)).to(torch.float64).to(device).detach()
                    # predict_x = predict_x.detach().cpu().numpy()

                    # x --> y
                    current_result = problem_list[task_id].evaluate(predict_x)
                    current_rmse, _ = rmse(front_list[task_id].to(device), current_result.to(device))
                    rmse_records.append(current_rmse)

            # record the ending time for each iteration
            time_t_iter = time.time()

            # print out the duration time
            print("The duration time for iteration {} is {}s".format(i_iter, time_t_iter - time_s_iter))
        
        # store the final performance
        # hv_list[test_ins] = hv_all_value

        # record the ending time
        time_t = time.time()
        time_list.append(time_t - time_s)
        print("*********The end of run id {}***********".format(run_iter))
        # At the end of each run
        pareto_records_list.append(pareto_records)
        pareto_tensors_list.append(pareto_tensors)

        igd_records_list.append(igd_records)
        rmse_records_list.append(rmse_records)

        # with open('hv_psl_mobo.pickle', 'wb') as output_file:
        #     pickle.dump([hv_list], output_file)

    # record the whole path file ready for analysis
    my_dict = dict()
    my_dict['record'] = pareto_tensors_list
    my_dict['pareto'] = pareto_records_list
    my_dict['igd'] = igd_records_list
    my_dict['rmse'] = rmse_records_list
    my_dict['time'] = time_list
    my_dict['dim'] = n_dim_list[0]
    my_dict['obj'] = n_obj_list[0]

    print("DEBUG")

    # torch.save(my_dict, "./psl_server/{}_obj{}_dim{}_{}.pth".
    #            format(problem_list[0].current_name,
    #                   my_dict['obj'],
    #                   my_dict['dim'],
    #                   "PSL-MOBO"))
    torch.save(my_dict, "result/{}_{}_{}_test.pth".format(range_id, problem_list[0].current_name, "PSL-MOBO"))
    # torch.save(my_dict, "./result/Benchmark_low_result_P{}_{}.pth".
    #            format(range_id+1, "PSL-MOBO"))
