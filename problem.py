import numpy
import torch
import numpy as np
# import tdc
# from smiles_config import *
import math
import matplotlib.pyplot as plt
# import the hyper-parameter tuning related packages
from yahpo_gym import local_config, list_scenarios
import ConfigSpace
from yahpo_gym import BenchmarkSet

DTLZ_pref_list = [62, 18, 12, 9, 9]
train_size_list = [1.0, 0.75, 0.5]


def read_matrix_from_file(filename):
    matrix = []
    with open(filename, 'r') as file:
        for line in file:
            row = list(map(float, line.split()))  # Assuming elements are integers
            matrix.append(row)
    return matrix


def generate_norm(source_tensor: torch.tensor):
    norm_tensor = torch.norm(source_tensor, dim=1).unsqueeze(1)
    target_tensor = source_tensor / norm_tensor
    return target_tensor


def ref_points(obj_num: int, granularity: int, if_norm: bool = True):
    """
    Return the reference tuple for decomposition-based methods
    like MOEA/D or NSGA-III
    :param obj_num: the objective size
    :param granularity: parameter H such that x_1 + x_2 +
    ... + x_obj_num = H
    :param if_norm: parameter boolean to indicate different modes
    :return: A torch.tensor in shape of [C(obj_num+H-1, H)]
    """

    # We solve this problem by DP-like algorithm
    dp_list = []
    for i in range(granularity + 1):
        dp_list.append(torch.tensor([i]).unsqueeze(0))

    for i in range(2, obj_num + 1):
        for j in range(granularity + 1):
            if j == 0:
                # prefix a zero simply
                dp_list[j] = torch.cat((torch.zeros_like(dp_list[j])[:, 0].unsqueeze(1), dp_list[j]), dim=1)
            else:
                # first prefix a zero simply
                dp_list[j] = torch.cat((torch.zeros_like(dp_list[j])[:, 0].unsqueeze(1), dp_list[j]), dim=1)
                # then plus one based on dp_list[j-1]
                dp_tmp = torch.zeros_like(dp_list[j - 1])
                dp_tmp[:, 0] = 1
                dp_tmp = dp_tmp + dp_list[j - 1]
                dp_list[j] = torch.cat((dp_list[j], dp_tmp), dim=0)

        # DEBUG:
        # print("shape {} in iteration {}.".format(dp_list[-1].shape, i))

    dp_list[-1] = dp_list[-1] / granularity

    if if_norm:
        return dp_list[-1] / torch.norm(dp_list[-1], dim=1).unsqueeze(1)
    else:
        return dp_list[-1]


def get_problem(name, *args, **kwargs):
    name = name.lower()

    PROBLEM = {
        'f1': F1,
        'f2': F2,
        'f3': F3,
        'f4': F4,
        'f5': F5,
        'f6': F6,
        'vlmop1': VLMOP1,
        'vlmop2': VLMOP2,
        'vlmop3': VLMOP3,
        'dtlz2': DTLZ2,
        're21': RE21,
        're23': RE23,
        're33': RE33,
        're36': RE36,
        're37': RE37,
        'mdtlz1_3_1': mDTLZ1(m=3, n=6, s=1.0, p=0.5, p_ind=0),
        'mdtlz1_3_2': mDTLZ1(m=3, n=6, s=0.85, p=0.5, p_ind=1),
        'mdtlz1_3_3': mDTLZ1(m=3, n=6, s=0.70, p=0.5, p_ind=2),
        'mdtlz1_4_1': mDTLZ1(m=3, n=6, s=0.35, p=0.5, p_ind=0),
        'mdtlz1_4_2': mDTLZ1(m=3, n=6, s=0.30, p=0.5, p_ind=1),
        'mdtlz1_4_3': mDTLZ1(m=3, n=6, s=0.25, p=0.5, p_ind=2),
        'mdtlz1_4_4': mDTLZ1(m=3, n=6, s=0.20, p=0.5, p_ind=3),
        'invdtlz1_4_1': invDTLZ1(m=3, n=6, s=1.0, p=0.5, p_ind=0),
        'invdtlz1_4_2': invDTLZ1(m=3, n=6, s=0.75, p=0.5, p_ind=1),
        'invdtlz1_4_3': invDTLZ1(m=3, n=6, s=0.50, p=0.5, p_ind=2),
        'invdtlz1_4_4': invDTLZ1(m=3, n=6, s=0.25, p=0.5, p_ind=3),
        'mdtlz2_3_1': mDTLZ2(m=3, n=6, s=1.0, p=0.5, p_ind=0),
        'mdtlz2_3_2': mDTLZ2(m=3, n=6, s=0.85, p=0.5, p_ind=1),
        'mdtlz2_3_3': mDTLZ2(m=3, n=6, s=0.70, p=0.5, p_ind=2),
        'mdtlz2_4_1': mDTLZ2(m=3, n=6, s=1.0, p=0.5, p_ind=0),
        'mdtlz2_4_2': mDTLZ2(m=3, n=6, s=0.75, p=0.5, p_ind=1),
        'mdtlz2_4_3': mDTLZ2(m=3, n=6, s=0.50, p=0.5, p_ind=2),
        'mdtlz2_4_4': mDTLZ2(m=3, n=6, s=0.25, p=0.5, p_ind=3),
        'invdtlz2_4_1': invDTLZ2(m=3, n=6, s=1.0, p=0.5, p_ind=0),
        'invdtlz2_4_2': invDTLZ2(m=3, n=6, s=0.75, p=0.5, p_ind=1),
        'invdtlz2_4_3': invDTLZ2(m=3, n=6, s=0.50, p=0.5, p_ind=2),
        'invdtlz2_4_4': invDTLZ2(m=3, n=6, s=0.25, p=0.5, p_ind=3),
        'mdtlz3_3_1': mDTLZ3(m=3, n=6, s=1.0, p=0.5, p_ind=0),
        'mdtlz3_3_2': mDTLZ3(m=3, n=6, s=0.85, p=0.5, p_ind=1),
        'mdtlz3_3_3': mDTLZ3(m=3, n=6, s=0.70, p=0.5, p_ind=2),
        'mdtlz3_4_1': mDTLZ3(m=3, n=6, s=0.20, p=0.5, p_ind=0),
        'mdtlz3_4_2': mDTLZ3(m=3, n=6, s=0.18, p=0.5, p_ind=1),
        'mdtlz3_4_3': mDTLZ3(m=3, n=6, s=0.16, p=0.5, p_ind=2),
        'mdtlz3_4_4': mDTLZ3(m=3, n=6, s=0.14, p=0.5, p_ind=3),
        'invdtlz3_4_1': invDTLZ3(m=3, n=6, s=1.0, p=0.5, p_ind=0),
        'invdtlz3_4_2': invDTLZ3(m=3, n=6, s=0.75, p=0.5, p_ind=1),
        'invdtlz3_4_3': invDTLZ3(m=3, n=6, s=0.50, p=0.5, p_ind=2),
        'invdtlz3_4_4': invDTLZ3(m=3, n=6, s=0.25, p=0.5, p_ind=3),
        'hyper1': hyper(task_id=0, if_gbtree=True),
        'hyper2': hyper(task_id=1, if_gbtree=True),
        'hyper3': hyper(task_id=2, if_gbtree=True),
        'hyper1_dart': hyper(task_id=0, if_gbtree=False),
        'hyper2_dart': hyper(task_id=1, if_gbtree=False),
        'hyper3_dart': hyper(task_id=2, if_gbtree=False),
        'hyper_r1': hyper(task_id=0, instance="iaml_ranger"),
        'hyper_r2': hyper(task_id=1, instance="iaml_ranger"),
        'hyper_r3': hyper(task_id=2, instance="iaml_ranger"),
        'method1_1': hyper(task_num=2, task_id=0, if_methods=True, problem_two_inner_id=0),
        'method1_2': hyper(task_num=2, task_id=1, if_methods=True, problem_two_inner_id=0),
        'method2_1': hyper(task_num=2, task_id=0, if_methods=True, problem_two_inner_id=1),
        'method2_2': hyper(task_num=2, task_id=1, if_methods=True, problem_two_inner_id=1),
        'method3_1': hyper(task_num=2, task_id=0, if_methods=True, problem_two_inner_id=2),
        'method3_2': hyper(task_num=2, task_id=1, if_methods=True, problem_two_inner_id=2),
        'fidelity_1': hyper(task_id=2, if_fidelity=True, fidelity=0.50, if_gbtree=True),
        'fidelity_2': hyper(task_id=2, if_fidelity=True, fidelity=0.50, if_gbtree=True),
        'fidelity_3': hyper(task_id=2, if_fidelity=True, fidelity=0.50, if_gbtree=True),
        'fidelity_1_dart': hyper(task_id=2, if_fidelity=True, fidelity=0.50, if_gbtree=False),
        'fidelity_2_dart': hyper(task_id=2, if_fidelity=True, fidelity=0.50, if_gbtree=False),
        'fidelity_3_dart': hyper(task_id=2, if_fidelity=True, fidelity=0.50, if_gbtree=False),
        're21_t1': RE21(),
        're21_t2': RE21(F=10, sigma=8, L=200, E=1.8e5),
        're21_t3': RE21(F=8, sigma=5, L=200, E=1.5e5),
        're24_t1': RE24(sigma_b_max=800, tau_max=350, delta_max=1.5),
        're24_t2': RE24(sigma_b_max=700, tau_max=450, delta_max=1.3),
        're24_t3': RE24(sigma_b_max=700, tau_max=350, delta_max=1.3),
        're25_t1': RE25(F_max=1000, l_max=14, sigma_pm=6),
        're25_t2': RE25(F_max=800, l_max=14, sigma_pm=6),
        're25_t3': RE25(F_max=1000, l_max=10, sigma_pm=4),
        'p1t1': P1T1(),
        'p1t2': P1T2(),
        'p2t1': P2T1(),
        'p2t2': P2T2(),
        'p3t1': P3T1(),
        'p3t2': P3T2(),
        'p4t1': P4T1(),
        'p4t2': P4T2(),
        'p5t1': P5T1(),
        'p5t2': P5T2(),
        'p6t1': P6T1(),
        'p6t2': P6T2(),
        'p7t1': P7T1(),
        'p7t2': P7T2(),
    }

    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name]


# class Oracle:
#     def __init__(self, name=None):
#         self.max_len = 255
#         self.qed_scorer = tdc.Oracle(name='qed')
#         self.sa_scorer = tdc.Oracle(name='SA')
#         if name is not None:
#             self.scorer = tdc.Oracle(name=name)
#             self.n_obj = 3
#         else:
#             self.scorer = None
#             self.n_obj = 2
#
#     def _transform_params(self, params):
#         # The input params in this case should a real-value vector
#         # We expect to transform the params into smiles or other representations
#         if isinstance(params, torch.Tensor):
#             params = params.cpu().numpy()
#         if not isinstance(params, np.ndarray):
#             params = params.numpy()
#
#         if len(params.shape) == 1:
#             batch_size = 1
#         else:
#             batch_size = params.shape[0]
#
#         # The continuous vector should be converted into combinatorial first
#         params = np.floor(params * self.max_len).astype(np.uint16)
#
#         # Convert to list
#         params = params.tolist()
#
#         # The combinatorial vector should be attached with corresponding
#         # SMILES or other representation
#         # The obtained vectors in class of numpy.ndarray
#         return params, batch_size
#
#     def evaluate(self, params):
#         raise NotImplementedError


# class OracleSmiles(Oracle):
#
#     def __init__(self, name=None):
#         super().__init__(self, name=None)
#         if name is not None:
#             self.current_name = f"smiles_{name}"
#         else:
#             self.current_name = "smiles"
#
#     def transform_smiles(self, params):
#
#         params, batch_size = self._transform_params(params)
#         # The output param, params, is a list or list of lists.
#         if batch_size == 1:
#             smiles = canonicalize(decode(gene_to_cfg(params)))
#         else:
#             smiles = []
#             for param in params:
#                 smiles.append(canonicalize(decode(gene_to_cfg(param))))
#
#         # Convert each list
#         return smiles, batch_size
#
#     def evaluate_from_smiles(self, smiles):
#         if self.n_obj == 2:
#             f1 = 1 - self.qed_scorer(smiles)
#             f2 = (10 - self.sa_scorer(smiles)) / (10 - 1)
#             obj = torch.stack([torch.Tensor(f1), torch.Tensor(f2)]).T
#
#         if self.n_obj == 3:
#             f1 = 1 - self.qed_scorer(smiles)
#             f2 = (10 - self.sa_scorer(smiles)) / (10 - 1)
#             f3 = 1 - self.scorer(smiles)
#             obj = torch.stack([torch.Tensor(f1), torch.Tensor(f2)]).T
#
#         return obj
#
#     def evaluate(self, params):
#         return self._transform


class hyper:
    def __init__(self, task_num=3, task_id=0, instance="iaml_xgboost",
                 if_methods: bool = False, problem_two_inner_id: int = 0,
                 if_fidelity: bool = False, fidelity: float = None,
                 if_gbtree: bool = False):
        local_config.init_config()
        local_config.set_data_path("yahpo_data")

        # print("Start Preparing Configure Space")
        # b is an instantiated benchmark called "lcbench" with multiple instances
        b = BenchmarkSet(scenario=instance)
        # Can set up a task_num parameter to control the multi-task scale
        b_list = []
        for i in range(task_num):
            b_temp = BenchmarkSet(scenario=instance)
            # In the new settings, we set up the problem indices all as 2 (the 3rd task)
            # b_temp.set_instance(b_temp.instances[2])
            # b_temp.set_instance(b_temp.instances[i])
            if if_methods:
                b_temp.set_instance(b_temp.instances[problem_two_inner_id])
            elif if_fidelity:
                b_temp.set_instance(b_temp.instances[2])
            else:
                b_temp.set_instance(b_temp.instances[i])
            b_list.append(b_temp)

        space_list = []
        for i in b_list:
            space_list.append(i.get_opt_space(drop_fidelity_params=False))

        info_list = b.config.hp_names
        info_xs = b._get_config_space()

        if instance == "iaml_xgboost":
            # info_list = info_list[2:-1]
            if if_gbtree:
                info_list = info_list[2:-2]
                # print("Before: {}.".format(info_list))
                info_list.remove("rate_drop")
                # print("After: {}.".format(info_list))
            else:
                info_list = info_list[2:-1]
        elif instance == "iaml_ranger":
            info_list = info_list[4:-1]
        else:
            info_list = info_list[1:]
        # print("Start Leaving Configure Space")

        hyper_info = dict()
        hyper_info["info_xs"] = info_xs
        hyper_info["info_list"] = info_list
        hyper_info["b_list"] = b_list
        hyper_info["space_list"] = space_list

        if_negate = torch.tensor([False, False, False])

        if instance == "iaml_xgboost":
            tmp_str_list = ["first", "second", "third"]
            if if_methods:
                self.current_name = "two_{}_new".format(
                    tmp_str_list[problem_two_inner_id])
            else:
                self.current_name = "hp"
        elif instance == "iaml_ranger":
            self.current_name = "hp_ranger"
        self.obj_list = ['mmce', 'rammodel', 'ias']

        if if_methods:
            self.n_dim = len(info_list) - 2
        else:
            self.n_dim = len(info_list)
        self.n_obj = len(self.obj_list)
        self.hyper_info = hyper_info
        self.task_id = task_id
        self.if_methods = if_methods
        self.if_fidelity = if_fidelity
        self.if_gbtree = if_gbtree
        self.fidelity = fidelity
        self.problem_two_inner_id = problem_two_inner_id

        if instance == "iaml_xgboost":
            self.lower_bounds = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            self.upper_bounds = torch.tensor([[1, 1, 1], [5, 10, 15], [1, 1, 1]])
        elif instance == "iaml_ranger":
            self.lower_bounds = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            self.upper_bounds = torch.tensor([[1, 1, 1], [35, 180, 180], [3.8, 1.6, 2.6]])
        self.negate_coef = torch.where(if_negate, -1, 1)
        self.negate_bias = torch.where(if_negate, 1, 0)

        # Setup Name:
        self.name = None
        if self.if_fidelity and self.if_gbtree:
            self.name = 'fidelity_1'
        elif self.if_fidelity and not self.if_gbtree:
            self.name = 'fidelity_1_dart'
        elif self.if_methods and problem_two_inner_id == 1:
            self.name = 'method2_1'
        elif self.if_methods and problem_two_inner_id == 2:
            self.name = 'method3_1'
        elif self.if_gbtree:
            self.name = 'hyper1'
        elif not self.if_gbtree:
            self.name = 'hyper1_dart'
        else:
            assert 1 == 2

    def _prepare_normalize(self, solution: torch.tensor):
        obj_num = self.n_obj
        task_id = self.task_id
        # solution = torch.from_numpy(solution)
        if not isinstance(solution, torch.Tensor):
            if isinstance(solution, np.ndarray):
                solution = torch.from_numpy(solution)
            if isinstance(solution, list):
                solution = torch.Tensor(solution)

        if len(solution.shape) == 1:
            solution = solution[None, :]

        if self.if_methods:
            tmp_id = self.problem_two_inner_id
            solution[:, :obj_num] = (solution[:, :obj_num] - self.lower_bounds[:, tmp_id]) / \
                                    (self.upper_bounds[:, tmp_id] - self.lower_bounds[:, tmp_id])
        elif self.if_fidelity:
            tmp_id = 2
            solution[:, :obj_num] = (solution[:, :obj_num] - self.lower_bounds[:, tmp_id]) / \
                                    (self.upper_bounds[:, tmp_id] - self.lower_bounds[:, tmp_id])

        else:
            solution[:, :obj_num] = (solution[:, :obj_num] - self.lower_bounds[:, task_id]) / \
                                    (self.upper_bounds[:, task_id] - self.lower_bounds[:, task_id])

        solution[:, :obj_num] = self.negate_coef * solution[:, :obj_num] + self.negate_bias

        solution[solution[:, :obj_num].max(1).values > 1, :obj_num] = 1.1

        return solution

    def transform_value(self, param, batch_size):
        """
        Transform a sequence of random parameters into valid trial vector
        """
        new_param = param.copy()
        # print("param in type of {}.".format(param.dtype))
        # print("new_param in type of {}.".format(new_param.dtype))
        if len(new_param.shape) == 1:
            new_param = new_param[None, :]

        info_list = self.hyper_info['info_list']
        info_xs = self.hyper_info['info_xs']
        bool_param = np.zeros((len(info_list),), dtype=bool)

        for i, attr_item in enumerate(info_list):
            if self.if_methods and (attr_item == "rate_drop" or attr_item == "skip_drop"):
                continue

            trans_float = isinstance(info_xs[attr_item], ConfigSpace.hyperparameters.UniformFloatHyperparameter)
            trans_int = isinstance(info_xs[attr_item], ConfigSpace.hyperparameters.UniformIntegerHyperparameter)
            trans_log = info_xs[attr_item].log
            trans_lower = info_xs[attr_item].lower
            trans_upper = info_xs[attr_item].upper

            # TODO:
            # Enable batch_size larger than 1 or equals to 1
            if trans_float:
                if trans_log:
                    new_item = np.exp(
                        np.log(trans_lower) + (np.log(trans_upper) - np.log(trans_lower)) * new_param[:, i])
                    new_item = np.clip(new_item, trans_lower, trans_upper)
                else:
                    new_item = trans_lower + (trans_upper - trans_lower) * new_param[:, i]
            if trans_int:
                bool_param[i] = True
                if trans_log:
                    new_item = np.round(np.exp(
                        np.log(trans_lower) + (np.log(trans_upper) - np.log(trans_lower)) * new_param[:, i]))
                    new_item = np.clip(new_item, trans_lower, trans_upper)
                else:
                    new_item = np.round(trans_lower + (trans_upper - trans_lower) * new_param[:, i])

            # print("For {} parameter, if_float is {}, if_int is {}".format(info_list[i],
            #                                                               trans_float,
            #                                                               trans_int))
            new_param[:, i] = new_item

        # print("result in type of {}.".format(new_param.dtype))
        # print(new_param)
        if batch_size == 1:
            # print(new_param[0])
            # print(bool_param)
            return new_param[0], bool_param
        else:
            return new_param, bool_param

    def evaluate(self, params):
        """
        task_id: int -- indicates the index of the task
        params: dict -- indicates the parameters for each dimension
        task_list: list -- the list of ConfigSpace objects
        set_list: list -- the list of BenchmarkSet objects
        """

        if isinstance(params, torch.Tensor):
            params = params.cpu()
        if not isinstance(params, np.ndarray):
            params = params.numpy()

        if len(params.shape) == 1:
            batch_size = 1
        else:
            batch_size = params.shape[0]
        # 1. Transfer the params into the legit form for each dimension
        # print("original params is {}".format(params))
        trans_params, bool_params = self.transform_value(params, batch_size)
        # print("transformed params is {}".format(trans_params))
        xs = self.hyper_info['space_list'][self.task_id].sample_configuration(size=batch_size)
        info_list = self.hyper_info['info_list']
        info_xs = self.hyper_info['info_xs']
        set_list = self.hyper_info['b_list']

        if batch_size == 1:
            if self.current_name == "hp":
                if self.if_gbtree:
                    xs["booster"] = "gbtree"
                else:
                    xs["booster"] = "dart"
            if self.current_name == "hp_ranger":
                xs["splitrule"] = 'extratrees'
                xs["replace"] = 'TRUE'
                xs["respect.unordered.factors"] = 'order'
            # xs["trainsize"] = train_size_list[self.task_id]
            if self.if_methods and self.task_id == 0:
                xs["booster"] = "gbtree"

            if self.if_methods and self.task_id == 1:
                xs["booster"] = "dart"

            if self.if_fidelity:
                xs["trainsize"] = self.fidelity
            else:
                xs["trainsize"] = 1.0

            # Assign the params
            for i, attr_item in enumerate(info_list):
                # print("Current attr {} with type {}.".format(xs[attr_item].__class__, trans_params[i].__class__))
                if self.if_methods and xs["booster"] == "gbtree" \
                        and (attr_item == "rate_drop" or attr_item == "skip_drop"):
                    continue
                if self.if_methods and xs["booster"] == "dart" \
                        and (attr_item == "rate_drop" or attr_item == "skip_drop"):
                    xs[attr_item] = 0.5
                    continue

                if bool_params[i]:
                    xs[attr_item] = int(trans_params[i])
                else:
                    tmp_float = float(trans_params[i])
                    if tmp_float < info_xs[attr_item].lower:
                        xs[attr_item] = info_xs[attr_item].lower
                    elif tmp_float > info_xs[attr_item].upper:
                        xs[attr_item] = info_xs[attr_item].upper
                    else:
                        xs[attr_item] = tmp_float

            # 2. Extract the corresponding BenchmarkSet object
            # 3. Set up the configuration space
            # 4. Obtain the evalutation vector
            # print("Check out the solution: {}".format(xs))

            tmp_obj = {'obj': [set_list[self.task_id].objective_function(xs)[0][i] for i in self.obj_list],
                       'sol': params}['obj']
            final_obj = self._prepare_normalize(tmp_obj)
            return final_obj
        else:
            for cur_id, cur_sample in enumerate(xs):
                if self.current_name == "hp":
                    if self.if_gbtree:
                        xs[cur_id]["booster"] = "gbtree"
                    else:
                        xs[cur_id]["booster"] = "dart"
                if self.current_name == "hp_ranger":
                    xs[cur_id]["splitrule"] = 'extratrees'
                    xs[cur_id]["replace"] = 'TRUE'
                    xs[cur_id]["respect.unordered.factors"] = 'order'
                # xs[cur_id]["trainsize"] = train_size_list[self.task_id]

                if self.if_methods and self.task_id == 0:
                    xs[cur_id]["booster"] = "gbtree"

                if self.if_methods and self.task_id == 1:
                    xs[cur_id]["booster"] = "dart"

                if self.if_fidelity:
                    xs[cur_id]["trainsize"] = self.fidelity
                else:
                    xs[cur_id]["trainsize"] = 1.0

                # Assign the params
                for i, attr_item in enumerate(info_list):
                    # print("Current attr {} with type {}.".format(xs[attr_item].__class__, trans_params[i].__class__))
                    if self.if_methods and xs[cur_id]["booster"] == "gbtree" \
                            and (attr_item == "rate_drop" or attr_item == "skip_drop"):
                        continue
                    if self.if_methods and xs[cur_id]["booster"] == "dart" \
                            and (attr_item == "rate_drop" or attr_item == "skip_drop"):
                        xs[cur_id][attr_item] = 0.5
                        continue

                    if bool_params[i]:
                        xs[cur_id][attr_item] = int(trans_params[cur_id][i])
                    else:
                        tmp_float = float(trans_params[cur_id][i])
                        if tmp_float < info_xs[attr_item].lower:
                            xs[cur_id][attr_item] = info_xs[attr_item].lower
                        elif tmp_float > info_xs[attr_item].upper:
                            xs[cur_id][attr_item] = info_xs[attr_item].upper
                        else:
                            xs[cur_id][attr_item] = tmp_float

            # evaluate all the solutions
            results = set_list[self.task_id].objective_function(xs)
            # prepare a numpy array to store all the info
            col_size = len(self.obj_list)
            result_obj = np.zeros((batch_size, col_size))
            for cur_batch in range(batch_size):
                for cur_col_id, cur_col in enumerate(self.obj_list):
                    result_obj[cur_batch, cur_col_id] = results[cur_batch][cur_col]

            tmp_obj = {'obj': result_obj, 'sol': params}['obj']
            final_obj = self._prepare_normalize(tmp_obj)
            return final_obj


class mDTLZ1:
    def __init__(self, m: int, n: int, s: float, p: float, p_ind: int = 0):
        assert n >= m
        assert s <= 1
        assert 0 <= p <= 1

        m = int(m)
        n = int(n)
        self.m = m
        self.n_obj = self.m
        self.n = n
        self.n_dim = self.n
        self.s = s
        self.p = p
        self.k = n + 1 - m
        self.p_ind = p_ind
        self.current_name = "DTLZ1"
        self.nadir_point = [5, 5, 5]
        if p_ind == 0:
            self.p_vec = None

        else:
            self.p_vec = torch.arange(1, self.k + 1) / (p_ind * self.k)

    def pareto_set(self, sample_size: int):
        """
        :return: the pareto set answer in shape of [sample_size, n]
        """

        # In the dimension from 1 to m-1, the variables are generated uniformly
        # In the dimension from m to n, the variables are obtained through equation
        # x_j = 0.9 * b(x_I; B) * cos(E * pi * l(x_I) + ((n + 2) * j * pi)/(2 * n))

        x_I = torch.rand(sample_size, self.m - 1)
        # in the biased searching space we need to guarantee
        # the biased transformation can obtain the uniform original counterpart
        x_I = torch.pow(x_I, 1.0 / self.s)

        if self.p_ind == 0:
            x_II = torch.ones(sample_size, self.k) * self.p
        else:
            x_II = torch.zeros(sample_size, self.k) + self.p_vec

        ps_value = torch.cat((x_I, x_II), dim=1)
        return ps_value

    def g(self, x_II: torch.tensor):
        """
        :param x_II: the distance variable torch.tensor in shape of [sample_size, self.k]
        :return: the g function results torch.tensor in shape of [sample_size, 1]
        """
        g_a, g_b, g_c = 1, 1, 2
        if self.p_ind == 0:
            g_result_1 = torch.pow(x_II - self.p, 2)
            g_result_2 = g_b * torch.cos(g_c * torch.pi * (x_II - self.p))
        else:
            g_result_1 = torch.pow(x_II - self.p_vec.to(x_II.device), 2)
            g_result_2 = g_b * torch.cos(g_c * torch.pi * (x_II - self.p_vec.to(x_II.device)))

        g_result_inter = torch.sum(g_result_1 - g_result_2, dim=1).unsqueeze(1)
        g_result = g_a * (self.k + g_result_inter)
        return g_result

    def h(self, x_I: torch.tensor):
        """
        :param x_I: the position variable torch.tensor in shape of [sample_size, self.m - 1]
        :return: the h function results torch.tensor in shape of [sample_size, self.m]
        """
        sample_size, _ = x_I.shape
        x_new_I = torch.pow(x_I, self.s)
        sample_prod = torch.cumprod(x_new_I, dim=1)
        sample_minus = 1 - x_new_I
        sample_ones = torch.ones(sample_size, 1).to(x_I.device)
        sample_new_prod = torch.cat((sample_ones, sample_prod), dim=1)
        sample_new_minus = torch.cat((sample_minus, sample_ones), dim=1)
        sample_result = sample_new_prod * sample_new_minus
        sample_ans = 0.5 * torch.flip(sample_result, [1])
        return sample_ans

    def evaluate(self, solution: torch.tensor = None,
                 if_ans: bool = True, sample_size: int = 5000):
        """
        evaluate the given vector x_I and x_II with sample size of "sample_size"
        and the class attributes p/s
        :param: if_ans, False returns Pareto front, True returns the evaluated solutions
        :return:
        """
        if if_ans:
            x_I = solution[:, :(self.m - 1)]
            x_II = solution[:, (self.m - 1):]
            g_result = self.g(x_II)
            # print("Within the class dta, g_result is {} in shape of {}.".format(g_result, g_result.shape))
            h_result = self.h(x_I)
            # print("Within the class dta, h_result is {} in shape of {}.".format(h_result, h_result.shape))
            # print("The g result here is {} with X {}.".format(g_result, x_II))
            return (g_result + 1) * h_result
        else:
            ps_value = self.pareto_set(sample_size=sample_size)
            x_I = ps_value[:, :(self.m - 1)]
            x_II = ps_value[:, (self.m - 1):]
            g_result = self.g(x_II)
            h_result = self.h(x_I)
            return (g_result + 1) * h_result

    def obj(self, solution: torch.tensor):
        """
        :param solution: torch.tensor in shape of [size, self.n] in the space of [0, 1]^n
        :return: normalized results in shape of [size, self.m] in the space of [0, 1]^m
        """
        res = self.evaluate(solution)
        return res

    def showcase_params(self):
        print("********showcase_params**********")
        print("Param m is {},\nParam n is {},\n"
              "Param s is {},\nParam p is {},\nParam p_ind is {}\n".format(self.m, self.n,
                                                                           self.s, self.p,
                                                                           self.p_vec))
        print("********ending_params**********")

    def ref_and_obj(self):
        assert 3 <= self.m <= 7
        ref_vec = ref_points(self.m, DTLZ_pref_list[self.m - 3], if_norm=False)
        # ref vectors sum to 1
        obj_vec = 0.5 * ref_vec
        return ref_vec, obj_vec


class invDTLZ1:
    def __init__(self, m: int, n: int, s: float, p: float, p_ind: int = 0):
        assert n >= m
        assert s <= 1
        assert 0 <= p <= 1

        m = int(m)
        n = int(n)
        self.m = m
        self.n_obj = self.m
        self.n = n
        self.n_dim = self.n
        self.s = s
        self.p = p
        self.k = n + 1 - m
        self.p_ind = p_ind
        self.current_name = "invDTLZ1"
        self.nadir_point = [5, 5, 5]
        if p_ind == 0:
            self.p_vec = None

        else:
            self.p_vec = torch.arange(1, self.k + 1) / (p_ind * self.k)

    def pareto_set(self, sample_size: int):
        """
        :return: the pareto set answer in shape of [sample_size, n]
        """

        # In the dimension from 1 to m-1, the variables are generated uniformly
        # In the dimension from m to n, the variables are obtained through equation
        # x_j = 0.9 * b(x_I; B) * cos(E * pi * l(x_I) + ((n + 2) * j * pi)/(2 * n))

        x_I = torch.rand(sample_size, self.m - 1)
        # in the biased searching space we need to guarantee
        # the biased transformation can obtain the uniform original counterpart
        x_I = torch.pow(x_I, 1.0 / self.s)

        if self.p_ind == 0:
            x_II = torch.ones(sample_size, self.k) * self.p
        else:
            x_II = torch.zeros(sample_size, self.k) + self.p_vec

        ps_value = torch.cat((x_I, x_II), dim=1)
        return ps_value

    def g(self, x_II: torch.tensor):
        """
        :param x_II: the distance variable torch.tensor in shape of [sample_size, self.k]
        :return: the g function results torch.tensor in shape of [sample_size, 1]
        """
        g_a, g_b, g_c = 1, 1, 2
        if self.p_ind == 0:
            g_result_1 = torch.pow(x_II - self.p, 2)
            g_result_2 = g_b * torch.cos(g_c * torch.pi * (x_II - self.p))
        else:
            g_result_1 = torch.pow(x_II - self.p_vec.to(x_II.device), 2)
            g_result_2 = g_b * torch.cos(g_c * torch.pi * (x_II - self.p_vec.to(x_II.device)))

        g_result_inter = torch.sum(g_result_1 - g_result_2, dim=1).unsqueeze(1)
        g_result = g_a * (self.k + g_result_inter)
        return g_result

    def h(self, x_I: torch.tensor):
        """
        :param x_I: the position variable torch.tensor in shape of [sample_size, self.m - 1]
        :return: the h function results torch.tensor in shape of [sample_size, self.m]
        """
        sample_size, _ = x_I.shape
        x_new_I = torch.pow(x_I, self.s)
        sample_prod = torch.cumprod(x_new_I, dim=1)
        sample_minus = 1 - x_new_I
        sample_ones = torch.ones(sample_size, 1).to(x_I.device)
        sample_new_prod = torch.cat((sample_ones, sample_prod), dim=1)
        sample_new_minus = torch.cat((sample_minus, sample_ones), dim=1)
        sample_result = sample_new_prod * sample_new_minus
        sample_ans = 0.5 * torch.flip(sample_result, [1])
        return sample_ans

    def evaluate(self, solution: torch.tensor = None,
                 if_ans: bool = True, sample_size: int = 5000):
        """
        evaluate the given vector x_I and x_II with sample size of "sample_size"
        and the class attributes p/s
        :param: if_ans, False returns Pareto front, True returns the evaluated solutions
        :return:
        """
        if if_ans:
            x_I = solution[:, :(self.m - 1)]
            x_II = solution[:, (self.m - 1):]
            g_result = self.g(x_II)
            # print("Within the class dta, g_result is {} in shape of {}.".format(g_result, g_result.shape))
            h_result = self.h(x_I)
            # print("Within the class dta, h_result is {} in shape of {}.".format(h_result, h_result.shape))
            # print("The g result here is {} with X {}.".format(g_result, x_II))
            return - (g_result + 1) * h_result
        else:
            ps_value = self.pareto_set(sample_size=sample_size)
            x_I = ps_value[:, :(self.m - 1)]
            x_II = ps_value[:, (self.m - 1):]
            g_result = self.g(x_II)
            h_result = self.h(x_I)
            return - (g_result + 1) * h_result

    def obj(self, solution: torch.tensor):
        """
        :param solution: torch.tensor in shape of [size, self.n] in the space of [0, 1]^n
        :return: normalized results in shape of [size, self.m] in the space of [0, 1]^m
        """
        res = self.evaluate(solution)
        return res

    def showcase_params(self):
        print("********showcase_params**********")
        print("Param m is {},\nParam n is {},\n"
              "Param s is {},\nParam p is {},\nParam p_ind is {}\n".format(self.m, self.n,
                                                                           self.s, self.p,
                                                                           self.p_vec))
        print("********ending_params**********")

    def ref_and_obj(self):
        assert 3 <= self.m <= 7
        ref_vec = ref_points(self.m, DTLZ_pref_list[self.m - 3], if_norm=False)
        # ref vectors sum to 1
        obj_vec = -0.5 * ref_vec
        return ref_vec, obj_vec


class mDTLZ2:
    def __init__(self, m: int, n: int, s: float, p: float, p_ind: int = 0):
        assert n >= m
        assert s <= 1
        assert 0 <= p <= 1

        m = int(m)
        n = int(n)
        self.m = m
        self.n_obj = self.m
        self.n = n
        self.n_dim = self.n
        self.s = s
        self.p = p
        self.k = n + 1 - m
        self.p_ind = p_ind
        self.current_name = "DTLZ2"
        self.nadir_point = [3.5, 3.5, 3.5]
        if p_ind == 0:
            self.p_vec = None
        else:
            self.p_vec = torch.arange(1, self.k + 1) / (p_ind * self.k)

    def pareto_set(self, sample_size: int):
        """
        :return: the pareto set answer in shape of [sample_size, n]
        """

        # In the dimension from 1 to m-1, the variables are generated uniformly
        # In the dimension from m to n, the variables are obtained through equation
        # x_j = 0.9 * b(x_I; B) * cos(E * pi * l(x_I) + ((n + 2) * j * pi)/(2 * n))

        x_I = torch.rand(sample_size, self.m - 1)

        if self.p_ind == 0:
            x_II = torch.ones(sample_size, self.k) * self.p
        else:
            x_II = torch.zeros(sample_size, self.k) + self.p_vec

        ps_value = torch.cat((x_I, x_II), dim=1)
        return ps_value

    def g(self, x_II: torch.tensor):
        """
        :param x_II: the distance variable torch.tensor in shape of [sample_size, self.k]
        :return: the g function results torch.tensor in shape of [sample_size, 1]
        """
        if self.p_ind == 0:
            g_result_1 = torch.pow(x_II - self.p, 2)
            # g_result_2 = torch.cos(20 * torch.pi * (x_II - self.p))
        else:
            g_result_1 = torch.pow(x_II - self.p_vec.to(x_II.device), 2)
            # g_result_2 = torch.cos(20 * torch.pi * (x_II - self.p_vec))
        g_result_inter = torch.sum(g_result_1, dim=1).unsqueeze(1)
        g_result = g_result_inter
        return g_result

    def h(self, x_I: torch.tensor):
        """
        :param x_I: the position variable torch.tensor in shape of [sample_size, self.m - 1]
        :return: the h function results torch.tensor in shape of [sample_size, self.m]
        """
        sample_size, _ = x_I.shape
        x_new_I = torch.pow(x_I, self.s) * torch.pi / 2
        sample_prod = torch.cumprod(torch.cos(x_new_I), dim=1)
        sample_minus = torch.sin(x_new_I)
        sample_ones = torch.ones(sample_size, 1).to(x_I.device)
        sample_new_prod = torch.cat((sample_ones, sample_prod), dim=1)
        sample_new_minus = torch.cat((sample_minus, sample_ones), dim=1)
        sample_result = sample_new_prod * sample_new_minus
        sample_ans = torch.flip(sample_result, [1])
        return sample_ans

    def evaluate(self, solution: torch.tensor = None, if_ans: bool = True, sample_size: int = 5000):
        """
        evaluate the given vector x_I and x_II with sample size of "sample_size"
        and the class attributes p/s
        :param: if_ans, False returns Pareto front, True returns the evaluated solutions
        :return:
        """
        if if_ans:
            x_I = solution[:, :(self.m - 1)]
            x_II = solution[:, (self.m - 1):]
            g_result = self.g(x_II)
            # print("Within the class dta, g_result is {} in shape of {}.".format(g_result, g_result.shape))
            h_result = self.h(x_I)
            # print("Within the class dta, h_result is {} in shape of {}.".format(h_result, h_result.shape))
            return (g_result + 1) * h_result
        else:
            ps_value = self.pareto_set(sample_size=sample_size)
            x_I = ps_value[:, :(self.m - 1)]
            x_II = ps_value[:, (self.m - 1):]
            g_result = self.g(x_II)
            h_result = self.h(x_I)
            return (g_result + 1) * h_result

    def obj(self, solution: torch.tensor):
        """
        :param solution: torch.tensor in shape of [size, self.n] in the space of [0, 1]^n
        :return: normalized results in shape of [size, self.m] in the space of [0, 1]^m
        """
        res = self.evaluate(solution)
        return res

    def showcase_params(self):
        print("********showcase_params**********")
        print("Param m is {},\nParam n is {},\n"
              "Param s is {},\nParam p is {},\n".format(self.m, self.n,
                                                        self.s, self.p_vec))
        print("********ending_params**********")

    def ref_and_obj(self):
        assert 3 <= self.m <= 7
        ref_vec = ref_points(self.m, DTLZ_pref_list[self.m - 3], if_norm=False)
        # ref vectors sum to 1
        obj_vec = generate_norm(ref_vec)
        # obj vectors norm equals to 1
        return ref_vec, obj_vec


class invDTLZ2:
    def __init__(self, m: int, n: int, s: float, p: float, p_ind: int = 0):
        assert n >= m
        assert s <= 1
        assert 0 <= p <= 1

        m = int(m)
        n = int(n)
        self.m = m
        self.n_obj = self.m
        self.n = n
        self.n_dim = self.n
        self.s = s
        self.p = p
        self.k = n + 1 - m
        self.p_ind = p_ind
        self.current_name = "invDTLZ2"
        self.nadir_point = [3.5, 3.5, 3.5]
        if p_ind == 0:
            self.p_vec = None
        else:
            self.p_vec = torch.arange(1, self.k + 1) / (p_ind * self.k)

    def pareto_set(self, sample_size: int):
        """
        :return: the pareto set answer in shape of [sample_size, n]
        """

        # In the dimension from 1 to m-1, the variables are generated uniformly
        # In the dimension from m to n, the variables are obtained through equation
        # x_j = 0.9 * b(x_I; B) * cos(E * pi * l(x_I) + ((n + 2) * j * pi)/(2 * n))

        x_I = torch.rand(sample_size, self.m - 1)

        if self.p_ind == 0:
            x_II = torch.ones(sample_size, self.k) * self.p
        else:
            x_II = torch.zeros(sample_size, self.k) + self.p_vec

        ps_value = torch.cat((x_I, x_II), dim=1)
        return ps_value

    def g(self, x_II: torch.tensor):
        """
        :param x_II: the distance variable torch.tensor in shape of [sample_size, self.k]
        :return: the g function results torch.tensor in shape of [sample_size, 1]
        """
        if self.p_ind == 0:
            g_result_1 = torch.pow(x_II - self.p, 2)
            # g_result_2 = torch.cos(20 * torch.pi * (x_II - self.p))
        else:
            g_result_1 = torch.pow(x_II - self.p_vec.to(x_II.device), 2)
            # g_result_2 = torch.cos(20 * torch.pi * (x_II - self.p_vec))
        g_result_inter = torch.sum(g_result_1, dim=1).unsqueeze(1)
        g_result = g_result_inter
        return - g_result

    def h(self, x_I: torch.tensor):
        """
        :param x_I: the position variable torch.tensor in shape of [sample_size, self.m - 1]
        :return: the h function results torch.tensor in shape of [sample_size, self.m]
        """
        sample_size, _ = x_I.shape
        x_new_I = torch.pow(x_I, self.s) * torch.pi / 2
        sample_prod = torch.cumprod(torch.cos(x_new_I), dim=1)
        sample_minus = torch.sin(x_new_I)
        sample_ones = torch.ones(sample_size, 1).to(x_I.device)
        sample_new_prod = torch.cat((sample_ones, sample_prod), dim=1)
        sample_new_minus = torch.cat((sample_minus, sample_ones), dim=1)
        sample_result = sample_new_prod * sample_new_minus
        sample_ans = torch.flip(sample_result, [1])
        return sample_ans

    def evaluate(self, solution: torch.tensor = None, if_ans: bool = True, sample_size: int = 5000):
        """
        evaluate the given vector x_I and x_II with sample size of "sample_size"
        and the class attributes p/s
        :param: if_ans, False returns Pareto front, True returns the evaluated solutions
        :return:
        """
        if if_ans:
            x_I = solution[:, :(self.m - 1)]
            x_II = solution[:, (self.m - 1):]
            g_result = self.g(x_II)
            # print("Within the class dta, g_result is {} in shape of {}.".format(g_result, g_result.shape))
            h_result = self.h(x_I)
            # print("Within the class dta, h_result is {} in shape of {}.".format(h_result, h_result.shape))
            return - (g_result + 1) * h_result
        else:
            ps_value = self.pareto_set(sample_size=sample_size)
            x_I = ps_value[:, :(self.m - 1)]
            x_II = ps_value[:, (self.m - 1):]
            g_result = self.g(x_II)
            h_result = self.h(x_I)
            return - (g_result + 1) * h_result

    def obj(self, solution: torch.tensor):
        """
        :param solution: torch.tensor in shape of [size, self.n] in the space of [0, 1]^n
        :return: normalized results in shape of [size, self.m] in the space of [0, 1]^m
        """
        res = self.evaluate(solution)
        return res

    def showcase_params(self):
        print("********showcase_params**********")
        print("Param m is {},\nParam n is {},\n"
              "Param s is {},\nParam p is {},\n".format(self.m, self.n,
                                                        self.s, self.p_vec))
        print("********ending_params**********")

    def ref_and_obj(self):
        assert 3 <= self.m <= 7
        ref_vec = ref_points(self.m, DTLZ_pref_list[self.m - 3], if_norm=False)
        # ref vectors sum to 1
        obj_vec = - generate_norm(ref_vec)
        # obj vectors norm equals to 1
        return ref_vec, obj_vec


class mDTLZ3:
    def __init__(self, m: int, n: int, s: float, p: float, p_ind: int = 0):
        assert n >= m
        assert s <= 1
        assert 0 <= p <= 1

        m = int(m)
        n = int(n)
        self.m = m
        self.n_obj = self.m
        self.n = n
        self.n_dim = self.n
        self.s = s
        self.p = p
        self.k = n + 1 - m
        self.p_ind = p_ind
        self.current_name = "DTLZ3"
        self.nadir_point = [2, 2, 2]
        if p_ind == 0:
            self.p_vec = None
        else:
            self.p_vec = torch.arange(1, self.k + 1) / (p_ind * self.k)

    def pareto_set(self, sample_size: int):
        """
        :return: the pareto set answer in shape of [sample_size, n]
        """

        # In the dimension from 1 to m-1, the variables are generated uniformly
        # In the dimension from m to n, the variables are obtained through equation
        # x_j = 0.9 * b(x_I; B) * cos(E * pi * l(x_I) + ((n + 2) * j * pi)/(2 * n))

        x_I = torch.rand(sample_size, self.m - 1)

        if self.p_ind == 0:
            x_II = torch.ones(sample_size, self.k) * self.p
        else:
            x_II = torch.zeros(sample_size, self.k) + self.p_vec

        ps_value = torch.cat((x_I, x_II), dim=1)
        return ps_value

    def g(self, x_II: torch.tensor):
        """
        :param x_II: the distance variable torch.tensor in shape of [sample_size, self.k]
        :return: the g function results torch.tensor in shape of [sample_size, 1]
        """
        g_a, g_b, g_c = 0.1, 0.1, 2
        if self.p_ind == 0:
            g_result_1 = torch.pow(x_II - self.p, 2)
            g_result_2 = g_b * torch.cos(g_c * torch.pi * (x_II - self.p))
        else:
            g_result_1 = torch.pow(x_II - self.p_vec.to(x_II.device), 2)
            g_result_2 = g_b * torch.cos(g_c * torch.pi * (x_II - self.p_vec.to(x_II.device)))
        g_result_inter = torch.sum(g_result_1 - g_result_2, dim=1).unsqueeze(1)
        g_result = g_a * (self.k + g_result_inter)
        return g_result

    def h(self, x_I: torch.tensor):
        """
        :param x_I: the position variable torch.tensor in shape of [sample_size, self.m - 1]
        :return: the h function results torch.tensor in shape of [sample_size, self.m]
        """
        sample_size, _ = x_I.shape
        x_new_I = torch.pow(x_I, self.s) * torch.pi / 2
        sample_prod = torch.cumprod(torch.cos(x_new_I), dim=1)
        sample_minus = torch.sin(x_new_I)
        sample_ones = torch.ones(sample_size, 1).to(x_I.device)
        sample_new_prod = torch.cat((sample_ones, sample_prod), dim=1)
        sample_new_minus = torch.cat((sample_minus, sample_ones), dim=1)
        sample_result = sample_new_prod * sample_new_minus
        sample_ans = torch.flip(sample_result, [1])
        return sample_ans

    def evaluate(self, solution: torch.tensor = None, if_ans: bool = True, sample_size: int = 5000):
        """
        evaluate the given vector x_I and x_II with sample size of "sample_size"
        and the class attributes p/s
        :param: if_ans, True returns Pareto front, False returns the evaluated solutions
        :return:
        """
        if if_ans:
            x_I = solution[:, :(self.m - 1)]
            x_II = solution[:, (self.m - 1):]
            g_result = self.g(x_II)
            # print("Within the class dta, g_result is {} in shape of {}.".format(g_result, g_result.shape))
            h_result = self.h(x_I)
            # print("Within the class dta, h_result is {} in shape of {}.".format(h_result, h_result.shape))
            return (g_result + 1) * h_result
        else:
            ps_value = self.pareto_set(sample_size=sample_size)
            x_I = ps_value[:, :(self.m - 1)]
            x_II = ps_value[:, (self.m - 1):]
            g_result = self.g(x_II)
            h_result = self.h(x_I)
            return (g_result + 1) * h_result

    def obj(self, solution: torch.tensor):
        """
        :param solution: torch.tensor in shape of [size, self.n] in the space of [0, 1]^n
        :return: normalized results in shape of [size, self.m] in the space of [0, 1]^m
        """
        res = self.evaluate(solution)
        return res

    def showcase_params(self):
        print("********showcase_params**********")
        print("Param m is {},\nParam n is {},\n"
              "Param s is {},\nParam p is {},\n".format(self.m, self.n,
                                                        self.s, self.p_vec))
        print("********ending_params**********")

    def ref_and_obj(self):
        assert 3 <= self.m <= 7
        ref_vec = ref_points(self.m, DTLZ_pref_list[self.m - 3], if_norm=False)
        # ref vectors sum to 1
        obj_vec = generate_norm(ref_vec)
        # obj vectors norm equals to 1
        return ref_vec, obj_vec


class invDTLZ3:
    def __init__(self, m: int, n: int, s: float, p: float, p_ind: int = 0):
        assert n >= m
        assert s <= 1
        assert 0 <= p <= 1

        m = int(m)
        n = int(n)
        self.m = m
        self.n_obj = self.m
        self.n = n
        self.n_dim = self.n
        self.s = s
        self.p = p
        self.k = n + 1 - m
        self.p_ind = p_ind
        self.current_name = "invDTLZ3"
        self.nadir_point = [2, 2, 2]
        if p_ind == 0:
            self.p_vec = None
        else:
            self.p_vec = torch.arange(1, self.k + 1) / (p_ind * self.k)

    def pareto_set(self, sample_size: int):
        """
        :return: the pareto set answer in shape of [sample_size, n]
        """

        # In the dimension from 1 to m-1, the variables are generated uniformly
        # In the dimension from m to n, the variables are obtained through equation
        # x_j = 0.9 * b(x_I; B) * cos(E * pi * l(x_I) + ((n + 2) * j * pi)/(2 * n))

        x_I = torch.rand(sample_size, self.m - 1)

        if self.p_ind == 0:
            x_II = torch.ones(sample_size, self.k) * self.p
        else:
            x_II = torch.zeros(sample_size, self.k) + self.p_vec

        ps_value = torch.cat((x_I, x_II), dim=1)
        return ps_value

    def g(self, x_II: torch.tensor):
        """
        :param x_II: the distance variable torch.tensor in shape of [sample_size, self.k]
        :return: the g function results torch.tensor in shape of [sample_size, 1]
        """
        g_a, g_b, g_c = 0.1, 0.1, 2
        if self.p_ind == 0:
            g_result_1 = torch.pow(x_II - self.p, 2)
            g_result_2 = g_b * torch.cos(g_c * torch.pi * (x_II - self.p))
        else:
            g_result_1 = torch.pow(x_II - self.p_vec.to(x_II.device), 2)
            g_result_2 = g_b * torch.cos(g_c * torch.pi * (x_II - self.p_vec.to(x_II.device)))
        g_result_inter = torch.sum(g_result_1 - g_result_2, dim=1).unsqueeze(1)
        g_result = g_a * (self.k + g_result_inter)
        return - g_result

    def h(self, x_I: torch.tensor):
        """
        :param x_I: the position variable torch.tensor in shape of [sample_size, self.m - 1]
        :return: the h function results torch.tensor in shape of [sample_size, self.m]
        """
        sample_size, _ = x_I.shape
        x_new_I = torch.pow(x_I, self.s) * torch.pi / 2
        sample_prod = torch.cumprod(torch.cos(x_new_I), dim=1)
        sample_minus = torch.sin(x_new_I)
        sample_ones = torch.ones(sample_size, 1).to(x_I.device)
        sample_new_prod = torch.cat((sample_ones, sample_prod), dim=1)
        sample_new_minus = torch.cat((sample_minus, sample_ones), dim=1)
        sample_result = sample_new_prod * sample_new_minus
        sample_ans = torch.flip(sample_result, [1])
        return sample_ans

    def evaluate(self, solution: torch.tensor = None, if_ans: bool = True, sample_size: int = 5000):
        """
        evaluate the given vector x_I and x_II with sample size of "sample_size"
        and the class attributes p/s
        :param: if_ans, True returns Pareto front, False returns the evaluated solutions
        :return:
        """
        if if_ans:
            x_I = solution[:, :(self.m - 1)]
            x_II = solution[:, (self.m - 1):]
            g_result = self.g(x_II)
            # print("Within the class dta, g_result is {} in shape of {}.".format(g_result, g_result.shape))
            h_result = self.h(x_I)
            # print("Within the class dta, h_result is {} in shape of {}.".format(h_result, h_result.shape))
            return - (g_result + 1) * h_result
        else:
            ps_value = self.pareto_set(sample_size=sample_size)
            x_I = ps_value[:, :(self.m - 1)]
            x_II = ps_value[:, (self.m - 1):]
            g_result = self.g(x_II)
            h_result = self.h(x_I)
            return - (g_result + 1) * h_result

    def obj(self, solution: torch.tensor):
        """
        :param solution: torch.tensor in shape of [size, self.n] in the space of [0, 1]^n
        :return: normalized results in shape of [size, self.m] in the space of [0, 1]^m
        """
        res = self.evaluate(solution)
        return res

    def showcase_params(self):
        print("********showcase_params**********")
        print("Param m is {},\nParam n is {},\n"
              "Param s is {},\nParam p is {},\n".format(self.m, self.n,
                                                        self.s, self.p_vec))
        print("********ending_params**********")

    def ref_and_obj(self):
        assert 3 <= self.m <= 7
        ref_vec = ref_points(self.m, DTLZ_pref_list[self.m - 3], if_norm=False)
        # ref vectors sum to 1
        tmp_pareto_set = self.pareto_set(sample_size=1)
        tmp_pareto_front = self.result(tmp_pareto_set)
        coef = torch.norm(tmp_pareto_front, dim=1)[0].item()
        obj_vec = - coef * generate_norm(ref_vec)
        # obj vectors norm equals to 1
        return ref_vec, obj_vec


class F1():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0.0
        count1 = count2 = 0.0

        for i in range(2, n + 1):
            yi = x[:, i - 1] - torch.pow(2 * x[:, 0] - 1, 2)
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class F2():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0.0
        count1 = count2 = 0.0

        for i in range(2, n + 1):
            theta = 1.0 + 3.0 * (i - 2) / (n - 2)
            yi = x[:, i - 1] - torch.pow(x[:, 0], 0.5 * theta)
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class F3():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0.0
        count1 = count2 = 0.0

        for i in range(2, n + 1):
            xi = x[:, i - 1]
            yi = xi - (torch.sin(4.0 * np.pi * x[:, 0] + i * np.pi / n) + 1) / 2
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0]))

        objs = torch.stack([f1, f2]).T

        return objs


class F4():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0
        count1 = count2 = 0

        for i in range(2, n + 1):
            xi = -1.0 + 2.0 * x[:, i - 1]

            if i % 2 == 0:
                yi = xi - 0.8 * x[:, 0] * torch.sin(4.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8 * x[:, 0] * torch.cos(4.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class F5():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0
        count1 = count2 = 0

        for i in range(2, n + 1):
            xi = -1.0 + 2.0 * x[:, i - 1]

            if i % 2 == 0:
                yi = xi - 0.8 * x[:, 0] * torch.sin(4.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8 * x[:, 0] * torch.cos((4.0 * np.pi * x[:, 0] + i * np.pi / n) / 3)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class F6():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0
        count1 = count2 = 0

        for i in range(2, n + 1):
            xi = -1.0 + 2.0 * x[:, i - 1]

            if i % 2 == 0:
                yi = xi - (0.3 * x[:, 0] ** 2 * torch.cos(12.0 * np.pi * x[:, 0] + 4 * i * np.pi / n) + 0.6 * x[:,
                                                                                                              0]) * torch.sin(
                    6.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - (0.3 * x[:, 0] ** 2 * torch.cos(12.0 * np.pi * x[:, 0] + 4 * i * np.pi / n) + 0.6 * x[:,
                                                                                                              0]) * torch.cos(
                    6.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class VLMOP1():
    def __init__(self, n_dim=1):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([-2.0]).float()
        self.ubound = torch.tensor([4.0]).float()
        self.nadir_point = [4, 4]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        f1 = torch.pow(x[:, 0], 2)
        f2 = torch.pow(x[:, 0] - 2, 2)

        objs = torch.stack([f1, f2]).T

        return objs


class VLMOP2():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0]).float()
        self.ubound = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        f1 = 1 - torch.exp(-torch.sum((x - 1 / np.sqrt(n)) ** 2, axis=1))
        f2 = 1 - torch.exp(-torch.sum((x + 1 / np.sqrt(n)) ** 2, axis=1))

        objs = torch.stack([f1, f2]).T

        return objs


class VLMOP3():
    def __init__(self, n_dim=2):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([-3.0, -3.0]).float()
        self.ubound = torch.tensor([3.0, 3.0]).float()
        self.nadir_point = [10, 60, 1]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1, x2 = x[:, 0], x[:, 1]

        f1 = 0.5 * (x1 ** 2 + x2 ** 2) + torch.sin(x1 ** 2 + x2 ** 2)
        f2 = (3 * x1 - 2 * x2 + 4) ** 2 / 8 + (x1 - x2 + 1) ** 2 / 27 + 15
        f3 = 1 / (x1 ** 2 + x2 ** 2 + 1) - 1.1 * torch.exp(-x1 ** 2 - x2 ** 2)

        objs = torch.stack([f1, f2, f3]).T

        return objs


class DTLZ2():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = torch.sum(torch.stack([torch.pow(x[:, i] - 0.5, 2) for i in range(2, n)]), axis=0)
        g = sum1

        f1 = (1 + g) * torch.cos(x[:, 0] * np.pi / 2) * torch.cos(x[:, 1] * np.pi / 2)
        f2 = (1 + g) * torch.cos(x[:, 0] * np.pi / 2) * torch.sin(x[:, 1] * np.pi / 2)
        f3 = (1 + g) * torch.sin(x[:, 0] * np.pi / 2)

        objs = torch.stack([f1, f2, f3]).T

        return objs


class PT:
    def __init__(self, n_dim=None):
        self.M_cm2 = torch.tensor(read_matrix_from_file("./mdata/M_CIMS_2.txt"))
        self.M_pm1 = torch.tensor(read_matrix_from_file("./mdata/M_PIMS_1.txt"))
        self.M_pm2 = torch.tensor(read_matrix_from_file("./mdata/M_PIMS_2.txt"))
        self.M_nm2 = torch.tensor(read_matrix_from_file("./mdata/M_NIMS_2.txt"))
        self.S_cm2 = torch.tensor(read_matrix_from_file("./mdata/S_CIMS_2.txt"))
        self.S_pm1 = torch.tensor(read_matrix_from_file("./mdata/S_PIMS_1.txt"))
        self.S_ph2 = torch.tensor(read_matrix_from_file("./mdata/S_PIHS_2.txt"))
        self.S_pl2 = torch.tensor(read_matrix_from_file("./mdata/S_PILS_2.txt"))


class P1T1(PT):
    def __init__(self, n_dim=50):
        super().__init__(self)
        self.current_name = "P1T1"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[0] = 0
        self.lbound[1:] = self.lbound[1:] * -100
        self.ubound = torch.ones(n_dim)
        self.ubound[1:] = self.ubound[1:] * 100
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.coef = 1

    def q(self, x):
        # Compute the q function to scale both objectives
        return 1 + self.coef * torch.sum(torch.square(x), dim=1)

    def pareto_x(self, sample_size=500):
        optimal_x = torch.ones(sample_size, self.n_dim) * 0.5
        optimal_x[:, 0] = torch.arange(sample_size)/sample_size

        return optimal_x

    def pareto_y(self):
        # Transform the pareto_x to the real variables
        x = self.pareto_x(2000)
        y = self.evaluate(x)
        y_plus = y
        y_sum = torch.sum(y_plus, dim=1)[:, None]
        y_ref = y_plus / y_sum

        return y, y_ref

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim) * 0.5
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        # f1_max = 6e4
        # f2_max = 6e4
        f1_max = 1
        f2_max = 1


        f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        f1 = f1 / f1_max
        f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs


class P1T2(PT):
    def __init__(self, n_dim=50):
        super().__init__(self)
        self.current_name = "P1T2"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[0] = 0
        self.lbound[1:] = self.lbound[1:] * -100
        self.ubound = torch.ones(n_dim)
        self.ubound[1:] = self.ubound[1:] * 100
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.coef = 1
        # self.coef = 1

    def q(self, x):
        # Compute the q function to scale both objectives
        return 1 + self.coef * torch.sum(torch.abs(x), dim=1) * 9 / (self.n_dim - 1)

    def pareto_x(self, sample_size=500):
        optimal_x = torch.ones(sample_size, self.n_dim) * 0.5
        optimal_x[:, 0] = torch.arange(sample_size)/sample_size

        return optimal_x

    def pareto_y(self):
        # Transform the pareto_x to the real variables
        x = self.pareto_x(2000)
        y = self.evaluate(x)
        y_plus = y
        y_sum = torch.sum(y_plus, dim=1)[:, None]
        y_ref = y_plus / y_sum

        return y, y_ref

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim) * 0.5
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        f1_max = 1
        f2_max = 1

        f1 = x[:, 0]
        # f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        f1 = f1 / f1_max

        f2 = self.q(x[:, 1:]) * (1 - torch.square(x[:, 0] / self.q(x[:, 1:])))
        # f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs


class P2T1(PT):
    def __init__(self, n_dim=10):
        super().__init__(self)
        self.current_name = "P2T1"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[0] = 0
        self.lbound[1:] = self.lbound[1:] * -5
        self.ubound = torch.ones(n_dim)
        self.ubound[1:] = self.ubound[1:] * 5
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.coef_1 = 100
        self.coef_2 = 1

    def q(self, x):
        # Compute the q function to scale both objectives
        x_plus = x[:, 1:]
        x_minus = x[:, :-1]
        return 1 + torch.sum(self.coef_1 * torch.square(torch.square(x_minus) - x_plus)
                             + self.coef_2 * torch.square(1 - x_minus), dim=1)

    def pareto_x(self, sample_size=500):
        optimal_x = torch.ones(sample_size, self.n_dim) * 0.6
        optimal_x[:, 0] = torch.arange(sample_size)/sample_size

        return optimal_x

    def pareto_y(self):
        # Transform the pareto_x to the real variables
        x = self.pareto_x(2000)
        y = self.evaluate(x)
        y_plus = y
        y_sum = torch.sum(y_plus, dim=1)[:, None]
        y_ref = y_plus / y_sum

        return y, y_ref

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim) * 0.6
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        f1_max = 1
        # f2_max = 8
        f2_max = 1

        f1 = x[:, 0]
        f1 = f1 / f1_max
        f2 = self.q(x[:, 1:]) * (1 - torch.square(x[:, 0] / self.q(x[:, 1:])))
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs


class P2T2(PT):
    def __init__(self, n_dim=10):
        super().__init__(self)
        self.current_name = "P2T2"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[0] = 0
        self.lbound[1:] = self.lbound[1:] * -5
        self.ubound = torch.ones(n_dim)
        self.ubound[1:] = self.ubound[1:] * 5
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.coef = 1

    def pareto_x(self, sample_size=500):
        optimal_x = torch.ones(sample_size, self.n_dim) * 0.6
        optimal_x[:, 0] = torch.arange(sample_size)/sample_size

        if optimal_x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        optimal_x[:, 1:] = (self.S_cm2 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])

        return optimal_x

    def pareto_y(self):
        # Transform the pareto_x to the real variables
        x = self.pareto_x(2000)
        y = self.evaluate(x)
        y_plus = y
        y_sum = torch.sum(y_plus, dim=1)[:, None]
        y_ref = y_plus / y_sum

        return y, y_ref

    def q(self, x):
        # Compute the q function to scale both objectives
        # Compute z first using M_cm2 and S_cm2
        # print("M shape of {}, x shape of {}, and S shape of {}.".format(M_cm2.shape,
        #                                                                 x.shape,
        #                                                                 S_cm2.shape))
        z = torch.matmul((x - self.S_cm2), self.M_cm2.T)
        return 1 + self.coef * torch.sum(torch.abs(z), dim=1) * 9 / (self.n_dim - 1)

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim)
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        # f1_max = 2
        # f2_max = 2
        f1_max = 1
        f2_max = 1

        f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        f1 = f1 / f1_max
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs


class P3T1(PT):
    def __init__(self, n_dim=50):
        super().__init__(self)
        self.current_name = "P3T1"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[0] = 0
        self.lbound[1:] = self.lbound[1:] * -2
        self.ubound = torch.ones(n_dim)
        self.ubound[1:] = self.ubound[1:] * 2
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.coef_1 = 1
        self.coef_2 = 10

    def q(self, x):
        # Compute the q function to scale both objectives
        return 1 + self.coef_1 * torch.sum(torch.square(x) - self.coef_2 * torch.cos(2 * torch.pi * x) + 10, dim=1)

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim) * 0.5
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        # f1_max = 6.5
        # f2_max = 6.5
        f1_max = 1
        f2_max = 1

        f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        f1 = f1 / f1_max
        f2 = f2 / f2_max
        # f1 = x[:, 0]
        # f1 = f1 / f1_max
        # f2 = self.q(x[:, 1:]) * (1 - torch.sqrt(x[:, 0] / self.q(x[:, 1:])))
        # f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs

    def pareto_x(self, sample_size=500):
        # optimal_x = torch.ones(sample_size, self.n_dim) * 0.6
        optimal_x = torch.ones(sample_size, self.n_dim) * 0.5
        optimal_x[:, 0] = torch.arange(sample_size)/sample_size

        # if optimal_x.device.type == 'cuda':
        #     self.lbound = self.lbound.cuda()
        #     self.ubound = self.ubound.cuda()
        #
        # optimal_x[:, 1:] = (S_cm2 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])

        return optimal_x

    def pareto_y(self):
        # Transform the pareto_x to the real variables
        x = self.pareto_x(2000)
        y = self.evaluate(x)
        y_plus = y
        y_sum = torch.sum(y_plus, dim=1)[:, None]
        y_ref = y_plus / y_sum

        return y, y_ref


class P3T2(PT):
    def __init__(self, n_dim=50):
        super().__init__(self)
        self.current_name = "P3T2"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[0] = 0
        self.lbound[1:] = self.lbound[1:] * -1
        self.ubound = torch.ones(n_dim)
        self.ubound[1:] = self.ubound[1:] * 1
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.coef_1 = 1
        self.coef_2 = 1

    def q(self, x):
        # Compute the q function to scale both objectives
        return 21 + math.exp(1) - \
            20 * torch.exp(-self.coef_1 * torch.sqrt(torch.sum(torch.square(x), dim=1) / (self.n_dim - 1))) - \
            self.coef_2 * torch.exp(torch.sum(torch.cos(2 * torch.pi * x), dim=1) / (self.n_dim - 1))
        # return 1 + self.coef * torch.sum(torch.square(x), dim=1)

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim) * 0.5
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        f1_max = 1
        f2_max = 1

        # f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        # f1 = f1 / f1_max
        # f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        # f2 = f2 / f2_max
        f1 = x[:, 0]
        f2 = self.q(x[:, 1:]) * (1 - torch.sqrt(x[:, 0] / self.q(x[:, 1:])))
        f1 = f1 / f1_max
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs

    def pareto_x(self, sample_size=500):
        # optimal_x = torch.ones(sample_size, self.n_dim) * 0.6
        optimal_x = torch.ones(sample_size, self.n_dim) * 0.5
        optimal_x[:, 0] = torch.arange(sample_size)/sample_size

        # if optimal_x.device.type == 'cuda':
        #     self.lbound = self.lbound.cuda()
        #     self.ubound = self.ubound.cuda()
        #
        # optimal_x[:, 1:] = (S_cm2 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])

        return optimal_x

    def pareto_y(self):
        # Transform the pareto_x to the real variables
        x = self.pareto_x(2000)
        y = self.evaluate(x)
        y_plus = y
        y_sum = torch.sum(y_plus, dim=1)[:, None]
        y_ref = y_plus / y_sum

        return y, y_ref


class P4T1(PT):
    def __init__(self, n_dim=50):
        super().__init__(self)
        self.current_name = "P4T1"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[0] = 0
        self.lbound[1:] = self.lbound[1:] * -100
        self.ubound = torch.ones(n_dim)
        self.ubound[1:] = self.ubound[1:] * 100
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.coef = 1

    def q(self, x):
        # Compute the q function to scale both objectives
        return 1 + self.coef * torch.sum(torch.square(x), dim=1)

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim) * 0.5
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        # f1_max = 1
        # f2_max = 70
        f1_max = 1
        f2_max = 1


        # f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        # f1 = f1 / f1_max
        # f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        # f2 = f2 / f2_max
        f1 = x[:, 0]
        f2 = self.q(x[:, 1:]) * (1 - torch.sqrt(x[:, 0] / self.q(x[:, 1:])))
        f1 = f1 / f1_max
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs

    def pareto_x(self, sample_size=500):
        # optimal_x = torch.ones(sample_size, self.n_dim) * 0.6
        optimal_x = torch.ones(sample_size, self.n_dim) * 0.5
        optimal_x[:, 0] = torch.arange(sample_size)/sample_size

        # if optimal_x.device.type == 'cuda':
        #     self.lbound = self.lbound.cuda()
        #     self.ubound = self.ubound.cuda()
        #
        # optimal_x[:, 1:] = (S_cm2 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])

        return optimal_x

    def pareto_y(self):
        # Transform the pareto_x to the real variables
        x = self.pareto_x(2000)
        y = self.evaluate(x)
        y_plus = y
        y_sum = torch.sum(y_plus, dim=1)[:, None]
        y_ref = y_plus / y_sum

        return y, y_ref


class P4T2(PT):
    def __init__(self, n_dim=50):
        super().__init__(self)
        self.current_name = "P4T2"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[0] = 0
        self.lbound[1:] = self.lbound[1:] * -100
        self.ubound = torch.ones(n_dim)
        self.ubound[1:] = self.ubound[1:] * 100
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.coef_1 = 1
        self.coef_2 = 10

    def q(self, x):
        # Compute the q function to scale both objectives
        z = x - self.S_ph2
        return 1 + self.coef_1 * torch.sum(torch.square(z) - self.coef_2 * torch.cos(2 * torch.pi * z) + 10, dim=1)

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim)
        x[:, 1:] = (self.S_ph2 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        # f1_max = 1
        # f2_max = 70
        f1_max = 1
        f2_max = 1

        # f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        # f1 = f1 / f1_max
        # f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        # f2 = f2 / f2_max
        f1 = x[:, 0]
        f2 = self.q(x[:, 1:]) * (1 - torch.sqrt(x[:, 0] / self.q(x[:, 1:])))
        f1 = f1 / f1_max
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs

    def pareto_x(self, sample_size=500):
        # optimal_x = torch.ones(sample_size, self.n_dim) * 0.6
        optimal_x = torch.ones(sample_size, self.n_dim) * 0.5
        optimal_x[:, 0] = torch.arange(sample_size)/sample_size

        if optimal_x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        optimal_x[:, 1:] = (self.S_ph2 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])

        return optimal_x

    def pareto_y(self):
        # Transform the pareto_x to the real variables
        x = self.pareto_x(2000)
        y = self.evaluate(x)
        y_plus = y
        y_sum = torch.sum(y_plus, dim=1)[:, None]
        y_ref = y_plus / y_sum

        return y, y_ref


class P5T1(PT):
    def __init__(self, n_dim=50):
        super().__init__(self)
        self.current_name = "P5T1"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[0] = 0
        self.lbound[1:] = self.lbound[1:] * 0
        self.ubound = torch.ones(n_dim)
        self.ubound[1:] = self.ubound[1:] * 1
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.coef = 1

    def q(self, x):
        # Compute the q function to scale both objectives
        z = torch.matmul((x - self.S_pm1), self.M_pm1.T)
        return 1 + self.coef * torch.sum(torch.square(z), dim=1)

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim)
        x[:, 1:] = (self.S_pm1 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        # f1_max = 1.3
        # f2_max = 1.3
        f1_max = 1
        f2_max = 1

        # f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        # f1 = f1 / f1_max
        # f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        # f2 = f2 / f2_max
        f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        f1 = f1 / f1_max
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs

    def pareto_x(self, sample_size=500):
        # optimal_x = torch.ones(sample_size, self.n_dim) * 0.6
        optimal_x = torch.ones(sample_size, self.n_dim) * 0.5
        optimal_x[:, 0] = torch.arange(sample_size)/sample_size

        if optimal_x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        optimal_x[:, 1:] = (self.S_pm1 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])

        return optimal_x

    def pareto_y(self):
        # Transform the pareto_x to the real variables
        x = self.pareto_x(2000)
        y = self.evaluate(x)
        y_plus = y
        y_sum = torch.sum(y_plus, dim=1)[:, None]
        y_ref = y_plus / y_sum

        return y, y_ref


class P5T2(PT):
    def __init__(self, n_dim=50):
        super().__init__(self)
        self.current_name = "P5T2"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[0] = 0
        self.lbound[1:] = self.lbound[1:] * 0
        self.ubound = torch.ones(n_dim)
        self.ubound[1:] = self.ubound[1:] * 1
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.coef_1 = 1
        self.coef_2 = 10

    def q(self, x):
        # Compute the q function to scale both objectives
        z = torch.matmul(x, self.M_pm2.T)
        return 1 + self.coef_1 * torch.sum(torch.square(z) - self.coef_2 * torch.cos(2 * torch.pi * z) + 10, dim=1)

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim) * 0
        # x[:, 1:] = (S_pm1 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        # f1_max = 1
        # f2_max = 2.5
        f1_max = 1
        f2_max = 1

        # f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        # f1 = f1 / f1_max
        # f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        # f2 = f2 / f2_max
        f1 = x[:, 0]
        f2 = self.q(x[:, 1:]) * (1 - torch.square(x[:, 0] / self.q(x[:, 1:])))

        f1 = f1 / f1_max
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs

    def pareto_x(self, sample_size=500):
        # optimal_x = torch.ones(sample_size, self.n_dim) * 0.6
        optimal_x = torch.ones(sample_size, self.n_dim) * 0
        optimal_x[:, 0] = torch.arange(sample_size)/sample_size

        # if optimal_x.device.type == 'cuda':
        #     self.lbound = self.lbound.cuda()
        #     self.ubound = self.ubound.cuda()
        #
        # optimal_x[:, 1:] = (S_cm2 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])

        return optimal_x

    def pareto_y(self):
        # Transform the pareto_x to the real variables
        x = self.pareto_x(2000)
        y = self.evaluate(x)
        y_plus = y
        y_sum = torch.sum(y_plus, dim=1)[:, None]
        y_ref = y_plus / y_sum

        return y, y_ref


class P6T1(PT):
    def __init__(self, n_dim=50):
        super().__init__(self)
        self.current_name = "P6T1"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[0] = 0
        self.lbound[1:] = self.lbound[1:] * -50
        self.ubound = torch.ones(n_dim)
        self.ubound[1:] = self.ubound[1:] * 50
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]

    def q(self, x):
        # Compute the q function to scale both objectives
        z = torch.matmul((x - self.S_pm1), self.M_pm1.T)
        return 2 + torch.sum(torch.square(x), dim=1) / 4000 - \
            torch.prod(torch.cos(x / torch.sqrt(1 + torch.arange(self.n_dim-1))), dim=1)
        # return 1 + torch.sum(torch.square(x), dim=1) / 4000

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim) * 0.5
        # x[:, 1:] = (S_pm1 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        # f1_max = 5.5
        # f2_max = 5.5
        f1_max = 1
        f2_max = 1

        # f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        # f1 = f1 / f1_max
        # f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        # f2 = f2 / f2_max
        f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        f1 = f1 / f1_max
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs

    def pareto_x(self, sample_size=500):
        # optimal_x = torch.ones(sample_size, self.n_dim) * 0.6
        optimal_x = torch.ones(sample_size, self.n_dim) * 0.5
        optimal_x[:, 0] = torch.arange(sample_size)/sample_size

        # if optimal_x.device.type == 'cuda':
        #     self.lbound = self.lbound.cuda()
        #     self.ubound = self.ubound.cuda()
        #
        # optimal_x[:, 1:] = (S_cm2 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])

        return optimal_x

    def pareto_y(self):
        # Transform the pareto_x to the real variables
        x = self.pareto_x(2000)
        y = self.evaluate(x)
        y_plus = y
        y_sum = torch.sum(y_plus, dim=1)[:, None]
        y_ref = y_plus / y_sum

        return y, y_ref


class P6T2(PT):
    def __init__(self, n_dim=50):
        super().__init__(self)
        self.current_name = "P6T2"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[0] = 0
        self.lbound[1:] = self.lbound[1:] * -100
        self.ubound = torch.ones(n_dim)
        self.ubound[1:] = self.ubound[1:] * 100
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.coef = 1

    def q(self, x):
        # Compute the q function to scale both objectives
        z = x - self.S_pl2
        return 21 + math.exp(1) - \
            20 * torch.exp(-0.2 * torch.sqrt(torch.sum(torch.square(z), dim=1) / (self.n_dim - 1))) - \
            torch.exp(torch.sum(torch.cos(2 * torch.pi * z), dim=1) / (self.n_dim - 1))
        # return 1 + self.coef * torch.sum(torch.square(z), dim=1)

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim) * 0
        x[:, 1:] = (self.S_pl2 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        # f1_max = 700
        # f2_max = 700
        f1_max = 1
        f2_max = 1

        # f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        # f1 = f1 / f1_max
        # f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        # f2 = f2 / f2_max
        f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        f1 = f1 / f1_max
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs

    def pareto_x(self, sample_size=500):
        # optimal_x = torch.ones(sample_size, self.n_dim) * 0.6
        optimal_x = torch.ones(sample_size, self.n_dim) * 0.5
        optimal_x[:, 0] = torch.arange(sample_size)/sample_size

        if optimal_x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        optimal_x[:, 1:] = (self.S_pl2 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])

        return optimal_x

    def pareto_y(self):
        # Transform the pareto_x to the real variables
        x = self.pareto_x(2000)
        y = self.evaluate(x)
        y_plus = y
        y_sum = torch.sum(y_plus, dim=1)[:, None]
        y_ref = y_plus / y_sum

        return y, y_ref


class P7T1(PT):
    def __init__(self, n_dim=50):
        super().__init__(self)
        self.current_name = "P7T1"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[0] = 0
        self.lbound[1:] = self.lbound[1:] * -80
        self.ubound = torch.ones(n_dim)
        self.ubound[1:] = self.ubound[1:] * 80
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.coef_1 = 100
        self.coef_2 = 1
        self.coef_3 = 1

    def q(self, x):
        # Compute the q function to scale both objectives
        # z = torch.matmul((x - S_pm1), M_pm1.T)
        x_plus = x[:, 1:]
        x_minus = x[:, :-1]
        return 1 + self.coef_3 * torch.sum(self.coef_1 * torch.square(torch.square(x_minus) - x_plus)
                             + self.coef_2 * torch.square(1 - x_minus), dim=1)

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim) * 0.5
        x[:, 1:] = (1 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        f1_max = 2.8e3
        f2_max = 2.8e3
        # f1_max = 1
        # f2_max = 2

        # f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        # f1 = f1 / f1_max
        # f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        # f2 = f2 / f2_max
        f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        f1 = f1 / f1_max
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs

    def pareto_x(self, sample_size=500):
        # optimal_x = torch.ones(sample_size, self.n_dim) * 0.6
        optimal_x = torch.ones(sample_size, self.n_dim) * 0.5
        optimal_x[:, 0] = torch.arange(sample_size)/sample_size

        if optimal_x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        optimal_x[:, 1:] = (1 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])
        # optimal_x[:, 1:] = (S_cm2 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])

        return optimal_x

    def pareto_y(self):
        # Transform the pareto_x to the real variables
        x = self.pareto_x(2000)
        y = self.evaluate(x)
        y_plus = y
        y_sum = torch.sum(y_plus, dim=1)[:, None]
        y_ref = y_plus / y_sum

        return y, y_ref


class P7T2(PT):
    def __init__(self, n_dim=50):
        super().__init__(self)
        self.current_name = "P7T2"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[0] = 0
        self.lbound[1:] = self.lbound[1:] * -80
        self.ubound = torch.ones(n_dim)
        self.ubound[1:] = self.ubound[1:] * 80
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        self.coef = 1

    def q(self, x):
        # Compute the q function to scale both objectives
        return 1 + self.coef * torch.sum(torch.square(x), dim=1)

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim) * 0.5
        # x[:, 1:] = (S_pl2 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        # f1_max = 1
        # f2_max = 250
        f1_max = 1
        f2_max = 1

        # f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        # f1 = f1 / f1_max
        # f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        # f2 = f2 / f2_max
        f1 = x[:, 0]
        f2 = self.q(x[:, 1:]) * (1 - torch.sqrt(x[:, 0] / self.q(x[:, 1:])))
        f1 = f1 / f1_max
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs

    def pareto_x(self, sample_size=500):
        # optimal_x = torch.ones(sample_size, self.n_dim) * 0.6
        optimal_x = torch.ones(sample_size, self.n_dim) * 0.5
        optimal_x[:, 0] = torch.arange(sample_size)/sample_size

        # if optimal_x.device.type == 'cuda':
        #     self.lbound = self.lbound.cuda()
        #     self.ubound = self.ubound.cuda()
        #
        # optimal_x[:, 1:] = (1 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])
        # optimal_x[:, 1:] = (S_cm2 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])

        return optimal_x

    def pareto_y(self):
        # Transform the pareto_x to the real variables
        x = self.pareto_x(2000)
        y = self.evaluate(x)
        y_plus = y
        y_sum = torch.sum(y_plus, dim=1)[:, None]
        y_ref = y_plus / y_sum

        return y, y_ref


class P8T1(PT):
    def __init__(self, n_dim=20):
        super().__init__(self)
        self.current_name = "P8T1"
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.ones(n_dim)
        self.lbound[:2] = 0
        self.lbound[2:] = self.lbound[2:] * -20
        self.ubound = torch.ones(n_dim)
        self.ubound[2:] = self.ubound[2:] * 20
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]

    def q(self, x):
        # Compute the q function to scale both objectives
        # z = torch.matmul((x - S_pm1), M_pm1.T)
        x_plus = x[:, 1:]
        x_minus = x[:, :-1]
        return 1 + torch.sum(100 * torch.square(torch.square(x_minus) - x_plus)
                             + torch.square(1 - x_minus), dim=1)

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim) * 0.5
        x[:, 2:] = (1 - self.lbound[2:]) / (self.ubound[2:] - self.lbound[2:])
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        f1_max = 8e7
        f2_max = 8e7
        f3_max = 8e7

        f1 = self.q(x[:, 2:]) * torch.cos(torch.pi * x[:, 0] / 2) * torch.cos(torch.pi * x[:, 1] / 2)
        f1 = f1 / f1_max
        f2 = self.q(x[:, 2:]) * torch.cos(torch.pi * x[:, 0] / 2) * torch.sin(torch.pi * x[:, 1] / 2)
        f2 = f2 / f2_max
        f3 = self.q(x[:, 2:]) * torch.sin(torch.pi * x[:, 0] / 2)
        f3 = f3 / f3_max

        objs = torch.stack([f1, f2, f3]).T

        return objs


class P8T2(PT):
    def __init__(self, n_dim=20):
        super().__init__(self)
        self.current_name = "P8T2"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim)
        self.lbound[:2] = 0
        self.lbound[2:] = self.lbound[2:] * -20
        self.ubound = torch.ones(n_dim)
        self.ubound[2:] = self.ubound[2:] * 20
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]

    def q(self, x):
        # Compute the q function to scale both objectives
        z = torch.matmul(x, self.M_nm2.T)
        return 1 + torch.sum(torch.square(z), dim=1)

    def evaluate(self, eval_x):
        # Feed in the effective size of solution vectors
        # Transform them into standard vectors with pending 0.5
        sample_size, sample_dim = eval_x.shape
        x = torch.ones(sample_size, self.n_dim) * 0.5
        # x[:, 1:] = (S_pl2 - self.lbound[1:]) / (self.ubound[1:] - self.lbound[1:])
        x[:, :sample_dim] = eval_x

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        f1_max = 1
        f2_max = 2e4

        # f1 = self.q(x[:, 1:]) * torch.cos(torch.pi * x[:, 0] / 2)
        # f1 = f1 / f1_max
        # f2 = self.q(x[:, 1:]) * torch.sin(torch.pi * x[:, 0] / 2)
        # f2 = f2 / f2_max
        f1 = (x[:, 0] + x[:, 1]) / 2
        f1 = f1 / f1_max
        f2 = self.q(x[:, 2:]) * (1 - torch.square((x[:, 0] + x[:, 1])/(2 * self.q(x[:, 2:]))))
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs


class RE21():
    def __init__(self, n_dim=4, F=10.0, sigma=10.0, L=200.0, E=2e5):
        # F = 10.0
        # E = 2.0 * 1e5
        # L = 200.0
        # sigma = 10.0
        tmp_val = F / sigma

        self.F = F
        self.E = E
        self.L = L

        self.current_name = "real_one"
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([tmp_val, np.sqrt(2.0) * tmp_val, np.sqrt(2.0) * tmp_val, tmp_val]).float()
        self.ubound = torch.ones(n_dim).float() * 3 * tmp_val
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]

    def evaluate(self, x):
        F = self.F
        E = self.E
        L = self.L

        if isinstance(x, numpy.ndarray):
            x = torch.from_numpy(x).float()

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        f1 = L * ((2 * x[:, 0]) + np.sqrt(2.0) * x[:, 1] + torch.sqrt(x[:, 2]) + x[:, 3])
        f1_max = L * ((2 * self.ubound[0]) + np.sqrt(2.0) * self.ubound[1] +
                      torch.sqrt(self.ubound[2]) + self.ubound[3])

        f2 = ((F * L) / E) * (
                    (2.0 / x[:, 0]) + (2.0 * np.sqrt(2.0) / x[:, 1]) - (2.0 * np.sqrt(2.0) / x[:, 2]) + (2.0 / x[:, 3]))
        f2_max = ((F * L) / E) * (
                    (2.0 / self.lbound[0]) + (2.0 * np.sqrt(2.0) / self.lbound[1]) -
                    (2.0 * np.sqrt(2.0) / self.ubound[2]) + (2.0 / self.lbound[3]))

        f1 = f1 / f1_max
        f2 = f2 / f2_max

        objs = torch.stack([f1, f2]).T

        return objs


class RE23():
    def __init__(self, n_dim=4):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([1, 1, 10, 10]).float()
        self.ubound = torch.tensor([100, 100, 200, 240]).float()
        self.nadir_point = [5852.05896876, 1288669.78054]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = 0.0625 * torch.round(x[:, 0])
        x2 = 0.0625 * torch.round(x[:, 1])
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        f1 = (0.6224 * x1 * x3 * x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)
        f1 = f1.float()

        # Original constraint functions
        g1 = x1 - (0.0193 * x3)
        g2 = x2 - (0.00954 * x3)
        g3 = (np.pi * x3 * x3 * x4) + ((4.0 / 3.0) * (np.pi * x3 * x3 * x3)) - 1296000

        g = torch.stack([g1, g2, g3])
        z = torch.zeros(g.shape).cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f2 = torch.sum(g, axis=0).to(torch.float64)

        objs = torch.stack([f1, f2]).T

        return objs


class RE24():
    def __init__(self, n_dim=2, sigma_b_max=700, tau_max=450, delta_max=1.5):
        self.sigma_b_max = sigma_b_max
        self.tau_max = tau_max
        self.delta_max = delta_max

        self.current_name = "real_two_shift"

        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([0.5, 4]).float()
        self.ubound = torch.tensor([4, 50]).float()
        self.nadir_point = [5852.05896876, 1288669.78054]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = x[:, 0]
        x2 = x[:, 1]

        # First original objective function
        # f1 = (0.6224 * x1 * x3 * x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)
        f1 = (x1 + 120 * x2) / (self.ubound[0] + 120 * self.ubound[1])
        f1 = f1.float()

        # Constraint variables
        E = 7 * 1e5

        sigma_b = 4500 / (x1 * x2)
        sigma_b_max = self.sigma_b_max

        tau = 1800 / x2
        tau_max = self.tau_max

        delta = 56.2 * 1e4 / (E * x1 * x2 * x2)
        delta_max = 1.5

        sigma_k = (E * x1 * x1) / 100

        # Original constraint functions
        # g1 = x1 - (0.0193 * x3)
        # g2 = x2 - (0.00954 * x3)
        # g3 = (np.pi * x3 * x3 * x4) + ((4.0 / 3.0) * (np.pi * x3 * x3 * x3)) - 1296000
        g1 = 1.0 - sigma_b / sigma_b_max
        g2 = 1.0 - tau / tau_max
        g3 = 1.0 - delta / delta_max
        g4 = 1.0 - sigma_b / sigma_k

        g = torch.stack([g1, g2, g3, g4])
        if x.device.type == 'cuda':
            z = torch.zeros(g.shape).cuda().to(torch.float64)
        else:
            z = torch.zeros(g.shape).to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f2 = torch.sum(g, axis=0).to(torch.float64)

        objs = torch.stack([f1, f2]).T

        return objs


class RE25():
    def __init__(self, n_dim=3, F_max=1000, l_max=14, sigma_pm=6):
        self.F_max = F_max
        self.l_max = l_max
        self.sigma_pm = sigma_pm

        self.current_name = "real_three"

        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([1, 0.6, 0.009]).float()
        self.ubound = torch.tensor([70, 30, 0.5]).float()
        self.nadir_point = [5852.05896876, 1288669.78054]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = torch.round(x[:, 0])
        x2 = x[:, 1]
        x3 = x[:, 2]

        # First original objective function
        # f1 = (0.6224 * x1 * x3 * x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)
        f1 = (torch.pi * torch.pi * x2 * x3 * x3 * (x1 + 2)) / \
             (torch.pi * torch.pi * self.ubound[1] * self.ubound[2] * self.ubound[2] * (self.ubound[0] + 2))
        f1 = f1.float()

        # Constraint variables
        F_max = self.F_max
        l_max = self.l_max
        sigma_pm = self.sigma_pm

        S = 1.89 * 1e5
        C_f = ((4 * (x2 / x3) - 1) / (4 * (x2 / x3) - 4)) + (0.615 * x3) / x2
        G = 11.5 * 1e6
        K = (G * x3 * x3 * x3 * x3) / (8 * x1 * x2 * x2 * x2)
        F_p = 300
        sigma_p = F_p / K
        l_f = (F_max / K) + (1.05 * (x1 + 2) * x3)
        sigma_w = 1.25

        # Original constraint functions
        g1 = S - (8 * C_f * F_max * x2) / (torch.pi * x3 * x3 * x3)
        g2 = l_max - l_f
        g3 = x2 / x3 - 3
        g4 = sigma_pm - sigma_p
        g5 = - sigma_p - (F_max - F_p) / K - 1.05 * (x1 + 2) * x3 + l_f
        g6 = - sigma_w + (F_max - F_p) / K
        g = torch.stack([g1, g2, g3, g4, g5, g6])
        if x.device.type == 'cuda':
            z = torch.zeros(g.shape).cuda().to(torch.float64)
        else:
            z = torch.zeros(g.shape).to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f2 = torch.sum(g, axis=0).to(torch.float64)
        f2 = f2 / 1e10

        objs = torch.stack([f1, f2]).T

        return objs


class RE33():
    def __init__(self, n_dim=4):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([55, 75, 1000, 11]).float()
        self.ubound = torch.tensor([80, 110, 3000, 20]).float()
        self.nadir_point = [5.3067, 3.12833430979, 25.0]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        f1 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        # Second original objective function
        f2 = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))

        # Reformulated objective functions
        g1 = (x2 - x1) - 20.0
        g2 = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
        g3 = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / torch.pow((x2 * x2 - x1 * x1), 2)
        g4 = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0

        g = torch.stack([g1, g2, g3, g4])
        z = torch.zeros(g.shape).float().cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f3 = torch.sum(g, axis=0).float()

        objs = torch.stack([f1, f2, f3]).T

        return objs


class RE36():
    def __init__(self, n_dim=4):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([12, 12, 12, 12]).float()
        self.ubound = torch.tensor([60, 60, 60, 60]).float()
        self.nadir_point = [5.931, 56.0, 0.355720675227]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        f1 = torch.abs(6.931 - ((x3 / x1) * (x4 / x2)))
        # Second original objective function (the maximum value among the four variables)
        l = torch.stack([x1, x2, x3, x4])
        f2 = torch.max(l, dim=0)[0]

        g1 = 0.5 - (f1 / 6.931)

        g = torch.stack([g1])
        z = torch.zeros(g.shape).float().cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)
        f3 = g[0]

        objs = torch.stack([f1, f2, f3]).T

        return objs


class RE37():
    def __init__(self, n_dim=4):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([0, 0, 0, 0]).float()
        self.ubound = torch.tensor([1, 1, 1, 1]).float()
        self.nadir_point = [0.98949120096, 0.956587924661, 0.987530948586]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        xAlpha = x[:, 0]
        xHA = x[:, 1]
        xOA = x[:, 2]
        xOPTT = x[:, 3]

        # f1 (TF_max)
        f1 = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (
                    0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (
                         0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (
                         0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
        # f2 (X_cc)
        f2 = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (
                    0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (
                         0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (
                         0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
        # f3 (TT_max)
        f3 = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (
                    0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (
                         0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (
                         0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (
                         0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (
                         0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)

        objs = torch.stack([f1, f2, f3]).T

        return objs


if __name__ == "__main__":
    # print("M_cm2 is {} with shape {}.".format(M_cm2, M_cm2.shape))
    # print("S_cm2 is {} with shape {}.".format(S_cm2, S_cm2.shape))

    # res = ref_points(3, 4)
    # print(res)

    problem_current = P3T1()
    effect_dim = 10
    sample_size = 30000
    sample_ans = torch.rand(sample_size, effect_dim)
    # # truncate the solution to the effective size version
    # # sample_ans[:, effect_dim:] = 0.5
    # # print(sample_ans)
    #
    sample_res = problem_current.evaluate(sample_ans)
    # import matplotlib.pyplot as plt
    #
    # plt.scatter(sample_res[:, 0], sample_res[:, 1])
    # plt.show()

    sample_min = torch.min(sample_res, dim=0).values
    sample_max = torch.max(sample_res, dim=0).values

    print("sample_min is {}.".format(sample_min))
    print("sample_max is {}.".format(sample_max))
    # print(problem_current.lbound)
    # print(problem_current.ubound)
