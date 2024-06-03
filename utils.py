import torch
import matplotlib.pyplot as plt
from typing import List
from sklearn.manifold import TSNE


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
                dp_tmp = torch.zeros_like(dp_list[j-1])
                dp_tmp[:, 0] = 1
                dp_tmp = dp_tmp + dp_list[j-1]
                dp_list[j] = torch.cat((dp_list[j], dp_tmp), dim=0)

        # DEBUG:
        # print("shape {} in iteration {}.".format(dp_list[-1].shape, i))

    dp_list[-1] = dp_list[-1] / granularity

    if if_norm:
        return dp_list[-1]/torch.norm(dp_list[-1], dim=1).unsqueeze(1)
    else:
        return dp_list[-1]


def generate_uniform(sample_size: int = 10, dim_size: int = 2):
    result = torch.rand(sample_size, dim_size)
    result = result / torch.sum(result, dim=1).unsqueeze(1)
    return result


def generate_norm(source_tensor: torch.tensor):
    norm_tensor = torch.norm(source_tensor, dim=1).unsqueeze(1)
    target_tensor = source_tensor / norm_tensor
    return target_tensor


def get_cor_mat(k_mat: torch.tensor):
    """
    :param k_mat: the index kernel matrix
    :return: we want to obtain the correlation matrix
    """
    diag_mat = torch.diag(k_mat)
    new_diag = torch.diag_embed(diag_mat**(-1/2))
    c = new_diag @ k_mat @ new_diag
    return c


def igd(front: torch.tensor, solution: torch.tensor):
    """
    IGD =\frac{1}{|Y^*|}\sum_{q=1}^{|Y^*|}
    \min\{\lVert y_q^*-y_1\rVert, ...,
    \lVert y_q^*-y_{n_q} \rVert\}
    :param front: The true pareto front or the ground truth x or y
    by the queried preference vector
    :param solution: The generated set of solutions to be measured
    by IGD metric.
    :return: scalar value to indicate how good the solution is
    (the smaller, the better)
    """
    front_s, front_dim = front.shape
    sol_s, sol_dim = solution.shape

    # the front and the solutions should be in same dimension
    assert front_dim == sol_dim

    # Compute vector-wise norm-2 distance
    tot_diff = front.unsqueeze(1) - solution.unsqueeze(0)
    tot_norm = torch.norm(tot_diff, dim=2)

    # tot_norm is in shape of [front_s, sol_s]
    tot_norm_min = torch.min(tot_norm, dim=1).values
    igd_ans = torch.sum(tot_norm_min)/front_s

    return igd_ans.item()


def igd_plus(front: torch.tensor, solution: torch.tensor):
    """
    IGD =\frac{1}{|Y^*|}\sum_{q=1}^{|Y^*|}
    \min\{\lVert y_q^*-y_1\rVert, ...,
    \lVert y_q^*-y_{n_q} \rVert\}
    :param front: The true pareto front or the ground truth x or y
    by the queried preference vector
    :param solution: The generated set of solutions to be measured
    by IGD metric.
    :return: scalar value to indicate how good the solution is
    (the smaller, the better)
    """
    front_s, front_dim = front.shape
    sol_s, sol_dim = solution.shape

    # the front and the solutions should be in same dimension
    assert front_dim == sol_dim

    # Compute vector-wise norm-2 distance
    tot_diff = front.unsqueeze(1) - solution.unsqueeze(0)
    tot_diff = torch.where(tot_diff > 0, tot_diff, 0)
    tot_norm = torch.norm(tot_diff, dim=2)

    # tot_norm is in shape of [front_s, sol_s]
    tot_norm_min = torch.min(tot_norm, dim=1).values
    igd_ans = torch.sum(tot_norm_min)/front_s

    return igd_ans.item()


def rmse(target: torch.tensor, source: torch.tensor):
    """
    RMSE_x = \sqrt{\frac{\sum_{q=1}^{n_q}
    \lVert x_a^* - x_q \rVert^2}{n_q}}
    :param target: The target tensors
    :param source: The source tensors
    :return: scalar value to indicate how close the source related
    to the target tensors (the smaller, the better)
    """
    t_sample, t_dim = target.shape
    s_sample, s_dim = source.shape
    assert t_sample == s_sample and t_dim == s_dim

    ans = torch.sqrt(torch.sum(torch.pow(torch.norm(target - source, dim=1), 2))/t_sample)
    detail_ans = torch.pow(torch.norm(target - source, dim=1), 2)

    return ans.item(), detail_ans.detach()


def debug_plot(target: torch.tensor, title=None):
    a, b = target.shape
    print("The shape is {}.".format(target.shape))
    # a for sample size, b for sol dim
    fig, ax = plt.subplots()
    ind = torch.arange(1, b+1)
    new_target = target.T
    for i in ind:
        ax.axvline(x=i, color='k', alpha=0.2)
    ax.plot(ind, new_target, 'b-', alpha=0.3)
    if title is not None:
        ax.set_title(title)
    plt.show()


def std_plot(uncertain: List[torch.tensor], multi_uncertain: List[torch.tensor]):
    tot_problems = len(uncertain)
    tot_dim = uncertain[0].shape[1]

    for i in range(tot_problems):
        fig, ax = plt.subplots()

        ax.plot(torch.arange(0, tot_dim), torch.min(uncertain[i].detach(), dim=0).values, '--', label="min")
        ax.plot(torch.arange(0, tot_dim), torch.max(uncertain[i].detach(), dim=0).values, '--', label="max")
        ax.plot(torch.arange(0, tot_dim), torch.median(uncertain[i].detach(), dim=0).values, '--', label="median")
        ax.plot(torch.arange(0, tot_dim), torch.mean(uncertain[i].detach(), dim=0), '--', label="mean")
        ax.plot(torch.arange(0, tot_dim), torch.min(multi_uncertain[i].detach(), dim=0).values,
                '--', label="multi_min")
        ax.plot(torch.arange(0, tot_dim), torch.max(multi_uncertain[i].detach(), dim=0).values,
                '--', label="multi_max")
        ax.plot(torch.arange(0, tot_dim), torch.median(multi_uncertain[i].detach(), dim=0).values,
                '--', label="multi_median")
        ax.plot(torch.arange(0, tot_dim), torch.mean(multi_uncertain[i].detach(), dim=0),
                '--', label="multi_mean")
        ax.set_title("problem_ind-{}".format(i+1))
        ax.legend()

    plt.show()


def tsne_plot(target_list: List[torch.tensor], source_list: List[torch.tensor]):
    tot_len = len(target_list)
    for cur_ind in range(tot_len):
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        tot = torch.cat((source_list[cur_ind], target_list[cur_ind]))
        cut_off_val, _ = source_list[cur_ind].shape
        embedded_data = tsne.fit_transform(tot)
        fig, ax = plt.subplots()
        ax.scatter(embedded_data[:cut_off_val, 0],
                    embedded_data[:cut_off_val, 1],
                    label='source')
        ax.scatter(embedded_data[cut_off_val:, 0],
                    embedded_data[cut_off_val:, 1],
                    label='target')
        ax.set_title("tsne visualization for problem_id-{}".format(cur_ind+1))
        ax.legend()
    plt.show()


def _merge_list(sample_list):
    merged_list = [item for sublist in zip(*sample_list) for item in sublist]
    return merged_list


def _sample_transfer_info(transfer_records: torch.Tensor,
                          budget_records: torch.Tensor,
                          budget_limit: int,
                          threshold: float,
                          if_adapt: bool):
    task_num = len(transfer_records)
    task_ids = torch.arange(task_num)

    # First, find out the valid task_id
    task_valid_ids = task_ids[budget_records < budget_limit]

    # Second, subtract the transfer records by a thresholding value
    sigmoid_factor = 3

    # Default 5
    adapt_bound = 3

    new_records = torch.where(transfer_records[task_valid_ids] < threshold,
                              0, transfer_records[task_valid_ids] - threshold)
    if if_adapt:
        new_records = transfer_records[task_valid_ids]
        min_records = torch.min(new_records).item()
        print(f"min_records is {min_records}")
        sigmoid_factor = 1 + (adapt_bound - 1) * (1 - min_records)
        print(f"sigmoid_factor is {sigmoid_factor}")

    new_prob = torch.exp(sigmoid_factor * new_records)
    new_prob = new_prob / torch.sum(new_prob)

    # Third, do the sampling to return the transfer records with corresponding id
    sampled_index = torch.multinomial(new_prob, 1).item()
    transfer_id = task_valid_ids[sampled_index].item()
    transfer_record = transfer_records[transfer_id].item()

    # Do Something!
    return transfer_record, transfer_id


if __name__ == "__main__":
    # Example usage
    # source_result = generate_uniform()
    # target_result = generate_norm(source_result)
    #
    # print("source_result in shape of {}.".format(source_result.shape))
    # print("target_result in shape of {}.".format(target_result.shape))
    #
    # plt.scatter(source_result[:, 0], source_result[:, 1])
    # plt.scatter(target_result[:, 0], target_result[:, 1])
    # plt.xlim([-0.2, 1.2])
    # plt.ylim([-0.2, 1.2])
    # plt.show()

    # test the sample method
    result = _sample_transfer_info(torch.Tensor([0.3, 0.79, 0.8]), torch.Tensor([25, 26, 80]), 100)
    print(result)

    # result = ref_points(3, 40, if_norm=False)
    # print(result)
    # print(result.shape)

    # tmp_sum = torch.sum(tmp_result, dim=1)
    # print(tmp_sum)
    # a = torch.rand(100, 5)
    # b = torch.rand(100, 5)
    # debug_plot(a, title="random a")
    # tsne_plot(a, b)
    # print("**********DEBUG*********")
