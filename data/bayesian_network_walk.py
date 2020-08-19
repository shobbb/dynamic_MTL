import numpy as np
import pickle as p

def get_params(seed=0):
    np.random.seed(seed=seed)
    logit_lint = np.random.rand(input_size, num_tasks**2) - 0.5
    initial_task_lint = np.random.rand(input_size, 1) - 0.5
    return (logit_lint, initial_task_lint)

def create_dataset(
    input_size, num_tasks, params, seed=0, sigma=1, sample_size=10000, alpha=1, eps=1e-1
):
    np.random.seed(seed=seed)
    logit_lint, initial_task_lint = params
    
    inputs = np.random.multivariate_normal(
        np.zeros(input_size), np.identity(input_size), sample_size
    )
    logits = np.matmul(inputs, alpha * logit_lint)
    bayesian_matrix = np.reshape(
        np.square(logits), (sample_size, num_tasks, num_tasks)
    )

    # normalize
    bayesian_matrix = np.transpose(bayesian_matrix, (0, 2, 1))
    bayesian_matrix = np.nan_to_num(
        bayesian_matrix/np.linalg.norm(bayesian_matrix, axis=-1).reshape(
            sample_size, num_tasks, 1
        )
    )
    bayesian_matrix = np.transpose(bayesian_matrix, (0, 2, 1))

    # upper triangular
    mask = np.array(
        np.tile(np.triu(np.ones((num_tasks, num_tasks))) - np.eye(num_tasks), (sample_size, 1, 1))
    )
    bayesian_matrix = bayesian_matrix * mask

    bayesian_matrix[bayesian_matrix < eps] = 0
    bayesian_matrix[:, 0, 0] += 1
    
    tasks = np.matmul(inputs, alpha * initial_task_lint)
    tasks = np.expand_dims(np.pad(tasks, ((0, 0), (0, num_tasks - 1))), 1)
    for i in range(1, num_tasks):
        mean = np.matmul(tasks, bayesian_matrix[:, :, :i+1])[:, :, i]
        sample = np.random.normal(
            mean.flatten(), np.ones(sample_size), sample_size
        )
        sample = mean
        tasks[:, :, i] += sample
    
    bayesian_matrix[:, 0, 0] -= 1

    return {
        'inputs': inputs, 'tasks': np.squeeze(tasks), 'relationships': bayesian_matrix
    }

if __name__ == '__main__':
    np.set_printoptions(linewidth=300)
    input_size, num_tasks, alpha, eps = (128, 10, 100, 2e-1)
    split = np.array([0.8, 0.1, 0.1]) * 100000
    split = split.astype(np.int)
    params = get_params()

    dataset = {
        'training': create_dataset(
            input_size, num_tasks, params, sample_size=split[0], alpha=alpha, eps=eps
        ),
        'valid': create_dataset(
            input_size, num_tasks, params, sample_size=split[1], alpha=alpha, eps=eps
        ),
        'test': create_dataset(
            input_size, num_tasks, params, sample_size=split[2], alpha=alpha, eps=eps
        ),
    }

    with open('/share/jolivaunc/data/bayesian_network_walk/data.p', 'wb') as f:
        p.dump(dataset, f)