import torch

from amfbo import PROJECT_ROOT


def save_dataset(dataloader, filepath, exp_name):
    all_batches = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            saved_batch = tuple(item.clone() for item in batch)
        elif isinstance(batch, dict):
            saved_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    saved_batch[key] = value.clone().detach()
                else:
                    saved_batch[key] = value
        else:
            saved_batch = batch
        all_batches.append(saved_batch)
    torch.save(all_batches, str(PROJECT_ROOT / 'dataloader' / exp_name / f'{filepath}.pt'))

def load_dataset(filepath, exp_name):
    all_batches = torch.load(str(PROJECT_ROOT / 'dataloader' / exp_name / f'{filepath}.pt'))
    return all_batches

def combination_to_number(combination):
    a_value, b_value = combination
    return a_value * 5 + b_value

def number_to_combination_tensor(number_tensor,):
    a_values = number_tensor // 5
    b_values = number_tensor % 5
    combinations = torch.stack([a_values, b_values], dim=1)
    return combinations

def split_task_probabilities(output, num_a=4, num_b=3):
    batch_size = output.shape[0]
    output_reshaped = output.view(batch_size, num_a, num_b)
    prob_a = output_reshaped.sum(dim=2)  # shape (batch_size, 10)
    prob_a = torch.softmax(prob_a, dim=1)
    prob_b = output_reshaped.sum(dim=1)  # shape (batch_size, 5)
    prob_b = torch.softmax(prob_b, dim=1)
    return prob_a, prob_b
