import torch
import argparse
import pickle
import numpy as np
import os
from decision_transformer.models.decision_transformer import KnapsackTransformer
from decision_transformer.training.seq_trainer import SequenceTrainer
from config import DATASETS_DIR, NO_MAX_ITEMS


def load_knapsack_trajectories(file_path):
    """
    Loads knapsack problem trajectories from a pickle file.
    Args:
        file_path (str): Path to the pickle file containing trajectories.

    Returns:
        dict: A dictionary with 'states', 'actions', 'rewards', and 'returns' (RTG).
    """
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def pad_trajectories(dataset, max_trajectory_length, state_dim):
    """
    Pads knapsack trajectories to ensure all trajectories have the same length.

    Args:
        dataset (dict): Contains 'states', 'actions', 'rewards', 'returns'.
        max_trajectory_length (int): The maximum number of steps in any trajectory.
        state_dim (int): The dimensionality of each state, representing features like value and weight.

    Returns:
        dict: A dataset with padded 'states', 'actions', 'rewards', and 'returns' of uniform length.
    """
    padded_dataset = {
        'states': [],
        'actions': [],
        'rewards': [],
        'returns': []
    }

    for i in range(len(dataset['states'])):
        # Retrieve trajectory data for the current knapsack instance
        states = dataset['states'][i]
        actions = dataset['actions'][i]
        rewards = dataset['rewards'][i]
        returns = dataset['returns'][i]

        # Determine how much padding is needed to match the max trajectory length
        padding_needed = max_trajectory_length - len(states)

        # Only apply padding if necessary
        if padding_needed > 0:
            # Create padding: empty states, actions, rewards, and returns
            empty_state = np.zeros((1, state_dim))  # A state filled with zeros as padding
            empty_action = np.array([-1])  # A placeholder for no action (-1 for empty)
            empty_reward = np.array([0.0])  # No reward for padded steps
            empty_return = np.array([0.0])  # No return-to-go for padded steps

            # Add the padding to the end of each trajectory
            padded_states = np.vstack([states, np.tile(empty_state, (padding_needed, 1))])
            padded_actions = np.concatenate([actions, np.tile(empty_action, padding_needed)])
            padded_rewards = np.concatenate([rewards, np.tile(empty_reward, padding_needed)])
            padded_returns = np.concatenate([returns, np.tile(empty_return, padding_needed)])
        else:
            # No padding needed, use the original trajectory data
            padded_states = states
            padded_actions = actions
            padded_rewards = rewards
            padded_returns = returns

        # Store the padded data in the padded dataset
        padded_dataset['states'].append(padded_states)
        padded_dataset['actions'].append(padded_actions)
        padded_dataset['rewards'].append(padded_rewards)
        padded_dataset['returns'].append(padded_returns)

    return padded_dataset


def train_model(args):
    # Load the knapsack dataset
    dataset = load_knapsack_trajectories(args.dataset)
    states, actions, rewards, returns = dataset['states'], dataset['actions'], dataset['rewards'], dataset['returns']

    # Calculate total_state_dim based on max_items and state_dim
    max_items = NO_MAX_ITEMS  # Defined elsewhere
    state_dim = 2  # Number of features per item (e.g., weight and value)

    total_state_dim = 1 + max_items * state_dim  # 1 for remaining capacity, + item features

    # Find the max trajectory length from the dataset
    max_trajectory_length = max([len(traj) for traj in states])

    # Apply manual padding to ensure uniform trajectory lengths
    padded_dataset = pad_trajectories(dataset, max_trajectory_length, state_dim=total_state_dim)

    # Convert the padded lists to PyTorch tensors
    states_tensor = torch.tensor(np.array(padded_dataset['states']), dtype=torch.float32)
    actions_tensor = torch.tensor(np.array(padded_dataset['actions']), dtype=torch.long)
    rewards_tensor = torch.tensor(np.array(padded_dataset['rewards']), dtype=torch.float32)
    returns_tensor = torch.tensor(np.array(padded_dataset['returns']), dtype=torch.float32)

    # Normalize the states no needed, because data is already normalized
    # state_mean, state_std = states.mean(axis=0), states.std(axis=0) + 1e-6

    print('=' * 50)
    print(f'Training on dataset: {args.dataset}')
    print(f'Number of trajectories: {len(states)}')
    print('=' * 50)

    # Model configuration
    model = KnapsackTransformer(
        state_dim=total_state_dim,  # Use the calculated total_state_dim (capacity + items)
        act_dim=1,  # Binary action (select/reject for each item, so it's 1-dimensional)
        hidden_size=args.embed_dim,
        max_length=args.K,  # Maximum sequence length (number of steps in the trajectory)
        max_ep_len=max_trajectory_length,  # Total number of steps in the longest trajectory
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=4 * args.embed_dim,
        activation_function=args.activation_function,
        resid_pdrop=args.dropout,
        attn_pdrop=args.dropout,
    ).to(args.device)

    # Setup optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / args.warmup_steps, 1)
    )

    # Initialize the sequence trainer
    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=args.batch_size,
        get_batch=lambda: get_batch(states, actions, rewards, returns, args.K, args.device, state_mean, state_std),
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),  # MSE Loss for actions
    )

    # Training loop
    for iter in range(args.max_iters):
        outputs = trainer.train_iteration(num_steps=args.num_steps_per_iter, iter_num=iter + 1, print_logs=True)

        # Save the model at every iteration or after a set number of steps
        if (iter + 1) % args.save_freq == 0:
            model_save_path = f'models/knapsack_transformer_{iter + 1}.pt'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model checkpoint saved at {model_save_path}")


def get_batch(states, actions, rewards, returns, K, device, state_mean, state_std):
    """
    Generates a batch of data from the stored trajectories.

    Args:
        states, actions, rewards, returns: Trajectories of knapsack instances.
        K: The number of steps in each sequence.
        device: The device to use for training (CPU or GPU).
        state_mean, state_std: Normalization values for states.

    Returns:
        Batch of (states, actions, rewards, returns-to-go, timesteps, masks) tensors.
    """
    batch_size = min(len(states), 64)
    batch_inds = np.random.choice(len(states), size=batch_size, replace=True)

    s, a, r, rtg, timesteps, mask = [], [], [], [], [], []
    for i in batch_inds:
        traj_states = states[i]
        traj_actions = actions[i]
        traj_rewards = rewards[i]

        # Select random starting point within trajectory
        start = np.random.randint(0, len(traj_states) - K)

        s.append((traj_states[start:start + K] - state_mean) / state_std)
        a.append(traj_actions[start:start + K])
        r.append(traj_rewards[start:start + K])

        # Calculate RTG for each step
        rtg.append(discount_cumsum(traj_rewards[start:], gamma=1.0)[:K])
        timesteps.append(np.arange(start, start + K))
        mask.append(np.ones(K))

    return torch.tensor(s).float().to(device), \
        torch.tensor(a).float().to(device), \
        torch.tensor(r).float().to(device), \
        torch.tensor(rtg).float().to(device), \
        torch.tensor(timesteps).long().to(device), \
        torch.tensor(mask).float().to(device)


def discount_cumsum(x, gamma):
    """
    Compute discounted cumulative sums of vectors.

    Args:
        x (array): The rewards.
        gamma (float): Discount factor.

    Returns:
        Discounted cumulative sum of rewards.
    """
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    pickle_file = 'knapsack_trajectories1.pkl'
    dataset = os.path.join(DATASETS_DIR, pickle_file)

    # Add all the training arguments
    parser.add_argument('--dataset', type=str, default=dataset)  # Dataset name
    parser.add_argument('--K', type=int, default=20)  # Sequence length
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--save_freq', type=int, default=5)  # Frequency of saving model checkpoints
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Train the model
    train_model(args)