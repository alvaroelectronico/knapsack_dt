import torch
import argparse
import os
from knapsack_transformer.models.knapsack_transformer import KnapsackTransformer
from knapsack_transformer.training.seq_trainer import SequenceTrainer
from knapsack_instance_generator import load_knapsack_dataset  # Your dataset loader


def train_model(args):
    # Load the knapsack dataset
    dataset = load_knapsack_dataset(f'data/{args.dataset}.pkl')  # Adjust path to dataset
    states, actions, rewards, returns = dataset['states'], dataset['actions'], dataset['rewards'], dataset['returns']

    # Normalize states for input
    state_mean, state_std = states.mean(axis=0), states.std(axis=0) + 1e-6
    num_timesteps = len(states)

    print('=' * 50)
    print(f'Training on dataset: {args.dataset}')
    print(f'{num_timesteps} timesteps found')
    print(f'Average return: {returns.mean():.2f}, std: {returns.std():.2f}')
    print(f'Max return: {returns.max():.2f}, min: {returns.min():.2f}')
    print('=' * 50)

    # Model configuration
    model = KnapsackTransformer(
        state_dim=states.shape[2],  # Adjust based on your knapsack instance format
        act_dim=actions.shape[2],  # Binary action (select/reject)
        hidden_size=args.embed_dim,
        max_length=args.K,  # Maximum sequence length
        max_ep_len=100,  # Total number of items
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=4 * args.embed_dim,
        activation_function=args.activation_function,
        resid_pdrop=args.dropout,
        attn_pdrop=args.dropout,
    )

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
        get_batch=lambda: get_batch(states, actions, rewards, returns, args.K, args.device),
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


def get_batch(states, actions, rewards, returns, K, device):
    """
    Function to generate a batch of data for training.
    This needs to sample from your knapsack trajectories.
    """
    # Implement the batch sampling logic here as in your preprocessing script
    # Similar to the get_batch function in experiment.py
    batch_size = 256  # This can be passed as an argument too
    # Add logic to fetch batches from knapsack data
    # For example, randomly selecting sequences of states, actions, rewards
    # Padding sequences to a fixed length (K)

    # Example structure (replace with your logic):
    # return torch.tensor(...).to(device), ...
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add all the training arguments
    parser.add_argument('--dataset', type=str, default='knapsack_medium')  # Dataset name
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
