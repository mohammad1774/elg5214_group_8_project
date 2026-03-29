import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.training.train_dqn import train_dqn

def main():
    agent, history = train_dqn(
        env_name="cartpole",
        reward_type="dense",
        seed=0,
        num_episodes=150,
        max_steps_per_episode=200,
        hidden_dim=64,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_capacity=10000,
        batch_size=64,
        min_buffer_size_before_training=500,
        train_freq=1,
        target_update_freq=100,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=100,
        num_eval_episodes=5,
        eval_every=25,
    )

    print("final eval episodes:", history["eval_episode_idx"])
    print("final eval returns:", history["eval_return_mean"])

    assert len(history["eval_return_mean"]) > 0, "No evaluation points recorded."
    assert history["eval_return_mean"][-1] >= 100.0, "DQN did not learn a reasonable CartPole policy."


if __name__ == "__main__":
    main()