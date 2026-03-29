from src.training.train_dqn_icm import train_dqn_icm


def main():
    agent, history = train_dqn_icm(
        env_name="cartpole",
        reward_type="sparse",
        seed=0,
        num_episodes=150,
        max_steps_per_episode=200,
        q_hidden_dim=64,
        icm_hidden_dim=64,
        icm_feat_dim=64,
        q_learning_rate=1e-3,
        icm_learning_rate=1e-3,
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
        icm_eta=0.01,
        icm_beta=0.2,
    )

    print("final eval episodes:", history["eval_episode_idx"])
    print("final eval returns:", history["eval_return_mean"])
    print("final intrinsic means (last 10):", history["train_intrinsic_reward_mean"][-10:])


if __name__ == "__main__":
    main()