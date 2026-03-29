from src.training.train_dqn_icm import train_dqn_icm

agent, history = train_dqn_icm(
    env_name="mountaincar",
    reward_type="dense",
    seed=0,
    num_episodes=1000,
    max_steps_per_episode=200,
    q_hidden_dim=64,
    icm_hidden_dim=64,
    icm_feat_dim=64,
    q_learning_rate=5e-4,
    icm_learning_rate=1e-4,
    gamma=0.99,
    buffer_capacity=50000,
    batch_size=64,
    min_buffer_size_before_training=500,
    train_freq=1,
    target_update_freq=500,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_episodes=700,
    num_eval_episodes=10,
    eval_every=50,
    icm_eta=1.0,
    icm_beta=0.2,
    icm_update_freq=10,    # only update ICM every 10 Q-updates
)

print("final eval episodes:", history["eval_episode_idx"])
print("final eval returns:", history["eval_return_mean"])
print("final intrinsic means (last 10):", history["train_intrinsic_reward_mean"][-10:])