from src.training.train_dqn import train_dqn

agent, history = train_dqn(
    env_name="cartpole",
    reward_type="sparse",
    seed=0,
    num_episodes=500,               # was 150 — need more for sparse
    max_steps_per_episode=500,       # CRITICAL FIX: was 200, must be 500 for CartPole
    hidden_dim=64,
    learning_rate=1e-3,
    gamma=0.99,
    buffer_capacity=50000,           # was 10000 — larger buffer helps
    batch_size=64,
    min_buffer_size_before_training=500,
    train_freq=4,                    # was 1 — train every 4 steps (faster, still effective)
    target_update_freq=500,          # was 100 — more stable with longer episodes
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_episodes=300,      # was 100 — slower decay for sparse exploration
    num_eval_episodes=10,
    eval_every=50,
)

print("final eval episodes:", history["eval_episode_idx"])
print("final eval returns:", history["eval_return_mean"])