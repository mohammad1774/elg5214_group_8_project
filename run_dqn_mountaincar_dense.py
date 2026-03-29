from src.training.train_dqn import train_dqn

agent, history = train_dqn(
    env_name="mountaincar",
    reward_type="dense",
    seed=0,
    num_episodes=1000,              # was 150 — MountainCar needs many more
    max_steps_per_episode=200,       # 200 is correct for MountainCar (env max)
    hidden_dim=64,
    learning_rate=5e-4,              # was 1e-3 — slightly lower for stability
    gamma=0.99,
    buffer_capacity=50000,
    batch_size=64,
    min_buffer_size_before_training=500,
    train_freq=1,
    target_update_freq=500,
    epsilon_start=1.0,
    epsilon_end=0.1,                 # was 0.05 — keep more exploration for MountainCar
    epsilon_decay_episodes=700,      # was 100 — much slower decay
    num_eval_episodes=10,
    eval_every=50,
)

print("final eval episodes:", history["eval_episode_idx"])
print("final eval returns:", history["eval_return_mean"])

#lr = 0.1 , 0.01, 0.001
#gamma = 0.99, 0.90 
#seeds = 1,2,3,4,5,6,7,8,9,0
#remove logging, run on GPU for increased speed
#Run parameters sweep through bash files stateless (Stable)
#XLA_PYTHON_CLIENT_PREALLOCATE=false 
#XLA_PYTHON_CLIENT_MEM_FRACTION=0
 

 