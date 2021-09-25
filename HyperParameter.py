from dataclasses import dataclass


# Default Hyperparameters
SCALE_REWARD:         float = 0.01
MIN_REWARD:           float = -1000.
HIDDEN_SIZE:          float = 128
BATCH_SIZE:           int   = 512
DISCOUNT:             float = 0.99
GAE_LAMBDA:           float = 0.95
PPO_CLIP:             float = 0.2
PPO_EPOCHS:           int   = 10
MAX_GRAD_NORM:        float = 1.
ENTROPY_FACTOR:       float = 0.
ACTOR_LEARNING_RATE:  float = 1e-4
CRITIC_LEARNING_RATE: float = 1e-4
RECURRENT_SEQ_LEN:    int = 8
RECURRENT_LAYERS:     int = 1    
ROLLOUT_STEPS:        int = 2048
PARALLEL_ROLLOUTS:    int = 8
PATIENCE:             int = 200
TRAINABLE_STD_DEV:    bool = False 
INIT_LOG_STD_DEV:     float = 0.0


@dataclass
class HyperParameters():
    scale_reward:         float = SCALE_REWARD
    min_reward:           float = MIN_REWARD
    hidden_size:          float = HIDDEN_SIZE
    batch_size:           int   = BATCH_SIZE
    discount:             float = DISCOUNT
    gae_lambda:           float = GAE_LAMBDA
    ppo_clip:             float = PPO_CLIP
    ppo_epochs:           int   = PPO_EPOCHS
    max_grad_norm:        float = MAX_GRAD_NORM
    entropy_factor:       float = ENTROPY_FACTOR
    actor_learning_rate:  float = ACTOR_LEARNING_RATE
    critic_learning_rate: float = CRITIC_LEARNING_RATE
    recurrent_seq_len:    int = RECURRENT_SEQ_LEN
    recurrent_layers:     int = RECURRENT_LAYERS 
    rollout_steps:        int = ROLLOUT_STEPS
    parallel_rollouts:    int = PARALLEL_ROLLOUTS
    patience:             int = PATIENCE
    # LSTM
    use_lstm:             bool = True
    # Apply to continous action spaces only 
    trainable_std_dev:    bool = TRAINABLE_STD_DEV
    init_log_std_dev:     float = INIT_LOG_STD_DEV
    # Stop condition
    max_iterations: int = 1000000
    noise:                float = 0.0