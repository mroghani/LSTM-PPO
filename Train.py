import argparse

from HyperParameter import HyperParameters
from Trainer import Trainer

def make_hp(args) -> HyperParameters:
    if args.env == "CartPole-v1" and args.mask_velocity:
        # Working perfectly with patience.
        hp = HyperParameters(parallel_rollouts=32, rollout_steps=512, batch_size=128, recurrent_seq_len=8, patience=200)
    elif args.env == "CartPole-v1" and not args.mask_velocity:
        # Working perfectly with patience.
        hp = HyperParameters(parallel_rollouts=32, rollout_steps=512, batch_size=128, recurrent_seq_len=8, patience=200)

    elif args.env == "Pendulum-v0" and args.mask_velocity:
        # Works well.     
        hp = HyperParameters(parallel_rollouts=32, rollout_steps=200, batch_size=512, recurrent_seq_len=8,
                            init_log_std_dev=1., trainable_std_dev=True, actor_learning_rate=1e-3, critic_learning_rate=1e-3)

    elif args.env == "LunarLander-v2" and args.mask_velocity:
        # Works well.
        hp = HyperParameters(parallel_rollouts=32, rollout_steps=1024, batch_size=512, recurrent_seq_len=8, patience=1000) 

    elif args.env == "LunarLanderContinuous-v2" and args.mask_velocity:
        # Works well.
        hp = HyperParameters(parallel_rollouts=32, rollout_steps=1024, batch_size=1024, recurrent_seq_len=8, trainable_std_dev=True,  patience=200)
    elif args.env == "LunarLanderContinuous-v2" and not args.mask_velocity:
        # Works well.
        hp = HyperParameters(parallel_rollouts=32, rollout_steps=1024, batch_size=1024, recurrent_seq_len=8, trainable_std_dev=True,  patience=100)
        
    elif args.env == "BipedalWalker-v2" and not args.mask_velocity:
        # Working :-D
        hp = HyperParameters(parallel_rollouts=8, rollout_steps=2048, batch_size=256, patience=1000, entropy_factor=1e-4,
                            init_log_std_dev=-1., trainable_std_dev=True, min_reward=-1.)
                            #init_log_std_dev=1., trainable_std_dev=True)
        
    elif args.env == "BipedalWalkerHardcore-v2" and not args.mask_velocity:
        # Working :-D
        hp = HyperParameters(batch_size=1024, parallel_rollouts=32, recurrent_seq_len=8, rollout_steps=2048, patience=10000, entropy_factor=1e-4, 
                            init_log_std_dev=-1., trainable_std_dev=True, min_reward=-1., hidden_size=256)
    else:
        raise NotImplementedError
    
    hp.use_lstm = args.use_lstm
    hp.noise = args.noise
    return hp

def train(args):
    hp = make_hp(args)
    experiment_name = f'{args.env}_{"LSTM" if args.use_lstm else "NoLSTM"}_{"NoVelocity" if args.mask_velocity else "Velocity"}_noise{args.noise}'
    trainer = Trainer(args.env, args.mask_velocity, experiment_name, hp)
    score = trainer.train()
    print(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default='CartPole-v1')
    parser.add_argument("-m", "--mask-velocity", action='store_true')
    parser.add_argument("-n", "--name", type=str, default='experiment')
    parser.add_argument("-R", "--use-lstm", action='store_true')
    parser.add_argument("--noise", type=float, default=0.0)

    args = parser.parse_args()

    train(args)