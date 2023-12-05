import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import load
from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL
from imitation.util.networks import RunningNorm, EMANorm
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from ptr_trainer import PTR_Trainer, NRN
from stable_baselines3.common.utils import set_random_seed

import argparse
import random
import torch as th

parser = argparse.ArgumentParser(description='Collect demonstrations for imitation learning')
parser.add_argument('--logdir',type=str,default='./logs/',help="Folder for logging")
parser.add_argument('--env',type=str,default='Pendulum-v1',help="gym environment")
parser.add_argument('--steps',type=int,default=500000,help="Number of training steps")
parser.add_argument('--episodes',type=int,default=100,help="Number of test episodes")
parser.add_argument('--epochs',type=int,default=500000,help="Number of policy training epochs")
parser.add_argument('--demos',type=int,default=100,help="Number of demos")
parser.add_argument('--method',type=str,default='PTR',help="Imitation learning approach")
parser.add_argument('--seed',type=int,default=42,help="Seed for reproducibility")
parser.add_argument('--lr',type=float,default=1e-4,help="PPO learning rate")
parser.add_argument('--batch_size',type=int,default=1024,help="PPO batch size")

args = parser.parse_args()

# Pendulum - lr = 5e-4 batch_size = 4-96


if __name__ == "__main__":

    seed = args.seed
    env = gym.make(args.env)
    env.seed(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    set_random_seed(seed)
    th.manual_seed(seed)
    venv = make_vec_env(args.env, n_envs=1, rng=rng)

    rollouts = load(args.logdir+'demos_%s%d_%d.npz'%(args.env,args.steps,args.demos))
    transitions = rollout.flatten_trajectories(rollouts)
    
    print("Training a policy using %s"%args.method)
    if args.method == 'BC':

        trainer = bc.BC(
            observation_space=env.observation_space, 
            action_space=env.action_space, 
            demonstrations=transitions, 
            rng=rng,
        )

        reward, _ = evaluate_policy(
            trainer.policy,  # type: ignore[arg-type]
            env,
            n_eval_episodes=args.episodes,
            render=False,
        )
        print(f"Reward before training: {reward}")
            
        trainer.train(n_epochs=20)

    elif args.method == 'GAIL':

        learner = PPO(env=venv, 
                    policy=MlpPolicy,
                    seed=seed,
                    )
        reward_net = BasicRewardNet(
            env.observation_space,
            env.action_space,
            normalize_input_layer=RunningNorm
        )
        trainer = GAIL(
            demonstrations=rollouts,
            demo_batch_size=1024,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=4,
            venv=venv,
            gen_algo=learner,
            reward_net=reward_net,
            allow_variable_horizon=True, 
            log_dir = args.logdir+''+args.method+'_'+args.env+'_%d_%d_%d/'%(args.steps,args.demos,args.epochs),
            init_tensorboard=True
        )
        reward, _ = evaluate_policy(
            trainer.policy,  # type: ignore[arg-type]
            env,
            n_eval_episodes=args.episodes,
            render=False,
        )
        print(f"Reward before training: {reward}")
        trainer.train(args.epochs)

    elif args.method == 'AIRL':

        learner = PPO(env=venv, 
                        policy=MlpPolicy,
                        seed=seed
                    )
        reward_net = BasicShapedRewardNet(
            venv.observation_space,
            venv.action_space,
            normalize_input_layer=RunningNorm
        )
        trainer = AIRL(
            demonstrations=rollouts,
            demo_batch_size=1024,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=4,
            venv=venv,
            gen_algo=learner,
            reward_net=reward_net,
            allow_variable_horizon=True,
            log_dir = args.logdir+'/'+args.method+'_'+args.env+'_%d_%d_%d/'%(args.steps,args.demos,args.epochs),
            init_tensorboard=True
        )
        reward, _ = evaluate_policy(
            trainer.policy,  # type: ignore[arg-type]
            env,
            n_eval_episodes=args.episodes,
            render=False,
        )
        print(f"Reward before training: {reward}")

        trainer.train(args.epochs)

    elif args.method == 'PTR':

        base = BasicRewardNet(
            venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm, use_action=False,use_next_state=False, use_done=False)

        reward_net = NRN(base)
        rtrainer = PTR_Trainer(reward_net=reward_net,lr=1e-4)

        print("Learing reward function.")
        rtrainer.train(rollouts,epochs=1000)
        
        rtrainer.reward.eval()
        learned_reward_venv = RewardVecEnvWrapper(venv, rtrainer.reward.predict)

        trainer= PPO(
            policy=MlpPolicy,
            env=learned_reward_venv,
            seed=seed,
            verbose=1,
            ent_coef=0.001,
            batch_size=1024,
            learning_rate=args.lr,
            tensorboard_log=args.logdir+'/'+args.method+'_'+args.env+'_%d_%d_%d'%(args.steps,args.demos,args.epochs)
        )

        reward, _ = evaluate_policy(
            trainer.policy,  # type: ignore[arg-type]
            env,
            n_eval_episodes=args.episodes,
            render=False,
        )
        print(f"Reward before training: {reward}")
        trainer.learn(args.epochs)
        trainer.save(args.logdir+'/'+args.method+'_'+args.env+'_%d_%d_%d/%d.zip'%(args.steps,args.demos,args.epochs,args.seed))

    mean_reward, std_reward = evaluate_policy(
        trainer.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=args.episodes,
        render=False
    )
    print(f"Reward after training: {mean_reward}")
    np.save(args.logdir+args.method+'_'+args.env+'_%d_%d_%d/performance_%d.npy'%(args.steps,args.demos,args.epochs,args.seed),np.array([mean_reward,std_reward]))