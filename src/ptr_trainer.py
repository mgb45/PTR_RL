import torch.nn as nn
from torch.optim import AdamW
import torch
import numpy as np
from imitation.rewards.reward_nets import  ForwardWrapper
from imitation.util.networks import RunningNorm, EMANorm

class NRN(ForwardWrapper):

    def __init__(self,base):
        super().__init__(base)

        self.base = base
        self.out = RunningNorm(1)

    def forward(self,st,ac,nst,done):
        return self.out(torch.nn.ReLU()(self.base(st,ac,nst,done)))

class PTR_Trainer:

    def __init__(self,reward_net,comparisons=64,lr=1e-2):

        self.comparisons = comparisons
        self.lr = lr

        self.reward = reward_net
        self.loss = torch.nn.BCELoss()
        self.optimizer = AdamW(self.reward.parameters(),lr=self.lr)

    def train(self,rollouts,epochs):

        for i in range(epochs):
            # Sample random rollout
            idxs = np.random.randint(0,len(rollouts),self.comparisons)
            
            # Sample a batch of pairs in time

            id_1 = np.array([np.random.randint(0,rollouts[idx].obs.shape[0]-2) for idx in idxs])
            id_2 = np.array([np.random.randint(id,rollouts[idx].obs.shape[0]-1,1) for id,idx in zip(id_1,idxs)])
            s1 = np.array([rollouts[idx].obs[id_] for idx,id_ in zip(idxs,id_1)])
            a1 = np.array([rollouts[idx].acts[id_] for idx,id_ in zip(idxs,id_1)])
            s2 = np.array([rollouts[idx].obs[id_] for idx,id_ in zip(idxs,id_2)])
            a2 = np.array([rollouts[idx].acts[id_] for idx,id_ in zip(idxs,id_2)])

            # Train
            # a = torch.from_numpy(rollouts[idx].acts[id_1]).reshape(self.comparisons,-1)
            # s = torch.from_numpy(rollouts[idx].obs[id_1,:]).reshape(self.comparisons,-1)
            st,at,sst,dt = self.reward.preprocess(state=s1,action=a1,next_state=s1,done=np.zeros((self.comparisons,1)))
            r1 = self.reward(st,at,sst,dt).reshape(-1,1)
        
            st,at,sst,dt = self.reward.preprocess(state=s2,action=a2,next_state=s2,done=np.zeros((self.comparisons,1)))
            r2 = self.reward(st,at,sst,dt).reshape(-1,1)
            
            score = torch.sigmoid(r2-r1)
            self.optimizer.zero_grad()
            out = self.loss(score,torch.ones_like(score))
            out.backward()
            self.optimizer.step()
            print("\rEpoch %d: %f"%(i, out.item()),end="")




