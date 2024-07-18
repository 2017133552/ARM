import numpy as np
import torch


class UniformCategoricalPolicy(object):
    def __init__(self,action_dim):
        self.action_dim=action_dim
        self.action_prob=np.float32(1/self.action_dim)

    def __call__(self, obs_buffer):
        if isinstance(obs_buffer,np.ndarray):#用于采样的计算动作的概率是，环境返回的状态是ndarray的形式
            batch_size=1
        elif isinstance(obs_buffer,torch.Tensor):#用于训练时策略的计算，批量化的
            batch_size=obs_buffer.size(0)#每个批量的大小，有多少个trnsiotion
        else:
            raise NotImplementedError

        action_indexs=[]
        action_probs=[]
        for i in range(batch_size):
            action_index=np.random.choice(self.action_dim)
            action_prob=self.action_prob
            action_indexs.append(action_index)
            action_probs.append(action_prob)

        return action_indexs,action_probs


class DiscreteARMPoicy(object):
    def __init__(self,action_dim,ccq_fun,v_fun):
        self.action_dim=action_dim
        self.ccq_fun=ccq_fun
        self.v_fun=v_fun

    def __call__(self,obs_buffer,only_action_index=None):
        # if only_action_index is not None:
        print("debug :  torch observation size : ",obs_buffer.size())
        print("arm policy : Unkonwn type : ",type(obs_buffer))
        self.ccq_fun.eval()
        self.v_fun.eval()
        with torch.no_grad():
            regrets_vector=torch.clamp(self.ccq_fun(obs_buffer)-self.v_fun(obs_buffer),min=0)
        regrets=regrets_vector.data.cpu().numpy()
        sum_regerts=np.sum(regrets,axis=1,keepdims=True)#axis=1 指定在每一行上进行求和操作。keepdims=True 保持求和后的数组的维度，这样结果数组在求和的那个轴上仍然有维度 1，而不是被压缩掉。
        if sum_regerts:
            action_prob_list=regrets/sum_regerts
        else:
            action_prob_list=torch.full([self.action_dim],1/self.action_dim)
        action=int(torch.multinomial(action_prob_list,1))

        return action_prob_list,action_prob_list





