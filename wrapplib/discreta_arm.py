import gym
import numpy as np
import torch
import torch.nn.functional as F
from wrapplib.sample import SampleBatch,SampleBatchCache
from wrapplib.utils import perf_counter
from wrapplib.policy import UniformCategoricalPolicy,DiscreteARMPoicy
class DiscretaARM(object):
    def __init__(self,arm_cfg,batch_config,device=torch.device("cpu")):
        self._cfg=arm_cfg
        self._batch_cfg = batch_config

        self._iteration_num=0
        self._steps_alleps=0
        self._batch_cache=SampleBatchCache(self._cfg["num_cached_batches"])





        self._v_fun=None
        self._target_v_fun=None
        self._prev_ccq_fun=None
        self._prev_v_fun=None
        self._prev_ccq_fun=None
        self._ccq_fun=None

        self._init_policy=None
        self._policy=None

        self.grad_clip=None

        self.device=device
    def reset(self,env,target_v_value_func,prev_ccq_func,prev_v_func,v_func,ccq_func,grad_clip=None):


        self._target_v_fun=target_v_value_func
        self._prev_ccq_fun=prev_ccq_func
        self._prev_v_fun=prev_v_func
        self._v_fun=v_func
        self._ccq_fun=ccq_func

        self.grad_clip=grad_clip

        self._init_policy=UniformCategoricalPolicy(env.action_space.n)
        self._policy=DiscreteARMPoicy(env.action_space.n,self._ccq_fun,self._v_fun)

    def average_v_para(self,tua):
        target_params=self._target_v_fun.parameters()
        params=self._v_fun.parameters()
        for target,net in zip(target_params,params):
            target.data.add_(tua*(net.data-target.data))

    def copy_param(self,dst_var,src_var):
        w_p=dst_var.parameters()
        v_p=src_var.parameters()
        for w,v in zip(w_p,v_p):
            w.data.copy_(v.data)

    def run(self,env):
        print("debug : arm: iteration :{} total steps : {}".format(self._iteration_num,self._steps_alleps))
        online_batch=SampleBatch()
        if self._iteration_num==0:#and self._cfg["initialize_uniform"]: 进行rollout收集数据刚开始第一轮迭代，采用随机均匀初始策略
            online_batch.resample_catagorical(self._batch_cfg,env,self._init_policy)
        else:
            online_batch.resample_catagorical(self._batch_cfg,env,self._policy)
        self._batch_cache.append(online_batch)
        self._batch_cache.vectorize_categorical()

        # if self._iteration_num==0:
        #     pass
        # else:
        #     self._batch_cache.reweight_categorical(self._policy)

        n_step=self._cfg["n_steps"]
        discount_rate=self._cfg["discounted_rate"]
        nsteps_discount=float(np.power(discount_rate,n_step))
        reward_scale=None
        if "reward_scaling" in self._cfg:
            reward_scale=self._cfg["reward_scaling"]
        else:
            reward_scale=1.0

        initial_num_arm_iters=self._cfg["initial_num_arm_iters"]
        num_arm_iters=self._cfg["num_arm_iters"]

        minbatch_size=self._cfg["minibatch_size"]
        tua_moving_average=self._cfg["tua"]


        if self._iteration_num==0:#根据是不是第一轮确定训练迭代的轮数
            curr_num_arm_iters=initial_num_arm_iters
        else:
            curr_num_arm_iters=num_arm_iters

        avg_v_loss=torch.zeros(1).cuda()
        avg_ccq_loss=torch.zeros(1).cuda()
        last_display_iter=0
        last_t=perf_counter()

        for iter in range(curr_num_arm_iters):
            idx,nstep_idx,observation_ks,observation_kpns,act_idx_ks,v_rets,q_rets,dones=self._batch_cache.sample_minibatch(minibatch_size=minbatch_size,n_steps=n_step,discounted_rate=discount_rate,reward_scale=reward_scale,categorical=True)
            # observation_ks.to(self.device)
            # observation_kpns.to(self.device)
            observation_ks=observation_ks.to(self.device)
            observation_kpns=observation_kpns.to(self.device)
            # print("self device : ",self.device)
            act_idx_ks=act_idx_ks.cuda()
            v_rets=v_rets.cuda()
            q_rets=q_rets.cuda()
            dones=dones.cuda()
            target_v_b=self._target_v_fun(observation_kpns)
            target_v_b=(1-dones)*torch.squeeze(target_v_b,1)
            target_v_value=v_rets+nsteps_discount*target_v_b
            if self._iteration_num==0:
                target_ccq_k=q_rets+nsteps_discount*target_v_b
            else:
                target_ccq_k=torch.clamp(torch.squeeze(torch.gather(self._prev_ccq_fun(observation_ks),1,torch.unsqueeze(act_idx_ks,dim=1)))-torch.squeeze(self._prev_v_fun(observation_ks)),min=0)+q_rets+nsteps_discount*target_v_b
            target_v_value=target_v_value.detach()#将张量从向量图中分离出来
            target_ccq_k=target_ccq_k.detach()
            print("debug : target_v_k sixe {} target_ccq_k size:{}".format(target_v_value.size(),target_ccq_k.size()))

            self._v_fun.optimizer.zero_grad()
            v_loss=self._v_fun.criterion(torch.squeeze(self._v_fun(observation_ks)),target_v_value)
            ccq_loss=self._ccq_fun.criterion(torch.squeeze(torch.gather(self._ccq_fun(observation_ks),1,torch.unsqueeze(act_idx_ks,dim=1))),target_ccq_k)
            v_loss.backward()
            ccq_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm(self._v_fun.parameters(),self.grad_clip)
                torch.nn.utils.clip_grad_norm(self._ccq_fun.parameters(),self.grad_clip)
            self._v_fun.optimizer.step()
            self._ccq_fun.optimizer.step()

            self.average_v_para(tua_moving_average)
            avg_v_loss.add_(v_loss)
            avg_ccq_loss.add_(ccq_loss)

            if (iter+1)%400==0 or (iter+1)==curr_num_arm_iters:#没个epoch迭代多少个批次; 没400个批次输出一下这400个批次以来的平均损失
                batch_num_now=iter+1-last_display_iter
                lap_t=perf_counter()
                elapsed_s=float(lap_t-last_t)
                print("debug : arm: iters:{} v_loss : {:.6f},ccq_loss : {:.6f} eplapsed : {:.3f}".format(
                    iter+1,v_loss.detach().cpu().item()/batch_num_now,ccq_loss.detach().cpu().item()/batch_num_now,elapsed_s
                ))

                avg_v_loss.zero_()
                avg_ccq_loss.zero_()
                last_display_iter=iter+1
                last_t=lap_t

        self._iteration_num+=1
        self._steps_alleps+=online_batch.step_count()

        self.copy_param(self._prev_v_fun,self._v_fun)
        self.copy_param(self._prev_ccq_fun,self._ccq_fun)

        return online_batch.step_count()










