import numpy as np
import torch,torchvision
from collections import deque
from wrapplib.utils import perf_counter


class SampleTransition(object):
    def __init__(self):
        self.action=None
        self.action_index=None #只要离散动作空间有
        self.action_prob=None
        self.reward=None
        self.terminal=None
        self.next_observation=None


class SampleTraj(object):
    def __init__(self):
        self.init_observation=None
        self.steps=[]

    def step_count(self):
        return len(self.steps)

    def sum_return(self):
        ret=0.0
        for transition in self.steps:
            ret+=transition.reward

        return ret
    def init(self,init_observation):
        self.init_observation=init_observation

    def append(self,action,reward,terminal,net_observation,action_pro=None):
        transition=SampleTransition()

        transition.action=action
        transition.action_pro=action_pro
        transition.reward=reward
        transition.terminal=terminal
        transition.next_observation=net_observation

        self.steps.append(transition)

    def append_categorical(self,action_index,action_prob,reward,terminal,next_obs):

        transition=SampleTransition()
        transition.action_index=action_index
        transition.action_prob=action_prob
        transition.reward=reward
        transition.terminal=terminal
        transition.next_observation=next_obs

        self.steps.append(transition)

class SampleBatch(object):
    def __init__(self):
        self._step_ct=0
        self.trajs=[]

        self.observation_buffer=None
        self.action_buffer=None
        self.action_index_buffer=None
        self.act_prob_buffer=None
        self.reward_buffer=None
        self.terminal_buffer=None
        self.end_epo_flag_buffer=None

    def step_count(self):#该cache所有轨迹的步骤总数
        return self._step_ct

    def traj_count(self):
        return len(self.trajs)
    def resample_catagorical(self,batch_cfg,env,cat_policy,epoch,writer):
        print("debug : batch: resampling categorical policy...")
        batch_start_t=perf_counter()
        self._step_ct=0
        self.trajs[:]=[]#使用 rajs[:] = [] 会将现有列表 rajs 中的所有元素移除，使其变为空列表。这个操作会原地修改列表，意味着它会改变原来列表对象的内容，而不改变列表的引用（或内存地址）
        avg_return=0#记录所有轨迹的平均回报值
        max_return=-float("inf")#记录所有轨迹中汇报最大的那条轨迹的回报值
        min_return=float("inf")#记录所有轨迹中回报最小的那条轨迹的回报值

        while self._step_ct<batch_cfg["sample_size"]:
            ep_start_t=perf_counter()
            traj=SampleTraj()#代表一条轨迹
            observation=env.reset()
            traj.init(observation)#注意，初始状态s1单独存的；一个transition(a1,r1,done,s2)
            # # convert to grayscale, scale to 84x84 and scale values between 0 and 1
            # pre_torchvision = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
            #                                                   torchvision.transforms.Grayscale(),
            #                                                   torchvision.transforms.Resize(
            #                                                       (84, 84)),
            #                                                   torchvision.transforms.ToTensor()])
            #
            # # remove channels and convert to numpy
            # def preprocess(img):
            #     return pre_torchvision(img)[0].numpy()
            #
            # observation=pre_torchvision(observation)
            while True:#术语 "simulation steps" 或简称 "sim steps" 常常被用来描述在仿真过程中发生的单个迭代或时间步。每个 simulation step 代表仿真时间中的一个离散单元，在这个时间单元内，系统的状态根据预定义的物理法则、规则或算法进行更新。
                action_indexl,action_probl=cat_policy(observation)#注意，返回的是一个列表，尽管元素只有一个[0-5]
                action_index=action_indexl[0]
                action_prob=float(action_probl[0])
                observation,reward,terminal,truncated,info=env.step(action_index)#某条轨迹中的一个transition
                traj.append_categorical(action_index,action_prob,reward,terminal,observation)
                if terminal:
                    break
            self._step_ct+=traj.step_count()
            self.trajs.append(traj)
            avg_return+=1.0/len(self.trajs)*(traj.sum_return()-avg_return)#增量式计算平均值，来一个数据计算一个数据
            max_return=max(max_return,traj.sum_return())
            min_return=min(min_return,traj.sum_return())

            ep_lap_t=perf_counter()
            ep_elapsed=float(ep_lap_t-ep_start_t)#这一条轨迹收集所需要的时间
            elapased_s=float(ep_lap_t-batch_start_t)#整个采样到这一条路径所需要的时间

            print("debug: batch:  traj_index :{} return of this traj:{:.3f}  steps of this traj:{}  steps of all trajs: {} elapsed of this traj: {:.3f} batch eplapsed {:.3f}".format(
                len(self.trajs),traj.sum_return(),traj.step_count(),self._step_ct,ep_elapsed,elapased_s))

        print("debug: batch: trajs:{} steps of all trajs:{} avg_return:{:.3f} max_return:{:.3f} min_return:{:.3f}  all trajs elapased {:.3f}".format(
            self.traj_count(),self.step_count(),avg_return,max_return,min_return,elapased_s))
        if writer is not None:
            writer.add_scalar("return : ",avg_return,epoch)

    def is_vectorized(self):
        return self.observation_buffer is not None

    def vectorize_categorical(self):
        xbatch_size=self.step_count()+self.traj_count()#该 cache batch 所有轨迹中的步骤总数加上，轨迹本身的数量【防止溢出的作用】

        #设置缓冲区的大小
        observation_buffer_h = torch.FloatTensor(*([xbatch_size] + list(self.trajs[0].init_observation.shape)))
        action_index_buffer_h=torch.LongTensor(xbatch_size)
        act_prob_buffer_h=torch.FloatTensor(xbatch_size)
        reward_buffer_h=torch.FloatTensor(xbatch_size)
        terminal_buffer_h=torch.FloatTensor(xbatch_size)
        end_epo_flag_buffer_h=torch.FloatTensor(xbatch_size)

        #全部初始化为0
        observation_buffer_h.zero_()
        action_index_buffer_h.zero_()#把所有的trajs排成一排
        act_prob_buffer_h.zero_()
        reward_buffer_h.zero_()
        terminal_buffer_h.zero_()
        end_epo_flag_buffer_h.zero_()

        traj_offsets=[]#用于存储每个轨迹在缓冲区中的起始位置。
        trajs_length=[]#记录每个轨迹的步骤数。
        traj_offset_ctr=0#是当前轨迹偏移的累加器，用于计算下一个轨迹的起始位置。
        for i ,traj in enumerate(self.trajs):
            traj_offsets.append(traj_offset_ctr)
            trajs_length.append(traj.step_count())####traj.step_count()===trajs_length[i]
            traj_terminal=False
            #observation_buffer_h[traj_offsets[i], :]=torch.from_numpy(traj.init_observation)
            observation_buffer_h[traj_offsets[i],:].copy_(torch.from_numpy(traj.init_observation))#在PyTorch中，如果你只索引行而不使用 [,] 进行列切片，仍然是有效的，意味着整行都将被选中。
            terminal_buffer_h[traj_offsets[i]]=0#初始化 done_buffer_h，表示轨迹起始时尚未结束，1表示轨迹结束。
            for k in range(traj.step_count()):
                if traj.steps[k].terminal:
                    traj_terminal=True
                if traj_terminal:#更新终端标志，并检查后续步骤是否正确标记为终端。
                    assert traj.steps[k].terminal#现在又变回来了transition=(s1,a1,r1,done)
                action_index_buffer_h[traj_offsets[i]+k]=traj.steps[k].action_index
                act_prob_buffer_h[traj_offsets[i]+k]=traj.steps[k].action_prob
                reward_buffer_h[traj_offsets[i]+k]=traj.steps[k].reward
                terminal_buffer_h[traj_offsets[i]+k+1]=1.0 if traj.steps[k].terminal else 0
                observation_buffer_h[traj_offsets[i]+k+1].copy_(torch.from_numpy(traj.steps[k].next_observation))
                #注意到，1表示轨迹结束。;结束的那个索引只存储了(st;)
            action_index_buffer_h[traj_offsets[i]+traj.step_count()]=-1
            act_prob_buffer_h[traj_offsets[i]+traj.step_count()]=0
            reward_buffer_h[traj_offsets[i]+traj.step_count()]=0
            end_epo_flag_buffer_h[traj_offsets[i]+traj.step_count()]=1#可以与terminal相互验证,1表示该轨迹结束标志;即最后一个索引(s2;-1,0,0,terminal=1,end=1)
            traj_offset_ctr+=traj.step_count()+1#加1是因为最后一个元素使用了存储有关轨迹结束的信息
        assert traj_offset_ctr==xbatch_size,"number is not eaqual"

        observation_buffer_d=torch.FloatTensor(*([xbatch_size]+list(self.trajs[0].init_observation.shape)))
        observation_buffer_d.copy_(observation_buffer_h)


        self.xbatch_size=xbatch_size
        self.observation_buffer=observation_buffer_d
        self.action_index_buffer=action_index_buffer_h
        self.act_prob_buffer=act_prob_buffer_h
        self.reward_buffer=reward_buffer_h
        self.terminal_buffer=terminal_buffer_h
        self.end_epo_flag_buffer=end_epo_flag_buffer_h

        print("debug : obersvation buffer size :{}".format(self.observation_buffer.size()))
        print("debug : action index buffer size :{}".format(self.action_index_buffer.size()))











    def reweight_categorical(self,online_policy,clip_weight=1.0,chunk_size=32):
        pass

    def sample_transition(self,nsteps=1,discounted_rate=1.0,reward_scale=None,reward_weight=True,strategy="uniform"):
        if strategy=="uniform":
            idx=None
            while True:#采到合法的序号
                idx=int(np.random.choice(self.xbatch_size))
                if self.terminal_buffer[idx]==1 or self.end_epo_flag_buffer[idx]==1:
                    continue
                break
            assert idx is not None,"there is  not sample in this batch cache"

            terminal_index=None
            end_index=None
            for step in range(1,nsteps+1):
                if terminal_index is None and end_index is None:
                    if self.terminal_buffer[idx+step]!=0:
                        terminal_index=idx+step#终端节点的序号
                    if self.end_epo_flag_buffer[idx+step]!=0:
                        end_index=idx+step
                        break
            assert terminal_index==end_index,"terminal_index not equal end_index !!! error "

            nstep_idx=None
            if terminal_index!=end_index:
                print("erroe: what ,not equal!!!!")
            if terminal_index is None and end_index is None:#再走n步任然还没到终点
                nstep_idx=idx+nsteps
            elif terminal_index is not None:
                assert terminal_index>=idx
                assert terminal_index<=idx+nsteps#再次检查终止节点序号在给定范围内
                nstep_idx=terminal_index#不能有n步了，只能到这个序号了
            elif end_index is not None:
                assert end_index>=idx
                assert end_index<=idx+nsteps
                nstep_idx=end_index
            else:
                raise NotImplemented
            assert nstep_idx is not None
            assert nstep_idx>=idx
            assert nstep_idx<=idx+nsteps#结束的索引要在指定范围内

            v_nres=0
            q_nres=0
            for step in reversed(range(nsteps)):
                if end_index is not None:
                    if idx+step>=end_index:
                        continue
                if end_index is not None:
                    if idx+step>=terminal_index:
                        continue
                if reward_weight:
                    w = 1.0
                    pass
                else:
                    w=1.0
                r=float(self.reward_buffer[idx+step])#n=1；则只有一个r
                v_nres=w*(r+discounted_rate*v_nres)
                if step!=0:
                    q_nres=w*(r+discounted_rate*q_nres)
                else:
                    q_nres=r+w*discounted_rate*q_nres#第一个奖励不进行甲醛，why？
            if reward_scale is not None:
                v_nres*=reward_scale
                q_nres*=reward_scale

            done=0
            if terminal_index is not None:
                done=1
            #c采样的一个transition序号，n步之后走到的序号，可能不能走n步如果done为1的话；n步回报
            return idx,nstep_idx,v_nres,q_nres,done

        else:
            raise NotImplementedError



























class SampleBatchCache(object):
    def __init__(self,max_num_batches=None):
        self._max_num_batches=max_num_batches
        self._steps_allcount=0
        self._trajs_count=0
        self.batches=deque((),maxlen=max_num_batches)
        self.online_batch_idx=None

    def traj_count(self):
        return self._trajs_count

    def step_count(self):
        return self._steps_allcount

    def batches_count(self):
        return len(self.batches)

    def append(self,new_online_batch):
        while len(self.batches)>=self._max_num_batches:#不能大于指定数量的cache，多了要先删去
            print("debug : batch cache:drop :trajs:{} steps : {} ".format(self._trajs_count,self._steps_allcount))
            pop_batch=self.batches.pop()
            self._steps_allcount-=pop_batch.step_count()
            self._trajs_count-=pop_batch.traj_count()
            assert self._trajs_count>=0
            assert self._steps_allcount>=0
        self._steps_allcount+=new_online_batch.step_count()
        self._trajs_count+=new_online_batch.traj_count()
        new_batch_idx=len(self.batches)#从0开始的序号
        self.batches.append(new_online_batch)
        self.online_batch_idx=new_batch_idx
        assert self.online_batch_idx==len(self.batches)-1
        print("debug: batch cache : all num cache batches : {} total trjas of all caches : {} total steps of all caches : {}".format(
            self.batches_count(),self.traj_count(),self.step_count()
        ))

    def vectorize_categorical(self):
        xbatch_sizes=[]
        total_xbatch_size=0

        for batch_idx in range(len(self.batches)):
            batch=self.batches[batch_idx]
            if not batch.is_vectorized():
                batch.vectorize_categorical()
            xbatch_size=batch.step_count()+batch.traj_count()#每一个cache的步骤总数和轨迹总数
            xbatch_sizes.append(xbatch_size)
            total_xbatch_size+=xbatch_size#所有cache的需要用来表示索引的大小
        #这行代码提供了一种非常高效的方式来计算在处理分批或分块数据时每个部分的起始偏移。这在数据加载、批处理处理、以及并行数据处理中非常有用，特别是在需要明确处理每个数据段起始点的场景中。
        xbatch_offsets=[0]+list(np.cumsum(xbatch_sizes))#每一个cache开始的索引【0】+【10,20,50，=【0,10,30,80】
        self.xbatch_sizes=xbatch_sizes
        self.xbatch_offsets=xbatch_offsets

    def reweight_categorical(self,online_policy,clip_weight=1.0,chunk_size=32):
        for batch_idx in range(len(self.batches)):
            if batch_idx==self.online_batch_idx:
                continue
            batch=self.batches[batch_idx]
            batch.reweight_categorical(online_policy,clip_weight=clip_weight,chunk_size=chunk_size)

    def sample_batch(self):
        num_cache_batches=len(self.batches)
        assert num_cache_batches<=self._max_num_batches
        batch_idx=int(np.random.choice(num_cache_batches))#[0, num_cache_batches)中随机输出一个随机数
        batch_offset=self.xbatch_offsets[batch_idx]

        return batch_idx,batch_offset,self.batches[batch_idx]
    def sample_minibatch(self,minibatch_size,n_steps=1,discounted_rate=1,reward_scale=None,reward_weight=True,categorical=False):
        num_bathches=len(self.batches)
        idxs=[]
        nstep_indexs=[]
        obs_ks=[]
        obs_kpns=[]
        act_ks=[]
        v_rets=[]
        q_rets=[]
        terminals=[]

        for _ in range(minibatch_size):
            batch_idx,batch_offset,batch=self.sample_batch()
            if batch_idx==self.online_batch_idx:#采样采到的是当前轮数的策略获得的样本，不进行样本加权
                batch_res_weight=False
            else:
                batch_res_weight=reward_weight
            # idx=int(np.random.choice(self.xbatch_size))
            idx,nstep_idx,v_nres,q_nres,terminal=batch.sample_transition(nsteps=n_steps,discounted_rate=discounted_rate,reward_scale=reward_scale,reward_weight=batch_res_weight,strategy="uniform")
            observation=torch.unsqueeze(batch.observation_buffer[idx,:],dim=0)#两种方式通常也会得到相同的结果
            observation_nstep=torch.unsqueeze(batch.observation_buffer[nstep_idx],dim=0)#如果 batch.obs_buffer[idx,:] 返回一个形状为 (C, H, W) 的张量，那么 torch.unsqueeze(batch.obs_buffer[idx,:], dim=0) 将返回一个形状为 (1, C, H, W) 的张量。在这个例子中，C 可能表示通道数，H 表示高度，W 表示宽度。
            #n步以后得观测
            if batch.action_buffer is not None:
                action=torch.unsqueeze(batch.action_buffer[idx],dim=0)
            else:
                action=int(batch.action_index_buffer[idx])
            idxs.append(batch_offset+idx)
            nstep_indexs.append(batch_offset+nstep_idx)
            obs_ks.append(observation)
            obs_kpns.append(observation_nstep)
            act_ks.append(action)
            v_rets.append(v_nres)
            q_rets.append(q_nres)
            terminals.append(terminal)
        obs_ks=torch.cat(obs_ks,dim=0)#注意，是将列表中的元素，在第0维度进行叠加
        obs_kpns=torch.cat(obs_kpns,dim=0)# 是一个批量数目的终止状态的数量
        # if batch.act_buffer is None:
        if categorical:
            act_ks=torch.from_numpy(np.array(act_ks)).type(torch.LongTensor)
        else:#使用的是动作的序号，说明是离散动作空间
            act_ks=torch.cat(act_ks,dim=0)

        v_rets=torch.from_numpy(np.array(v_rets)).type(torch.FloatTensor)#元素是标量
        q_rets=torch.from_numpy(np.array(q_rets)).type(torch.FloatTensor)
        terminals=torch.from_numpy(np.array(terminals)).type(torch.FloatTensor)

        return idxs,nstep_indexs,obs_ks,obs_kpns,act_ks,v_rets,q_rets,terminals

























