import sys
from wrapplib.wrapper import wrap_atari
sys.path.append("./")# Add the current directory to the system path
import matplotlib.pyplot as plt
plt.ion()
import gym
import torch
from wrapplib.Network import VNet,CQNet
from wrapplib.discreta_arm import DiscretaARM,DiscreteARMPoicy
def main():
    if len(sys.argv)>2:
        total_step_limit=int(sys.argv[2])
    else:
        total_step_limit=10000000
    if len(sys.argv)>1:
        env_id=sys.argv[1]
#     sys.argv 是一个列表，其中包含了命令行通过命令提示符（如终端或命令行界面）传递给
#      Python 脚本的参数。这个列表的第一个元素（即 sys.argv[0]）是脚本的名称，
#     或者是脚本的完整路径，这取决于你如何运行脚本。
    else:
        env_id=("PongNoFrameskip-v4")

    env=gym.make(env_id,render_mode="rgb_array")#,render_mode="human"


    frame_skip=4
    #把上面一行注释掉会默认不跳过，即默认frame_skip=1；也可以换成跳过其它帧数，不一定是4
    preproc_pong_mask=True
    #preproc_pong_mask=False
    #是否使用遮盖，营造部分可观环境的参数
    history_len=4
    #history_len=1
    #该选项用于控制在observation中药堆叠的最近的帧数，为1表示只用当前帧作为观测，不使用过去的历史帧叠加起来

    config={"env_id":env_id,"frame_skip":frame_skip,"preproc_pong_mask":preproc_pong_mask,"history_len":history_len}
    print("config : ",config)

    env=wrap_atari(env,frame_skip=frame_skip,prepro_pong_mask=preproc_pong_mask,history_len=history_len)

    arm_config={
        #如果多于一个cached batch ，那就是off-policy的缓冲区，总sim steps有batch_cfg=【"smple_size"】*num_cached_batches
        "num_cached_batches":1,
        #Should be kept as "uniform" (prioritized sampling is not implemented).
        "sampling_strategy":"uniform",
        #使用n-steps的时序差分学习估计器
        "n_steps":1,
        #回报的折扣因子
        "discounted_rate":0.99,
        #奖励（reward）进行调整的过程，以帮助学习算法更有效地学习。这通常涉及到将原始奖励值乘以一个常数因子（scaling factor），或是应用更复杂的数学变换，以使奖励值的范围适应学习算法的优化过程。
        #“None is the default unit scaling.”时，意味着在没有进行任何额外的奖励调整的默认情况下，奖励的缩放比例是1（或不进行缩放）。这里的“unit scaling”基本上是说，每一个给予的奖励值都按照它原本的比例和大小被直接用于算法的训练过程中，没有经过修改或缩放。
        "reward_scaling":None,
        #对应随即策略采得的数据，Adam优化器在第一个batch=batch_cfg=【"smple_size"】，迭代训练的轮数
        "initial_num_arm_iters":3000,
        #在之后的每一个batch上，batch_cfg=【"smple_size"】，Adam优化器迭代训练的轮数
        "num_arm_iters":3000,
        #Adam优化器的小批量随机梯度下降的小批量的大小
        "minibatch_size":32,
        #MOVING AVERAGE (τ )Target value function parameters are updated via moving average with this rate.对应公式（17）
        "tua":0.01,
    }
    print("debug : arm config : {}".format(arm_config))
    #Batchsize ,batch_configuration,主要设置每批中transitions的数量,1个batch cache所存储的数量
    batch_cfg={
        "sample_size":12500,
        "num_trajs":None,
    }
    print("debug of batch cfg : ",batch_cfg)
    grad_clip=None


    input_channel=history_len
    action_dim=env.action_space.n
    device=device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    v_fun=VNet(input_channel,1,device=device)
    ccq_fun=CQNet(input_channel,action_dim,device=device)
    prev_v_fun=VNet(input_channel,1,device=device)
    prev_ccq_fun=CQNet(input_channel,action_dim,device=device)
    target_v_fun=VNet(input_channel,1,device=device)
    arm=DiscretaARM(arm_config,batch_cfg,device=device)
    arm.reset(env,target_v_fun,prev_ccq_fun,prev_v_fun,v_fun,ccq_fun,grad_clip)

    total_step_counts=0
    while total_step_counts<=total_step_counts:
        total_step_counts+=arm.run(env)



    # num_episodes=5
    # for i in range(num_episodes):
    #     observation=env.reset()
    #     terminated=False
    #     step_counter=0
    #     while not terminated:
    #         action=env.action_space.sample()
    #         observation,reward,terminated,truncated,info=env.step(action)
    #         print(observation)
    #         step_counter+=1
    #         # plt.imshow(observation)
    #         # plt.title("Initial Frame from Breakout")
    #         # plt.axis('off')  # 关闭坐标轴
    #         # plt.pause(0.5)
    #
    #     print(f"Episode {i + 1} finished after {step_counter} steps.")


if __name__=="__main__":
    main()