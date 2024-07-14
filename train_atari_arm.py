import sys
from wrapplib.wrapper import wrap_atari
sys.path.append("./")# Add the current directory to the system path
import matplotlib.pyplot as plt
plt.ion()
import gym
import torch
def main():
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

    num_episodes=5
    for i in range(num_episodes):
        observation=env.reset()
        terminated=False
        step_counter=0
        while not terminated:
            action=env.action_space.sample()
            observation,reward,terminated,truncated,info=env.step(action)
            print(observation)
            step_counter+=1
            # plt.imshow(observation)
            # plt.title("Initial Frame from Breakout")
            # plt.axis('off')  # 关闭坐标轴
            # plt.pause(0.5)

        print(f"Episode {i + 1} finished after {step_counter} steps.")


if __name__=="__main__":
    main()