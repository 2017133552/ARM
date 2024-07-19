import gym
import torch

from wrapplib.Network import VNet,CQNet
from wrapplib.wrapper import wrap_atari
from wrapplib.discreta_arm import DiscreteARMPoicy

def test_model():
    vmodel_path="model_tase/v_model194.pth"
    ccq_model_path="model_tase/ccq_model194.pth"
    history_len=4
    frame_skip=4
    prepro_pong_mask=True
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env=gym.make("PongNoFrameskip-v4",render_mode="human")
    action_dim=env.action_space.n

    v_func=VNet(history_len,1,device=device)
    ccq_func=CQNet(history_len,action_dim,device=device)
    v_func.load_state_dict(torch.load(vmodel_path))
    ccq_func.load_state_dict(torch.load(ccq_model_path))


    env=wrap_atari(env,frame_skip,prepro_pong_mask=prepro_pong_mask,history_len=history_len)
    observation=env.reset()
    policy=DiscreteARMPoicy(action_dim,ccq_func,v_func)
    R=0
    while True:
        action_indexl,action_probl=policy(observation)
        action_index=action_indexl[0]
        action_prob=float(action_probl[0])

        observation,reward,terminal,truncated,info=env.step(action_index)
        R+=reward
        if terminal:
            break


    print("the score/Return of  teat episode : ",R)









if __name__=="__main__":
    test_model()
