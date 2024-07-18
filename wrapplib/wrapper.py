import gym
import numpy as np
from collections import  deque
import torchvision
class SkipFrameEnv(gym.Wrapper):
    def __init__(self,env=None,skip=2):
        '''

        :param env:  The environment to wrap
        :param skip: Return only every `skip`-th frame"
        '''
        super(SkipFrameEnv,self).__init__(env)
        self._skip=skip
        ###注意父类的初始化中有语句：self.env = env

    def step(self,action):# _ 暗示这是一个内部或“私有”方法，意味着它应该只在类的内部被调用，而不是从类外部调用。
        total_reward=0.0
        done=None
        for i in range(self._skip):#跳过指定帧数
            observation,reward,terminated,truncated,info=self.env.step(action)
            total_reward+=reward
            if terminated:
                break
        return observation,total_reward,terminated,truncated,info

    def reset(self):
        """
        除过去的帧缓存并初始化为内部环境的第一个观察。这通常意味着将环境重置到其初始状态。
        清除过去的帧缓存并初始化为内部环境的第一个观察。这通常意味着将环境重置到其初始状态。清除帧缓存：如果环境使用了帧缓存（例如，为了处理视频帧或其他连续的观察数据），重置时清除缓存是常见的做法，以确保新的episode不会受到旧状态数据的影响。
        重置环境后获取初始观察是启动新episode的关键步骤，它为代理提供了决策的起点。"""
        observation=self.env.reset()

        return observation

class PreprocAtariPongMask(gym.ObservationWrapper):
    '''
    ObservationWrapper类没有定义自己的 __init__ 方法,因此自动使用了他的父类Wrapper 的 __init__ 方法；
    这说明了如果你不需要在子类中添加任何特殊的初始化逻辑，就不必编写自己的 __init__ 方法或调用 super()。
    '''

    def __init__(self,env=None):
        super(PreprocAtariPongMask,self).__init__(env)# Ensures that parent's __init__ is called
        self.observation_space=gym.spaces.Box(low=0,high=255,shape=(210,160,3))
        #gym.spaces.Box 是用来定义一个连续空间的类，它表示在给定范围内的所有 n 维数组的空间。这种空间类型常用于定义观测空间（observations space）和动作空间（action space），特别是在涉及视觉输入或多维连续动作的环境中。
    def observation(self,obs):
        if obs.size==210*160*3:#该属性返回向量中所有标量的个数
            img=np.reshape(obs,(210,160,3))
        else:
            assert False,"Unknown resolution[未知的分辨率]: {}".format(obs.resize)
        img[34:194,55:105,0]=144
        img[34:194,55:105,1]=72
        img[34:194,55:105,2]=17#更改以上区域的像素值，造成遮盖，部分可观
        return img

class LazyFrames(object):
    '''
     LazyFrames 的目的是为了在内存使用上进行优化，
     特别是对于需要存储大量历史观测数据的场景（如DQN算法中的重放缓冲区）。
     LazyFrames 是一种用于优化内存使用的工具，它通过延迟合并和转换操作，
     直到这些操作真正需要执行时才进行，从而减少了不必要的数据复制和存储。
     这在处理大规模数据集（如视频帧）时尤其重要，可以显著降低内存占用，并提高数据处理效率。
     It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
     这个类特别适用于机器学习和计算机视觉中的应用，其中经常需要处理和分析大量的图像数据。
    '''
    def __init__(self,stack_frames):

        self._frames=stack_frames#_frames 属性被用来存储这些帧的引用。由于 Python 中对象的引用机制，这种方式允许 LazyFrames 对象在多个地方被引用时共享相同的帧数据，而不需要在内存中重复存储相同的数据。

    #This object should only be converted to numpy array before being passed to the model.
    def __array__(self,dtype=None):#当你尝试将一个 LazyFrames 对象传递到需要 NumPy 数组的函数或模型中时，这个方法会被自动调用。
        out=np.concatenate(self._frames,axis=0)# 将存储在 _frames 中的帧沿着第一个维度（axis=0）拼接起来。这意味着假设每个帧都是一个独立的数组，所有这些帧将被垂直（在第一个轴）堆叠成一个单一的数组
        if dtype is not None:
            out=out.astype(dtype)#当你尝试将一个 LazyFrames 对象传递到需要 NumPy 数组的函数或模型中时，这个方法会被自动调用。

        return out

class Preprecess(object):
    def __init__(self):
        self.pre_torchvision = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                      torchvision.transforms.Grayscale(),
                                                      torchvision.transforms.Resize(
                                                          (84, 84)),
                                                      torchvision.transforms.ToTensor()])
    # remove channels and convert to numpy
    def __call__(self,img):
        return self.pre_torchvision(img).numpy()
class StackFrameEnv(gym.Wrapper):
    def __init__(self,env=None,history_len=1):
        super(StackFrameEnv,self).__init__(env)
        """
        堆叠过去的history_len帧
        返回一个延迟数组，其存储效率更高
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        self.preprogress=Preprecess()
        self.history_len=history_len
        self.frames=deque((),maxlen=self.history_len)#使用了 collections 模块中的 deque 类来初始化一个双端队列（deque），具有固定的最大长度 k。这个双端队列被用来存储有限数量的元素（在这个上下文中可能是图像帧或其他数据），并且自动管理其大小，以确保不超过设定的最大长度。
        #[]：这是传递给 deque 的初始迭代器。在这个例子中，deque 是用一个空列表初始化的，意味着它一开始是空的。
        #maxlen=k：这个参数设置 deque 的最大长度为 k。这意味着队列可以存储最多 k 个元素。当新元素被添加到已满的 deque 中时，另一端的元素将自动被移除，保持元素总数不超过 k。
        '''
        空元组 () 和空列表 [] 都是有效的空的可迭代对象。在初始化 deque 或其他需要可迭代对象的数据结构时，它们都可以用来表示没有任何初始元素的情况。选择使用空列表还是空元组通常取决于个人偏好或特定的编码风格，
        因为在这种情况下，它们的功能完全相同。这种方式为以后的数据操作提供了一个干净的起点，确保了队列在任何元素被添加之前是空的。
        '''
        shp=env.observation_space.shape

        self.observation_space=gym.spaces.Box(low=0,high=255,shape=(shp[0]*history_len,shp[1],shp[2]))

    def reset(self):
        observation,info=self.env.reset()

        observation=self.preprogress(observation)
        for i in range(self.history_len):
            self.frames.append(observation)
        return self._get_ob()

    def step(self,action):
        observation,reward,terminated,truncated,info=self.env.step(action)
        observation=self.preprogress(observation)
        self.frames.append(observation)
        return self._get_ob(),reward,terminated,truncated,info

    def _get_ob(self):
        assert len(self.frames)==self.history_len,"unequal the frames {} to the requested_len{}".format(len(self.frames),self.history_len)
        return np.array(LazyFrames(list(self.frames)))

class ClippedRewardsWrapper(gym.RewardWrapper):
    def reward(self,reward):
        """Change all the positive rewards to 1, negative to -1 and keep zero."""
        return np.sign(reward)
def wrap_atari(env,frame_skip=1,prepro_pong_mask=False,history_len=1):
    '''
    :param env: 是一个环境实例
    :param frame_skip:跳帧的帧数，可用于加快游戏进程
    :param prepro_pong_mask:是否对环境画面进行遮盖，掩膜
    :return:
    '''
    assert "NoFrameskip" in env.spec.id#这个 assert 语句是为了确保环境确实是 'NoFrameskip' 版本;，研究者可能希望确保他们使用的环境是没有帧跳过的版本。这样可以保证模型在做出决策时能观察到环境的每一帧，从而使得学习过程更透明，也有助于调查如何处理每个时刻的视觉信息。
    #env.spec 是环境的 "specification"（规格说明），包含有关环境的元数据。
    #env.spec.id 是环境的唯一标识符，通常是一个描述性的字符串，如 'CartPole-v1' 或 'PongNoFrameskip-v4'。


    if frame_skip>1:
        env=SkipFrameEnv(env,skip=frame_skip)

    if prepro_pong_mask:
        env=PreprocAtariPongMask(env)

    if history_len>1:
        env=StackFrameEnv(env,history_len)
    env=ClippedRewardsWrapper(env)

    return env


