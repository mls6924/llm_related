from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# 构建dataset
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer = tokenizer
        
        self.final_prompts = []
        
        for prompt in prompts:
            # Instruct 模型（指令微调模型），目标是让模型“听得懂人话、会按指令办事”，而不是只会胡言乱语或续写文本。
            # 让模型不要写文章，而是像对话一样回答。
            if apply_chat_template:  # instruct模型都需要apply_chat_template=True
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
                # add_generation_prompt=True会让它加上了 assistant\n 或 <|im_start|>assistant，告诉模型：“现在轮到你生成了”。
            else:
                prompt = self.tokenizer.bos_token + prompt
                
            self.final_prompts.append(prompt)
        
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.final_prompts[index]

# 价值（评论家）模型，用于预测每一步（生成token）的动作产生的收益，使用演员模型进行初始化，并外加一个回归头，输出shape为：(batch_size, seq_len， 1)
class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask, num_actions):
        # Step 1: 用 base_model 编码整个输入序列
        hidden_state = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        # shape: [batch_size, seq_len, hidden_size]

        # Step 2: 用 value_head 把每个 token 映射成 1 个 value
        value_model_output = self.value_head(hidden_state)
        # shape: [batch_size, seq_len, 1]

        # Step 3: 去掉最后一维（1 → 标量） x
        values = value_model_output.squeeze(-1)
        # values = value_model_output.squeeze(-1) 
        
        # Step 4: 只取最后 num_actions 个 token 的 values   回答部分的values
        values = values[:, -num_actions:]
        # shape: [batch_size, num_actions]
        return values



def compute_policy_loss(log_probs, old_log_probs, advantages, action_mask=None, clip_eps=0.2):
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2)
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


def compute_value_loss(values, old_values, returns, action_mask=None, clip_eps: float = None):
    if clip_eps is not None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        loss = (values - returns) ** 2
        
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []
    
    def append(self, experiences):
        batch = [{} for _ in range(len(experiences))]
        keys = (
        "seqs",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "num_actions"
    )
        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value
          
        self.buffer.extend(batch)
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer)-self.limit:]
        
    def get_batches(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        self.buffer = []
        
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]
    

@dataclass
class Samples:
    seqs: torch.Tensor                  # 完整序列（提示+生成内容）
    attention_mask: Optional[torch.LongTensor]  # 注意力掩码
    action_mask: Optional[torch.BoolTensor]     # 动作掩码
    num_actions: Union[int, torch.Tensor]       # 最大动作数
    packed_seq_lens: Optional[torch.Tensor]     # 序列打包长度（高级优化）
    response_length: torch.Tensor        # 每个样本的实际响应长度
    total_length: torch.Tensor           # 每个样本的总有效长度（包括提示和生成部分）

@dataclass
class Experience:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None

def compute_approx_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
):

    log_ratio = log_probs.float() - ref_log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio

# A(t) = R(t) + gam*V(t+1) - V(t)
# gae:A(t) = R(t) + gam*V(t+1) - V(t) + gam*lam*A(t+1)
# 最后一个时刻的未来优势和未来收益为0：A(T+1) = 0, V(T+1) = 0,  则A(T) = R(T) - V(T), 得出A(T)
# A(T-1) = R(T-1) + gam*V(T) - V(T-1) + gam*lam*A(T) 知道A(T)可计算A(T-1) 依次类推
# returns(t) = A(t) + V(t) = = R(t) + gam * (V(t+1) + lam * A(t+1))
def get_advantages_and_returns(
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float):
    
    lastgaelam = 0
    advantages_reversed = []
    response_length = rewards.size(1)
    
    if action_mask is not None:
        values = action_mask * values
        rewards = action_mask * rewards

    for t in reversed(range(response_length)):
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages.detach(), returns

def generate_samples(prompts, model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size):
    samples_list = []
    model.eval()
    all_prompts = sum([[prompt]*n_samples_per_prompt for prompt in prompts], [])  # 8个菜，每个菜做两次，2 * 8为samples总数
    for i in range(0, len(all_prompts), micro_rollout_batch_size):  # 两口锅
        prompts = all_prompts[i:i+micro_rollout_batch_size]  # 一次做两份
        """
        padding='max_length', max_length=max_length，统一填充到固定长度max_length（不是填充到批次最长序列）。所有人用同样尺寸的碗，但也并非向最大分量的菜品看齐。
        padding参数值可用'max_length' 或 'longest'
        # 为什么用 'max_length'？PPO 训练需要固定批次输入，不能因为某位客人点菜太长导致批次大小不一致。

        truncation=True  超过最大长度的截断，超长提示词不报错
        return_tensors='pt'   返回 PyTorch 张量（torch.tensor）  你的模型是 PyTorch 写的，必须用 pt
        # 我的micro_rollout_batch_size=2，max_length=256，所以inputs的shape为2x256
        """
        inputs = actor_tokenizer(prompts, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt') 

        input_ids = inputs['input_ids']  # 只有两个核心键。input_ids：真实饭粒的数字编号，attention_mask：标记哪些饭粒是真实内容

        """
        model.generate() 的作用就是“让模型生成回答”
        **inputs.to(device) 不是 Python 语法的特殊用法，而是 Hugging Face 的 BatchEncoding 特有设计。  
        **符号可以理解为“拆快递”。model.generate 需要的是 键值对，不是变量。

        max_new_tokens 是生成回答中新 token 的最大数量，不包括输入的提示词（prompt）
        """
        seqs = model.generate(**inputs.to(device), 
                            max_new_tokens = max_new_tokens, 
                            eos_token_id = eos_token_id, 
                            pad_token_id = pad_token_id)
        
        """
        “把模型生成的菜谱（seqs）整理成标准格式：
        1. 确保总长度 = 最大提示词长度 + 最大回答长度  256 + 50 = 306
        2. 标记哪些是真实菜（attention_mask）
        3. 提取厨师做的菜（ans）
        4. 标记哪些菜是有效生成的（action_mask）”
        """
        # seqs.size(0)为处理份数，seqs.size(1)为填充长度，理想情况 = max_new_tokens + max_length = 256 + 50 = 306
        # 这段代码不是冗余的，而是处理现实场景中序列长度不确定性的必要保障，确保了pipeline的鲁棒性。在实际深度学习应用中，这种防御性编程是非常重要的实践。
        if seqs.size(1) >= max_new_tokens + max_length:
            seqs = seqs[:, :max_new_tokens + max_length]  # 超过了就截断
        else:
            """
            torch.cat 是 PyTorch 中的核心函数，用于沿着指定维度连接多个张量。在这里，它用于将原始序列和填充部分连接起来，确保所有序列达到统一的长度。
            torch.cat(tensors, dim) 接收两个参数：
                tensors: 一个张量列表，这些张量必须在除连接维度外的所有维度上形状一致
                dim: 沿着哪个维度进行连接（0表示行方向，1表示列方向，以此类推）
            """
            seqs = torch.cat(
                        [seqs, torch.full((seqs.size(0), max_new_tokens + max_length - seqs.size(1)),   # 填充部分张量的shape
                                    fill_value=pad_token_id, 
                                    device=seqs.device)],
                        dim=1)
        
        
        """
        # 1.注意力掩码 (attention_mask)
        # 作用：创建一个二进制掩码，标识哪些token是真实内容，哪些是填充部分
        # 工作原理：
            # seqs.ne(pad_token_id)：对每个位置检查是否不等于填充值，返回布尔张量
            # .to(dtype=torch.long)：将布尔值转换为0/1整数 (True→1, False→0)
        # 结果形状：与seqs相同，例如[2, 306]
        # 示例：
            # seqs:           [12, 34, 56, 0, 0, 0]  (0是pad_token_id)
            # attention_mask: [1,  1,  1, 0, 0, 0]
        # 用途：在注意力计算中，模型会忽略掩码为0的位置，确保填充部分不影响模型内部表示
        """
        attention_mask = (seqs.ne(pad_token_id)).to(dtype=torch.long)

        # 分离回答部分 (ans)
        ans = seqs[:, input_ids.size(1):]

        """
        3. 动作掩码 (action_mask)
        作用：创建掩码标识哪些生成的token是有效的"动作"，需要在策略梯度计算中考虑
        工作原理：
            ans.ne(eos_token_id)：标记 不是 结束符的位置
            ans.ne(pad_token_id)：标记 不是 填充符的位置
            &：按位与，只保留同时满足两个条件的位置
        转换为长整型(0/1)
        结果形状：与ans相同，例如[2, 50]
        示例：
            ans:         [123, 456, 789, eos_id, pad_id, pad_id]
            action_mask: [1,   1,   1,    0,      0,      0  ]
        用途：在强化学习中，只有有效的token（既不是EOS也不是PAD）才被视为模型做出的"决策"，需要计算策略梯度
        """
        action_mask = (ans.ne(eos_token_id) & ans.ne(pad_token_id)).to(dtype=torch.long)
       

        samples = Samples(
            seqs=seqs,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),
            packed_seq_lens=None,  # packed_seq_lens: 设为None，表示没有使用序列打包优化。在高性能实现中，这个字段可以存储变长序列的实际长度，用于内存优化
            # .float()转为浮点数，sum(dim=-1) 沿着最后一个维度（序列长度维度）求和
            response_length=action_mask.float().sum(dim=-1),  # 有效回答动作数
            total_length=attention_mask.float().sum(dim=-1),  # 总有效数
        )
        samples_list.append(samples)

    return samples_list


def compute_rewards(kl, r, action_mask, kl_ctl, clip_reward_value):#

        kl_divergence_estimate = -kl_ctl * kl
        rewards = kl_divergence_estimate

        ends = action_mask.sum(1) + 1
        
        if not isinstance(clip_reward_value, torch.Tensor):
            clip_reward_value = torch.tensor(clip_reward_value).to(r.device)
    
        reward_clip = torch.clamp(r, -clip_reward_value,
                                  clip_reward_value)
        batch_size = r.size(0)
        for j in range(batch_size):
            rewards[j, :ends[j]][-1] += reward_clip[j, 0]

        return rewards

def generate_experiences(samples_list):
    # 这段代码将所有模型切换到评估/推理模式，是强化学习经验收集阶段的关键准备步骤。
    # 训练模式下(train())，Dropout层会随机丢弃神经元(例如50%)，BatchNorm层仅使用当前batch统计量；
    # 评估模式下(eval())，Dropout层完全禁用，所有神经元参与计算，BatchNorm层使用全局统计量（训练时累积的均值/方差）；
    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()

    experiences = []
    
    for samples in samples_list:
        seqs = samples.seqs
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        # torch.no_grad()意为，这个上下文管理器禁用梯度计算，减少内存消耗并加速计算。
        # 由于我们只是在收集经验，而不是更新模型，因此不需要计算梯度。
        with torch.no_grad():
            # 计算策略模型输出token的概率
            # 这行代码是强化学习与语言模型结合的核心操作，执行策略模型的前向传播计算。
            # 等价于：output = actor_model.forward(seqs, attention_mask=attention_mask)
            # {
                # 'logits': tensor([batch_size, seq_len, vocab_size]),  # 核心输出
                # 'past_key_values': [...],  # 可选，用于加速生成
                # 'hidden_states': [...],    # 可选，各层隐藏状态
                # 'attentions': [...]        # 可选，注意力权重
            # }
            output = actor_model(seqs, attention_mask=attention_mask)
            """
            Q：所以前向传播就是回答的过程，输入输出就是回答问题？可模型回答不是生成样本时通过seqs = model.generate(...)完成了吗？为什么又回答一次？
            A：generate是在创作一整篇文章，这个过程对应多次前向传播。总前向次数：生成N个新token需要N次前向传播。
            而output = actor_model(seqs, attention_mask=attention_mask)这一次前向传播不是为了预测新词，
            而是让模型"回忆"它在生成这些序列时的决策过程，给出每个token的概率。这就像让作家回看自己的手稿，分析当时为什么选择每个词。

            Q：这是一次特殊的前向传播吗？因为它评估的是各个词的概率，和生成过程有很大不同。
            A：在 生成模式 下，如果我们一个词一个词地生成，确实需要多次前向传播（每生成一个新词就需要一次前向传播）。
            但在 评估模式 下（如你的代码所示），模型只需要一次前向传播就能处理整个序列，评估所有位置的预测概率。
            """

            # logits张量形状会是 [2, 306, vocab_size]，包含了所有位置的预测分数
            logits = output.logits
            # 去掉最后一个位置的预测（[:, :-1, :]），因为最后一个token后没有下一个token
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # 对前面所有位置的预测应用log_softmax，转换为对数概率
            log_probs_labels = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            """
            unsqueeze(-1) 函数  在张量的末尾插入一个维度（大小为1的维度）
            gather 函数  沿着指定维度，根据索引收集张量中的值

            一个理解上面句子的具体实例：
            # 假设：
            # - 词汇表大小 = 5 (token IDs: 0,1,2,3,4)
            # - batch_size = 1
            # - 序列: [0, 1, 3, 2, 4] (BOS, "你好", "天气", "今天", "EOS")

            # 1. 假设log_probs的形状是 [1, 4, 5]
            #    (batch_size=1, 4个位置需要预测, 5个词汇)
            log_probs = torch.tensor([
                [
                    [-1.0, -0.1, -2.0, -3.0, -4.0],  # 位置0: "BOS"后预测"你好"(ID=1)的概率最高
                    [-3.0, -2.0, -0.5, -1.0, -4.0],  # 位置1: "你好"后预测"天气"(ID=2)的概率最高
                    [-2.0, -3.0, -4.0, -0.3, -1.0],  # 位置2: "天气"后预测"今天"(ID=3)的概率最高
                    [-1.0, -2.0, -3.0, -4.0, -0.2]   # 位置3: "今天"后预测"EOS"(ID=4)的概率最高
                ]
            ])

            # 2. seqs[:, 1:] 是原始序列去掉第一个token
            #    原始序列: [0, 1, 3, 2, 4]
            #    seqs[:, 1:]: [1, 3, 2, 4]
            seqs_next = torch.tensor([[1, 3, 2, 4]])  # 形状 [1, 4]

            # 3. unsqueeze(-1) 使其形状变为 [1, 4, 1]
            index_tensor = seqs_next.unsqueeze(-1)  # 现在形状是 [1, 4, 1]
            print(index_tensor)
            # tensor([[[1],
            #          [3],
            #          [2],
            #          [4]]])

            # 4. gather操作: 沿着最后一个维度(dim=-1)收集指定索引位置的值
            #    对于每个位置，从log_probs中提取对应实际token的对数概率
            log_probs_labels = log_probs.gather(dim=-1, index=index_tensor)
            print(log_probs_labels)
            # tensor([[[-0.1],  # 位置0预测ID=1("你好")的对数概率
            #          [-0.3],  # 位置1预测ID=3("今天"?)的对数概率
            #          [-4.0],  # 位置2预测ID=2("天气"?)的对数概率 
            #          [-0.2]]]) # 位置3预测ID=4("EOS")的对数概率
            """

            # 取出回答部分（长度num_actions）的对数概率
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]  # 1. squeeze(-1) 函数  移除张量中指定位置（维度）大小为1的维度

            # 计算参考模型输出token的概率
            ref_output = ref_model(seqs, attention_mask=attention_mask)
            ref_logits = ref_output.logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_log_probs_labels = ref_log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]
            # 计算价值
            # 假设生成回答是：“阳光 明媚 EOS”，最终输出可能为[[2.3, 2.7, 3.1]]
            value = critic_model.forward(seqs, attention_mask, num_actions).to(device)
            # 转换成文本
            seq_texts = actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            # 计算奖励模型的奖励值
            reward_model_inputs = reward_tokenizer(seq_texts, return_tensors="pt", padding=True)
            r = reward_model(**reward_model_inputs.to(device)).logits # 奖励模型的输出，相当于生成最后一个token的奖励（结果奖励模型）
            # 计算kl散度
            kl = compute_approx_kl(
                    action_log_probs,
                    ref_action_log_probs,
                    action_mask=action_mask).to(device)
            # 计算实际奖励
            rewards = compute_rewards(kl, r, action_mask, kl_ctl=0.1, clip_reward_value=0.2)
            # 计算优势和回报
            advantages, returns = get_advantages_and_returns(value, rewards, action_mask, gamma=0.1, lambd=0.2)
        # actor_model.train()
        # critic_model.train()

        experiences.append(Experience(seqs,
                    action_log_probs.detach(),
                    value.detach(),
                    returns.detach(),
                    advantages.detach(),
                    attention_mask,
                    action_mask,
                    r.detach(),
                    samples.response_length,
                    samples.total_length,
                    num_actions,
                    kl.detach(),
        ))

    return experiences

@dataclass
class BufferItem:

    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]

def collate_fn(batch):

    seqs = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []
    
    for x in batch:
        seqs.append(x['seqs'])
        action_log_probs.append(x['action_log_probs'])
        values.append(x['values'])
        returns.append(x['returns'])
        advantages.append(x['advantages'])
        attention_mask.append(x['attention_mask'])
        action_mask.append(x['action_mask'])

    seqs = torch.cat(seqs, dim=0)
    action_log_probs = torch.cat(action_log_probs, dim=0)
    values = torch.cat(values, dim=0)
    returns = torch.cat(returns, dim=0)
    advantages = torch.cat(advantages, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    action_mask = torch.cat(action_mask, dim=0)
    
    return BufferItem(seqs, action_log_probs, values, returns, advantages, attention_mask, action_mask, action_mask.size(1))
    
def train_step(experience, steps):
    
    actor_model.train()
    optimizer_actor.zero_grad()

    
    sequences = experience.seqs
    old_action_log_probs = experience.action_log_probs
    advantages = experience.advantages
    num_actions = experience.num_actions
    attention_mask = experience.attention_mask
    action_mask = experience.action_mask
    old_values = experience.values
    returns = experience.returns
    
    logits = actor_model(
            sequences,
            attention_mask=attention_mask).logits
    
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=sequences[:, 1:].unsqueeze(-1))
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
  

    
    policy_loss = compute_policy_loss(action_log_probs, old_action_log_probs, advantages,action_mask=action_mask)
    policy_loss.backward()
    optimizer_actor.step()  
    writer.add_scalar("policy_loss", policy_loss.item(), steps)
    
    critic_model.train()
    optimizer_critic.zero_grad()
    values = critic_model.forward(sequences, attention_mask, num_actions)
    value_loss = compute_value_loss(values, old_values, returns, action_mask)
    value_loss.backward()
    optimizer_critic.step()
    writer.add_scalar("value_loss", value_loss.item(), steps)
    print(f"step: {steps}  policy_loss: {policy_loss.item():.4f}  value_loss: {value_loss.item():.4f}")
    

def train():
    # 初始化经验池
    buffer = ExperienceBuffer(limit=100)  #  每轮（episode）的经验只用于当轮训练  8 × 2 = 16 条 << 100
    steps = 0
    for episode in range(episodes):
        for rand_prompts in prompts_dataloader:  # 从处理好的prompts池子中随机抽取8个。
            # 生成样本（获取模型推理结果）
            samples = generate_samples(rand_prompts, actor_model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size)
            # 生成经验（获取优势、奖励、回报等）
            experiences = generate_experiences(samples)
            buffer.append(experiences)
            dataloader = DataLoader(buffer, batch_size=micro_train_batch_size, shuffle=True, collate_fn=collate_fn)
            torch.cuda.empty_cache()
            for epoch in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1
            
            buffer.clear()
        
            torch.cuda.empty_cache()
            

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """
    你的厨房规则（参数）：
        episodes = 3 → 总共营业 3 天
        rollout_batch_size = 8 → 每天要服务 8 位客人
        micro_rollout_batch_size = 2 → 但灶台只有 2 口锅，一次最多炒 2 份
        n_samples_per_prompt = 2 → 每位客人要点 2 碗饭（探索不同做法）
        max_epochs = 5 → 每天收工后，你要复盘 5 轮

    对于复盘阶段，
    等 8 位客人都吃完、所有 16 碗饭（8×2）的评分都拿到后再复盘。
    你把今天所有的记录摊开：
        每碗饭的做法（token sequence）
        评委总分（reward）
        副厨当时的预估分（values）
    然后你和副厨一起 复盘 5 轮（max_epochs=5）：
        第1轮复盘：用全部 16 条经验，更新你的炒饭手法（Actor） + 副厨的打分标准（Critic）
        第2轮复盘：再用同样的 16 条经验，再更新一次
        ...
        第5轮复盘：最后一次微调
    """

    # 一共迭代多少轮
    episodes = 3
    # 生成一次经验，训练的轮数
    max_epochs = 5
    # 一次从提示词数据集中取多少条数据用于生成经验
    rollout_batch_size = 8  # （一轮ppo服务8个客人）
    # 一次取多少条数据生成经验（生成经验需要多个模型推理，对显存要求高）
    micro_rollout_batch_size = 2  # （但我只有两口锅）
    # 一个提示词生成多少个样本
    n_samples_per_prompt = 2
    # 生成的最大长度，相当于最大动作数，数值越大，模型探索的可能性越多
    max_new_tokens = 50
    # 最大长度
    max_length = 256
    # 实际训练的batch_size大小，一次取多少条数据用于更新参数
    micro_train_batch_size = 2
    # 记录日志
    writer = SummaryWriter('./runs')

    # .to(device) 是什么意思？这是 PyTorch 的核心操作，作用是：把模型（或张量）移动到指定设备上（CPU / GPU）
    # 模型默认加载在 CPU，如果你的 input tensors 在 GPU（比如 input_ids.to('cuda')），但模型还在 CPU → 会报错：device mismatch!

    # 策略模型
    actor_model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct').to(device)
    # 参考模型
    ref_model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct').to(device)
    
    # 好奖励模型的特征：
    # 1. 在人类偏好数据上训练过   2. 输出是“标量打分”，AutoModelForSequenceClassification(num_labels=1)，输出一个 float 分数（越高越好）
    # 3. 支持长文本，能处理 prompt + response 的总长度（DeBERTa-v3 支持 512 tokens，够用）
    # 4. 与你的任务对齐  如果你训练的是中文助手，最好用中文奖励模型；你是英文/通用任务，DeBERTa-v3 很合适
    # 奖励模型
    reward_model = AutoModelForSequenceClassification.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2').to(device)
    actor_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct')
    reward_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')
    # 价值模型
    critic_model = Critic(actor_model.base_model).to(device)
    """
    关于价值模型critic和奖励模型的reward的区别：
    reward是针对整条轨迹做出打分
    critic给的是每个token的val，至于我如何拿这些vals对齐整个轨迹的得分、好与reward得分相比，又涉及到另外的方法，比方说累加或者GAE

    过程：
    选手Actor回答，裁判reward打分（分数比赛时不展示，比赛后教练和选手才知道得分），
    教练critic旁边观察并针对Actor的每一个动作步骤内心评估。一轮（rollout_batch）比赛结束后，教练和选手都对自己的表现复盘。
    """
    
    # 初始化优化器（锅，跟我关心的食谱ppo算法无关）
    optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=0.00005)
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.00005)
    
    # 填充方式为左填充
    # （我想让大模型一次回答多个问题，但是神经网络要求prompts长度一样，那就需要对短的prompt填充）
    actor_tokenizer.padding_side = 'left'
    eos_token_id = actor_tokenizer.eos_token_id  # 出现它，或者模型回答到达最大长度，就代表模型说完了
    pad_token_id = actor_tokenizer.pad_token_id  # 填充时出现的占位符，（有时有些模型没有专门的 [PAD] token，而是用 eos_token 兼做 pad_token。
    prompt_list = [  # 好比点菜单，最好种类多、数量多。并非都会用到，而是每轮训练随机抽取8个。
        '请问1+1等于多少？',
        'PowerShell，如何知道BIOS中的虚拟化是否已禁用',
        '为什么人们喜欢在水族馆里游泳，而不是在游泳池里？',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '为什么所有的镜子都是矩形的？',
        '我们在受感染的植物根部可以找到哪一种，臭氧还是金子？'
    ]
    prompts_dataset = PromptDataset(prompt_list, actor_tokenizer, apply_chat_template=True)
    prompts_dataloader = DataLoader(prompts_dataset, batch_size=rollout_batch_size, shuffle=True)
    # Dataset 和 DataLoader 是 PyTorch 中处理数据的两个核心组件
    
    # PromptDataset：把你的原始提示列表（如 ["你好", "1+1=？"]）变成模型能吃的“预制菜”（比如 token IDs、attention mask 等）。

    # DataLoader：负责按批次（batch）把“预制菜”端上桌（送进 GPU），还能打乱顺序、多进程加载等。
    # 作用有：批量取数据、自动填充（padding）、打乱顺序（shuffle=True）、（可选）多进程加载：加速数据准备（虽然你没开
   
    train()
    

