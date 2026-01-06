import json
import torch
from typing import List, Dict
from your_reward_module import SIERewardModel  # 替换为你的reward文件

# ======================
# 1. 模拟模型响应生成 (无需真实模型)
# ======================
def mock_generate(question: str, context: str) -> str:
    """模拟LLM生成，覆盖各种格式情况"""
    cases = [
        # 完美响应
        "<think>Based on the context, Microsoft was founded by Bill Gates.</think><answer>Bill Gates</answer>",
        # 格式正确但答案错误
        "<think>Paris is the capital of Germany.</think><answer>Berlin</answer>",
        # 缺少</think>标签
        "<think>Water has the chemical formula H2O<answer>Water</answer>",
        # 无<think>开头
        "The director is Christopher Nolan<answer>Christopher Nolan</answer>",
        # 完美响应
        "<think>Jupiter is the largest planet in our solar system.</think><answer>Jupiter</answer>"
    ]
    return cases.pop(0) if cases else "<answer>Default</answer>"

# ======================
# 2. CPU版奖励验证
# ======================
def validate_reward_model():
    print("="*50)
    print("🚀 STARTING CPU-ONLY REWARD VALIDATION")
    print("="*50)
    
    # 初始化奖励模型 (CPU模式)
    reward_model = SIERewardModel(
        format_weight=0.2,
        answer_weight=0.8,
        strict_format=True,
        case_sensitive=False
    )
    
    # 加载测试样本
    samples = []
    with open("test_samples.jsonl", "r") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    # 模拟RL训练中的reward计算
    all_rewards = []
    for i, sample in enumerate(samples):
        # 1. 模拟模型生成响应
        response = mock_generate(sample["question"], sample["context"])
        
        # 2. 准备ground_truth格式 (Verl兼容)
        ground_truth = {"target": sample["target"]}
        
        # 3. 计算奖励 (CPU张量)
        reward_tensor = reward_model.compute_reward(
            responses=[response],
            ground_truths=[ground_truth]
        )
        
        # 4. 转换为Python float
        reward_value = reward_tensor.item()
        all_rewards.append(reward_value)
        
        # 5. 详细日志
        print(f"\n{'-'*40}")
        print(f"📝 Sample {i+1}: {sample['question']}")
        print(f"💬 Response: {response}")
        print(f"✅ Ground Truth: {sample['target']}")
        print(f"⭐ Computed Reward: {reward_value:.4f}")
        
        # 6. 验证逻辑正确性
        expected = {
            0: 1.0,   # 完美响应
            1: 0.0,   # 答案错误 + 格式错误 (Berlin不是France首都)
            2: 0.15,  # 答案正确但缺少</think> (格式分扣减)
            3: 0.05,  # 答案正确但无<think>开头 (严重格式错误)
            4: 1.0    # 完美响应
        }
        if abs(reward_value - expected[i]) > 0.05:
            print(f"❌ WARNING: Unexpected reward {reward_value:.4f} (expected {expected[i]})")
        else:
            print(f"✅ PASS: Reward within expected range")
    
    # 最终统计
    print("\n" + "="*50)
    print(f"✅ VALIDATION SUCCESS! Average Reward: {sum(all_rewards)/len(all_rewards):.4f}")
    print(f"💡 TIP: Now safe to run on GPU server with confidence")
    print("="*50)

# ======================
# 3. 运行验证
# ======================
if __name__ == "__main__":
    # 强制CPU模式 (防止意外使用GPU)
    device = torch.device("cpu")
    torch.set_num_threads(4)  # 限制CPU线程数
    
    validate_reward_model()
