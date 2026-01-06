import torch
import random
import re
from verl.reward import BaseRewardModel
from typing import List, Dict, Union, Tuple

class SIERewardModel(BaseRewardModel):
    """
    结构化推理环境(SIE)专用奖励模型
    结合格式奖励(Format Reward)和答案奖励(Answer Reward)
    """
    
    def __init__(self, 
                 format_weight: float = 0.2,
                 answer_weight: float = 0.8,
                 case_sensitive: bool = False,
                 strict_format: bool = True):
        """
        Args:
            format_weight: 格式奖励的最大权重 (0.0~1.0)
            answer_weight: 答案奖励的最大权重 (0.0~1.0)
            case_sensitive: 答案匹配是否区分大小写
            strict_format: 是否严格校验格式结构
        """
        super().__init__()
        self.format_weight = format_weight
        self.answer_weight = answer_weight
        self.case_sensitive = case_sensitive
        self.strict_format = strict_format
        
        # 预编译正则表达式提高效率
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        self.think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        
    def extract_solution(self, solution_str: str) -> Union[str, None]:
        """从响应中提取<answer>标签内的内容"""
        matches = self.answer_pattern.findall(solution_str)
        if not matches:
            return None
        # 取第一个匹配的answer内容并清理
        answer = matches[0].strip()
        # 移除内部可能存在的标签
        answer = re.sub(r'<[^>]+>', '', answer)
        return answer if answer else None
    
    def count_answer_tags(self, text: str) -> Tuple[int, int]:
        """统计<answer>和</answer>标签数量"""
        open_tags = text.count('<answer>')
        close_tags = text.count('</answer>')
        return open_tags, close_tags
    
    def em_check(self, prediction: str, ground_truth: str) -> bool:
        """精确匹配检查，支持大小写不敏感"""
        if prediction is None or ground_truth is None:
            return False
        
        # 标准化处理
        pred = prediction.strip()
        gt = ground_truth.strip()
        
        if not self.case_sensitive:
            pred = pred.lower()
            gt = gt.lower()
        
        # 基础匹配
        if pred == gt:
            return True
        
        # 高级匹配：处理列表/多答案情况
        if ';' in gt or ',' in gt:
            gt_items = [item.strip() for item in re.split(r'[;,]', gt) if item.strip()]
            pred_items = [item.strip() for item in re.split(r'[;,]', pred) if item.strip()]
            return set(pred_items) == set(gt_items)
        
        return False
    
    def validate_response_structure(self, response: str) -> Tuple[float, float]:
        """
        验证响应结构，返回两个分数：
        1. format_score: 完整格式正确性 (0.0~1.0)
        2. begin_score: 开头格式正确性 (0.0~1.0)
        """
        # 检查开头是否包含 <think> 标签
        begin_score = 1.0 if response.lstrip().startswith("<think>") else 0.0
        
        # 严格模式：检查标签存在性和顺序
        if self.strict_format:
            # 检查必需标签
            has_think_start = "<think>" in response
            has_think_end = "</think>" in response
            has_answer_start = "<answer>" in response
            has_answer_end = "</answer>" in response
            
            # 检查标签顺序
            think_start_idx = response.find("<think>")
            think_end_idx = response.find("</think>")
            answer_start_idx = response.find("<answer>")
            answer_end_idx = response.find("</answer>")
            
            order_valid = (
                think_start_idx != -1 and 
                think_end_idx != -1 and
                answer_start_idx != -1 and
                answer_end_idx != -1 and
                think_start_idx < think_end_idx < answer_start_idx < answer_end_idx
            )
            
            # 计算格式分数（0.75权重给完整结构，0.25权重给开头）
            tag_validity = (has_think_start + has_think_end + has_answer_start + has_answer_end) / 4.0
            format_score = 0.75 * (1.0 if order_valid else 0.0) + 0.25 * tag_validity
            
        else:
            # 宽松模式：只检查关键标签存在性
            required_tags = ["<think>", "</think>", "<answer>", "</answer>"]
            present_tags = sum(1 for tag in required_tags if tag in response)
            format_score = present_tags / len(required_tags)
        
        return format_score, begin_score
    
    def compute_reward(self, 
                      responses: List[str], 
                      ground_truths: List[Dict[str, str]],
                      **kwargs) -> torch.Tensor:
        """
        Verl 兼容的奖励计算接口
        
        Args:
            responses: 模型生成的响应列表
            ground_truths: 真实答案列表，每个元素为字典 {'target': '正确答案'}
            
        Returns:
            torch.Tensor: 形状为 [batch_size] 的奖励张量
        """
        rewards = []
        
        for resp, gt in zip(responses, ground_truths):
            # 采样日志（每64次记录一次）
            do_log = random.randint(1, 64) == 1
            
            # 1. 提取答案
            answer = self.extract_solution(resp)
            
            # 2. 验证格式
            format_valid, begin_valid = self.validate_response_structure(resp)
            
            # 3. 检查答案正确性
            answer_correct = False
            if answer is not None:
                answer_correct = self.em_check(answer, gt["target"])
            
            # 4. 计算格式奖励 (0.0 ~ format_weight)
            format_score = self.format_weight * (
                0.75 * format_valid + 
                0.25 * begin_valid
            )
            
            # 5. 计算最终奖励
            if answer is None:
                total_reward = 0.0
            elif answer_correct:
                # 答案正确：基础分 + 格式分
                total_reward = self.answer_weight + format_score
            else:
                # 答案错误：只给格式分
                total_reward = format_score
            
            # 6. 调试日志
            if do_log:
                print(f"\n[REWARD DEBUG] Format: {format_score:.3f}/{self.format_weight}, "
                      f"Answer: {'CORRECT' if answer_correct else 'WRONG'}, "
                      f"Total: {total_reward:.3f}")
                print(f"Response: {resp[:200]}..." if len(resp) > 200 else resp)
                print(f"Extracted Answer: '{answer}' vs GT: '{gt['target']}'")
            
            rewards.append(total_reward)
        
        return torch.tensor(rewards, dtype=torch.float32)