import numpy as np
from typing import Dict, Tuple

class CombatSystem:
    def __init__(self):
        """初始化战斗系统"""
        self.attribute_matrix = {
            'speed': {'speed': 0.5, 'power': 0.3, 'defense': 0.2},
            'power': {'speed': 0.2, 'power': 0.5, 'defense': 0.3},
            'defense': {'speed': 0.3, 'power': 0.2, 'defense': 0.5}
        }
        self.k = 0.02  # 曲线陡度因子

    def calculate_combat_result(self, attacker: Dict[str, float], 
                              defender: Dict[str, float]) -> Tuple[bool, float]:
        """
        计算对抗结果
        返回: (是否获胜, 胜率)
        """
        # 计算加权属性总和
        attacker_score = self._calculate_weighted_score(attacker)
        defender_score = self._calculate_weighted_score(defender)
        
        # 使用双曲线衰减模型计算胜率
        win_probability = 1 / (1 + np.exp(-self.k * (attacker_score - defender_score)))
        
        # 根据胜率决定胜负
        is_winner = np.random.random() < win_probability
        
        return is_winner, win_probability

    def _calculate_weighted_score(self, attributes: Dict) -> float:
        """
        计算加权得分
        :param attributes: 属性字典
        :return: 加权得分
        """
        score = 0.0
        for attr, value in attributes.items():
            weights = self.attribute_matrix[attr]
            for target_attr, weight in weights.items():
                score += value * weight * attributes[target_attr]
        return score

    def validate_combat_system(self) -> Dict[str, bool]:
        """验证对抗系统的合理性"""
        results = {
            'boundary_test': self._test_boundary_conditions(),
            'balance_test': self._test_balance(),
            'counter_test': self._test_counter_chain()
        }
        return results

    def _test_boundary_conditions(self) -> bool:
        """测试边界条件"""
        strong_attacker = {'speed': 100, 'power': 100, 'defense': 100}
        weak_defender = {'speed': 10, 'power': 10, 'defense': 10}
        
        _, win_prob = self.calculate_combat_result(strong_attacker, weak_defender)
        return win_prob >= 0.999

    def _test_balance(self) -> bool:
        """测试平衡性"""
        identical_creature = {'speed': 50, 'power': 50, 'defense': 50}
        wins = 0
        total_tests = 1000
        
        for _ in range(total_tests):
            is_winner, _ = self.calculate_combat_result(identical_creature, identical_creature)
            if is_winner:
                wins += 1
                
        win_rate = wins / total_tests
        return 0.48 <= win_rate <= 0.52

    def _test_counter_chain(self) -> bool:
        """测试克制链"""
        speed_creature = {'speed': 100, 'power': 0, 'defense': 0}
        defense_creature = {'speed': 0, 'power': 0, 'defense': 100}
        
        _, win_prob = self.calculate_combat_result(speed_creature, defense_creature)
        return win_prob >= 0.65  # Speed应该对Defense有15%以上的优势

    def resolve_combat(self, attacker_attributes: Dict, defender_attributes: Dict) -> int:
        """
        解决战斗并返回胜者
        :param attacker_attributes: 攻击者属性
        :param defender_attributes: 防御者属性
        :return: 0表示攻击者胜利，1表示防御者胜利
        """
        attacker_score = self._calculate_weighted_score(attacker_attributes)
        defender_score = self._calculate_weighted_score(defender_attributes)
        
        # 添加随机因素
        attacker_score *= np.random.uniform(0.8, 1.2)
        defender_score *= np.random.uniform(0.8, 1.2)
        
        return 0 if attacker_score > defender_score else 1