from typing import List, Dict, Optional, TYPE_CHECKING, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from .gene import Chromosome, Gene
from .combat import CombatSystem
from .neural_network import BioRL
import random

if TYPE_CHECKING:
    from .ecosystem import Agent

class Agent:
    def __init__(self, position, ecosystem, brain=None, ctype=None):
        self.position = np.array(position)
        self.ecosystem = ecosystem  # 添加ecosystem引用
        self.energy = 100
        self.age = 0
        self.id = np.random.randint(1000, 9999)  # 生成4位数的ID
        self.chromosome = Chromosome()  # 使用默认基因组
        self.speed = 1.0  # 添加移动速度属性
        
        # 根据基因设置属性
        traits = self.chromosome.calculate_dominant_trait()
        self.type = ctype
        self.attributes = {
            'producer_ability': traits['trait_0'],
            'consumer_ability': traits['trait_1'],
            'defense': traits['trait_2'],
            'energy_efficiency': traits['trait_3'],
            'lifespan': 50 + traits['trait_4'] * 10,  # 基础寿命100，基因影响0-100
            'reproduction_rate': traits['trait_5']
        }
        
        # 只有消费者需要神经网络
        self.brain = brain if brain and self.type != 'producer' else None
        
    def decide(self, snapshot_agents: List[Dict], resources: np.ndarray, width: int, height: int, view_range: int):
        """基于快照做出决策，不修改任何状态"""
        # 局部视野：只看一定范围内的个体
        local_agents = []
        for other in snapshot_agents:
            if other["id"] == self.id:
                continue
            dist = _toroidal_distance(self.position, other["position"], width, height)
            if dist <= view_range:
                local_agents.append((other, dist))

        move_target = self.position.copy()
        intended_prey_id = None
        arrival_score = 0.0

        if self.type == "low_consumer":
            nearest_producer = None
            nearest_producer_dist = float("inf")
            nearest_high_consumer = None
            nearest_high_consumer_dist = float("inf")
            for other, dist in local_agents:
                if other["type"] == "producer" and dist < nearest_producer_dist:
                    nearest_producer = other
                    nearest_producer_dist = dist
                elif other["type"] == "high_consumer" and dist < nearest_high_consumer_dist:
                    nearest_high_consumer = other
                    nearest_high_consumer_dist = dist

            move_direction = np.zeros(2)
            if nearest_producer is not None:
                move_direction += np.array(nearest_producer["position"]) - np.array(self.position)
                intended_prey_id = nearest_producer["id"]
                arrival_score = 1.0 / (nearest_producer_dist + 1.0)
            if nearest_high_consumer is not None:
                move_direction -= np.array(nearest_high_consumer["position"]) - np.array(self.position)

            if np.any(move_direction):
                move_direction = move_direction / np.linalg.norm(move_direction)
                move_target = np.array(self.position) + move_direction * self.speed

        elif self.type == "high_consumer":
            nearest_low_consumer = None
            nearest_low_consumer_dist = float("inf")
            for other, dist in local_agents:
                if other["type"] != "low_consumer":
                    continue
                if dist < nearest_low_consumer_dist:
                    nearest_low_consumer = other
                    nearest_low_consumer_dist = dist

            if nearest_low_consumer is not None:
                move_direction = np.array(nearest_low_consumer["position"]) - np.array(self.position)
                if np.any(move_direction):
                    move_direction = move_direction / np.linalg.norm(move_direction)
                    move_target = np.array(self.position) + move_direction * self.speed
                intended_prey_id = nearest_low_consumer["id"]
                arrival_score = 1.0 / (nearest_low_consumer_dist + 1.0)

        # 生产者不移动
        elif self.type == "producer":
            pass

        # 繁殖决策：统一执行阶段处理能量扣减
        reproduce = False
        if self.energy >= 150:
            base_rate = self.attributes["reproduction_rate"] * 0.8
            if self.type == "producer":
                local_producers = sum(1 for other, _ in local_agents if other["type"] == "producer")
                area = (2 * view_range + 1) ** 2
                density_factor = max(0.1, 1.0 - (local_producers / max(1, area)))
                reproduce = np.random.random() < base_rate * density_factor
            else:
                reproduce = np.random.random() < base_rate

        return AgentDecision(
            agent_id=self.id,
            move_target=move_target,
            intended_prey_id=intended_prey_id,
            arrival_score=arrival_score,
            reproduce=reproduce,
        )
        
    def _get_environment_state(self, environment_state):
        """获取环境状态"""
        return environment_state
        
    def reproduce(self) -> Optional["Agent"]:
        """生成子代（执行阶段调用）"""
        # 创建子代，位置在父代周围随机偏移
        offset = np.random.randint(-2, 3, size=2)
        child_position = np.clip(
            self.position + offset,
            [0, 0],
            [self.ecosystem.width - 1, self.ecosystem.height - 1],
        )

        # 只有消费者需要神经网络
        child_brain = BioRL(input_size=5) if self.type != "producer" else None
        child = Agent(child_position, self.ecosystem, child_brain, self.type)

        # 遗传基因
        child.chromosome = self.chromosome.reproduce()

        # 根据基因更新属性
        traits = child.chromosome.calculate_dominant_trait()
        child.type = self.type
        child.attributes = {
            "producer_ability": traits["trait_0"],
            "consumer_ability": traits["trait_1"],
            "defense": traits["trait_2"],
            "energy_efficiency": traits["trait_3"],
            "lifespan": 50 + traits["trait_4"] * 10,
            "reproduction_rate": traits["trait_5"],
        }

        # 只有消费者需要转移知识
        if self.brain is not None and child.brain is not None:
            child.brain.transfer_knowledge(self.brain)

        # 消耗能量
        self.energy -= 80
        return child

    def die(self):
        """处理生物死亡"""
        # 将能量返还给环境
        x, y = int(self.position[0]), int(self.position[1])
        if 0 <= x < self.ecosystem.width and 0 <= y < self.ecosystem.height:
            self.ecosystem.resources[x, y] = min(1.0, self.ecosystem.resources[x, y] + self.energy * 0.9)
        
        # 从生态系统中移除
        if self in self.ecosystem.agents:
            self.ecosystem.agents.remove(self)

class Ecosystem:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.agents: List[Agent] = []
        self.resources = np.zeros((width, height))
        self.combat_system = CombatSystem()
        self.view_range = 6
        self.predation_energy_factor = 0.6
        self.predation_arrival_factor = 0.4
        
    def initialize(self, num_agents: int):
        """初始化生态系统"""
        # 初始化资源
        self.resources = np.random.random((self.width, self.height))
        
        # 计算各类型生物的数量
        num_producers = int(num_agents * 0.95)
        num_low_consumers = int(num_agents * 0.04)
        num_high_consumers = num_agents - num_producers - num_low_consumers
        
        # 创建生产者（随机分布）
        for _ in range(num_producers):
            position = [
                np.random.randint(0, self.width),
                np.random.randint(0, self.height)
            ]
            agent = Agent(position, self, ctype='producer')  # 传入self作为ecosystem
            # 强制设置为生产者类型
            agent.type = 'producer'
            agent.attributes['producer_ability'] = 0.8 + np.random.random() * 0.2
            agent.attributes['consumer_ability'] = np.random.random() * 0.2
            self.agents.append(agent)
            
        # 创建低级消费者（扎堆分布）
        # 选择3-5个中心点
        num_centers = np.random.randint(3, 6)
        centers = []
        for _ in range(num_centers):
            center = [
                np.random.randint(0, self.width),
                np.random.randint(0, self.height)
            ]
            centers.append(center)
            
        # 在中心点周围生成低级消费者
        for _ in range(num_low_consumers):
            # 随机选择一个中心点
            center = centers[np.random.randint(0, len(centers))]
            # 在中心点周围随机生成位置
            position = [
                int(np.clip(center[0] + np.random.normal(0, 5), 0, self.width-1)),
                int(np.clip(center[1] + np.random.normal(0, 5), 0, self.height-1))
            ]
            agent = Agent(position, self, ctype='low_consumer')  # 传入self作为ecosystem
            # 强制设置为低级消费者类型
            agent.type = 'low_consumer'
            agent.attributes['producer_ability'] = np.random.random() * 0.3
            agent.attributes['consumer_ability'] = 0.6 + np.random.random() * 0.2
            self.agents.append(agent)
            
        # 创建高级消费者（随机分布）
        for _ in range(num_high_consumers):
            position = [
                np.random.randint(0, self.width),
                np.random.randint(0, self.height)
            ]
            agent = Agent(position, self, ctype='high_consumer')  # 传入self作为ecosystem
            # 强制设置为高级消费者类型
            agent.type = 'high_consumer'
            agent.attributes['producer_ability'] = np.random.random() * 0.4
            agent.attributes['consumer_ability'] = 0.8 + np.random.random() * 0.2
            self.agents.append(agent)
            
        print(f"初始化完成：")
        print(f"生产者数量: {num_producers}")
        print(f"低级消费者数量: {num_low_consumers}")
        print(f"高级消费者数量: {num_high_consumers}")
        
    def update(self):
        """更新生态系统状态"""
        # 更新资源
        self.resources += 0.005  # 减少资源恢复速度
        np.clip(self.resources, 0, 1, out=self.resources)

        if not self.agents:
            return

        # 决策阶段：快照 + 并行决策
        snapshot_agents = [
            {
                "id": agent.id,
                "type": agent.type,
                "position": np.array(agent.position),
                "energy": agent.energy,
                "attributes": agent.attributes,
                "age": agent.age,
            }
            for agent in self.agents
        ]

        decisions = []
        max_workers = min(len(self.agents), max(1, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    agent.decide,
                    snapshot_agents,
                    self.resources,
                    self.width,
                    self.height,
                    self.view_range,
                )
                for agent in self.agents
            ]
            for future in futures:
                decisions.append(future.result())

        decision_map = {d.agent_id: d for d in decisions}

        # 统计决策（可用于调试/日志）
        self.last_decision_stats = {
            "move": sum(1 for d in decisions if d.move_target is not None),
            "attack": sum(1 for d in decisions if d.intended_prey_id is not None),
            "reproduce": sum(1 for d in decisions if d.reproduce),
        }

        # 执行阶段：统一更新
        # 1) 移动
        for agent in self.agents:
            decision = decision_map.get(agent.id)
            if decision and decision.move_target is not None:
                agent.position = np.array(
                    np.clip(
                        decision.move_target,
                        [0, 0],
                        [self.width - 1, self.height - 1],
                    )
                )

        # 2) 代谢与年龄
        for agent in self.agents:
            agent.age += 1
            agent.energy -= 0.1

        # 3) 生产者采集资源
        for agent in self.agents:
            if agent.type != "producer":
                continue
            x, y = int(agent.position[0]), int(agent.position[1])
            if 0 <= x < self.width and 0 <= y < self.height:
                energy_gain = self.resources[x, y] * agent.attributes["producer_ability"] * 1.2
                agent.energy += energy_gain
                self.resources[x, y] = max(0.0, self.resources[x, y] - energy_gain * 0.6)

        # 4) 捕食裁决（基于能量 + 先到先得）
        id_to_agent = {agent.id: agent for agent in self.agents}
        prey_to_predators: Dict[int, List[Tuple[Agent, AgentDecision]]] = {}
        for agent in self.agents:
            decision = decision_map.get(agent.id)
            if not decision or decision.intended_prey_id is None:
                continue
            if agent.energy <= 0:
                continue
            prey = id_to_agent.get(decision.intended_prey_id)
            if not prey or prey.energy <= 0:
                continue
            if not self._are_adjacent(agent, prey):
                continue
            # 限制可捕食类型
            if agent.type == "low_consumer" and prey.type != "producer":
                continue
            if agent.type == "high_consumer" and prey.type not in ["low_consumer", "producer"]:
                continue
            prey_to_predators.setdefault(prey.id, []).append((agent, decision))

        dead_agents = set()
        for prey_id, predator_list in prey_to_predators.items():
            prey = id_to_agent.get(prey_id)
            if not prey or prey in dead_agents:
                continue
            energies = np.array([p.energy for p, _ in predator_list], dtype=float)
            arrivals = np.array([d.arrival_score for _, d in predator_list], dtype=float)
            if energies.max() > energies.min():
                energies = (energies - energies.min()) / (energies.max() - energies.min())
            if arrivals.max() > arrivals.min():
                arrivals = (arrivals - arrivals.min()) / (arrivals.max() - arrivals.min())
            weights = (
                self.predation_energy_factor * energies
                + self.predation_arrival_factor * arrivals
            )
            if np.all(weights <= 0):
                weights = np.ones_like(weights)
            chosen_idx = random.choices(range(len(predator_list)), weights=weights, k=1)[0]
            predator, _ = predator_list[chosen_idx]

            energy_gained = prey.energy * 0.9
            energy_to_environment = prey.energy * 0.1
            predator.energy += energy_gained

            x, y = int(prey.position[0]), int(prey.position[1])
            self.resources[x, y] = min(1.0, self.resources[x, y] + energy_to_environment)
            dead_agents.add(prey)

        # 5) 移除死亡
        survivors = []
        for agent in self.agents:
            if agent in dead_agents or agent.energy <= 0 or agent.age >= agent.attributes["lifespan"]:
                x, y = int(agent.position[0]), int(agent.position[1])
                self.resources[x, y] = min(1.0, self.resources[x, y] + agent.energy * 0.8)
                continue
            survivors.append(agent)
        self.agents = survivors

        # 6) 统一繁殖
        new_agents = []
        for agent in self.agents:
            decision = decision_map.get(agent.id)
            if not decision or not decision.reproduce:
                continue
            if agent.energy < 150:
                continue
            child = agent.reproduce()
            if child:
                new_agents.append(child)
        self.agents.extend(new_agents)
        
    def _get_environment_state(self, agent: Agent) -> np.ndarray:
        """获取生物周围的环境状态"""
        x, y = int(agent.position[0]), int(agent.position[1])  # 确保使用整数索引
        state = np.zeros(5)
        
        # 当前位置的资源
        state[0] = self.resources[x, y]
        
        # 四个相邻位置的资源
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 上下左右
        for i, (dx, dy) in enumerate(directions):
            new_x = (x + dx) % self.width
            new_y = (y + dy) % self.height
            state[i + 1] = self.resources[new_x, new_y]
            
        return state
        
    def _are_adjacent(self, agent1: Agent, agent2: Agent) -> bool:
        """检查两个生物是否相邻"""
        x1, y1 = agent1.position
        x2, y2 = agent2.position
        dx = min((x1 - x2) % self.width, (x2 - x1) % self.width)
        dy = min((y1 - y2) % self.height, (y2 - y1) % self.height)
        return dx <= 1 and dy <= 1

    def get_statistics(self) -> Dict:
        """获取生态系统统计信息"""
        if not self.agents:
            return {
                'num_agents': 0,
                'avg_energy': 0,
                'avg_age': 0,
                'resource_level': np.mean(self.resources),
                'producer_count': 0,
                'low_consumer_count': 0,
                'high_consumer_count': 0
            }
            
        # 统计不同类型生物的数量
        producer_count = sum(1 for agent in self.agents if agent.type == 'producer')
        low_consumer_count = sum(1 for agent in self.agents if agent.type == 'low_consumer')
        high_consumer_count = sum(1 for agent in self.agents if agent.type == 'high_consumer')
            
        return {
            'num_agents': len(self.agents),
            'avg_energy': np.mean([agent.energy for agent in self.agents]),
            'avg_age': np.mean([agent.age for agent in self.agents]),
            'resource_level': np.mean(self.resources),
            'producer_count': producer_count,
            'low_consumer_count': low_consumer_count,
            'high_consumer_count': high_consumer_count
        }


@dataclass
class AgentDecision:
    agent_id: int
    move_target: np.ndarray
    intended_prey_id: Optional[int]
    arrival_score: float
    reproduce: bool


def _toroidal_distance(pos1: np.ndarray, pos2: np.ndarray, width: int, height: int) -> float:
    dx = min((pos1[0] - pos2[0]) % width, (pos2[0] - pos1[0]) % width)
    dy = min((pos1[1] - pos2[1]) % height, (pos2[1] - pos1[1]) % height)
    return float(np.hypot(dx, dy))
