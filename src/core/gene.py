from typing import List, Dict
import numpy as np

class Gene:
    def __init__(self, dominant: str = None, recessive: str = None, mutation_rate: float = 0.1):
        self.dominant = dominant if dominant else ('A' if np.random.random() < 0.5 else 'a')
        self.recessive = recessive if recessive else ('A' if np.random.random() < 0.5 else 'a')
        self.mutation_rate = mutation_rate
        self.value = np.random.random()  # 用于神经网络的连续值

    def __str__(self):
        return f"({self.dominant}, {self.recessive}, {self.mutation_rate}, {self.value:.2f})"

class Chromosome:
    def __init__(self, genes: List[Gene] = None):
        if genes is None:
            # 创建默认基因组
            self.genes = [
                Gene(),  # 生产者能力
                Gene(),  # 消费者能力
                Gene(),  # 被动防御能力
                Gene(),  # 能量转化效率
                Gene(),  # 寿命基因
                Gene()   # 繁殖能力
            ]
        else:
            self.genes = genes

    def crossover(self, other: 'Chromosome') -> 'Chromosome':
        """染色体交叉"""
        new_genes = []
        for g1, g2 in zip(self.genes, other.genes):
            if np.random.random() < 0.5:
                new_gene = Gene(g1.dominant, g2.recessive, g1.mutation_rate)
                new_gene.value = (g1.value + g2.value) / 2 + np.random.normal(0, 0.1)
                new_genes.append(new_gene)
            else:
                new_gene = Gene(g2.dominant, g1.recessive, g2.mutation_rate)
                new_gene.value = (g1.value + g2.value) / 2 + np.random.normal(0, 0.1)
                new_genes.append(new_gene)
        return Chromosome(new_genes)

    def mutate(self) -> 'Chromosome':
        """基因突变"""
        new_genes = []
        for gene in self.genes:
            if np.random.random() < gene.mutation_rate:
                # 随机选择一个基因进行突变
                if np.random.random() < 0.5:
                    new_gene = Gene(
                        self._mutate_allele(gene.dominant),
                        gene.recessive,
                        gene.mutation_rate
                    )
                else:
                    new_gene = Gene(
                        gene.dominant,
                        self._mutate_allele(gene.recessive),
                        gene.mutation_rate
                    )
                # 突变连续值
                new_gene.value = gene.value + np.random.normal(0, 0.2)
                new_gene.value = max(0, min(1, new_gene.value))  # 限制在0-1范围内
                new_genes.append(new_gene)
            else:
                new_genes.append(gene)
        return Chromosome(new_genes)

    def _mutate_allele(self, allele: str) -> str:
        """等位基因突变"""
        if allele.isupper():
            return 'a' if np.random.random() < 0.3 else 'A'
        else:
            return 'A' if np.random.random() < 0.3 else 'a'

    def calculate_dominant_trait(self) -> Dict[str, float]:
        """计算显性性状"""
        traits = {}
        for i, gene in enumerate(self.genes):
            # 根据孟德尔遗传定律计算显性性状
            if gene.dominant.isupper() and gene.recessive.isupper():
                base_value = 1.0  # 纯合显性
            elif gene.dominant.islower() and gene.recessive.islower():
                base_value = 0.0  # 纯合隐性
            else:
                base_value = 0.75  # 杂合显性
            
            # 结合连续值
            traits[f'trait_{i}'] = (base_value + gene.value) / 2
            
        # 计算生物类型
        producer_ability = traits['trait_0']
        consumer_ability = traits['trait_1']
        
        # if producer_ability > 0.7 and consumer_ability < 0.3:
        #     traits['type'] = 'producer'
        # elif consumer_ability > 0.7:
        #     if producer_ability > 0.4:
        #         traits['type'] = 'high_consumer'
        #     else:
        #         traits['type'] = 'low_consumer'
        # else:
        #     traits['type'] = 'producer'  # 默认类型
            
        return traits

    def reproduce(self) -> 'Chromosome':
        """生成后代染色体"""
        new_genes = []
        for gene in self.genes:
            # 创建新基因
            new_gene = Gene(gene.dominant, gene.recessive, gene.mutation_rate)
            # 添加随机变异
            new_gene.value = max(0, min(1, gene.value + np.random.normal(0, 0.1)))
            new_genes.append(new_gene)
        return Chromosome(new_genes)