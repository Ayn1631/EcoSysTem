import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from core.ecosystem import Ecosystem
from core.gene import Chromosome, Gene
import time
from PIL import Image
import io
import imageio

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class EcosystemVisualizer:
    def __init__(self):
        """初始化可视化器"""
        # 创建主图形
        self.fig = plt.figure(figsize=(15, 10))
        
        # 创建子图
        self.ax1 = self.fig.add_subplot(221)  # 资源分布
        self.ax2 = self.fig.add_subplot(222)  # 生物分布
        self.ax3 = self.fig.add_subplot(223)  # 属性分布
        self.ax4 = self.fig.add_subplot(224)  # 年龄分布
        
        # 创建统计图表
        self.stats_fig = plt.figure(figsize=(12, 6))
        self.stats_ax = self.stats_fig.add_subplot(111)
        
        # 创建种群可视化
        self.population_fig = plt.figure(figsize=(12, 6))
        self.population_ax = self.population_fig.add_subplot(111)
        
        # 初始化数据
        self.species_history = []
        self.frames = []
        self.population_frames = []
        
        # 初始化颜色映射
        self.color_map = {}
        self.resource_colorbar = None
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        print("初始化可视化器...")
        
    def _calculate_gene_similarity(self, genes1, genes2):
        """计算两个基因序列的相似度"""
        if len(genes1) != len(genes2):
            return 0.0
        
        total_diff = 0
        for g1, g2 in zip(genes1, genes2):
            try:
                diff = abs(float(g1.value) - float(g2.value))
                total_diff += min(diff, 1.0)  # 限制单个基因的差异最大为1
            except (TypeError, ValueError) as e:
                print(f"计算基因差异时出错: {str(e)}")
                return 0.0
                
        avg_diff = total_diff / len(genes1)
        similarity = 1.0 - avg_diff
        return max(0.0, min(1.0, similarity))  # 确保相似度在0到1之间

    def _get_species_id(self, agent, species_representatives):
        """获取生物所属的物种ID"""
        try:
            # 相似度阈值
            SIMILARITY_THRESHOLD = 0.85
            
            # 获取当前生物的基因值
            agent_genes = agent.chromosome.genes
            
            # 与现有物种代表比较
            max_similarity = 0
            best_species_id = None
            
            for species_id, rep_genes in species_representatives.items():
                similarity = self._calculate_gene_similarity(agent_genes, rep_genes)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_species_id = species_id
            
            # 如果找到足够相似的物种，返回该物种ID
            if max_similarity >= SIMILARITY_THRESHOLD:
                return best_species_id
                    
            # 如果没有找到匹配的物种，创建新物种
            new_species_id = len(species_representatives)
            species_representatives[new_species_id] = agent_genes
            return new_species_id
            
        except Exception as e:
            print(f"获取物种ID时出错: {str(e)}")
            # 发生错误时返回一个默认的物种ID
            return 0

    def _get_agent_color(self, agent, species_representatives):
        """获取或创建生物的颜色"""
        species_id = self._get_species_id(agent, species_representatives)
        if species_id not in self.color_map:
            # 使用HSV颜色空间生成不同的颜色
            hue = species_id / 100  # 使用100种不同的色调
            self.color_map[species_id] = plt.cm.hsv(hue)
        return self.color_map[species_id]

    def _count_species(self, ecosystem):
        """统计不同物种的数量"""
        species_counts = {}
        species_representatives = {}  # 用于存储每个物种的代表基因
        
        for agent in ecosystem.agents:
            species_id = self._get_species_id(agent, species_representatives)
            species_counts[species_id] = species_counts.get(species_id, 0) + 1
            
        return len(species_counts), species_counts, species_representatives

    def update(self, ecosystem: Ecosystem, stats_history: list):
        """更新可视化"""
        try:
            start_time = time.time()
            
            # 统计物种数量
            num_species, species_counts, species_representatives = self._count_species(ecosystem)
            self.species_history.append(num_species)
            
            # 更新资源分布
            self.ax1.clear()
            self.resource_im = self.ax1.imshow(ecosystem.resources, cmap='viridis')
            if self.resource_colorbar is not None:
                self.resource_colorbar.remove()
            self.resource_colorbar = self.fig.colorbar(self.resource_im, ax=self.ax1)
            self.ax1.set_title('资源分布')
            
            # 更新生物分布
            self.ax2.clear()
            positions = np.array([agent.position for agent in ecosystem.agents])
            if len(positions) > 0:
                # 使用颜色表示类型
                type_colors = {
                    'producer': 'green',
                    'low_consumer': 'blue',
                    'high_consumer': 'red'
                }
                
                # 使用不同形状表示不同类型的生物
                type_markers = {
                    'producer': '^',
                    'low_consumer': 'v',
                    'high_consumer': 's'
                }
                
                # 计算能量的最小值和最大值
                energies = [agent.energy for agent in ecosystem.agents]
                min_energy = min(energies)
                max_energy = max(energies)
                
                # 将能量值归一化到20-200的范围内
                sizes = [20 + (energy - min_energy) / (max_energy - min_energy + 1e-6) * 180 
                        for energy in energies]
                
                # 绘制每种类型的生物
                for type_name in type_colors:
                    mask = [agent.type == type_name for agent in ecosystem.agents]
                    if any(mask):
                        self.ax2.scatter(
                            positions[mask, 0], positions[mask, 1],
                            c=type_colors[type_name],
                            s=[s for i, s in enumerate(sizes) if mask[i]],
                            marker=type_markers[type_name],
                            alpha=0.6,
                            label=type_name
                        )
                        
                        
                        # 添加ID标签
                        for i, agent in enumerate(ecosystem.agents):
                            if mask[i]:
                                self.ax2.annotate(
                                    str(agent.id),
                                    (positions[i, 0], positions[i, 1]),
                                    xytext=(5, 5),
                                    textcoords='offset points',
                                    fontsize=8
                                )
                
                # 添加图例
                type_legend = [
                    plt.scatter([], [], c=type_colors[t], marker=type_markers[t], 
                              s=100, alpha=0.6, label=f"{t}") 
                    for t in type_colors
                ]
                type_legend_ax = self.ax2.legend(handles=type_legend, title="生物类型", 
                                               loc='upper right', fontsize='small')
                self.ax2.add_artist(type_legend_ax)
                
                
                # 调整子图布局以适应图例
                self.fig.tight_layout()
                
            self.ax2.set_title(f'生物分布 (物种数: {num_species})')
            self.ax2.set_xlim(0, ecosystem.width)
            self.ax2.set_ylim(0, ecosystem.height)
            
            # 更新属性分布
            self.ax3.clear()
            if len(ecosystem.agents) > 0:
                producer_ability = [agent.attributes['producer_ability'] for agent in ecosystem.agents]
                consumer_ability = [agent.attributes['consumer_ability'] for agent in ecosystem.agents]
                defense = [agent.attributes['defense'] for agent in ecosystem.agents]
                
                self.ax3.hist([producer_ability, consumer_ability, defense], 
                             label=['生产者能力', '消费者能力', '防御能力'],
                             bins=20, alpha=0.7)
                self.ax3.legend()
            self.ax3.set_title('属性分布')
            
            # 更新年龄分布
            self.ax4.clear()
            if len(ecosystem.agents) > 0:
                ages = [agent.age for agent in ecosystem.agents]
                self.ax4.hist(ages, bins=20, alpha=0.7)
            self.ax4.set_title('年龄分布')
            
            # 更新统计图表
            self._update_stats(stats_history)
            
            # 更新种群可视化
            self._update_population_visualization(ecosystem, species_representatives)
            
            # 保存当前帧
            self._save_frame()
            
            # 刷新显示
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()
            # self.stats_fig.canvas.draw()
            # self.stats_fig.canvas.flush_events()
            # self.population_fig.canvas.draw()
            # self.population_fig.canvas.flush_events()
            
            end_time = time.time()
            print(f"\n第 {len(self.frames)} 代更新:")
            print(f"更新可视化耗时: {end_time - start_time:.2f}秒")
            print(f"当前物种数量: {num_species}")
            print("物种分布:")
            
            # 按数量排序显示物种信息
            # sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
            # for species_id, count in sorted_species:
            #     # 计算该物种的平均属性
            #     species_agents = [agent for agent in ecosystem.agents 
            #                     if self._get_species_id(agent, species_representatives) == species_id]
            #     if species_agents:
            #         avg_producer = np.mean([agent.attributes['producer_ability'] for agent in species_agents])
            #         avg_consumer = np.mean([agent.attributes['consumer_ability'] for agent in species_agents])
            #         avg_defense = np.mean([agent.attributes['defense'] for agent in species_agents])
            #         avg_energy = np.mean([agent.energy for agent in species_agents])
            #         types = [agent.type for agent in species_agents]
            #         type_counts = {t: types.count(t) for t in set(types)}
                    
            #         print(f"物种 {species_id}: {count}个")
            #         print(f"  类型分布: {type_counts}")
            #         print(f"  平均生产者能力: {avg_producer:.2f}")
            #         print(f"  平均消费者能力: {avg_consumer:.2f}")
            #         print(f"  平均防御能力: {avg_defense:.2f}")
            #         print(f"  平均能量: {avg_energy:.2f}")
            print("-" * 50)
            
        except Exception as e:
            print(f"更新可视化时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
    def _update_population_visualization(self, ecosystem: Ecosystem, species_representatives):
        """更新种群可视化"""
        try:
            self.population_ax.clear()
            if len(ecosystem.agents) > 0:
                positions = np.array([agent.position for agent in ecosystem.agents])
                colors = [self._get_agent_color(agent, species_representatives) for agent in ecosystem.agents]
                
                # 计算能量的最小值和最大值
                energies = [agent.energy for agent in ecosystem.agents]
                min_energy = min(energies)
                max_energy = max(energies)
                
                # 将能量值归一化到20-200的范围内
                sizes = [20 + (energy - min_energy) / (max_energy - min_energy + 1e-6) * 180 
                        for energy in energies]
                
                self.population_scatter = self.population_ax.scatter(
                    positions[:, 0], positions[:, 1],
                    c=colors, s=sizes, alpha=0.6
                )
                
                # 添加图例说明
                legend_elements = [
                    plt.scatter([], [], c='gray', s=20, label=f'最小能量: {min_energy:.0f}'),
                    plt.scatter([], [], c='gray', s=110, label=f'中等能量: {(min_energy + max_energy)/2:.0f}'),
                    plt.scatter([], [], c='gray', s=200, label=f'最大能量: {max_energy:.0f}')
                ]
                self.population_ax.legend(handles=legend_elements, loc='upper right')
                
            self.population_ax.set_title('种群分布 (点大小表示能量)')
            self.population_ax.set_xlim(0, ecosystem.width)
            self.population_ax.set_ylim(0, ecosystem.height)
            
        except Exception as e:
            print(f"更新种群可视化时出错: {str(e)}")
            raise
        
    def _save_frame(self):
        """保存当前帧"""
        # 将生物分布图转换为PIL图像
        buf = io.BytesIO()
        self.ax2.figure.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        image = Image.open(buf)
        
        # 调整图像大小
        image = image.resize((800, 800), Image.Resampling.LANCZOS)
        
        # 将图像转换为RGB模式并添加到帧列表
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        self.frames.append(image)
        buf.close()
        
    def _update_stats(self, stats_history: list):
        """更新统计图表"""
        try:
            self.stats_ax.clear()
            
            if len(stats_history) > 0:
                # 准备数据
                generations = range(len(stats_history))
                num_agents = [s['num_agents'] for s in stats_history]
                avg_energy = [s['avg_energy'] for s in stats_history]
                avg_age = [s['avg_age'] for s in stats_history]
                resource_level = [s['resource_level'] for s in stats_history]
                
                # 绘制多条线
                self.stats_ax.plot(generations, num_agents, label='生物数量')
                self.stats_ax.plot(generations, avg_energy, label='平均能量')
                self.stats_ax.plot(generations, avg_age, label='平均年龄')
                self.stats_ax.plot(generations, resource_level, label='资源水平')
                
                # 添加图例和标题
                self.stats_ax.legend()
                self.stats_ax.set_title('生态系统统计数据')
                self.stats_ax.set_xlabel('代数')
                self.stats_ax.set_ylabel('数值')
                
                # 添加网格
                self.stats_ax.grid(True)
                
        except Exception as e:
            print(f"更新统计图表时出错: {str(e)}")
            raise
            
    def save_animation(self, filename='ecosystem_evolution.gif'):
        """保存动画"""
        if not self.frames:
            print("没有帧可以保存")
            return
        
        print(f"正在保存动画,共 {len(self.frames)} 帧...")
        # 保存为GIF
        self.frames[0].save(
            filename,
            save_all=True,
            append_images=self.frames[1:],
            duration=500,  # 每帧持续时间(毫秒)
            loop=1  # 0表示无限循环
        )
        print(f"动画已保存为 {filename}")
            
    def close(self):
        """关闭可视化窗口"""
        try:
            plt.close(self.fig)
            plt.close(self.stats_fig)
            plt.close(self.population_fig)
            plt.ioff()
            print("已关闭所有窗口")
        except Exception as e:
            print(f"关闭窗口时出错: {str(e)}")
            raise

def main():
    try:
        print("开始生态系统模拟...")
        
        # 创建生态系统
        ecosystem = Ecosystem(width=500, height=500)
        ecosystem.initialize(num_agents=1000)
        print(f"生态系统初始化完成，初始生物数量: {len(ecosystem.agents)}")
        
        # 创建可视化器
        visualizer = EcosystemVisualizer()
        
        # 运行模拟
        num_generations = 1000
        stats_history = []
        
        for generation in range(num_generations):
            try:
                ecosystem.update()
                stats = ecosystem.get_statistics()
                stats_history.append(stats)
                
                # 每10代可视化一次
                if generation % 10 == 0:
                    print(f"\n第 {generation} 代:")
                    print(f"生物数量: {stats['num_agents']}")
                    print(f"生产者数量: {stats['producer_count']}")
                    print(f"低级消费者数量: {stats['low_consumer_count']}")
                    print(f"高级消费者数量: {stats['high_consumer_count']}")
                    print(f"平均能量: {stats['avg_energy']:.2f}")
                    print(f"平均年龄: {stats['avg_age']:.2f}")
                    print(f"资源水平: {stats['resource_level']:.2f}")
                    print("-" * 50)
                    
                    visualizer.update(ecosystem, stats_history)
                    # plt.pause(0.1)  # 短暂暂停以允许窗口更新
                    
            except Exception as e:
                print(f"第 {generation} 代更新时出错: {str(e)}")
                raise
                
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        raise
        
    finally:
        try:
            # 保存动画
            visualizer.save_animation()
            visualizer.close()
            plt.show()
            print("程序运行完成")
        except Exception as e:
            print(f"程序结束时出错: {str(e)}")
            raise

if __name__ == "__main__":
    main() 
