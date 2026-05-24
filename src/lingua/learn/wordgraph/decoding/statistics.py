#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 word-lingua 图中的边标签条件概率：
1. p(child_edge_label | (parent_edge_label, node_pos)): 给定父边标签 + 当前节点 POS，子边标签的条件概率
2. p(child_edge_label | other_child_edge_label): 给定另一个子边标签，某个子边标签的条件概率
3. p(child_edge_label | node_pos): 给定节点 POS，子边标签的条件概率
4. p(parent_edge_label | child_edge_label): 给定子边标签，父边标签的条件概率
"""

import json
import jsonlines
from collections import defaultdict
from typing import Dict, Iterable
from lingua.structure.gpgraph import GPGraph
from lingua.learn.wordgraph.modeler.gp2word import lingua2wordlingua

import traceback


class WordLinguaStatistics:
    """
    WordLingua 统计信息类，用于统计和计算边标签条件概率。
    
    支持的功能：
    1. 从训练数据计算统计信息
    2. 保存统计信息到文件
    3. 从文件加载统计信息
    4. 计算给定条件的概率（包括未观察到的条目，使用 add-one smoothing）
    """
    
    # 需要过滤的 upper/lower 边标签
    UPPER_LOWER_EDGES = {"@upper", "@lower"}
    
    def __init__(self):
        """初始化空的统计信息对象"""
        # 已计算的条件概率（仅包含观察到的条目）
        self.p_child_given_parent: Dict[str, float] = {}  # "parent_label|node_pos|child_label" -> prob
        self.p_child_given_sibling: Dict[str, float] = {}  # "other_label|child_label" -> prob
        self.p_child_given_pos: Dict[str, float] = {}  # "node_pos|child_label" -> prob
        self.p_parent_given_child: Dict[str, float] = {}  # "child_label|parent_label" -> prob
        
        # Smoothing 信息
        self.vocab_size_parent_child: int = 0
        self.vocab_size_sibling_child: int = 0
        self.vocab_size_pos_child: int = 0
        self.vocab_size_child_parent: int = 0
        self.parent_label_counts: Dict[str, int] = {}  # "parent_label|node_pos" -> count
        self.other_child_label_counts: Dict[str, int] = {}  # other_label -> count
        self.pos_counts: Dict[str, int] = {}  # node_pos -> count
        self.child_label_counts: Dict[str, int] = {}  # child_label -> count
        
        # 统计信息
        self.statistics: Dict[str, int] = {}
    
    def compute_statistics(self, input_file: str, verbose: bool = True, remove_upper_lower: bool = False):
        """
        从训练数据计算统计信息。
        
        Args:
            input_file: 输入的 JSONL 文件路径
            verbose: 是否打印详细信息
            remove_upper_lower: 是否在统计前移除 @upper 和 @lower 边
        """
        # 统计计数
        parent_child_counts = defaultdict(int)
        sibling_child_counts = defaultdict(int)
        pos_child_counts = defaultdict(int)  # (node_pos, child_label) -> count
        child_parent_counts = defaultdict(int)  # (child_label, parent_label) -> count
        parent_label_counts = defaultdict(int)  # (parent_label, node_pos) -> count
        other_child_label_counts = defaultdict(int)  # other_child_label -> count
        pos_counts = defaultdict(int)  # node_pos -> count
        child_label_counts = defaultdict(int)  # child_label -> count
        
        # 收集所有可能的 child 标签，用于 add-one smoothing
        all_child_labels = set()
        all_sibling_child_labels = set()
        all_pos_child_labels = set()
        all_child_parent_labels = set()
        
        total_samples = 0
        error_samples = 0
        max_in_degree = 0
        
        def filter_children(children):
            """过滤子节点，如果需要移除 @upper/@lower 边"""
            if not remove_upper_lower:
                return children
            return [(child, edge) for child, edge in children 
                    if edge.label not in self.UPPER_LOWER_EDGES]
        
        def filter_parents(parents):
            """过滤父节点，如果需要移除 @upper/@lower 边"""
            if not remove_upper_lower:
                return parents
            return [(parent, edge) for parent, edge in parents 
                    if edge.label not in self.UPPER_LOWER_EDGES]
        
        if verbose:
            print(f"Reading from {input_file}...")
            if remove_upper_lower:
                print(f"Filtering out @upper and @lower edges...")
        with jsonlines.open(input_file, "r") as f:
            for idx, item in enumerate(f):
                total_samples += 1
                if verbose and (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1} samples...")
                
                try:
                    # 解析 linguagraph
                    lingua_graph = GPGraph.parse(item["lingua"])
                    
                    # 转换为 word-lingua
                    word_lingua_graph, _, _ = lingua2wordlingua(lingua_graph, debug=False)
                    
                    # 遍历所有节点
                    for node in word_lingua_graph.nodes():
                        parents = filter_parents(list(word_lingua_graph.parents(node)))
                        children = filter_children(list(word_lingua_graph.children(node)))
                        
                        in_degree = len(parents)
                        if in_degree > max_in_degree:
                            max_in_degree = in_degree
                        
                        # 统计 p(child_edge_label | node_pos) - 只要有子节点就统计
                        if children:
                            node_pos = getattr(node, "pos", None) or "UNK"
                            for child, child_edge in children:
                                child_label = child_edge.label
                                pos_child_counts[(node_pos, child_label)] += 1
                                pos_counts[node_pos] += 1
                                all_pos_child_labels.add(child_label)
                        
                        # 统计 p(child_edge_label | (parent_edge_label, node_pos)) 和 p(parent_edge_label | child_edge_label)
                        if parents and children:
                            node_pos = getattr(node, "pos", None) or "UNK"
                            for parent, parent_edge in parents:
                                parent_label = parent_edge.label
                                parent_label_counts[(parent_label, node_pos)] += len(children)
                                
                                for child, child_edge in children:
                                    child_label = child_edge.label
                                    parent_child_counts[(parent_label, node_pos, child_label)] += 1
                                    all_child_labels.add(child_label)
                                    
                                    # 统计 p(parent_edge_label | child_edge_label)
                                    child_parent_counts[(child_label, parent_label)] += 1
                                    child_label_counts[child_label] += 1
                                    all_child_parent_labels.add(parent_label)
                        
                        # 统计 p(child_edge_label | other_child_edge_label)
                        if len(children) > 1:
                            for i, (child1, edge1) in enumerate(children):
                                other_label = edge1.label
                                all_sibling_child_labels.add(other_label)
                                for j, (child2, edge2) in enumerate(children):
                                    if i != j:
                                        child_label = edge2.label
                                        sibling_child_counts[(other_label, child_label)] += 1
                                        all_sibling_child_labels.add(child_label)
                            
                            for i, (child1, edge1) in enumerate(children):
                                other_label = edge1.label
                                other_child_label_counts[other_label] += (len(children) - 1)
                                        
                except Exception as e:
                    error_samples += 1
                    if verbose:
                        print(f"Error processing sample {idx}: {e}")
                        traceback.print_exc()
                    continue
        
        if verbose:
            print(f"\nTotal samples: {total_samples}")
            print(f"Error samples: {error_samples}")
            print(f"Successfully processed: {total_samples - error_samples}")
            print("\nComputing conditional probabilities...")
        
        # 计算条件概率（使用 add-one smoothing）
        self.vocab_size_parent_child = len(all_child_labels)
        self.vocab_size_sibling_child = len(all_sibling_child_labels)
        self.vocab_size_pos_child = len(all_pos_child_labels)
        self.vocab_size_child_parent = len(all_child_parent_labels)
        
        if verbose:
            print(f"Vocabulary size for p(child|parent,pos): {self.vocab_size_parent_child}")
            print(f"Vocabulary size for p(child|sibling): {self.vocab_size_sibling_child}")
            print(f"Vocabulary size for p(child|pos): {self.vocab_size_pos_child}")
            print(f"Vocabulary size for p(parent|child): {self.vocab_size_child_parent}")
        
        # p(child_edge_label | (parent_edge_label, node_pos))
        self.p_child_given_parent = {}
        for (parent_label, node_pos, child_label), count in parent_child_counts.items():
            denominator = parent_label_counts.get((parent_label, node_pos), 0)
            prob = (count + 1) / (denominator + self.vocab_size_parent_child)
            self.p_child_given_parent[f"{parent_label}|{node_pos}|{child_label}"] = prob
        
        # p(child_edge_label | other_child_edge_label)
        self.p_child_given_sibling = {}
        for (other_label, child_label), count in sibling_child_counts.items():
            denominator = other_child_label_counts.get(other_label, 0)
            prob = (count + 1) / (denominator + self.vocab_size_sibling_child)
            self.p_child_given_sibling[f"{other_label}|{child_label}"] = prob
        
        # p(child_edge_label | node_pos)
        self.p_child_given_pos = {}
        for (node_pos, child_label), count in pos_child_counts.items():
            denominator = pos_counts.get(node_pos, 0)
            prob = (count + 1) / (denominator + self.vocab_size_pos_child)
            self.p_child_given_pos[f"{node_pos}|{child_label}"] = prob
        
        # p(parent_edge_label | child_edge_label)
        self.p_parent_given_child = {}
        for (child_label, parent_label), count in child_parent_counts.items():
            denominator = child_label_counts.get(child_label, 0)
            prob = (count + 1) / (denominator + self.vocab_size_child_parent)
            self.p_parent_given_child[f"{child_label}|{parent_label}"] = prob
        
        # 保存 smoothing 信息
        self.parent_label_counts = {
            f"{parent_label}|{node_pos}": count
            for (parent_label, node_pos), count in parent_label_counts.items()
        }
        self.other_child_label_counts = dict(other_child_label_counts)
        self.pos_counts = dict(pos_counts)
        self.child_label_counts = dict(child_label_counts)
        
        # 保存统计信息
        self.statistics = {
            "total_samples": total_samples,
            "error_samples": error_samples,
            "successful_samples": total_samples - error_samples,
            "total_parent_child_pairs": sum(parent_child_counts.values()),
            "total_sibling_child_pairs": sum(sibling_child_counts.values()),
            "total_pos_child_pairs": sum(pos_child_counts.values()),
            "total_child_parent_pairs": sum(child_parent_counts.values()),
            "max_in_degree": max_in_degree,
        }
        
        if verbose:
            self._print_statistics()
    
    def compute_statistics_from_generator(
        self, 
        lingua_graphs: Iterable[GPGraph], 
        remove_upper_lower: bool = False,
        verbose: bool = True
    ):
        """
        从 lingua-graph 生成器计算统计信息。
        
        Args:
            lingua_graphs: GPGraph 对象的生成器或可迭代对象
            remove_upper_lower: 是否在统计前移除 @upper 和 @lower 边
            verbose: 是否打印详细信息
        """
        # 统计计数
        parent_child_counts = defaultdict(int)
        sibling_child_counts = defaultdict(int)
        pos_child_counts = defaultdict(int)  # (node_pos, child_label) -> count
        child_parent_counts = defaultdict(int)  # (child_label, parent_label) -> count
        parent_label_counts = defaultdict(int)  # (parent_label, node_pos) -> count
        other_child_label_counts = defaultdict(int)  # other_child_label -> count
        pos_counts = defaultdict(int)  # node_pos -> count
        child_label_counts = defaultdict(int)  # child_label -> count
        
        # 收集所有可能的 child 标签，用于 add-one smoothing
        all_child_labels = set()
        all_sibling_child_labels = set()
        all_pos_child_labels = set()
        all_child_parent_labels = set()
        
        total_samples = 0
        error_samples = 0
        max_in_degree = 0
        
        def filter_children(children):
            """过滤子节点，如果需要移除 @upper/@lower 边"""
            if not remove_upper_lower:
                return children
            return [(child, edge) for child, edge in children 
                    if edge.label not in self.UPPER_LOWER_EDGES]
        
        def filter_parents(parents):
            """过滤父节点，如果需要移除 @upper/@lower 边"""
            if not remove_upper_lower:
                return parents
            return [(parent, edge) for parent, edge in parents 
                    if edge.label not in self.UPPER_LOWER_EDGES]
        
        for idx, lingua_graph in enumerate(lingua_graphs):
            total_samples += 1
            if verbose and (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} samples...")
            
            try:
                # 转换为 word-lingua
                word_lingua_graph, _, _ = lingua2wordlingua(lingua_graph, debug=False)
                
                # 遍历所有节点
                for node in word_lingua_graph.nodes():
                    parents = filter_parents(list(word_lingua_graph.parents(node)))
                    children = filter_children(list(word_lingua_graph.children(node)))
                    
                    in_degree = len(parents)
                    if in_degree > max_in_degree:
                        max_in_degree = in_degree
                    
                    # 统计 p(child_edge_label | node_pos) - 只要有子节点就统计
                    if children:
                        node_pos = getattr(node, "pos", None) or "UNK"
                        for child, child_edge in children:
                            child_label = child_edge.label
                            pos_child_counts[(node_pos, child_label)] += 1
                            pos_counts[node_pos] += 1
                            all_pos_child_labels.add(child_label)
                    
                    # 统计 p(child_edge_label | (parent_edge_label, node_pos)) 和 p(parent_edge_label | child_edge_label)
                    if parents and children:
                        node_pos = getattr(node, "pos", None) or "UNK"
                        for parent, parent_edge in parents:
                            parent_label = parent_edge.label
                            parent_label_counts[(parent_label, node_pos)] += len(children)
                            
                            for child, child_edge in children:
                                child_label = child_edge.label
                                parent_child_counts[(parent_label, node_pos, child_label)] += 1
                                all_child_labels.add(child_label)
                                
                                # 统计 p(parent_edge_label | child_edge_label)
                                child_parent_counts[(child_label, parent_label)] += 1
                                child_label_counts[child_label] += 1
                                all_child_parent_labels.add(parent_label)
                    
                    # 统计 p(child_edge_label | other_child_edge_label)
                    if len(children) > 1:
                        for i, (child1, edge1) in enumerate(children):
                            other_label = edge1.label
                            all_sibling_child_labels.add(other_label)
                            for j, (child2, edge2) in enumerate(children):
                                if i != j:
                                    child_label = edge2.label
                                    sibling_child_counts[(other_label, child_label)] += 1
                                    all_sibling_child_labels.add(child_label)
                        
                        for i, (child1, edge1) in enumerate(children):
                            other_label = edge1.label
                            other_child_label_counts[other_label] += (len(children) - 1)
                                    
            except Exception as e:
                error_samples += 1
                if verbose:
                    print(f"Error processing sample {idx}: {e}")
                    traceback.print_exc()
                continue
        
        if verbose:
            print(f"\nTotal samples: {total_samples}")
            print(f"Error samples: {error_samples}")
            print(f"Successfully processed: {total_samples - error_samples}")
            if remove_upper_lower:
                print("Note: @upper and @lower edges were removed before computing statistics")
            print("\nComputing conditional probabilities...")
        
        # 计算条件概率（使用 add-one smoothing）
        self.vocab_size_parent_child = len(all_child_labels)
        self.vocab_size_sibling_child = len(all_sibling_child_labels)
        self.vocab_size_pos_child = len(all_pos_child_labels)
        self.vocab_size_child_parent = len(all_child_parent_labels)
        
        if verbose:
            print(f"Vocabulary size for p(child|parent,pos): {self.vocab_size_parent_child}")
            print(f"Vocabulary size for p(child|sibling): {self.vocab_size_sibling_child}")
            print(f"Vocabulary size for p(child|pos): {self.vocab_size_pos_child}")
            print(f"Vocabulary size for p(parent|child): {self.vocab_size_child_parent}")
        
        # p(child_edge_label | (parent_edge_label, node_pos))
        self.p_child_given_parent = {}
        for (parent_label, node_pos, child_label), count in parent_child_counts.items():
            denominator = parent_label_counts.get((parent_label, node_pos), 0)
            prob = (count + 1) / (denominator + self.vocab_size_parent_child)
            self.p_child_given_parent[f"{parent_label}|{node_pos}|{child_label}"] = prob
        
        # p(child_edge_label | other_child_edge_label)
        self.p_child_given_sibling = {}
        for (other_label, child_label), count in sibling_child_counts.items():
            denominator = other_child_label_counts.get(other_label, 0)
            prob = (count + 1) / (denominator + self.vocab_size_sibling_child)
            self.p_child_given_sibling[f"{other_label}|{child_label}"] = prob
        
        # p(child_edge_label | node_pos)
        self.p_child_given_pos = {}
        for (node_pos, child_label), count in pos_child_counts.items():
            denominator = pos_counts.get(node_pos, 0)
            prob = (count + 1) / (denominator + self.vocab_size_pos_child)
            self.p_child_given_pos[f"{node_pos}|{child_label}"] = prob
        
        # p(parent_edge_label | child_edge_label)
        self.p_parent_given_child = {}
        for (child_label, parent_label), count in child_parent_counts.items():
            denominator = child_label_counts.get(child_label, 0)
            prob = (count + 1) / (denominator + self.vocab_size_child_parent)
            self.p_parent_given_child[f"{child_label}|{parent_label}"] = prob
        
        # 保存 smoothing 信息
        self.parent_label_counts = {
            f"{parent_label}|{node_pos}": count
            for (parent_label, node_pos), count in parent_label_counts.items()
        }
        self.other_child_label_counts = dict(other_child_label_counts)
        self.pos_counts = dict(pos_counts)
        self.child_label_counts = dict(child_label_counts)
        
        # 保存统计信息
        self.statistics = {
            "total_samples": total_samples,
            "error_samples": error_samples,
            "successful_samples": total_samples - error_samples,
            "total_parent_child_pairs": sum(parent_child_counts.values()),
            "total_sibling_child_pairs": sum(sibling_child_counts.values()),
            "total_pos_child_pairs": sum(pos_child_counts.values()),
            "total_child_parent_pairs": sum(child_parent_counts.values()),
            "max_in_degree": max_in_degree,
            "remove_upper_lower": remove_upper_lower,
        }
        
        if verbose:
            self._print_statistics()
    
    def _print_statistics(self):
        """打印统计信息"""
        print(f"\nStatistics:")
        print(f"  p(child|parent,pos) entries: {len(self.p_child_given_parent)}")
        print(f"  p(child|sibling) entries: {len(self.p_child_given_sibling)}")
        print(f"  p(child|pos) entries: {len(self.p_child_given_pos)}")
        print(f"  p(parent|child) entries: {len(self.p_parent_given_child)}")
        print(f"  Total parent-child pairs: {self.statistics.get('total_parent_child_pairs', 0)}")
        print(f"  Total sibling-child pairs: {self.statistics.get('total_sibling_child_pairs', 0)}")
        print(f"  Total pos-child pairs: {self.statistics.get('total_pos_child_pairs', 0)}")
        print(f"  Total child-parent pairs: {self.statistics.get('total_child_parent_pairs', 0)}")
        print(f"  Max in-degree: {self.statistics.get('max_in_degree', 0)}")
        
        print("\nSample p(child|parent,pos) probabilities (top 10):")
        sorted_parent_child = sorted(self.p_child_given_parent.items(), key=lambda x: x[1], reverse=True)[:10]
        for key, prob in sorted_parent_child:
            print(f"  {key}: {prob:.4f}")
        
        print("\nSample p(child|sibling) probabilities (top 10):")
        sorted_sibling_child = sorted(self.p_child_given_sibling.items(), key=lambda x: x[1], reverse=True)[:10]
        for key, prob in sorted_sibling_child:
            print(f"  {key}: {prob:.4f}")
        
        print("\nSample p(child|pos) probabilities (top 10):")
        sorted_pos_child = sorted(self.p_child_given_pos.items(), key=lambda x: x[1], reverse=True)[:10]
        for key, prob in sorted_pos_child:
            print(f"  {key}: {prob:.4f}")
        
        print("\nSample p(parent|child) probabilities (top 10):")
        sorted_child_parent = sorted(self.p_parent_given_child.items(), key=lambda x: x[1], reverse=True)[:10]
        for key, prob in sorted_child_parent:
            print(f"  {key}: {prob:.4f}")
    
    def save(self, output_file: str, verbose: bool = True):
        """
        保存统计信息到文件。
        
        Args:
            output_file: 输出的 JSON 文件路径
            verbose: 是否打印信息
        """
        result = {
            "p_child_given_parent": self.p_child_given_parent,
            "p_child_given_sibling": self.p_child_given_sibling,
            "p_child_given_pos": self.p_child_given_pos,
            "p_parent_given_child": self.p_parent_given_child,
            "smoothing_info": {
                "vocab_size_parent_child": self.vocab_size_parent_child,
                "vocab_size_sibling_child": self.vocab_size_sibling_child,
                "vocab_size_pos_child": self.vocab_size_pos_child,
                "vocab_size_child_parent": self.vocab_size_child_parent,
                "parent_label_counts": self.parent_label_counts,
                "other_child_label_counts": self.other_child_label_counts,
                "pos_counts": self.pos_counts,
                "child_label_counts": self.child_label_counts,
            },
            "statistics": self.statistics,
        }
        
        if verbose:
            print(f"\nSaving results to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if verbose:
            print("Saved successfully!")
    
    @classmethod
    def load(cls, input_file: str, verbose: bool = True) -> "WordLinguaStatistics":
        """
        从文件加载统计信息。
        
        Args:
            input_file: 输入的 JSON 文件路径
            verbose: 是否打印信息
        
        Returns:
            WordLinguaStatistics 实例
        """
        if verbose:
            print(f"Loading statistics from {input_file}...")
        
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        stats = cls()
        stats.p_child_given_parent = data.get("p_child_given_parent", {})
        stats.p_child_given_sibling = data.get("p_child_given_sibling", {})
        stats.p_child_given_pos = data.get("p_child_given_pos", {})
        stats.p_parent_given_child = data.get("p_parent_given_child", {})
        
        smoothing_info = data.get("smoothing_info", {})
        stats.vocab_size_parent_child = smoothing_info.get("vocab_size_parent_child", 0)
        stats.vocab_size_sibling_child = smoothing_info.get("vocab_size_sibling_child", 0)
        stats.vocab_size_pos_child = smoothing_info.get("vocab_size_pos_child", 0)
        stats.vocab_size_child_parent = smoothing_info.get("vocab_size_child_parent", 0)
        stats.parent_label_counts = smoothing_info.get("parent_label_counts", {})
        stats.other_child_label_counts = smoothing_info.get("other_child_label_counts", {})
        stats.pos_counts = smoothing_info.get("pos_counts", {})
        stats.child_label_counts = smoothing_info.get("child_label_counts", {})
        
        stats.statistics = data.get("statistics", {})
        
        if verbose:
            print("Loaded successfully!")
            stats._print_statistics()
        
        return stats
    
    def get_prob_child_given_parent(
        self, 
        parent_label: str, 
        node_pos: str, 
        child_label: str
    ) -> float:
        """
        计算 p(child_label | parent_label, node_pos) 的概率。
        
        如果条目已观察到，返回存储的概率（已应用 add-one smoothing）。
        如果条目未观察到，使用 add-one smoothing 计算：prob = 1 / (denominator + vocab_size)
        
        Args:
            parent_label: 父边标签
            node_pos: 节点 POS
            child_label: 子边标签
        
        Returns:
            条件概率
        """
        key = f"{parent_label}|{node_pos}|{child_label}"
        
        # 如果已观察到，直接返回存储的概率
        if key in self.p_child_given_parent:
            return self.p_child_given_parent[key]
        
        # 如果未观察到，使用 add-one smoothing 计算
        parent_key = f"{parent_label}|{node_pos}"
        denominator = self.parent_label_counts.get(parent_key, 0)
        prob = 1.0 / (denominator + self.vocab_size_parent_child)
        return prob
    
    def get_prob_child_given_sibling(
        self, 
        other_label: str, 
        child_label: str
    ) -> float:
        """
        计算 p(child_label | other_label) 的概率。
        
        如果条目已观察到，返回存储的概率（已应用 add-one smoothing）。
        如果条目未观察到，使用 add-one smoothing 计算：prob = 1 / (denominator + vocab_size)
        
        Args:
            other_label: 另一个子边标签
            child_label: 子边标签
        
        Returns:
            条件概率
        """
        key = f"{other_label}|{child_label}"
        
        # 如果已观察到，直接返回存储的概率
        if key in self.p_child_given_sibling:
            return self.p_child_given_sibling[key]
        
        # 如果未观察到，使用 add-one smoothing 计算
        denominator = self.other_child_label_counts.get(other_label, 0)
        prob = 1.0 / (denominator + self.vocab_size_sibling_child)
        return prob
    
    def get_prob_child_given_pos(
        self, 
        node_pos: str, 
        child_label: str
    ) -> float:
        """
        计算 p(child_label | node_pos) 的概率。
        
        如果条目已观察到，返回存储的概率（已应用 add-one smoothing）。
        如果条目未观察到，使用 add-one smoothing 计算：prob = 1 / (denominator + vocab_size)
        
        Args:
            node_pos: 节点 POS
            child_label: 子边标签
        
        Returns:
            条件概率
        """
        key = f"{node_pos}|{child_label}"
        
        # 如果已观察到，直接返回存储的概率
        if key in self.p_child_given_pos:
            return self.p_child_given_pos[key]
        
        # 如果未观察到，使用 add-one smoothing 计算
        denominator = self.pos_counts.get(node_pos, 0)
        prob = 1.0 / (denominator + self.vocab_size_pos_child)
        return prob
    
    def get_prob_parent_given_child(
        self, 
        child_label: str, 
        parent_label: str
    ) -> float:
        """
        计算 p(parent_label | child_label) 的概率。
        
        如果条目已观察到，返回存储的概率（已应用 add-one smoothing）。
        如果条目未观察到，使用 add-one smoothing 计算：prob = 1 / (denominator + vocab_size)
        
        Args:
            child_label: 子边标签
            parent_label: 父边标签
        
        Returns:
            条件概率
        """
        key = f"{child_label}|{parent_label}"
        
        # 如果已观察到，直接返回存储的概率
        if key in self.p_parent_given_child:
            return self.p_parent_given_child[key]
        
        # 如果未观察到，使用 add-one smoothing 计算
        denominator = self.child_label_counts.get(child_label, 0)
        prob = 1.0 / (denominator + self.vocab_size_child_parent)
        return prob

