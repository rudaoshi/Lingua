from typing import Union, List

from lingua.structure.gpgraph import GPGraph, GPGAuxNode, TextGPGraph, GPGTextNode, GPGPhraseNode
import re


from lingua.structure.gpgraph import get_word_positions

# def span2text(words, sent, with_placeholder=True):

#     assert isinstance(words, (list, tuple))
#     w_list = [w if isinstance(w, str) else sent[w] for w in words]
#     if not with_placeholder:
#         w_list = [x for x in w_list if not re.match('\{\d+\}', x)]
#     return " ".join(w_list)

def _canonical_spans(spans):
    canonical = []
    for span in spans:
        if isinstance(span, tuple):
            canonical.append(span)
        else:
            canonical.append(tuple(span))
    return tuple(canonical)


def get_node_info(node, graph: GPGraph, *, with_type: bool = True, with_spans: bool = False):
    if isinstance(node, GPGAuxNode):
        node_text = node.label
    elif isinstance(node, GPGTextNode):
        node_text = node.text
    else:
        assert isinstance(node, GPGPhraseNode)
        node_text = graph.node_text(node)

    node_info = {"text": node_text}
    if with_type:
        node_info["type"] = getattr(node, "pos", None)
    if with_spans:
        node_info["spans"] = _canonical_spans(node.spans) if isinstance(node, GPGPhraseNode) else None

    return node_info

def remove_placeholders(text):
    """移除 {1}, {2} 等占位符并清理字符串"""
    return re.sub(r'\{\d+\}', '', text).strip()

def calculate_node_match_score(pred_node, true_node):
    """
    计算预测节点和真实节点之间的匹配度分数。
    分数 = 匹配的单词共同组成的字符串长度 / 清理后的真实节点字符串长度。
    """
    # 清理字符串，移除占位符和多余空格
    cleaned_pred = remove_placeholders(pred_node)
    cleaned_true = remove_placeholders(true_node)
    
    # 如果真实节点为空，无法计算分数
    if not cleaned_true:
        return 0.0

    # 如果清理后完全相同，则匹配度为1.0（最高分）
    if cleaned_pred == cleaned_true:
        return 1.0
        
    # 将节点字符串拆分为单词集合
    words_pred = set(cleaned_pred.split())
    words_true = set(cleaned_true.split())
    
    # 找到共同的单词
    intersecting_words = words_pred & words_true
    
    # 如果没有共同的单词，则匹配度为0
    if not intersecting_words:
        return 0.0
    
    # 计算匹配部分的长度。通过将共同单词排序后用空格连接，来模拟它们组成的字符串。
    # 排序是为了保证结果的一致性。
    matched_len = len(" ".join(sorted(list(intersecting_words))))
    
    # 计算真实节点的总长度
    true_len = len(cleaned_true)
    
    # 返回匹配度分数
    return matched_len / true_len

def loose_matching_edges(pred_edges, true_edges):
    """
    比较预测边和真实边，返回匹配的边数。
    匹配逻辑：对于每个预测边，选择与之匹配度分数最高的真实边进行匹配。
    """
    matched_pred = []
    unmatched_pred = []
    unmatched_true = true_edges.copy()  # 初始化为所有真实边，后续移除匹配的

    for pedge in pred_edges:
        p_node1, p_label, p_node2 = pedge
        
        best_match_candidate = None
        highest_score = 0.0

        # 遍历所有尚未匹配的真实边，找到分数最高的一个
        for tedge in unmatched_true:
            t_node1, t_label, t_node2 = tedge
            
            # 1. 边的标签必须完全相同
            # if p_label != t_label:
            #     continue
            
            # 2. 计算两个节点的匹配分数
            score1 = calculate_node_match_score(p_node1, t_node1)
            score2 = calculate_node_match_score(p_node2, t_node2)
            
            # 3. 两个节点都必须有部分匹配（分数 > 0）才被认为是有效候选
            if score1 > 0 and score2 > 0:
                # 边的总匹配分数定义为两个节点分数的平均值
                combined_score = (score1 + score2) / 2.0
                
                # 如果当前分数更高，则更新最佳匹配候选
                if combined_score > highest_score:
                    highest_score = combined_score
                    best_match_candidate = tedge

        # 如果找到了最佳匹配（即最高分>0）
        if best_match_candidate:
            matched_pred.append(pedge)
            # 从待匹配池中移除已匹配的真实边，防止重复匹配
            unmatched_true.remove(best_match_candidate)
        else:
            unmatched_pred.append(pedge)

    # 返回最终匹配到的边的数量
    return len(matched_pred)


def freq_similarity(list1, list2):

    import numpy as np
    all_elements = set(list1) | set(list2)
    if len(all_elements) == 0:
        return 1.0
    elif len(list1) == 0 or len(list2) == 0:
        return 0.0
        
    freq_vec_1 = [list1.count(element)/len(list1) for element in all_elements]
    freq_vec_2 = [list2.count(element)/len(list2) for element in all_elements]

    return np.dot(freq_vec_1, freq_vec_2)/np.linalg.norm(freq_vec_1)/np.linalg.norm(freq_vec_2)

def match_elements(elements1, elements2, match_func):
    import numpy as np
    matching_matrix = np.zeros((len(elements1), len(elements2)))
    for i in range(len(elements1)):
        for j in range(len(elements2)):
            matching_matrix[i, j] = match_func(elements1[i], elements2[j])

    matched_elements = []
    while np.any(matching_matrix > 0):
        max_idx = np.unravel_index(np.argmax(matching_matrix), matching_matrix.shape)
        i, j = max_idx
        matching_matrix[i, :] = 0
        matching_matrix[:, j] = 0
        matched_elements.append((i, j))

    return matched_elements

def node_exact_match_func(node1_info, node2_info):
    assert tuple(node1_info.keys()) == tuple(node2_info.keys())

    return all(node1_info[key] == node2_info[key] for key in node1_info.keys())

def node_loose_match_func(node1_info, node2_info):
    if "type" in node1_info or "type" in node2_info:
        if node1_info.get("type") != node2_info.get("type"):
            return False

    node1_words = [x.strip() for x in node1_info["text"].split() if x.strip()]
    node2_words = [x.strip() for x in node2_info["text"].split() if x.strip()]
    return freq_similarity(node1_words, node2_words) > 0 

def edge_exact_match_func(edge1_info, edge2_info):
    node1_info_1, edge_info_1, node2_info_1 = edge1_info
    node1_info_2, edge_info_2, node2_info_2 = edge2_info
    return node_exact_match_func(node1_info_1, node1_info_2) and \
           node_exact_match_func(node2_info_1, node2_info_2) and \
           edge_info_1 == edge_info_2

def edge_loose_match_func(edge1_info, edge2_info):
    node1_info_1, edge_info_1, node2_info_1 = edge1_info
    node1_info_2, edge_info_2, node2_info_2 = edge2_info
    return node_loose_match_func(node1_info_1, node1_info_2) and \
           node_loose_match_func(node2_info_1, node2_info_2) and \
           edge_info_1 == edge_info_2


def graph_match_metric(pred_graph: Union[GPGraph, TextGPGraph], 
                       truth_graph: Union[GPGraph, TextGPGraph],
                       edge_match: str = "exact",
                       with_type=True,
                       with_spans: bool = False):
    """

    :param predict:
    :param truth:
    :return:
    """

    assert edge_match in ["exact", "loose"]


    pred_node2info = {
        n: get_node_info(n, pred_graph, with_type=with_type, with_spans=with_spans)
        for n in pred_graph.nodes()
    }
            
    truth_node2info = {
        n: get_node_info(n, truth_graph, with_type=with_type, with_spans=with_spans)
        for n in truth_graph.nodes()
    }

    pred_nodes = list(pred_node2info.values())
    true_nodes = list(truth_node2info.values())

    matched_nodes = match_elements(pred_nodes, true_nodes, node_exact_match_func)
    node_match_num = len(matched_nodes)

    node_true_num = len(true_nodes)
    node_pred_num = len(pred_nodes)

    matched_pred_node_idx = set([matched_nodes[i][0] for i in range(len(matched_nodes))])
    matched_true_node_idx = set([matched_nodes[i][1] for i in range(len(matched_nodes))])

    unmatched_pred_nodes = [pred_nodes[i] for i in range(len(pred_nodes)) if i not in matched_pred_node_idx]
    unmatched_true_nodes = [true_nodes[i] for i in range(len(true_nodes)) if i not in matched_true_node_idx]

    pred_edges = [(pred_node2info[n1], edge.label, pred_node2info[n2])
                  for n1, edge, n2 in pred_graph.edges()]
    true_edges = [(truth_node2info[n1], edge.label, truth_node2info[n2])
                  for n1, edge, n2 in truth_graph.edges()]

    matched_edges = match_elements(pred_edges, true_edges, edge_exact_match_func if edge_match == "exact" else edge_loose_match_func)
    edge_match_num = len(matched_edges)

    matched_pred_edge_idx = set([matched_edges[i][0] for i in range(len(matched_edges))])
    matched_true_edge_idx = set([matched_edges[i][1] for i in range(len(matched_edges))])

    unmatched_pred_edges = [pred_edges[i] for i in range(len(pred_edges)) if i not in matched_pred_edge_idx]
    unmatched_true_edges = [true_edges[i] for i in range(len(true_edges)) if i not in matched_true_edge_idx]

    edge_true_num = len(true_edges)
    edge_pred_num = len(pred_edges)

    exact_same = node_match_num == node_true_num == node_pred_num and \
                 edge_match_num == edge_true_num == edge_pred_num

    node_f1 = 2 * node_match_num / (node_true_num + node_pred_num) if node_true_num + node_pred_num > 0 else 1
    edge_f1 = 2 * edge_match_num / (edge_true_num + edge_pred_num) if edge_true_num + edge_pred_num > 0 else 1


    return {"pred_node_num": node_pred_num,
            "true_node_num": node_true_num,
            "matched_node_num": node_match_num,
            "node_f1": node_f1,
            "pred_edge_num": edge_pred_num,
            "true_edge_num": edge_true_num,
            "matched_edge_num": edge_match_num,
            "edge_f1": edge_f1,
            "unmatched_pred_nodes": unmatched_pred_nodes,
            "unmatched_true_nodes": unmatched_true_nodes,
            "unmatched_pred_edges": unmatched_pred_edges,
            "unmatched_true_edges": unmatched_true_edges,
            "exact_same": exact_same}
