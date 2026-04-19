from lingua.concept.standard import LinguaGraphNodePoses

from lingua.structure.gpgraph import GPGraph, GPGAuxNode, GPGPhraseNode, GPGNode

priority_map = {

    "AttributePredicator": 80,
    "AppositionalPredicator": 80,
    "FactualPredicator": 80,
    "LogicalPredicator": 80,
    "ReferentialPredicator": 80,
    "ConjunctionalFunctor": 80,
    "ExpressionFunctor": 40,
    "GeneralFunctor": 40,
    "ListFunctor": 40,
    "ModificationalFunctor": 80,
    "DeterminerConstant": 80,
    "InterjectionConstant": 80              ,
    "ModificationalConstant": 80,
    "NominalConstant": 80,
    "OtherConstant": 80,
    "PunctuationalConstant": 80,
    "SymbolConstant": 80
}


def min_node_distance(n1: GPGNode, n2: GPGNode):
    if isinstance(n1, GPGAuxNode) or isinstance(n2, GPGAuxNode):
        return 10000

    assert isinstance(n1, GPGPhraseNode) and isinstance(n2, GPGPhraseNode)

    import numpy as np

    n1_words = np.array(list(n1.words(with_aux=False)))
    n2_words = np.array(list(n2.words(with_aux=False)))

    from scipy.spatial.distance import cdist

    return np.min(cdist(n1_words[..., np.newaxis], n2_words[..., np.newaxis])) # word index difference between n1 and n2



def get_modification_priority(graph: GPGraph, node, modification_phrases):

    base_priority = 20

    node_words = list(node.words(with_aux=False))

    min_node_pos = min(node_words)
    
    phrase_scores = []
    for phrase in modification_phrases:
        min_modification_pos = 10000
        min_dist = 10000
        for c in phrase:
            min_dist = min(min_dist, min_node_distance(node, c))
            if isinstance(c, GPGPhraseNode):
                c_words = list(c.words(with_aux=False))
                min_modification_pos = min(min_modification_pos, min(c_words))
            # 如果修饰词在中心词之前，给予奖励；否则不奖励
            positional_bonus = 100 if min_modification_pos < min_node_pos else 0
            # 如果距离超过1，每超过1个 token 扣除一定分数
            distance_penalty_factor = 5
            distance_penalty = (min_dist - 1) * distance_penalty_factor if min_dist > 1 else 0

            effective_score = base_priority + positional_bonus - distance_penalty
        phrase_scores.append(effective_score)
        
    return phrase_scores
    
