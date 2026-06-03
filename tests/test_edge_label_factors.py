import torch

from lingua.learn.wordgraph.decoding.inference import (
    build_edge_factor_targets_for_labels,
    combine_rel_logits_with_edge_factor_logits,
)
from lingua.learn.wordgraph.model.biaffine import BiaffineLayer
from lingua.learn.wordgraph.model.edge_label_factors import (
    NO_FACTOR,
    build_edge_factor_maps,
    decompose_edge_label,
)


def test_decompose_compound_edge_label_shares_atomic_targets():
    factors = decompose_edge_label("@modification_of+pred.arg.1")

    assert factors["component_1"] == "@modification_of"
    assert factors["component_2"] == "pred.arg.1"
    assert factors["semantic"] == "modification"
    assert factors["role"] == "pred.arg.1"
    assert factors["hierarchy"] == "@modification_of"
    assert factors["surface"] == NO_FACTOR
    assert factors["compound"] == "compound"


def test_build_factor_maps_aligns_targets_to_edge_vocab():
    edge_types = [
        "pred.arg.1",
        "@modification_of",
        "@modification_of+pred.arg.1",
        "as:punctuation",
        "func.arg+func.arg",
    ]

    maps = build_edge_factor_maps(edge_types)
    targets = maps.id_targets_by_label(edge_types)
    component_1_ids = {
        value: idx for idx, value in enumerate(maps.factor_vocabs["component_1"])
    }
    component_2_ids = {
        value: idx for idx, value in enumerate(maps.factor_vocabs["component_2"])
    }

    assert targets["component_1"][0] == component_1_ids["pred.arg.1"]
    assert targets["component_1"][2] == component_1_ids["@modification_of"]
    assert targets["component_2"][0] == component_2_ids[NO_FACTOR]
    assert targets["component_2"][2] == component_2_ids["pred.arg.1"]


def test_biaffine_layer_outputs_edge_factor_logits():
    edge_types = ["pred.arg.1", "@modification_of+pred.arg.1", "as:punctuation"]
    maps = build_edge_factor_maps(edge_types)
    layer = BiaffineLayer(
        input_size=8,
        label_num=len(edge_types),
        arc_hidden_size=4,
        rel_hidden_size=5,
        edge_factor_vocabs=maps.factor_vocabs,
    )

    x = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    out = layer(x, mask)

    assert out["arc_logits"].shape == (2, 3, 3)
    assert out["rel_logits"].shape == (2, 3, 3, len(edge_types))
    assert set(out["edge_factor_logits"]) == set(maps.factor_vocabs)
    for name, values in maps.factor_vocabs.items():
        assert out["edge_factor_logits"][name].shape == (2, 3, 3, len(values))
        assert torch.all(out["edge_factor_logits"][name][0, 2, :, :] == -1e9)
        assert torch.all(out["edge_factor_logits"][name][0, :, 2, :] == -1e9)


def test_joint_decode_combines_only_legal_labels():
    edge_types = ["pred.arg.1", "@modification_of+pred.arg.1"]
    id2label = {idx: label for idx, label in enumerate(edge_types)}
    maps = build_edge_factor_maps(edge_types)
    targets = build_edge_factor_targets_for_labels(
        id2label=id2label,
        edge_factor_vocabs=maps.factor_vocabs,
        edge_label_factors=maps.label_factors,
    )

    rel_logits = torch.tensor([[[[2.0, 1.9]]]])
    factor_logits = {}
    for name, values in maps.factor_vocabs.items():
        logits = torch.zeros(1, 1, 1, len(values))
        wanted = targets[name][1]
        logits[..., wanted] = 5.0
        factor_logits[name] = logits

    combined = combine_rel_logits_with_edge_factor_logits(
        rel_logits=rel_logits,
        edge_factor_logits=factor_logits,
        edge_factor_targets_by_label=targets,
        weight=1.0,
    )

    assert rel_logits.argmax(dim=-1).item() == 0
    assert combined.argmax(dim=-1).item() == 1
