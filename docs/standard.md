# Lingua-Graph 标注评估标准规范

版本：`v0.1`  
用途：指导 LLM 判断一个候选标注是否是正确的 Lingua / Lingua-Graph 标注。  
适用对象：句子级 Lingua 标注、Lingua parser 输出、人工标注质检、LLM 自动评估。

---

## 0. 依据与定位

本规范依据 Lingua-Graph 论文中的表示模型、schema、建模原则和示例制定，并参考 OIA / OIA-Graph 的 Predicate-Function-Argument 标注思想。

Lingua-Graph 的核心目标不是抽取孤立三元组，也不是生成普通依存树，而是为 Analytic NLP 提供一个统一的、可解释的、可复用的中间结构层。这个结构层应能表达句子中的实体、事实、属性、修饰、指代、逻辑关系和嵌套结构，并使下游任务能够通过选择节点自身或节点 descendant subgraph 来恢复任务所需的 analytic units。

因此，一个候选标注即使符合 OIA 风格，也不一定符合 Lingua-Graph；如果它过度扁平化，导致复杂实体、嵌套事实、属性结构或修饰结构不能作为局部子图恢复，则应视为 Lingua 标注不足。

### 0.1 核心图语义：节点与 descendant subgraph

Lingua-Graph 的基本语义单位不是孤立 token，也不是单个 node label，而是以某个节点为局部 root 的 descendant subgraph。评估器在解释一个节点引用时，应同时考虑：

1. `SELF(v)`：节点 `v` 自身承载的词面、类型或 symbol 语义。
2. `DESC(v)`：从 `v` 沿图边可达的完整局部子图，表示以 `v` 为 root 的 analytic unit。

当某条边指向节点 `v` 时，默认应先判断 `DESC(v)` 是否能恢复该边所需的实体、事实、修饰结构或逻辑结构，而不是只看 `SELF(v)` 的词面或 node type。只有当某类边或规则明确要求节点自身语义时，才应把判断限制在 `SELF(v)`。

这也是 Lingua 紧凑表示的核心：一个修饰节点、标点节点、`Ref`、`Expression`、`Modification` 或其他 wrapper node 可以作为另一个结构的论元或被修饰对象，只要它的 `DESC(v)` 能恢复对应 analytic unit。例如，一个介词修饰节点已经包装了 `blow on the weekend` 后，另一个修饰节点可以把它作为 `variable`，表示继续修饰“已经带有前一层修饰的事件”。

因此，判定 wrapper argument、链式修饰、标点 root、`as:` 反向边和冗余结构时，必须以 `DESC(v)` 的可恢复性为准。若 `DESC(v)` 无法恢复应有的 head、scope、核心论元或内部结构，才应判为结构错误；若只是 `SELF(v)` 看起来像 wrapper，但 `DESC(v)` 清楚可恢复目标语义，则属于合法紧凑结构。

但 `DESC(v)` 可恢复不等于任意链式嵌套都正确。多重修饰仍必须遵守语义亲疏和 scope 大小：更贴近 head/event、与 head 结合更紧的修饰应先进入内层；scope 更大、更疏远的修饰应作为外层。评估器应同时检查 `DESC(v)` 的可恢复性和修饰组合顺序。若链式结构虽然能恢复核心事件，但把近层修饰放到外层、把远层修饰放到内层，导致 modifier scope 或组合语义不符合原句，应判为修饰层级错误。

---

## 1. 总体判定原则

一个正确的 Lingua-Graph 标注应同时满足以下要求：

1. **形式合法**：图必须是 single-rooted DAG，即单根有向无环图。
2. **token partition 完整**：句子中的每个 token 必须被且只被一个 lexical node 覆盖；辅助 symbol node 可以没有 token。
3. **schema 合规**：节点类型、辅助 symbol label、边标签必须符合 Lingua-Graph schema，或可明确映射为 schema 中的规范标签。
4. **语义忠实**：不得增添原句没有表达的信息，也不得遗漏主事实、核心论元、实体、属性、指代、逻辑关系等关键语义。
5. **PFA 结构正确**：谓词、函数、论元之间的方向、边标签和论元编号应符合 Predicate-Function-Argument 语义。
6. **层级结构正确**：复杂实体、修饰、相对从句、属性、表达式、逻辑关系应形成清晰的局部子图，而不是全部压平成一个 phrase node。
7. **可恢复性良好**：任务相关的实体、事实、嵌套实体、属性值等应尽量能够通过 `SELF(v)` 或 `DESC(v)` 恢复。
8. **Lingua 特性保留**：应显式建模 `Modification`、`Expression`、`Attribute`、`Ref`、`List`、`Parataxis` 等 Lingua 结构，而不是退化为普通依存边或 OIA 式扁平 `mod`。

---

## 2. 推荐输入格式

评估器建议接收如下 JSON：

```json
{
  "sentence": "the film we saw yesterday is wonderful",
  "tokens": ["the", "film", "we", "saw", "yesterday", "is", "wonderful"],
  "root": "n1",
  "nodes": [
    {
      "id": "n1",
      "label": "is",
      "type": "FactualPredicator",
      "tokens": [5]
    }
  ],
  "edges": [
    {
      "source": "n1",
      "label": "pred.arg.1",
      "target": "n2"
    }
  ]
}
```

字段要求：

| 字段 | 要求 |
|---|---|
| `sentence` | 原句文本。 |
| `tokens` | 分词结果。若未提供，评估器可根据候选标注中的 token span 推断，但应降低置信度。 |
| `root` | 图的唯一根节点 ID。 |
| `nodes` | 节点列表。每个节点应有 `id`、`label`、`type`、`tokens`。 |
| `edges` | 有向边列表。每条边应有 `source`、`label`、`target`。 |

### 2.1 现有 JSONL serialization 兼容格式

已发布训练数据使用紧凑 serialization，也应作为合法输入格式支持：

```json
{
  "sentence": "the film we saw yesterday is wonderful",
  "lingua": {
    "meta": {},
    "words": ["the", " film", " we", " saw", " yesterday", " is", " wonderful"],
    "graph": {
      "nodes": [
        {"spans": [[5, 5]], "type": "FactualPredicator"},
        {"spans": "Modification", "type": "ModificationalFunctor"}
      ],
      "edges": [
        {"start": 0, "label": "pred.arg.1", "end": 1}
      ]
    }
  }
}
```

兼容规则：

1. `lingua.words` 是按 token index 对齐的分词序列，字符串可保留前导空格；token partition 以 index 为准。
2. lexical node 使用 `spans: [[start, end], ...]` 覆盖 token；同一 lexical token 必须且只能被覆盖一次。
3. aux symbol node 使用 `spans: "Modification"`、`"Ref"`、`"Expression"` 等字符串表示，等价于无 token anchor 的 symbol node。
4. `edges.start` 和 `edges.end` 是 `graph.nodes` 中的节点下标，等价于推荐格式中的 `source` / `target` ID。
5. 训练数据可以不显式存储 `root` 字段；若图中唯一入度为 0 的节点存在，则该节点即为 inferred root。
6. 评估器应先将该 serialization 规范化为统一内部图，再执行 schema、DAG、PFA 和语义检查。

---

## 3. Lingua-Graph Schema

### 3.1 辅助 symbol node labels

辅助 symbol node 用于表示非词面或结构性节点。它们通常可以没有 token anchor。

| Symbol label | 用途 |
|---|---|
| `Apposition` | 同位、别名、再描述关系，如 `John, a doctor`。 |
| `Attribute` | 属性-值结构，尤其是隐式属性表达。 |
| `Copula` | 系词性 predication。 |
| `Discourse` | 话语标记、插入语、话语层连接。 |
| `Expression` | 复合表达，如范围、公式、多 token 符号表达。 |
| `List` | 异质元素列表，不预设共享谓词。 |
| `Missing` | 结构中需要但文本未显式出现的省略成分。 |
| `Modification` | 修饰结构，将被修饰对象与修饰体打包。 |
| `Parataxis` | 并列、并置、协调结构。 |
| `Ref` | 指代、照应、相对从句中的 reference。 |
| `Vocative` | 呼语。 |
| `Whether` | 极性疑问或 whether-clause 的结构节点。 |

### 3.2 Node types

| Node type | 用途 |
|---|---|
| `FactualPredicator` | 事实、事件、普通子句谓词。 |
| `AttributePredicator` | 属性谓词，如 `battery capacity`。 |
| `AppositionalPredicator` | 同位谓词。 |
| `LogicalPredicator` | 逻辑、时间、因果、条件等关系谓词，如 `if`、`because`、`therefore`。 |
| `ReferentialPredicator` | 指代结构谓词。 |
| `ConjunctionalFunctor` | 协调、并列函数，如 `and`、`or`。 |
| `ExpressionFunctor` | 复合表达函数，如 `to` in `48 to 49`。 |
| `GeneralFunctor` | 一般函数，常用于 wh-word。 |
| `ListFunctor` | 异质列表函数。 |
| `ModificationalFunctor` | 修饰函数，通常对应 `Modification`。 |
| `DeterminerConstant` | 限定词，如 `the`、`a`、`this`。 |
| `NominalConstant` | 名词、专名、数词等实体性常量。 |
| `ModificationalConstant` | 形容词、副词、名词性前置修饰等。 |
| `InterjectionConstant` | 感叹词。 |
| `PunctuationalConstant` | 标点。 |
| `SymbolConstant` | 数学符号、单位、特殊字符等。 |
| `OtherConstant` | 无法归入上述类别的 fallback。 |

功能词的 node type 应按其在句中的语义功能判断，而不是按词形固定。例如同一个词形 `if`、`that` 在不同上下文中可对应不同类型；评估器应根据它是在连接命题、引导名词性从句、指代实体，还是限定名词来判定。

### 3.3 Edge labels

| Edge label | 用途 |
|---|---|
| `pred.arg.1` | 谓词第 1 论元，常为 subject-like argument。 |
| `pred.arg.2` | 谓词第 2 论元，常为 object-like argument。 |
| `pred.arg.3` | 第 3 论元，如间接宾语。 |
| `pred.arg.4` | 谓词内部第 4 论元，如宾补、结果补语、被谓词选择的补足语等；不用于普通目的、原因、时间、方式状语。 |
| `func.arg` | 函数论元。 |
| `variable` | 被修饰、被约束或被操作的对象。 |
| `body` | 修饰体、约束体或操作体。 |
| `modification` | 修饰关系。 |
| `attribute` | 属性关系。 |
| `appositive` | 同位关系。 |
| `ref` | 指代关系。 |
| `discourse` | 话语关系。 |
| `punctuation` | 标点附着。 |
| `vocative` | 呼语附着。 |
| `index` | 索引、对齐、结构 bookkeeping。 |
| `repeat` | 重复成分或结构规范化中的重复连接。 |
| `nonsense` | 无法归入既有语义边、但为保持训练数据结构或占位关系而保留的 fallback edge。 |

### 3.4 规范化规则

| 写法 | 处理 |
|---|---|
| `var` | 可作为 `variable` 的别名接受，但推荐统一规范化为 `variable`。 |
| `as:X` | 表示边 `X` 的反向关系。例如 `as:ref` 表示实际语义方向为 `ref` 的反向。 |
| `mod` | OIA 风格旧写法；Lingua 中应优先规范化为 `modification` 或显式 `Modification` 结构。简单无歧义场景可接受为轻微不规范。 |
| 词形还原 | 如 `gave -> give`、`was -> be`，可接受，只要 token anchor 保持正确。 |

---

## 4. 形式合法性检查

### 4.1 Token partition

候选标注必须满足：

```text
所有 lexical node 的 token 集合并集 = 全部 token
任意两个 lexical node 的 token 集合不相交
辅助 symbol node 可以 tokens = []
```

判错标准：

| 情况 | 严重程度 |
|---|---|
| 原句 token 被遗漏 | major / fatal |
| 一个 token 被多个 lexical node 覆盖 | major / fatal |
| 辅助 symbol node 没有 token | 正常 |
| 普通句号未标注 | minor / 可忽略 |
| 括号、引号、逗号等承载结构意义但未标注 | major |
| 错误合并复杂短语，导致内部结构不可恢复 | major |

### 4.2 单根 DAG

图必须满足：

1. 有且只有一个 `root`。
2. 从 `root` 沿有向边可达所有非游离节点。
3. 不允许有向环。
4. 不允许多个互不连接的事实图并列存在；若句子中有多个事实，应通过 logical predicate、parataxis、discourse、list 或主从层级连接成单根图。
5. 若自然语义边方向会破坏单根性，应使用 `as:` 反向边。

现有 JSONL 数据中，物理 root 可以是 `PunctuationalConstant` 或其他 wrapper node，只要从该 root 沿有向边可达主语义图，且主事实、实体和特殊结构可通过 `DESC(root)` 恢复。评估主事实时不应仅因 root 是标点而判错，应继续追踪 `as:punctuation` 等 wrapper 边。

判错标准：

| 情况 | 严重程度 |
|---|---|
| 多个 root | fatal |
| 有向环 | fatal |
| 核心事实节点不可达 | fatal |
| 只有标点或无意义节点不可达 | minor |
| 用 `as:` 解决单根方向问题 | 正确 |
| 滥用 `as:` 导致语义方向错误 | major |

### 4.3 Schema 合规

所有节点类型、symbol label、边标签必须在 Lingua schema 中，或可明确映射为 schema 中的规范标签。

`nonsense` 属于合法边标签，但仅表示低语义信息的 fallback / placeholder 关系。评估时不得把它当作高置信语义关系；若它承载了应由 `pred.arg.*`、`modification`、`ref` 等核心边表达的信息，可在语义忠实性、PFA 结构或可读性维度中按上下文轻罚。

判错标准：

| 情况 | 严重程度 |
|---|---|
| 标签拼写轻微差异，如 `FactualPredicate` vs `FactualPredicator` | minor |
| OIA 旧标签可一一映射 | minor |
| 使用完全未知标签且无法解释 | major |
| 关键结构标签错误，如把 `FactualPredicator` 标成 `NominalConstant` | major |
| 边类型错误导致事实结构错误 | major / fatal |

---

## 5. 语义忠实性检查

### 5.1 不得遗漏核心语义

必须覆盖：

1. 主事实或主事件。
2. 谓词的核心论元。
3. 实体内部修饰。
4. 属性-值结构。
5. 指代、相对从句、同位关系。
6. 逻辑关系、时间关系、因果关系、条件关系。
7. 明显的并列、枚举、列表结构。

遗漏判定：

| 遗漏内容 | 严重程度 |
|---|---|
| 主谓词 | fatal |
| 主体、客体等核心论元 | major |
| 关键修饰导致实体边界错误 | major |
| 非关键状语 | minor / major，取决于是否影响事实 |
| 标点 | minor，除非结构性标点 |
| discourse marker | minor / major，取决于是否表达逻辑关系 |

### 5.2 不得添加原句没有的信息

错误示例：

```text
原句：John bought a car.
错误标注：John bought a red car.
```

判错标准：

| 添加内容 | 严重程度 |
|---|---|
| 添加不存在的实体、属性、事实 | major |
| 添加不存在的逻辑关系 | major |
| 推断常识但文本未表达 | major |
| 规范化词形，如 `gave` 标为 `give` | 可接受 |
| 合理补出省略 `Missing` | 可接受，但需结构上有依据 |

---

## 6. 结构建模规范

### 6.1 简单事实

简单事实通常由 `FactualPredicator` 表示。

示例：

```text
he gave me a book
```

推荐结构：

```text
give/FactualPredicator
  pred.arg.1 -> he
  pred.arg.2 -> a book
  pred.arg.3 -> me
```

要求：

1. 谓词节点应覆盖谓词短语。
2. 主语、施事、经历者、被赋属性者等 subject-like argument 通常为 `pred.arg.1`。
3. 除非原句确实没有对应成分、该谓词语义上不需要 `arg1`，或文本省略且无法由结构恢复，所有 predicate node 原则上都应补足 `pred.arg.1`。
4. 对非限定谓词、分词、目的从句、相对从句、补足从句和并列共享谓词，也必须检查 `pred.arg.1`。若主语或施事可由控制、提升、并列共享、被动结构、前文 wrapper 或 `DESC(v)` 恢复，应显式连接到可恢复的节点或 descendant subgraph。
5. 直接宾语通常为 `pred.arg.2`。
6. 间接宾语通常为 `pred.arg.3`。
7. 宾补、结果补语、被谓词选择的补足语等可用 `pred.arg.4`。
8. 复杂论元应是一个节点或一个 descendant subgraph，而不是散乱 token。

在现有 JSONL serialization 中，谓词论元有时会指向 determiner、modifier、punctuation、`Modification`、`Expression`、`Ref` 等 wrapper node。若对应实体或事实可通过 `DESC(v)` 清晰恢复，这属于合法结构变体；只有 descendant subgraph 无法恢复核心论元时，才应判为 PFA 或语义错误。

### 6.1.1 系词、`be + prep` 与 `there be`

为减少历史数据中 copula 结构的合并/拆分不一致，以下结构按统一原则处理：

1. **简单系表结构合并为谓词短语**。`is true`、`was interesting`、`are happy`、`be ready` 等简单 `be + adjective/state complement` 应作为一个 `FactualPredicator`。主语仍以 `pred.arg.1` 显式连接；该简单状态补语不再另作 `pred.arg.2`。例如：

```text
it is true

is true/FactualPredicator
  pred.arg.1 -> it
```

2. **形容词/状态补语内部有副词修饰时，副词修饰必须可恢复**。若简单系表结构中出现修饰补语的副词，如 `very true`、`really interesting`、`fairly basic`、`so happy`，应避免把副词与形容词无结构地压成不可分析片段。推荐做法是将整个 `be + complement` 保持为事实谓词，同时拆出副词对形容词/状态核心的修饰，或用等价结构确保 `DESC` 可恢复副词 scope。例如：

```text
that is fairly basic

is basic/FactualPredicator
  pred.arg.1 -> that
fairly/ModificationalConstant
  as:modification -> is basic
```

若实现中将 `is fairly basic` 合并为一个 `FactualPredicator`，但无法恢复 `fairly` 修饰 `basic` 而非整个外层事实，应视为过度合并。

3. **`be` 直接加介词谓词时合并**。若 `be` 与介词连续构成状态/位置/关系谓词，如 `is in`、`was at`、`is about`、`are against`、`be with`，应合并为一个 `FactualPredicator`，介词宾语作为该合并谓词的论元。例如：

```text
he is in Houston

is in/FactualPredicator
  pred.arg.1 -> he
  pred.arg.2 -> Houston
```

4. **`be` 与介词之间有其他成分时，介词必须单独拆出**。若 `be` 和介词之间插入副词、程度词、否定、补语或其他成分，如 `is clearly in trouble`、`is not in Houston`、`was already at home`、`is all about this`，不得把整段无结构地合并为一个 predicate。应将 `be` 或可恢复的状态谓词作为主体，并将介词短语作为单独 `ModificationalFunctor`、`Modification` 或等价结构挂接，使插入成分和介词宾语的 scope 可恢复。

5. **`there be` 存在句拆分，`there` 作为 `pred.arg.1`**。`there is/are/was/were ...` 不应把 `there be` 合并为一个不可分析谓词。`be` 应作为 `FactualPredicator`，`there` 作为 `pred.arg.1`，存在物或其可恢复 descendant subgraph 作为 `pred.arg.2`。例如：

```text
there are bears

are/FactualPredicator
  pred.arg.1 -> there
  pred.arg.2 -> bears
```

若存在物内部有修饰、限定、相对从句或并列结构，应保留其内部结构，并让 `pred.arg.2` 指向可恢复完整存在物的节点或 descendant subgraph。

常见错误：

| 错误 | 严重程度 |
|---|---|
| 简单系表结构被拆成 `be -> pred.arg.2 -> adjective`，且无其他结构需求 | minor / consistency error |
| `is very true` 等把副词和补语压成不可恢复的单节点 | major |
| `be` 直接接介词谓词却拆成不必要 wrapper，导致关系谓词不清 | minor / major |
| `be` 与介词之间有插入成分却整体合并，导致插入成分或 PP scope 不可恢复 | major |
| `there be` 合并为不可分析谓词，未把 `there` 作为 `pred.arg.1` | major |

`pred.arg.4` 与状语修饰必须区分：

1. 若成分是谓词语义价的一部分，或表达宾补、结果状态、被谓词选择的非限定补足语，可作为 `pred.arg.4`。例如 `painted the wall red` 中 `red`、`elected him president` 中 `president`、`attempt to leave` 中 `to leave`。
2. 若成分回答目的、原因、时间、方式、条件等问题，且主谓宾事实在没有该成分时仍然完整，应建模为 `Modification`、介词/状语修饰，或在有明确逻辑关系时建模为 `LogicalPredicator`。
3. 目的状语，尤其是可改写为 `in order to ...` 的不定式短语，不应作为主谓词的 `pred.arg.4`；它应作为修饰体修饰完整事件或相应事件子图。

常见错误：

| 错误 | 严重程度 |
|---|---|
| 把谓词标成普通 nominal constant | major |
| 主客体论元反了 | major |
| 可由文本或结构恢复的 `pred.arg.1` 缺失 | major |
| 漏掉核心论元 | major |
| 把状语当核心论元 | minor / major |
| 轻微论元编号差异但语义可恢复 | minor |

---

### 6.2 实体与名词短语

Lingua 与 OIA 的重要区别是：Lingua 不应把复杂名词短语无条件压成一个扁平节点。复杂实体应尽量保留内部结构，使实体本身或其 descendant subgraph 能成为下游任务候选单元。

正确原则：

1. 简单专名或不可分实体可作为单节点。
2. 复杂实体应显式表示限定词、修饰语、中心词、介词短语、相对从句等。
3. 嵌套实体应可通过 `SELF(v)` 或 `DESC(v)` 恢复。
4. entity span 的边界必须准确。

示例：

```text
that old wooden chair
```

推荐完整结构：

```text
Modification
  variable -> Modification
    variable -> Modification
      variable -> chair
      body -> wooden
    body -> old
  body -> that
```

简化结构在无歧义时可接受：

```text
chair
  as:modification -> wooden
  as:modification -> old
  as:modification -> that
```

不可接受：

```text
that old wooden chair/NominalConstant
```

除非任务场景明确只要求粗粒度节点，否则这种标注丢失了 Lingua 希望保留的实体内部结构，应判为 major。

---

### 6.3 Modification

`Modification` 用于表示修饰结构，其核心是：

```text
Modification
  variable -> 被修饰对象
  body -> 修饰体
```

适用对象：

1. 形容词修饰名词。
2. 副词修饰动词或形容词。
3. 名词修饰名词。
4. 介词短语修饰。
5. 相对从句修饰。
6. 事实被状语或介词短语修饰。

要求：

| 要求 | 说明 |
|---|---|
| `variable` | 必须指向被修饰对象。 |
| `body` | 必须指向修饰成分。 |
| 层级 | 多个修饰语应按语义组合顺序形成层级。 |
| 可简化 | 无歧义时可用直接 `as:modification`。 |
| 不可过度简化 | 如果简化会导致 head 或 scope 不清，必须保留完整 `Modification`。 |

错误类型：

| 错误 | 严重程度 |
|---|---|
| `variable` 和 `body` 反了 | major |
| 多个修饰语 scope 错误 | major |
| 复杂修饰被压平导致实体无法恢复 | major |
| 简单修饰使用简化边 | correct / minor |
| 把修饰当作事实核心论元 | minor / major |

### 6.3.1 修饰层级与 scope 检查

修饰结构的关键不是线性位置，而是 semantic scope。所有 `Modification`、`as:modification`、介词修饰、状语修饰、相对从句修饰、分词短语修饰等，都必须检查其 head、scope 和内部层级是否正确。

要求：

1. 对每个修饰成分，必须明确它修饰的对象，即 `variable`。
2. 多个修饰成分同时出现时，应按语义组合顺序形成层级，而不是机械按词序或最近节点附着。
3. 修饰层级应体现语义亲疏：更贴近 head/event 的修饰进入内层，scope 更大、更疏远的修饰进入外层。
4. 当一个修饰节点作为另一个修饰节点的 `variable` 时，应将该 `variable` 解释为其完整 `DESC(variable)`，即“已被内层修饰后的对象或事件”。只有在 `DESC(variable)` 能恢复正确 head 和 scope，且链式顺序符合语义亲疏时，这种嵌套才是合法紧凑结构。
5. 修饰体内部结构必须保留；重接修饰层级时不得丢失修饰体自己的论元。
6. 若修饰成分位于句首或分句首，应作为高风险情况重点检查：它可能修饰整句、主事实、后续分句，也可能只修饰局部实体或局部事实，不能仅凭位置自动决定。
7. 若所谓“修饰”实际表达条件、让步、因果、时间等逻辑关系，应优先建模为逻辑结构，而不是普通 `Modification`。

错误类型：

| 错误 | 严重程度 |
|---|---|
| 修饰成分的 `DESC(variable)` 无法恢复正确 head，导致 scope 错误 | major |
| 多层修饰顺序错误，导致组合语义改变 | major |
| 近层修饰和远层修饰的内外顺序反了 | major |
| 修饰体内部论元在重接层级时丢失 | major |
| 前置修饰误接到后续嵌套事实或误作核心论元 | major |
| 简单修饰只存在可恢复的包装差异 | minor / correct variant |

---

### 6.4 介词修饰

介词短语可作为修饰体，也可在无歧义时以介词为局部 root。

示例：

```text
report from research lab
```

可接受结构：

```text
Modification
  variable -> report
  body -> from
    pred.arg.1 -> report
    pred.arg.2 -> research lab
```

或在无歧义时：

```text
from
  pred.arg.1 -> report
  pred.arg.2 -> research lab
```

评估原则：

1. 介词连接的 head 必须清楚。
2. 介词的两个论元不能反。
3. 若介词短语只修饰实体，不应误标为主句事实。
4. 若多个介词短语嵌套，应保留层级。

---

### 6.5 相对从句

相对从句修饰实体时，应先建模相对从句内部事实，再通过 `Modification` 包装。若句中存在显式 wh-word、`that` 或其他指代性关系标记，应使用 `ref` / `as:ref` 或等价 `Ref` 节点连接该显式指代成分与被修饰实体。

示例：

```text
the scientist who discovered the new element
```

推荐结构：

```text
Modification
  variable -> scientist
  body -> discovered/FactualPredicator
    pred.arg.1 -> who
    pred.arg.2 -> the new element

scientist
  as:ref -> who
```

现有 JSONL 数据也接受如下等价结构：

```text
Ref/ReferentialPredicator
  pred.arg.1 -> head entity / antecedent
  pred.arg.2 -> wh-word / referring expression / relative-clause Modification
```

要求：

1. 相对从句内部谓词必须正常建模。
2. wh-word、关系从句 `that` 或其他显式指代成分如果是论元，应参与谓词论元结构。
3. 显式 wh-word、关系从句 `that` 或其他指代成分与 head entity 之间必须有 `ref`、`as:ref` 或等价 `Ref(pred.arg.1, pred.arg.2)` 结构。
4. 整个相对从句作为修饰体修饰 head entity。
5. 零关系从句没有显式 wh-word / `that` / 指代词时，不应凭空添加 `Ref` 节点。此时应让 head entity 直接填补从句内部的隐含 gap，或在确有必要时使用有结构依据的 `Missing`。

零关系从句示例：

```text
the ocean we swim in
```

推荐结构：

```text
Modification
  variable -> ocean
  body -> in/ModificationalFunctor
    variable -> swim/FactualPredicator
      pred.arg.1 -> we
    body -> ocean
```

错误：

| 错误 | 严重程度 |
|---|---|
| 漏掉相对从句内部事实 | major |
| 显式 wh-word / `that` 与 head 的 ref 关系遗漏 | major |
| 零关系从句中凭空添加 `Ref`，把修饰后的名词误当作原名词的指代 | major |
| 把相对从句直接压成普通 modifier token | major |
| `ref` 方向因单根需要使用 `as:ref` | 正确 |
| 指代对象错误 | major / fatal |

---

### 6.6 Expression

`Expression` 用于非修饰性的组合表达，例如范围、公式、化学式、版本号、符号组合等。

示例：

```text
48 to 49
```

推荐结构：

```text
to/ExpressionFunctor
  func.arg -> 48
  func.arg -> 49
```

示例：

```text
Ca 2+
```

推荐结构：

```text
Expression
  func.arg -> Ca
  func.arg -> 2+
```

评估原则：

1. 如果组合关系不是修饰关系，不应误用 `Modification`。
2. 表达式整体应可作为一个 descendant subgraph 被恢复。
3. 数学、化学、范围、单位等符号应使用 `SymbolConstant` 或合适 constant type。
4. 独立出现的日期、时间、日期时间、编号、比分、版本号等符号串，若内部只是格式组合而不是语义修饰，应优先作为一个 `Expression`；不得把日期修饰时间、AM/PM 修饰日期时间等格式部件误标为 `Modification` 或 `as:modification` 链。

---

### 6.7 Attribute

属性结构用于表示：

```text
<entity, attribute_name, value>
```

显式属性示例：

```text
Tesla Model S battery capacity 100 kWh
```

推荐：

```text
battery capacity/AttributePredicator
  pred.arg.1 -> Tesla Model S
  pred.arg.2 -> 100 kWh
```

隐式属性示例：

```text
François Simon (actor)
```

推荐：

```text
Attribute
  pred.arg.1 -> François Simon
  pred.arg.2 -> actor
```

评估原则：

1. 属性名显式出现时，属性名应作为 `AttributePredicator`。
2. 属性名隐式时，可用 symbol node `Attribute`。
3. `pred.arg.1` 通常为被赋予属性的实体。
4. `pred.arg.2` 通常为属性值。
5. 括号中的身份、类型、别名需根据语义区分 `Attribute` 与 `Apposition`。

常见错误：

| 错误 | 严重程度 |
|---|---|
| 把属性值当作普通修饰词 | major |
| 实体和值反了 | major |
| 属性结构遗漏 | major |
| Attribute 与 Apposition 混淆但语义可恢复 | minor / major |

---

### 6.8 Apposition

同位结构用于表示同一实体的再描述、别名、身份说明。

示例：

```text
John Miller, the coach
```

推荐：

```text
Apposition
  pred.arg.1 -> John Miller
  pred.arg.2 -> the coach
```

或：

```text
John Miller
  appositive -> the coach
```

评估原则：

1. 两个成分应指向同一实体或同一 referent。
2. 同位不是普通修饰；不能简单标为 adjective-like modifier。
3. 若括号内容是类型、职业，也可能使用 `Attribute`，评估时应允许语义等价变体。

---

### 6.9 LogicalPredicator

逻辑、因果、条件、时间连接词应作为事实间关系处理。

示例：

```text
because it rained, the match was canceled
```

推荐：

```text
because/LogicalPredicator
  pred.arg.1 -> the match was canceled
  pred.arg.2 -> it rained
```

评估原则：

1. `if`、`because`、`therefore`、`so`、`although` 等不应简单丢弃。
2. 它们通常连接两个事实或命题。
3. 若一个主事实连接多个从句关系，应明确建模层级。
4. 逻辑关系的方向要符合语义：原因、结果、条件、结论不能反。

错误：

| 错误 | 严重程度 |
|---|---|
| 丢掉关键逻辑连接词 | major |
| 原因和结果反了 | major |
| 条件从句误作实体修饰 | major |
| discourse marker 无事实作用时未标 | minor |

---

### 6.9.1 歧义功能词定类

`but`、`if`、`that` 等高频功能词应按语义功能定类。标注时不应只按词形固定类型，也不应在相同语义功能下随意漂移。

| 词形 / 用法 | 推荐 node type | 说明 |
|---|---|---|
| `but` 表示转折、对比、让步，连接两个事实或命题 | `LogicalPredicator` | 如 `A happened, but B happened`。不应标为普通并列 `ConjunctionalFunctor`。 |
| `if` 表示“如果”，连接条件从句与主句 | `LogicalPredicator` | 如 `if it rains, the match is canceled`。 |
| `if` 表示“是否”，引导名词性从句 | `GeneralFunctor` | 如 `ask if he came`，整体从句作为被询问内容。 |
| `that` 引导名词性从句 | `GeneralFunctor` | 如 `said that he left`、`think that it is true`。 |
| `that` 独立指代实体、事件或命题 | `NominalConstant` | 如 `I want that`、`That is true`。 |
| `that` 限定名词 | `DeterminerConstant` | 如 `that book`、`that environment`；不应标为 `ModificationalConstant`。 |
| `that` 作为关系从句标记 | `GeneralFunctor` | 如 `the place that sells food`，并应通过 `Ref` 或 `ref/as:ref` 与 head entity 建立指代关系。 |

关系从句中的 `that` 不应仅因“指代名词”而标为 `NominalConstant`。推荐结构与 `who/which/where` 类似：`that` 作为从句中的论元参与内部谓词结构，head entity 与该关系结构通过 `Ref` 或 `ref/as:ref` 连接。

错误：

| 错误 | 严重程度 |
|---|---|
| 将转折 `but` 标为普通并列，导致转折语义丢失 | major |
| 将条件 `if` 与“是否” `if` 混淆 | major |
| 将限定名词的 `that` 标为 `ModificationalConstant` | minor / major，取决于是否影响实体结构 |
| 将关系从句 `that` 标为普通实体而没有 `Ref` 结构 | major |

---

### 6.10 Conjunction / Parataxis

并列、协调结构用 `Parataxis` 或 `ConjunctionalFunctor`。

示例：

```text
John and Mary arrived
```

可建模为：

```text
and/ConjunctionalFunctor
  func.arg -> John
  func.arg -> Mary
```

然后整体作为 `arrived` 的论元。

评估原则：

1. 并列项应全部出现。
2. 并列项应保持 parallel status。
3. 不能只保留第一个并列项。
4. 共享谓词或共享修饰应作用于整个并列结构。
5. 对异质枚举，不强调共享结构时可用 `List`。

---

### 6.11 List

`List` 用于枚举异质元素，不强制它们共享谓词、属性或行为。

示例：

```text
The package includes documents, keys, and a USB drive.
```

推荐：

```text
include/FactualPredicator
  pred.arg.1 -> The package
  pred.arg.2 -> List
    func.arg -> documents
    func.arg -> keys
    func.arg -> a USB drive
```

评估原则：

1. 同质并列偏 `Parataxis` / `ConjunctionalFunctor`。
2. 异质枚举偏 `List`。
3. 列表项不得遗漏。
4. 列表整体应能作为一个论元被主谓词引用。

---

### 6.12 Reference / Coreference

指代关系可用 `ref`、`as:ref` 或 `Ref/ReferentialPredicator` 节点表示。

示例：

```text
John said he was tired.
```

推荐：

```text
he
  ref -> John
```

或因单根方向需要：

```text
John
  as:ref -> he
```

等价 `Ref` 节点结构：

```text
Ref/ReferentialPredicator
  pred.arg.1 -> antecedent / head entity
  pred.arg.2 -> referring expression / referential structure
```

评估原则：

1. 指代词与先行词必须连接。
2. 相对从句中显式出现的 wh-word、`that` 或其他指代性关系标记也应通过 `ref/as:ref` 或等价 `Ref` 节点连接 head entity；零关系从句不应凭空添加 `Ref`。
3. 指代关系不应代替谓词论元关系；二者都应存在。
4. Coreference 的语义对象错误是严重错误。

---

### 6.13 Missing

`Missing` 用于处理省略、椭圆、省略主语等结构中需要但未显式出现的论元。

使用条件：

1. 句法或语义结构确实需要一个隐含成分。
2. 文本中没有对应 token。
3. 使用 `Missing` 能让 PFA 结构完整。
4. 不应滥用 `Missing` 添加推测信息。

判错：

| 情况 | 严重程度 |
|---|---|
| 合理补出省略论元 | 正确 |
| 用 `Missing` 添加文本没有暗示的实体 | major |
| 省略结构不补但事实仍可理解 | minor / major |
| 关键省略导致事实结构断裂 | major |

---

### 6.14 Discourse 与 Vocative

话语标记、插入语、呼语不应混入核心事实论元。

示例：

```text
John, please close the door.
```

推荐：

```text
close/FactualPredicator
  pred.arg.1 -> Missing/you
  pred.arg.2 -> the door
  vocative -> John
  discourse -> please
```

评估原则：

1. 呼语不是主语，除非句子语义确实如此。
2. `please` 等 discourse marker 不应作为事实核心论元。
3. 话语关系影响语气但不改变核心事实时，错误可判 minor。

---

### 6.15 Document Metadata / Header Fragments

训练数据中可能出现 section header、speaker label、field label、key-value label、UI fragment 等文档元数据片段与正文内容拼接在同一个样本中。此类片段应保留 token coverage，但不得强行解释为正文命题的普通 predicate、argument 或 modifier。

基本原则：

1. 若片段只是栏目名、页面标题、字段名、角色标签或界面控件文本，且不参与正文事实语义，应作为低语义权重的 discourse metadata 处理。
2. 这类 metadata token 可以建成 `NominalConstant` 或必要时的轻量 wrapper，并通过 `as:discourse` / `discourse` 连接到正文子图，以保持 single-root DAG 与 token coverage。
3. metadata 不得作为正文事实谓词的主语、宾语、补语、普通修饰体或逻辑连接词，除非文本明确表达它参与该事实。
4. `Overview Since 1992, P`、`Biography X was born...`、`History P` 等结构中，`Overview` / `Biography` / `History` 是 header，不应修饰 `Since`、`X` 或 `P` 的核心谓词；正文命题应独立恢复。
5. 冒号、破折号等分隔符若只是 header 与正文的边界，应作为局部 punctuation / discourse boundary，而不是事实谓词。
6. 若样本本身是纯 key-value / field-value 记录，没有独立正文谓词，可使用 `AttributePredicator` 或 symbol node `Attribute` 表示该元数据关系；key / field label 对应属性名，value 对应属性值，冒号、破折号仅作为 boundary punctuation。
7. 若 key-value 中缺少被赋予属性的外部实体，可用当前记录或可恢复 value 作为 `pred.arg.1`，但不得虚构未出现的业务实体；无法恢复的上下文实体可用 `Missing`，并在评估中视为低语义信息。

示例：

```text
Overview Since 1992, the festival has striven...
```

推荐：

```text
./PunctuationalConstant
  as:punctuation -> Since/ModificationalFunctor

Since/ModificationalFunctor
  variable -> has striven
  body -> 1992
  as:discourse -> Overview
```

或等价地由正文 root 通过 `discourse` / `as:discourse` 挂接 `Overview`，只要 `DESC(root)` 能恢复正文命题且 `Overview` 不进入核心 PFA。

判错：

| 情况 | 严重程度 |
|---|---|
| header 被当作正文主语、宾语或核心修饰体 | major |
| speaker label 被当作说话内容的事实谓词 | major |
| field label / key label 被当作内容实体的一部分 | major |
| 纯 key-value 记录用 `Attribute` / `AttributePredicator` 表示 | 正确 |
| UI fragment 被强行解析为普通命题 | major |
| metadata 仅作为 discourse metadata 挂接 | 正确 / minor |

#### 6.15.1 Pure Structured Fragments

有些样本本身不是正文句子，而是完整样本仅由 header、表格列名、比分、赛程、地址、联系方式、文件名、页面引用、版权行、decorative separator、列表项或 UI 控件文本构成，例如：

```text
LATEST PREVIOUS ACTUAL
W L T PF PA
p. 36
Preseason Game B: Thursday, May 24 @ 7:30 PM vs. Detroit SHOCK
Chicago, IL 60622
---->===}*{===<----
```

这类样本应先判定为 structured fragment，而不是套用普通句子的主事实谓词要求。纯 fragment 仍必须满足 token coverage、legal labels、single-root DAG 和 recoverability，但不要求虚构 `FactualPredicator`、主语、宾语或完整命题。

表示规则：

1. 平行字段、列名、菜单项、标题列表可用 `ListFunctor` / `List` 挂接每个项目；项目之间没有事实性 predicate-argument 关系时，不得强行建立谓词。
2. 页码、编号、比分、战绩、日期时间、时间范围、电话号码、地址、文件名、百分比、货币数值、模型号等常规符号串可用 `ExpressionFunctor` / `Expression` 打包为一个可恢复 expression unit。
3. `key: value`、`field - value`、`Game 11: Tuesday...`、`Preseason Game B: Thursday...`、`Coach: Doug Sewell`、`Email: ...`、`Location: ...`、`Attendance : 15,150`、`Pin #: ...` 等字段记录可用 `AttributePredicator` 或 `Attribute` 表示字段名与字段值；若外部实体缺失，不得虚构实体，可用当前 record 或可恢复 value 作为局部 host，必要时用 `Missing` 并降权。字段名不是普通事实谓词时，不应为了连接 key/value 而把它标成 `FactualPredicator`。
4. 冒号、破折号、逗号、斜杠、括号、句点若只是 fragment 内部 boundary 或 close marker，应作为局部 punctuation / expression boundary，不得成为事实谓词。
5. 纯装饰符号、乱码、截断严重、问候/签名、媒体边界、版权声明等低信息片段仍有片段语义，应优先保留为低语义信息 fragment；除非上游数据集有明确的非语义清洗策略，不应仅因信息量低而过滤。
6. 对问候、签名、感谢、残句等仍能看出普通语法或省略句结构的片段，应先按省略结构解析，必要时用 `Missing` 补出可由结构支持的省略论元；不得因为它是邮件结束语、寒暄语或低信息文本就直接平铺为 `Expression`。
7. `Expression`、`List`、`Attribute`、局部 punctuation root 或 `nonsense` 只应作为 fallback：当片段内部没有可负责任恢复的谓词、修饰、呼语、介词短语、字段、并列或省略关系时，才把它作为低语义信息 fragment 打包。不得把 `nonsense` 用于可按上述规则解释的正常字段或表达式。
8. 纯 fragment 的 root 可以是普通语法结构、省略结构、`ListFunctor`、`ExpressionFunctor`、`AttributePredicator`、对应 aux node，或能关闭该 fragment 的 `PunctuationalConstant`；只要 `DESC(root)` 恢复完整 fragment 即可。
9. 纯日期、时间、日期时间和时间范围片段，如 `02/27/2001 08:23 AM`、`Feb 2005`、`September 21, 2004`、`Tuesday, March 20th.`、`September 20, 1888?`、`4:00 pm - 6:00 pm`，应表示为一个 `Expression`，其子项可按 date / time / meridiem / range boundary 或更细粒度符号部件恢复；不要把 date-time 内部格式关系标成普通修饰，也不要把时间范围分隔符误标为事实谓词。
10. 新闻 dateline 或来源行，如 `DUBLIN 1996-08-27`、`LOS ANGELES 1996-12-05`，应表示为 `List(location-expression, date-expression)` 或等价可恢复结构：location 和 date 是同一元数据行的两个字段，不是彼此的修饰体，也不是正文事实谓词的论元。
11. 纯页码/页引用片段，如 `p. 36`、`pp. 12-15`，应作为 page-reference `Expression` 或等价可恢复表达式；不要把 `p.` 和页码误当成并列 `List` 项。

判错：

| 情况 | 严重程度 |
|---|---|
| 纯 fragment 没有主事实谓词，但用 `List` / `Expression` / `Attribute` 可恢复 | 正确 |
| 为表格列名、页码、比分、文件名虚构普通事实谓词 | major |
| 把 field label 当作 value 的普通修饰或实体组成部分 | major |
| 可解析字段记录未区分 key 与 value | minor / major |
| 纯 key-value 字段记录被标成普通 `FactualPredicator` 结构 | minor / major |
| 用 `nonsense` 代替可解析的 expression、list 或 attribute | minor / major |
| 纯日期时间或时间范围被标成 `Modification` / `as:modification` 链或普通事实谓词 | minor / major |
| dateline 中 location 与 date 被误标为修饰、事实论元或互相省略 | minor / major |
| 页引用被误标为并列 `List` 或普通命题 | minor / major |
| 纯装饰、乱码、截断、问候、签名、媒体边界、版权声明被保留为低语义 fragment | 正确 |
| 可按省略句解析的问候/签名/感谢/残句被无差别平铺为 `Expression` | minor / major |

#### 6.15.2 Proper Names, Titles, and Nominal Fragments

低置信样本中还常见多词专名、地名、作品名、标题、名单项、职位/角色短语和括号消歧义，例如：

```text
Donna Sells
Marquis de Condorcet
Beetham Tower, Manchester
Cedar Hill School (disambiguation)
Great Sandwiches, Great Prices
Koichi Morita (army officer)
```

这类片段缺少显式谓词时，不应为了满足普通句子模板而补出 copula、`is`、`be named` 或其他未出现的事实谓词。它们应作为 entity / title / nominal expression 处理。

表示规则：

1. 多词专名若内部组合是 lexicalized name，推荐用 `ExpressionFunctor` / `Expression` 或一个 head `NominalConstant` 加可恢复的 name-part 结构表示；不得把名和姓误作 predicate-argument。
2. 含有职称、角色、括号说明、地理限定或消歧义说明时，可用 `Attribute`、`Apposition`、`as:modification` 或 discourse metadata 表示其对核心实体的限定；选择取决于该说明是身份属性、同位描述、地理限定还是数据源消歧义。
3. 逗号连接的地名/机构名如 `Jordan, New York`、`Beetham Tower, Manchester` 通常不是并列列表，而是 entity + location / qualifier；不能简单当作两个无关 list items，除非上下文确实表达列表。
4. 标题式短语如 `PHOTOS DONE WELL`、`Great Sandwiches, Great Prices` 可作为 title/list/expression；只有当上下文明确表达命题时，才解析为普通事实句。
5. 如果专名或标题内部边界无法可靠判断，只要 token coverage 完整、root 可恢复整个名词性片段、没有添加未出现事实，可判为 acceptable variant 或 minor，不应直接判 major。

判错：

| 情况 | 严重程度 |
|---|---|
| 多词专名作为 `Expression` / entity fragment 可恢复 | 正确 |
| 为专名/标题虚构普通事实谓词 | major |
| 把地理限定、括号消歧义、角色说明误作核心事实论元 | major |
| 将 entity + qualifier 误作无关平行列表并丢失限定关系 | major |
| 专名内部结构风格不同但整体 entity 可恢复 | minor / acceptable variant |

---

### 6.16 Punctuation

标点通常是辅助结构，但在以下情况中很重要：

1. 括号表达属性或同位。
2. 逗号标记同位、插入语、列表。
3. 引号标记引用内容。
4. 冒号、分号标记解释、枚举或并列事实。

若某个标点可以确定为 close 某个结构，该 `PunctuationalConstant` 应作为该结构的局部 root，并用 `as:punctuation` 指向被关闭的内容。close 的范围必须尽量精确：句尾标点若只关闭第二个分句，就应成为第二分句的 root，而不应上提为整句或外层 discourse relation 的 root。只有无法确定 close 范围、或标点只是非结构性装饰时，才可作为普通附着项处理。

判错：

| 情况 | 严重程度 |
|---|---|
| 普通句号未标 | minor / 可忽略 |
| 括号属性未建模 | major |
| 逗号同位误解析 | major |
| 列表标点导致枚举项遗漏 | major |

---

## 7. 可接受变体

Lingua 标注允许一定程度的结构变体。LLM 评估时不应要求唯一答案。

### 7.1 词形规范化

```text
gave -> give
was -> be
```

可接受，只要 token anchor 保持原词位置。

### 7.2 Modification 简化

对于简单、无歧义的修饰，可以接受：

```text
chair
  as:modification -> wooden
```

代替完整：

```text
Modification
  variable -> chair
  body -> wooden
```

但复杂修饰、多个 scope、介词嵌套、相对从句等场景不能过度简化。

### 7.2.1 冗余 `Modification` 的判定与去除

`Modification` 的作用是显式打包“被修饰对象 + 修饰体”，帮助恢复 head、scope 和 analytic unit。若一个 `Modification` 节点没有提供新的 head、scope 或 recoverability 信息，只是把已有关系再包一层，则应视为冗余结构。

可判为冗余的情况：

1. `Modification` 只有 `variable` 和 `body` 两条核心边，且二者关系已经由 `body` 节点自身的语义边完整表达。
2. `body` 是 `LogicalPredicator`、`ConjunctionalFunctor`、`AttributePredicator`、`ReferentialPredicator` 等非修饰关系节点，而 `Modification` 只是把该关系重新包装成普通修饰。
3. 删除该 `Modification` 后，把原来指向它的 parent 改指向真正的关系 root，图仍保持 single-root DAG，且所有核心事实、实体、scope 都可通过 `DESC` 恢复。
4. `Modification` 的存在只是为了让某个标点、wrapper 或 root 连接到图中，而不是表达真实修饰关系。

不可判为冗余的情况：

1. `Modification` 是唯一表达 modifier 与 head 关系的结构，例如 `Modification(variable=chair, body=red)`。
2. `Modification` 用于区分多个修饰语的 scope 或组合顺序。
3. `Modification` 用于包装复杂介词短语、相对从句、分词短语、事实性修饰等，使其作为局部 analytic unit 可恢复。
4. 删除后会导致 head、scope、实体边界或修饰体内部结构不清。

去除规则：

1. 找到冗余 `Modification` 的真正语义 root，通常是其 `body`。
2. 将原本指向 `Modification` 的 incoming edge 改指向该语义 root。
3. 删除 `Modification.variable` / `Modification.body` 两条边和该 aux node。
4. 检查 single-root DAG、reachability、token partition、核心语义 recoverability。

### 7.3 Attribute vs Apposition

以下情况可能存在可接受变体：

```text
François Simon (actor)
```

可以视为：

```text
Attribute(entity=François Simon, value=actor)
```

也可能视为：

```text
Apposition(François Simon, actor)
```

评估时应根据上下文判断；若二者均保持同一实体身份信息，可判 minor variant，不应直接判错。

### 7.4 `ref` 与 `as:ref`

若只是为了满足单根 DAG 方向要求，`ref` 和 `as:ref` 的物理方向可不同，但语义方向必须可恢复。

### 7.5 介词修饰完整结构 vs 简化结构

简单介词短语可用介词作 root：

```text
from
  pred.arg.1 -> report
  pred.arg.2 -> lab
```

复杂场景应显式包装为 `Modification`。

---

## 8. 不可接受偏差

以下情况通常应判为 major 或 fatal：

1. 把整句压成一个节点。
2. 把复杂实体无条件压成一个 simple phrase，导致内部结构无法恢复。
3. 主谓宾核心论元关系错误。
4. 丢掉主事实。
5. 把条件、因果、时间关系当作普通修饰或直接丢弃。
6. 相对从句没有内部事实结构。
7. 显式 wh-word / `that` / 指代性关系标记与 head entity 没有 `ref`。
8. 句子中每个 token 没有形成 partition。
9. 图有环或多个 root。
10. 使用非 Lingua schema 且不可映射的标签。
11. 添加文本未表达的信息。
12. 指代对象明显错误。
13. 属性实体和值反向。
14. 并列项遗漏或共享谓词 scope 错误。
15. `as:` 前缀使用后语义方向不可恢复。

---

## 9. 评分标准

总分 100。

| 维度 | 分值 | 说明 |
|---|---:|---|
| 形式合法性 | 20 | 单根 DAG、无环、可达、schema 合法。 |
| Token partition | 15 | token 覆盖完整、不重叠、anchor 正确。 |
| 语义忠实性 | 20 | 不遗漏、不添加、不误解原句。 |
| PFA 结构 | 15 | 谓词、函数、论元、边方向和论元编号正确。 |
| Lingua 层级结构 | 20 | Modification、Expression、Attribute、Ref、List 等建模正确。 |
| 可读性与规范化 | 10 | 标签规范、冗余少、结构清晰；`nonsense` 合法但应按低语义信息边处理。 |

### 9.1 Verdict 阈值

| Verdict | 分数范围 | 含义 |
|---|---:|---|
| `correct` | 90–100 | 基本正确，仅有可接受变体或极小问题。 |
| `minor_errors` | 75–89 | 有轻微标签、简化、标点、非核心结构问题。 |
| `major_errors` | 40–74 | 有核心结构错误，但仍可识别部分 Lingua 图。 |
| `invalid` | 0–39 | 形式不合法或语义严重错误，不能视为 Lingua 标注。 |

### 9.2 Fatal errors

出现以下任一情况，通常直接判 `invalid`：

1. 图不是 DAG。
2. 多根且无法通过轻微修改连接。
3. 主事实完全缺失。
4. token partition 严重错误。
5. 大量节点或边不符合 schema。
6. 标注与原句表达的语义大面积不符。
7. 使用普通依存树、三元组列表或 AMR 风格图冒充 Lingua。

---

## 10. LLM 评估流程

LLM 应按以下顺序评估：

1. **读取句子与候选标注**：确认 token、node、edge、root 是否齐全。
2. **Schema 规范化**：将可映射标签标准化，例如 `var -> variable`、`mod -> modification`、`Predicate -> FactualPredicator`。
3. **形式检查**：检查 token partition、单根、DAG、可达性、重复节点、非法边。
4. **主事实检查**：找出句子的主谓词和核心事实，确认是否被正确建模。
5. **论元检查**：检查 `pred.arg.1`、`pred.arg.2`、`pred.arg.3`、`pred.arg.4` 是否符合语义角色。
6. **实体内部结构检查**：检查复杂名词短语是否具有 Lingua 层级结构。
7. **特殊结构检查**：检查 `Modification`、`Expression`、`Attribute`、`Apposition`、`Ref`、`LogicalPredicator`、`Parataxis`、`List`、`Discourse`、`Vocative`、`Missing`。
8. **可接受变体判断**：判断候选标注是否只是使用了等价简化，而非真正错误。
9. **评分与输出**：给出 verdict、分数、错误列表和最小修正建议。

---

## 11. 推荐输出格式

LLM 评估器应输出 JSON，便于自动处理。

```json
{
  "verdict": "minor_errors",
  "score": 82,
  "is_valid_lingua_graph": true,
  "fatal_errors": [],
  "summary": "The annotation is mostly valid, but it over-flattens one complex nominal modification and misses a ref edge in the relative clause.",
  "errors": [
    {
      "category": "Modification",
      "severity": "major",
      "location": {
        "nodes": ["n4", "n5"],
        "edges": []
      },
      "description": "The complex noun phrase is represented as a flat nominal node, so its internal modification structure is not recoverable as a descendant subgraph.",
      "expected_fix": "Introduce a Modification node with variable pointing to the head noun and body pointing to the modifier."
    }
  ],
  "accepted_variants": [
    {
      "description": "The use of var instead of variable is accepted as a shorthand alias."
    }
  ],
  "minimal_correction_skeleton": {
    "nodes_to_add": [],
    "edges_to_add": [],
    "nodes_to_change": [],
    "edges_to_change": []
  }
}
```

---

## 12. 错误类别

| Category | 说明 |
|---|---|
| `TokenPartition` | token 覆盖、重叠、anchor 错误。 |
| `GraphForm` | root、DAG、可达性、连通性错误。 |
| `Schema` | 节点类型、symbol label、边标签错误。 |
| `PredicateArgument` | 谓词-论元结构错误。 |
| `Modification` | 修饰结构、scope、variable/body 错误。 |
| `EntityStructure` | 实体内部结构、嵌套实体、head 错误。 |
| `Expression` | 范围、公式、符号组合错误。 |
| `Attribute` | 属性名、实体、属性值错误。 |
| `Apposition` | 同位关系错误。 |
| `Reference` | 指代、相对从句 ref 错误。 |
| `LogicalRelation` | 条件、因果、时间、让步等关系错误。 |
| `Coordination` | 并列、parataxis、list 错误。 |
| `Discourse` | 话语标记、插入语、呼语错误。 |
| `DocumentFragment` | header、field label、表格、UI、联系方式、文件名等 structured fragment 错误。 |
| `ProperNameFragment` | 专名、标题、名词性 fragment 被误解析为普通命题或丢失限定关系。 |
| `SemanticFidelity` | 添加、遗漏、误解原句语义。 |
| `OverFlattening` | OIA 式过度扁平化，未体现 Lingua 层级。 |
| `Redundancy` | 重复节点、重复边、无意义结构。 |

---

## 13. Severity 定义

| Severity | 定义 |
|---|---|
| `minor` | 不影响主语义和主要结构，可由规范化或局部修正解决。 |
| `major` | 影响核心实体、事实、修饰 scope 或 Lingua 子结构 recoverability。 |
| `fatal` | 导致图不合法或语义完全不可用。 |

---

## 14. LLM System Prompt 模板

```text
You are an expert evaluator for Lingua / Lingua-Graph annotations.

Your task is to judge whether a candidate annotation is a correct Lingua annotation for the given sentence.

Use the following principles:

1. A valid Lingua annotation must be a single-rooted DAG.
2. Lexical nodes must form a token partition of the input sentence: every token is covered exactly once; auxiliary symbol nodes may have empty token anchors.
3. Node types and edge labels must conform to the Lingua schema.
4. The graph must faithfully represent the sentence without adding unsupported information or omitting core information.
5. Predicate-Function-Argument structure must be correct: factual predicates, logical predicates, functions, and their arguments must be represented with correct edge labels.
6. Lingua is more hierarchical than OIA. Do not accept excessive OIA-style flattening when the sentence contains complex entity-internal structure, nested modification, relative clauses, attributes, expressions, or logical relations.
7. If the input is a document fragment, table/header field, UI fragment, title, or pure proper-name fragment, evaluate it under the fragment rules before requiring a main factual predicate.
8. Accept harmless variants, including:
   - var as an alias of variable;
   - as: prefix for reversed edges when needed for single-root DAG;
   - simplified modification edges when there is no ambiguity;
   - minor label spelling variants if they are clearly mappable to the Lingua schema.
8. Penalize:
   - missing main facts;
   - wrong predicate-argument roles;
   - wrong modification scope;
   - missing ref edges in relative clauses or anaphora;
   - wrong attribute entity/value direction;
   - invalid graph form;
   - token overlap or token omission;
   - unsupported semantic additions.

Return only JSON with:
- verdict: correct | minor_errors | major_errors | invalid
- score: integer from 0 to 100
- is_valid_lingua_graph: boolean
- fatal_errors: list
- summary: concise explanation
- errors: list of error objects
- accepted_variants: list
- minimal_correction_skeleton: object
```

---

## 15. LLM User Prompt 模板

```text
Evaluate the following candidate Lingua annotation.

Sentence:
{sentence}

Tokens:
{tokens}

Candidate annotation:
{candidate_annotation_json}

Use the Lingua annotation evaluation standard. Check:
1. token partition
2. single-root DAG validity
3. schema validity
4. predicate-function-argument correctness
5. entity and modification hierarchy
6. expression, attribute, apposition, reference, logical relation, coordination/list structures
7. document fragments, field labels, titles, proper names, and other non-sentential fragments
8. semantic faithfulness
9. acceptable variants

Return the evaluation JSON only.
```

---

## 16. 快速判定清单

```text
[ ] 所有 token 被覆盖且不重叠
[ ] 辅助 symbol node 合理
[ ] 有且只有一个 root
[ ] 图无环
[ ] 所有核心节点从 root 可达
[ ] 节点类型合法
[ ] 边标签合法
[ ] as: 前缀语义方向可恢复
[ ] 节点引用按 SELF/DESC 语义解释，wrapper 的 DESC 可恢复目标 analytic unit
[ ] 主事实存在，或样本已按合法 structured/name/title fragment 处理
[ ] 核心论元完整
[ ] 每个 predicate 的 pred.arg.1 已补足；无 arg1 的情况有文本或结构依据
[ ] 论元编号基本正确
[ ] 复杂实体没有被过度压平
[ ] Modification 的 variable/body 正确
[ ] Modification 的 head、scope、层级正确
[ ] 多重修饰的内外层顺序符合语义亲疏，近层修饰在内，远层修饰在外
[ ] 前置修饰成分没有误接到后续嵌套事实或误作核心论元
[ ] 冗余 Modification 已在不损失可恢复性的前提下去除
[ ] 相对从句有内部事实结构
[ ] 指代关系 ref/as:ref 正确
[ ] 属性结构 entity/value 正确
[ ] Expression 未误标为 Modification
[ ] 逻辑连接词未遗漏
[ ] 并列/列表项未遗漏
[ ] 纯 header、field label、table row、UI fragment、文件名、地址、联系方式等没有被强行解析为普通命题
[ ] 专名、标题、名词性 fragment 没有被虚构 copula 或普通事实谓词
[ ] 没有添加原句没有的信息
[ ] 没有遗漏关键语义
```

---

## 17. 最小修正原则

当候选标注有错误时，评估器应优先给出最小修正，而不是重写整张图。

最小修正优先级：

1. 修改错误 edge label。
2. 调整 edge direction 或添加 `as:`。
3. 添加遗漏的 `ref`、`attribute`、`modification` 边。
4. 将错误 node type 改为正确 type。
5. 为复杂结构添加必要 symbol node。
6. 拆分过度合并的复杂短语节点。
7. 只有在结构整体错误时，才建议重建子图。

---

## 18. Lingua 与 OIA 的评估差异

| 项目 | OIA 倾向 | Lingua 评估要求 |
|---|---|---|
| 基础骨架 | Predicate-Function-Argument | 保留 PFA，但要求更强层级化。 |
| 名词短语 | 可较扁平 | 复杂实体应显式内部结构。 |
| 修饰 | 可直接 `mod` | 推荐 `Modification(variable, body)`。 |
| 实体任务适配 | 不是核心目标 | 实体或嵌套实体应能由 `SELF` / `DESC` 恢复。 |
| 属性 | 较弱 | 显式 `Attribute` / `AttributePredicator`。 |
| 表达式 | 较弱 | 显式 `Expression` / `ExpressionFunctor`。 |
| 相对从句 | 使用 ref 与反向边 | 保留，并应包装为层级 modification。 |
| 评估重点 | OIE/OIX 表达 | 跨任务 analytic units 的 hostability。 |

核心判断：

> 一个标注即使符合 OIA，也不一定符合 Lingua；如果它无法通过节点或 descendant subgraph 恢复复杂实体、属性、嵌套事实和修饰结构，就应被视为 Lingua 标注不足。

---

## 19. 面向自动化评估的强制规则摘要

以下规则建议作为自动化评估器的 hard constraints：

1. `root` 必须唯一；现有 JSONL serialization 可由唯一入度为 0 的节点推断 root。
2. 图必须无环。
3. 所有 lexical tokens 必须被 exactly once 覆盖。
4. 所有 edge 的 source 和 target 必须存在。
5. 所有 node type 和 edge label 必须合法或可规范化。
6. 主事实谓词必须存在；但纯 structured fragment、标题、专名、field label、UI fragment 等无事实谓词样本按 fragment 规则处理，不得虚构事实谓词。
7. 核心 predicate arguments 不得缺失；predicate node 原则上必须补足 `pred.arg.1`，除非原句确实无对应成分、该谓词语义上不需要 `arg1`，或文本省略且无法由结构恢复。
8. 简单系表结构应合并为 `be + adjective/state complement` 谓词；补语内部副词修饰必须拆出并保持 scope 可恢复。`be` 直接接介词谓词时合并；`be` 与介词之间有其他成分时介词必须拆出；`there be` 必须拆分并以 `there` 作 `pred.arg.1`。
9. 相对从句必须有内部谓词结构；显式 wh-word / `that` / 指代词存在时必须有 `ref/as:ref` 或等价 `Ref` 节点结构，零关系从句不得凭空添加 `Ref`。
10. 属性结构必须明确区分 entity 与 value。
11. 复杂修饰结构不得因过度压平而失去 head 和 scope。
12. 所有修饰结构必须检查 head、scope 和层级；前置修饰成分出现时应作为高风险情况重点检查。
13. 冗余 `Modification` 应在不损失 recoverability 的前提下去除；不得用 `Modification` 包装已经自足的逻辑、并列、属性或指代关系。
14. 评估论元、修饰和 wrapper 结构时必须依据 `DESC(v)` 的可恢复性；不得仅因 `SELF(v)` 是标点、修饰词、`Modification`、`Ref` 或其他 wrapper 就判错。
15. 多重修饰的链式嵌套必须符合语义亲疏和 scope 顺序；`DESC(v)` 可恢复不能豁免近层/远层修饰内外顺序错误。
16. header、speaker label、field label、key-value label、UI fragment 等文档元数据片段不得进入正文核心 PFA；应作为 discourse metadata 或局部 boundary 保留。
17. 纯 structured fragment 可以用 `List`、`Expression`、`Attribute` 等结构作为 root，不要求普通句子的 `FactualPredicator`。
18. 多词专名、标题和名词性 fragment 不得被强行补成 copula/事实命题；其内部结构只需能恢复 entity/title 及必要限定关系。

以下规则建议作为 soft constraints：

1. 普通标点可忽略或轻罚。
2. 简单修饰可接受简化边。
3. Attribute 与 Apposition 在边界情形中可作为可接受变体。
4. 轻微标签拼写差异可规范化。
5. 论元编号在语义完全可恢复时可轻罚。

---

## 20. 示例评估输出

输入句子：

```text
the scientist who discovered the new element won the prize
```

如果候选标注将 `the scientist who discovered the new element` 整体压成一个 `NominalConstant`，则输出可为：

```json
{
  "verdict": "major_errors",
  "score": 62,
  "is_valid_lingua_graph": true,
  "fatal_errors": [],
  "summary": "The main fact is represented, but the relative-clause entity is over-flattened and cannot recover the nested fact 'who discovered the new element'.",
  "errors": [
    {
      "category": "OverFlattening",
      "severity": "major",
      "location": {
        "nodes": ["n2"],
        "edges": []
      },
      "description": "The complex entity containing an explicit relative marker is represented as a single nominal node. Lingua requires the relative clause to be represented as an internal factual subgraph and the explicit wh-word/that/referring expression to be linked to the head entity via ref/as:ref.",
      "expected_fix": "Split the entity into a Modification structure whose variable is 'scientist' and whose body is a FactualPredicator 'discovered' with arguments 'who' and 'the new element'; add an as:ref edge from scientist to who."
    }
  ],
  "accepted_variants": [],
  "minimal_correction_skeleton": {
    "nodes_to_add": ["Modification", "discovered/FactualPredicator", "who/GeneralFunctor"],
    "edges_to_add": ["Modification.variable -> scientist", "Modification.body -> discovered", "discovered.pred.arg.1 -> who", "discovered.pred.arg.2 -> the new element", "scientist.as:ref -> who"],
    "nodes_to_change": [],
    "edges_to_change": []
  }
}
```

---

## 21. 版本说明

`v0.1` 是面向 LLM 自动质检的初始规范，强调：

1. 形式合法性。
2. 语义忠实性。
3. Lingua 层级结构。
4. 与 OIA 的差异。
5. 可直接用于 prompt 的评估流程和 JSON 输出格式。

后续版本可继续补充：

1. 更细的介词语义分类。
2. 更细的逻辑关系方向规则。
3. 引语、比较级、否定、量词、范围、数学公式的专门规范。
4. 中英文混合句、中文分词与短语边界规则。
5. parser 输出格式与 graph serialization 的严格定义。
