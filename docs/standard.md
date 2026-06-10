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
| `Attribute` | 属性-值结构的无词面 symbol node，仅用于属性名未显式出现或无法由文本 token 承载的隐式属性关系。 |
| `Copula` | 系词性 predication。 |
| `Discourse` | 话语标记、插入语、话语层连接。 |
| `Expression` | 复合表达，如范围、公式、多 token 符号表达。 |
| `List` | 异质元素列表，不预设共享谓词。 |
| `Modification` | 修饰结构，将被修饰对象与修饰体打包。 |
| `Parataxis` | 并列、并置、协调结构。 |
| `Ref` | 指代、照应、相对从句中的 reference。 |
| `Vocative` | 呼语。 |
| `Whether` | 极性疑问或 whether-clause 的结构节点。 |

辅助 symbol node 的 label 与 node type 必须保持一致：

| Symbol label | Required node type |
|---|---|
| `Apposition` | `AppositionalPredicator` |
| `Attribute` | `AttributePredicator` |
| `Copula` | `FactualPredicator` |
| `Discourse` | `ModificationalFunctor` |
| `Expression` | `ExpressionFunctor` |
| `List` | `ListFunctor` |
| `Modification` | `ModificationalFunctor` |
| `Parataxis` | `ConjunctionalFunctor` |
| `Ref` | `ReferentialPredicator` |
| `Vocative` | `ModificationalFunctor` |
| `Whether` | `GeneralFunctor` |

`Expression`、`List`、`Parataxis` 等 `func.arg` 型 symbol node 必须至少有两个有效 `func.arg` 子项。若只有一个子项，该 wrapper 不提供可恢复结构信息，应删除 wrapper，并让外层关系直接指向唯一子项或能 close 该子项的 wrapper。显式 operator token 可直接作为 `ExpressionFunctor` / `ListFunctor` root，不需要再额外创建同名 symbol node 包住 operator。

当前标准不允许 `Missing` node。任何 `spans: "Missing"`、symbol label `Missing`，或用 `Missing` 作为隐式实体 / 省略论元占位的节点，都应视为非法标注。若论元无法由句内 token、控制关系、并列共享、被动/提升关系或 `DESC(v)` 负责任恢复，应让该论元缺省；不得为了补齐 schema 创建 `Missing`。

### 3.2 Node types

| Node type | 用途 |
|---|---|
| `FactualPredicator` | 事实、事件、普通子句谓词。 |
| `AttributePredicator` | 属性谓词或显式字段名，如 `battery capacity`、`Email`、`Attendance`。 |
| `AppositionalPredicator` | 同位谓词。 |
| `LogicalPredicator` | 逻辑、时间、因果、条件等关系谓词，如 `if`、`because`、`therefore`。 |
| `ReferentialPredicator` | 指代结构谓词。 |
| `ConjunctionalFunctor` | 协调、并列函数，如 `and`、`or`。 |
| `ExpressionFunctor` | 复合表达函数，如 `to` in `48 to 49`，以及日期、时间、地址、人名 / 专名、代码、公式等格式化表达里的格式 functor。 |
| `GeneralFunctor` | 一般函数，常用于 wh-word。 |
| `ListFunctor` | 异质列表函数。 |
| `ModificationalFunctor` | 修饰函数，通常对应 `Modification`。 |
| `DeterminerConstant` | 限定词，如 `the`、`a`、`this`，以及修饰名词的 possessive determiners（`my`、`your`、`his`、`her`、`their`、`its`、`our`）。 |
| `NominalConstant` | 名词、专名、数词等实体性常量。 |
| `ModificationalConstant` | 形容词、副词、名词性前置修饰等。 |
| `InterjectionConstant` | 感叹词。 |
| `PunctuationalConstant` | 标点。 |
| `SymbolConstant` | 数学符号、单位、特殊字符等。 |
| `OtherConstant` | 无法归入上述类别的 fallback。 |

功能词的 node type 应按其在句中的语义功能判断，而不是按词形固定。例如同一个词形 `if`、`that` 在不同上下文中可对应不同类型；评估器应根据它是在连接命题、引导名词性从句、指代实体，还是限定名词来判定。

纯标点或纯符号 token 不得标为任何 `*Predicator`。也就是说，`:`、`-`、`--`、`...`、括号、引号、斜杠、`&`、`@`、`#` 等节点，即使在文本中承担分隔、并置、解释、范围、格式或连接作用，也不应作为 `FactualPredicator`、`LogicalPredicator`、`AppositionalPredicator`、`AttributePredicator` 或 `ReferentialPredicator`。它们应按功能标为 `PunctuationalConstant`、`SymbolConstant`、`ExpressionFunctor` / `Expression` 的内部成分，或在 `&` 明确等同 `and` 时标为 `ConjunctionalFunctor`。若需要表达标点两侧的语义关系，应由 lexical predicate / functor 或 `Discourse`、`Parataxis`、`Apposition`、`Attribute`、`Expression` 等结构承载；标点本身只负责边界、close scope 或表达式组成。

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
| `attribute` | `Attribute` 关系的边级简写，语义上等价于显式 `Attribute` 结构的一次属性连接；不得和同一个 `Attribute` / `AttributePredicator` 节点叠加使用，也不得在一个 `Attribute` symbol node 下再用 `attribute` 边重复挂载该节点的字段名。 |
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

Dummy / extraposition `it` 也是可分析结构，不应使用 `nonsense` 粗略挂接。`It is interesting/easy/fair to ...`、`it is possible that ...` 等结构中，若 `it` 指向后置不定式或 content clause，应使用 `Ref`、`Modification` 或其他可恢复结构把占位 `it` 与真实内容关联起来。

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
| 添加 `Missing` node 表示省略或隐式实体 | major |

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
3. 除非原句确实没有对应成分、该谓词语义上不需要 `arg1`，或文本省略且无法由句内结构负责任恢复，所有 predicate node 原则上都应补足 `pred.arg.1`。
4. 对非限定谓词、分词、目的从句、相对从句、补足从句和并列共享谓词，也必须检查 `pred.arg.1`。若主语或施事可由控制、提升、并列共享、被动结构、前文 wrapper 或 `DESC(v)` 恢复，应显式连接到可恢复的节点或 descendant subgraph。
5. 不允许创建 `Missing` node 作为 `pred.arg.1` 或其他论元。若省略主语只是泛指、祈使语气、标题/fragment、目的不定式或无法从句内确定的参与者，允许该 predicate 不带 `pred.arg.1`，并在语义评分中按上下文判断；若 controller 可恢复，则应连接到 controller 或其 `DESC`。
6. 直接宾语通常为 `pred.arg.2`。
7. 间接宾语通常为 `pred.arg.3`。
8. 宾补、结果补语、被谓词选择的补足语等可用 `pred.arg.4`。
9. 复杂论元应是一个节点或一个 descendant subgraph，而不是散乱 token。

在现有 JSONL serialization 中，谓词论元有时会指向 determiner、modifier、punctuation、`Modification`、`Expression`、`Ref` 等 wrapper node。若对应实体或事实可通过 `DESC(v)` 清晰恢复，这属于合法结构变体；只有 descendant subgraph 无法恢复核心论元时，才应判为 PFA 或语义错误。

### 6.1.1 系词、`be + adjective`、`be + prep` 与 `there be`

为减少历史数据中 copula 结构的合并/拆分不一致，以下结构按统一原则处理：

1. **系词和形容词/状态补语拆开**。`is true`、`was interesting`、`are happy`、`be ready`、`is 6 years old` 等 `be + adjective/state complement` 中，`be` 或 modal/aux + `be` 保持为 `FactualPredicator`，主语为 `pred.arg.1`，形容词或可恢复的复合状态补语作为 `pred.arg.2` 的 `ModificationalConstant`。例如：

```text
it is true

is/FactualPredicator
  pred.arg.1 -> it
  pred.arg.2 -> true/ModificationalConstant
```

复合状态补语如果整体表示一个 adjectival unit，应保留为一个补语节点，例如：

```text
it is 6 years old

is/FactualPredicator
  pred.arg.1 -> it
  pred.arg.2 -> 6 years old/ModificationalConstant
```

若 `be + adjective/state complement` 的补语内部存在同质并列，`be` 仍是顶层谓词，并列结构作为唯一 `pred.arg.2`。并列中的所有 adjective/state item 应保持平行 status，通常都是 `ModificationalConstant`，不得只因为是后续 conjunct 就升格为 `FactualPredicator`。例如：

```text
they are professional, neat and clean

are/FactualPredicator
  pred.arg.1 -> they
  pred.arg.2 -> and/ConjunctionalFunctor

and/ConjunctionalFunctor
  func.arg -> professional/ModificationalConstant
  func.arg -> neat/ModificationalConstant
  func.arg -> clean/ModificationalConstant
```

纯被动、进行或 lexical predicate 不按该规则拆分；`was indicted`、`are looking`、`has been approved` 等仍作为 `FactualPredicator` 谓词短语，除非上下文和词性明确表明后项是 adjectival/state complement。

2. **副词、程度词和否定从谓词节点拆出，并保持 scope 可恢复**。`not`、`n't`、`nt`、分裂的 `n ' t`、普通 adverb/degree modifier 不应无结构地留在 `FactualPredicator` 内。它们应作为 `ModificationalConstant`，用 `as:modification` 修饰其真实 scope。对于 `be + adjective/state complement`，这些修饰通常外包在 `be` 谓词外层；补语本身仍在 `pred.arg.2`。例如：

```text
that is fairly basic

is/FactualPredicator
  pred.arg.1 -> that
  pred.arg.2 -> basic/ModificationalConstant
fairly/ModificationalConstant
  as:modification -> is
```

若实现中将 `is fairly basic`、`is not available`、`will no longer have access` 等整段压成一个 `FactualPredicator`，导致 modifier/negation scope 不可恢复，应视为过度合并。

3. **`be` 直接加介词谓词时合并**。若 `be` 与介词连续构成状态/位置/关系谓词，如 `is in`、`was at`、`is about`、`are against`、`be with`，应合并为一个 `FactualPredicator`，介词宾语作为该合并谓词的论元。例如：

```text
he is in Houston

is in/FactualPredicator
  pred.arg.1 -> he
  pred.arg.2 -> Houston
```

若 `be` 后面的介词部分本身是后文清单中的复合介词 / 固定修饰引导词，如 `out of`、`due to`、`close to`、`consistent with`，优先按该固定短语整体建 `ModificationalFunctor`，不要把 `be + 固定短语` 整段压成不可拆的 `FactualPredicator`。

4. **`be + scope modifier + PP` 中，modifier 修饰 PP relation，不修饰裸 `be`**。若 `be` 和介词之间插入副词、程度词、否定或 `no longer` 等 scope modifier，如 `is not in Houston`、`was already at home`、`is no longer in business`、`is all about this`，介词必须单独拆出为 `ModificationalFunctor`。该 PP relation 以 `variable -> be`、`body -> PP object` 表示；插入的 modifier 通过 `as:modification` 指向 PP relation 的 root。外层 parent 若选择整个被修饰的 PP predication，应指向 modifier root。例如：

```text
this location is no longer in business

is/FactualPredicator
  pred.arg.1 -> this location
in/ModificationalFunctor
  variable -> is
  body -> business
no longer/ModificationalConstant
  as:modification -> in
```

这不同于普通 verbal/passive 结构：`were no longer expressed in monocytes` 中，`no longer` 修饰 `were expressed`，`in monocytes` 是该谓词的 PP adjunct，不应改成 `no longer -> in`。

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
| `be + adjective/state complement` 被压成不可恢复的单个 predicate，导致补语或修饰 scope 不可恢复 | minor / major |
| `is very true`、`is not available` 等把副词/否定和补语压成不可恢复的单节点 | major |
| `be` 直接接介词谓词却拆成不必要 wrapper，导致关系谓词不清 | minor / major |
| `be + modifier + PP` 中 modifier 挂到裸 `be`，或 PP 的 `variable` 指向 modifier 而非 `be`，导致 PP scope 错误 | major |
| `be` 与介词之间有插入成分却整体合并，导致插入成分或 PP scope 不可恢复 | major |
| `there be` 合并为不可分析谓词，未把 `there` 作为 `pred.arg.1` | major |

`pred.arg.4` 与状语修饰必须区分：

1. 若成分是谓词语义价的一部分，或表达宾补、结果状态、被谓词选择的非限定补足语，可作为 `pred.arg.4`。例如 `painted the wall red` 中 `red`、`elected him president` 中 `president`、`attempt to leave` 中 `to leave`。
   对 `make/keep/leave/have + object + adjective/state complement` 这类宾语补语 / result-state 构式，matrix predicate 必须保留 object 为 `pred.arg.2`，并用 `pred.arg.4` 指向补语；补语本身通常是 `ModificationalConstant` 或可恢复的并列/修饰 subgraph，而不是另起一个以 object 为 `pred.arg.1` 的 `FactualPredicator`。例如：

```text
make this short

make/FactualPredicator
  pred.arg.2 -> this
  pred.arg.4 -> short/ModificationalConstant

leave soldiers dead

leave/FactualPredicator
  pred.arg.2 -> soldiers
  pred.arg.4 -> dead/ModificationalConstant
```

2. 若成分回答目的、原因、时间、方式、条件等问题，且主谓宾事实在没有该成分时仍然完整，应建模为 `Modification`、介词/状语修饰，或在有明确逻辑关系时建模为 `LogicalPredicator`。
3. 目的状语，尤其是可改写为 `in order to ...` 的不定式短语，不应作为主谓词的 `pred.arg.4`；它应作为修饰体修饰完整事件或相应事件子图。
4. 若普通目的、时间、方式状语有显式引导词（如 `to`、`on`、`in`、`for`），优先用该 `to`/介词作为有 token anchor 的 `ModificationalFunctor`：`variable` 指向核心事件或已包装的核心事件，`body` 指向状语内容。不要为了这种有锚点的结构再额外包一层无 token 的 `Modification` aux。`in order to` 等固定引导短语应作为一个整体的 `ModificationalFunctor`。
5. 当上层谓词选择的是“带有状语修饰的完整补足事件”时，上层边可以指向该介词/`to` 修饰结构的根节点；但不能把状语的 `body` 本身误当成被选择的 `pred.arg.4`。

复合介词 / 固定修饰引导词按 `in order to` 的方式处理：若整个短语在句中作为普通目的、原因、让步、依据、关系、空间、时间、方式、替代、交换、补充等修饰引导词，应覆盖连续 token 作为一个 `ModificationalFunctor`，`variable` 指向被修饰的核心事件/实体/已包装结构，`body` 指向该短语引出的内容。不要把短语内部拆成多层普通介词、名词或 `to`，也不要额外添加无 token 的 `Modification` wrapper，除非确有额外 scope 需要。

强制按整体处理的常见清单包括：

| 类型 | 固定短语 |
|---|---|
| 目的 / 用途 | `in order to`, `so as to`, `in order for`, `for the purpose of`, `for purposes of`, `for the sake of` |
| 原因 / 根据结果 | `because of`, `due to`, `owing to`, `thanks to`, `on account of`, `as a result of`, `by reason of`, `in view of`, `in light of` |
| 让步 / 例外 / 替代 | `in spite of`, `regardless of`, `irrespective of`, `apart from`, `aside from`, `except for`, `instead of`, `rather than` |
| 依据 / 符合 / 约束 | `according to`, `pursuant to`, `in accordance with`, `in compliance with`, `in line with`, `consistent with`, `subject to`, `contrary to` |
| 话题 / 关系 / 方面 | `with respect to`, `with regard to`, `in regard to`, `in relation to`, `relative to`, `in terms of`, `in connection with`, `with reference to`, `in reference to`, `as to`, `as for` |
| 空间 / 时间位置 | `in front of`, `in back of`, `on top of`, `at the top of`, `at the bottom of`, `at the end of`, `at the beginning of`, `in the middle of`, `next to`, `close to`, `far from`, `away from`, `out of`, `inside of`, `outside of`, `ahead of`, `prior to`, `subsequent to` |
| 方式 / 交换 / 代表 / 补充 | `by means of`, `by way of`, `in exchange for`, `in return for`, `in place of`, `on behalf of`, `in favor of`, `for lack of`, `in case of`, `in addition to`, `as well as`, `together with`, `along with` |

该清单不是封闭词表。遇到相同构式的固定短语（如 `prep + noun + prep`、`adjective/participle + prep`、`adverb + prep`）且整体语义不可由内部普通组合稳定恢复时，也按整体 `ModificationalFunctor` 处理。若同形短语在具体句中不是修饰引导词，例如 `as well as` 明确承担并列、`rather than` 承担逻辑对比，或 `in order` 只是普通名词短语“按顺序”，则仍按实际功能标为 `ConjunctionalFunctor`、`LogicalPredicator`、普通 nominal/modifier 等。

同形例外必须按实际构式判断，不按字符串盲合并。`is/was due to start/compete/appear` 一类是 `due` + infinitive 表示“预定/应当”，不是原因介词 `due to`；`back/drop/come/go out of`、`get/go along with`、`shy away from`、`come away from` 等 phrasal / prepositional verb 中的 particle 和介词仍按动词结构处理；`5 out of 5` 是比例/评分结构；`do next to file` 中 `next` 和目的 `to file` 分属不同成分。这些不强制合并成清单中的固定修饰引导词。

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

1. 简单专名或不可分实体可作为单节点；多词专名、标题、代码和含有可恢复中心词的专名按 6.15.2 的 `Expression` / `Modification` 标准判定。
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

简化结构在无歧义时可接受：每个单叶子修饰体直接指向被修饰对象。

```text
wooden
  as:modification -> chair
old
  as:modification -> chair
that
  as:modification -> chair
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
| 可简化 | 只有 `body` 是单个叶子节点、且无歧义时，才可用直接 `as:modification`。 |
| 不可过度简化 | 如果简化会导致 head 或 scope 不清，必须保留完整 `Modification`。 |

`Modification(variable=H, body=B)` 与 `B --as:modification--> H` 不是无条件等价。只有当该 `Modification` 只含一条 `variable` 和一条 `body` 核心边，且 `B` 是没有任何 outgoing edge 的单个叶子节点时，才允许把它压平为直接 `as:modification`。若 `body` 本身有内部结构，例如介词/`to` 修饰结构、相对从句、分词短语、带标点关闭的局部 wrapper、`ExpressionFunctor`/`ListFunctor` 子图、谓词论元结构或另一个 `Modification`，必须保留显式 `Modification`，不得仅因表面上是修饰关系就压成 `as:modification`。

若被压平的 `Modification` 原本被外部边引用，外部边必须重接到 `B`，而不是 `H`。因为压平后的等价子图是 `B --as:modification--> H`，其中 `B` 是能恢复整个修饰短语的局部 root；若外部边接到 `H`，`B` 会变成额外 root，图的单 root/DAG 约束会被破坏。

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

相对从句修饰实体时，应先建模相对从句内部事实，再通过 `Modification` 包装。若句中存在显式 wh-word、`that` 或其他指代性关系标记，应使用 `Ref` 节点表达“head entity 被该定语从句限定后形成的整体 referent”。

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

Ref/ReferentialPredicator
  pred.arg.1 -> scientist
  pred.arg.2 -> Modification
    variable -> who
    body -> discovered/FactualPredicator
```

在 compact JSONL 中，显式关系词如果在从句内部充当 subject、object、oblique 等论元成分，relative-clause `Ref` 的标准结构是：

```text
Ref/ReferentialPredicator
  pred.arg.1 -> head entity / antecedent
  pred.arg.2 -> relative-clause Modification

Modification
  variable -> explicit wh-word / that / referring expression
  body -> relative-clause predicate / functor / clause root
```

`Ref.pred.arg.1` 和 `Ref.pred.arg.2` 必须指向各自参数的可恢复根，而不是参数内部的任意 leaf。若 antecedent 后有逗号、括号、引号等局部 close wrapper，且该 wrapper 只关闭 antecedent，则 `Ref.pred.arg.1` 可以指向该 punctuation wrapper，并通过 `DESC(arg1)` 恢复 antecedent。外层谓词、修饰、属性、同位、列表等结构如果选择的是“被相对从句限定后的整体 referent”，应把语义边指向 `Ref` 本身，而不是继续指向 `Ref.pred.arg.1` 的内部节点；只有从句内部谓词/关系词角色才直接连到 wh-word、relative adverb 或 clause root 等 `Ref` 内部成分。

`who`、`which`、`whom`、`whose`、以及在从句中充当 subject/object 等论元的 relative `that` 不应直接作为 `Ref.pred.arg.2`；它们应作为 relative-clause `Modification.variable`，从句谓词或 clause root 作为 `Modification.body`。这些显式关系词如果是从句内部论元，仍必须在从句谓词中占据对应 `pred.arg.*`；不得用 `Ref` 节点替代从句内部的 wh/that 论元。

`when`、`where` 等 relative adverb 如果在从句中充当时间、地点、方式等状语成分，不应再额外创建 `Modification(variable=when/where, body=...)`。此时 `Ref.pred.arg.2` 直接指向该 relative adverb；relative adverb 必须是 `GeneralFunctor`，并用 `func.arg` 指向其引导的从句 root。这个 `func.arg` 结构已经表达了 relative adverb 与从句 root 的关系，从句 root 不应再以 `pred.arg.*`、`func.arg` 或其他子边反向指回该 relative adverb，否则会形成循环或重复角色。例如：

```text
Ref/ReferentialPredicator
  pred.arg.1 -> the place
  pred.arg.2 -> where/GeneralFunctor

where/GeneralFunctor
  func.arg -> sells/FactualPredicator
```

进入 `Ref` 相对从句结构的 wh-word / relative adverb 本身不应同时标为 `LogicalPredicator`；例如时间相对从句中的 `when` 应作为关系词 / relative adverb，而不是再用 `when.pred.arg.1/2` 承载主从句逻辑关系。

若 `that` 表达同位语从句，本身不在其引导的从句中充当 subject、object、oblique、location、time 等任何成分，则 `that` 节点本身应标为 `AppositionalPredicator`，直接表达 head noun 与该从句内容之间的同位关系，而不是创建 `Ref` aux node 或 lexical `Ref`。例如 `the fact that Tusmore was not required...`、`a statement that he had been diagnosed...` 中，`that` 本身是 lexical `Apposition`：`that.pred.arg.1 -> fact/statement`，`that.pred.arg.2 -> content clause root`。相反，`terrain that the line traverses` 中 `that` 是 `traverses` 的宾语，仍应作为从句内部论元出现。

要求：

1. 相对从句内部谓词必须正常建模。
2. wh-word、关系从句 `that` 或其他显式指代成分如果是论元，应参与谓词论元结构。
3. `who/which/whom/whose/that` 等显式关系词若在从句内部充当论元成分，必须进入 `Ref.pred.arg.2` 所指向的 `Modification.variable`，不能直接作为 `Ref.pred.arg.2`。
4. 对论元型关系词，整个相对从句作为 `Ref.pred.arg.2` 的 `Modification`，该 `Ref` 作为被定语从句限定后的整体实体参与外层谓词或修饰结构。
5. 外层结构若语义上选择完整 relative referent，应连到 `Ref`；不得一边创建 `Ref`，一边让外层谓词或修饰继续连到 `Ref.pred.arg.1` 的内部 root/leaf。
6. `when/where` 等 relative adverb 在从句中充当状语成分时，直接作为 `Ref.pred.arg.2`，并以 `func.arg` 指向从句 root；从句内部不得再用子边指回该 relative adverb。
7. `that` 表达同位语从句且不在从句中充当成分时，不走 `Modification.variable=that`；该 `that` 本身是 lexical `AppositionalPredicator`。
8. 零关系从句没有显式 wh-word / `that` / 指代词时，不应凭空添加 `Ref` 节点。此时应让 head entity 直接填补从句内部的隐含 gap；若 gap 无法由显式 head 或结构关系恢复，则保持缺省，不得添加 `Missing` node。

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
5. 当句面存在显式 expression operator 时，该 operator 节点本身就是这个 expression 的 root / functor。`,`、`:`、`-`、`/`、`.`、`@`、`?`、`=`、`&`、括号等若负责组合表达式部件，应标为 `ExpressionFunctor` 并直接以 `func.arg` 指向 operands；不要再额外创建 `spans: "Expression"` aux 只为了包住这个 operator。
6. 一个 `ExpressionFunctor` 节点可以覆盖多个分隔符 token，共同表示一个 n-ary expression；`spans: "Expression"` aux 只在没有单一显式 operator，或需要组合多个已成形子表达式且句面没有可复用 functor 时使用。
7. `spans: "Expression"` symbol node 不得只有一个 `func.arg`。只有一个成分时，外层关系应直接指向该成分或其 punctuation close wrapper。
8. hyphenated proper name / code / measure / format expression，如 `paddle-steamer`、`Al-Anbar`、`99-tonne`、`one-week`、`NF-kappa`，其中 `-` 只负责格式组合左右两项，应将 `-` 标为 `ExpressionFunctor`，以 `func.arg` 指向左右项，表达式整体再作为外层修饰体或实体部分参与结构。相反，标题、字段、话题或 discourse delimiter 中的 dash（如 `SOCCER - ...`、`P.S. - ...`）不是 compound operator，应按 punctuation close、`Discourse`、`Parataxis`、`Attribute` 或 `List` 等结构另行分析。
9. `a-b` 中若 `a` 是形容词性 / 立场性 / 否定性前置修饰词，且语义是“`a-` 修饰 `b`”而不是纯格式组合，则不使用 `ExpressionFunctor` 表达 `a` 与 `b` 的关系。此时 `a` 应标为 `ModificationalConstant`，`-` 应标为 `PunctuationalConstant` 并 close `a`：`- --as:punctuation--> a`；再用内层 `Modification(variable=b, body=-)` 表示 `a-` 对 `b` 的修饰。若整个 `a-b` 继续修饰外层 head，应再用外层 `Modification(variable=head, body=inner Modification)` 表示整体修饰关系；不要让同一个 `Modification` 节点既有 `variable/body` 又发出 `as:modification`。典型例子包括 `non-nuclear Austria`、`pro-choice rally`、`anti-war demonstrations`、`anti-DNP antibodies`。
10. 统计项 / value-label fragment，如 `42 - Number of nationalities...`、`49 - Percentage of Americans...`、`63 - Percentage of oil...`，前面的数字是 value，后面的 `Number/Percentage/...` 短语解释该数字的统计口径。这里的 dash 是格式分隔 operator，应标为 `ExpressionFunctor`：`dash.func.arg -> 数字`、`dash.func.arg -> 统计口径短语`。不得标为 `Apposition(pred.arg.1=数字-, pred.arg.2=统计口径)`，因为数字和统计口径短语并非同一 referent。除非句内另有明确 host / field 属性结构，否则也不要为了这类 fragment 添加 `Missing` host。

---

### 6.7 Attribute

属性结构用于表示：

```text
<entity, attribute_name, value>
```

其中三个角色必须区分：

| 角色 | 推荐边/表示 | 说明 |
|---|---|---|
| attribute bearer / host entity | `pred.arg.1` | 被赋予属性的实体、记录、对象或上下文 host。 |
| attribute name / field label | lexical `AttributePredicator` 自身 | 属性名，如 `Email`、`Tel`、`Job Title`、`Attendance`。 |
| attribute value | `pred.arg.2` | 属性值，如 email address、电话号码、职位值、人数值。 |

显式属性名必须由 lexical `AttributePredicator` 自身承载：也就是说，`AttributePredicator` 的 `spans` 应直接覆盖 `Email`、`Tel`、`Job Title`、`Attendance` 等字段名 token。不得创建一个 `spans="Attribute"` 的 symbol node，再用 `attribute -> field label` 把字段名挂回去；这会把同一个属性关系表达两次。

`Attribute` symbol node 只用于属性名确实未在文本中显式出现，或属性关系是结构性隐含关系的情况。此时 `Attribute` 节点自身就表示“某个未命名属性关系”，不应再添加隐式属性名或 `Missing` node。

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
6. `field label: value` 形式中，field label 是 attribute name，不是 `pred.arg.1`。若 host entity 在样本中缺失，`pred.arg.1` 应缺省，只用 `pred.arg.2` 指向字段值；不得添加 `Missing` node。
7. `value field label` 形式（如 `312-666-2372 Tel`）中，`Tel` 是 lexical `AttributePredicator`，电话号码是 `pred.arg.2`；host 缺失时 `pred.arg.1` 缺省。
8. `entity value` 形式（如 `Kim 713-853-3098`）中，实体是 `pred.arg.1`，电话号码等值是 `pred.arg.2`；若 attribute name 缺失，可用 `Attribute` symbol node 表示隐式属性关系，但不得再添加隐式属性名或 `Missing` node。不得把电话号码当作实体的同位语。
9. 单独出现的值（如单独电话号码、URL、文件名）若 host 和 attribute name 都缺失，不应强行建成 Attribute，可保留为 `NominalConstant`、`Expression` 或 fragment。
10. 冒号、破折号等只用于分隔 field label 和 value 时，是 delimiter punctuation，应挂在属性结构上作为 `punctuation`，不应作为 attribute name、host、value 或任何 `*Predicator`；除非该标点还明确 close 整个结构，否则不应上提为 root。

常见错误：

| 错误 | 严重程度 |
|---|---|
| 把属性值当作普通修饰词 | major |
| 实体和值反了 | major |
| 把 field label 当作 `pred.arg.1` host | major |
| 把 `entity value` 误标为 Apposition | major |
| 用 `Attribute` symbol node 加 `attribute -> field label` 重复表示显式字段名 | major |
| 添加 `Missing` node 作为 host、属性名或论元占位 | major |
| 把 delimiter 冒号、破折号等当作 host、attribute name、value 或 `*Predicator` | major |
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
  pred.arg.1 -> comma
  pred.arg.2 -> the coach

comma/PunctuationalConstant
  as:punctuation -> John Miller
```

或：

```text
John Miller
  appositive -> the coach
```

同位语从句中的 `that` 本身应作为 lexical `AppositionalPredicator`。例如：

```text
the fact that Tusmore was not required
```

推荐：

```text
that/AppositionalPredicator
  pred.arg.1 -> fact
  pred.arg.2 -> was not required/FactualPredicator
```

这里的 `that` 不在从句中充当 subject/object/oblique/time/location 等成分，而是连接 head noun 与其内容从句；不得标为 `Ref/ReferentialPredicator`，也不得把它放入 `Ref.pred.arg.2 -> Modification.variable`。

`Apposition.pred.arg.1` 表示同位结构的 host / base expression，判定时应按 `DESC(arg1)` 恢复 host，而不是只看 `SELF(arg1)`。若 host 后面的逗号、冒号、破折号等局部标点只 close 这个 host，例如 `Lopez, the secretary`，则 `pred.arg.1` 应指向该 close wrapper：`comma --as:punctuation--> Lopez`，`DESC(comma)` 才是带边界的 host。若某个 punctuation wrapper 的 `as:punctuation` 目标是整个 `Apposition`，该 wrapper 位于 `Apposition` 外层，只表示外层 close scope；它不是 `Apposition.pred.arg.1` 的另一种标法，也不应被塞回内部 `pred.arg.1`。节点覆盖多个标点 token 本身不产生特殊规则，必须只按实际 `as:punctuation` close target 判断 scope。

评估原则：

1. 两个成分应指向同一实体或同一 referent。
2. `pred.arg.1` 可以是实体节点、表达式节点、修饰 wrapper、或只 close host 的 punctuation wrapper；只要 `DESC(arg1)` 能恢复 host 即可。
3. 同位不是普通修饰；不能简单标为 adjective-like modifier。
4. 若括号内容是类型、职业，也可能使用 `Attribute`，评估时应允许语义等价变体。
5. `value - label` 统计项不是同位；数字和 `Number/Percentage/...` 统计口径短语不是同一 referent，应按 `Expression` 处理。

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
4. `LogicalPredicator` 的 `pred.arg.1` / `pred.arg.2` 是句法关系标签，不是原因、结果、条件、结论等语义角色标签。统一原则是：主句、矩阵句、宿主命题或未被该逻辑连接词标记的一侧作 `pred.arg.1`；由该逻辑连接词直接标记、引导或依存的一侧作 `pred.arg.2`。
5. 句首位置本身不改变连接词的语义功能。`And`、`But`、`So`、`Or`、`However` 等若仍承担并列、转折、因果、结论或选择关系，应尽量按其本意标为 `ConjunctionalFunctor` 或 `LogicalPredicator`，不得仅因位于句首、左侧论元在样本外或只显式出现一侧内容而改标为普通 `ModificationalConstant` / `as:modification`。
6. 外部或隐含论元不应为了凑二元结构而臆造；但显式可见的论元方向必须按句法功能放对。`because/if/although/though/when/while/unless/until/before/after/since/as soon as/so that` 等引导的从句作 `pred.arg.2`，主句作 `pred.arg.1`，不随语序变化。`so as to` 属于目的复合修饰引导词，按 `in order to` 一类 `ModificationalFunctor` 处理，不进入本条 `LogicalPredicator` 规则。`A so/therefore/thus B`、`A but/however B` 这类非从属连接不应分析为 “B 是主句、A 是从句”；通常以前文或左侧未标记命题 `A` 为 `pred.arg.1`，以连接词标记的后项 / 当前命题 `B` 为 `pred.arg.2`。若句首 `So/Therefore/Thus/But/However` 只有当前命题显式出现，不臆造外部前项，当前命题仍按连接词标记的一侧保留为 `pred.arg.2`。
7. `Whether` symbol node 只用于极性疑问、whether-clause 或“是否”类内容从句，类型必须是 `GeneralFunctor`，其子边为 `func.arg`。条件、因果、让步、时间等命题关系不得用 `Whether<LogicalPredicator>` 表示；应使用句面存在的 `if`、`because`、`should` 条件倒装 marker 等 lexical `LogicalPredicator`，按第 4-6 条确定 `pred.arg.1` / `pred.arg.2`。
   - `Whether` 是外层极性 / 是否 scope wrapper，不替代被询问的命题结构。`Whether.func.arg` 应指向可恢复的内部命题、短语或片段 root；该内部 root 仍按自身语义标为 `FactualPredicator`、`LogicalPredicator`、`Modification`、`Conjunction`、`NominalConstant` 等。因此在层级训练样本中，`Whether` 节点和它的 `func.arg` 子节点可以拥有完全相同的 token span；这是一组父子层级监督，不是同一 surface 的标注冲突。
   - 一般疑问 / 极性问句中若有显式问号，`Whether` 是问号的顶层结构，而问号仍必须作为 punctuation wrapper close 被询问的内部命题；不得把问号降成没有子边的 leaf。标准写法是 `Whether --func.arg--> ?/PunctuationalConstant`，再由 `? --as:punctuation--> 内部命题` 恢复问号关闭的内容；不得写成 `? --as:punctuation--> Whether`，也不得写成 `Whether --punctuation--> ?`。例如 `Are you ready?` 应为：

     ```text
     Whether/GeneralFunctor
       func.arg -> ?/PunctuationalConstant

     ?/PunctuationalConstant
       as:punctuation -> ready/FactualPredicator
     ```

   - 若句末标点是 `...?`、`??`、`!!!?`、`? "` 等包含问号的标点串，`Whether.func.arg` 应指向最外层 punctuation wrapper；标点之间继续用 `as:punctuation` 串联，最内层包含问号的 wrapper close 内部命题。问号在这里实现极性疑问 scope，但它仍是被询问内容的 close wrapper，不是 leaf。
   - 直接 wh-question 不因为句末问号而额外创建 `Whether`。`what/who/which/whom/whose/where/when/why/how` 等显式 wh-word 自身应作为 lexical `GeneralFunctor` 进入从句内部结构；若它在相对从句中作论元或状语，按相对从句 / `Ref` 规则处理。
   - 若极性疑问整体内部含有条件、因果、时间等 lexical logical marker，例如 `If X, would Y?`，外层可以是 `Whether.func.arg -> if/LogicalPredicator`，但 `if` 本身仍必须保留为 `LogicalPredicator`，不得被 `Whether` 吞掉。
   - 省略式或片段式极性问句可以使用 `Whether` 包住片段 root，例如 `Europe?`、`Any suggestions?`、`Right?`，但片段 root 的 POS / 内部结构仍按普通标准标注。
8. 只有当该词确实只是话语填充、语气组织、格式标记，且不表达并列、选择、转折、因果、条件、结论等事实关系时，才应标为 discourse / modification 类节点。

错误：

| 错误 | 严重程度 |
|---|---|
| 丢掉关键逻辑连接词 | major |
| 主句和连接词标记的从句 / 依存命题方向反了 | major |
| 条件从句误作实体修饰 | major |
| 仅因句首或缺少显式左侧论元，把有转折、结论、并列或选择语义的连接词改成普通 modification | major |
| 将 `because/if/although/when/so that` 等标记的从句放成 `pred.arg.1`，或将 `A so/therefore/thus B` 的后项 / 句首单边 `So/Therefore/Thus B` 放成 `pred.arg.1` | major |
| 把纯标点或纯符号分隔符（如 `:`, `-`, `--`, `...`, `&`）当作 `LogicalPredicator` | major |
| discourse marker 无事实作用时未标 | minor |

---

### 6.9.1 歧义功能词定类

`but`、`if`、`that` 等高频功能词应按语义功能定类。标注时不应只按词形固定类型，也不应在相同语义功能下随意漂移。

| 词形 / 用法 | 推荐 node type | 说明 |
|---|---|---|
| `but` 表示转折、对比、让步，连接两个事实或命题 | `LogicalPredicator` | 如 `A happened, but B happened`。不应标为普通并列 `ConjunctionalFunctor`。 |
| 句首 `But/However` 承接前文并表达转折、对比、让步 | `LogicalPredicator` | 不应仅因左侧命题在样本外而改成普通 modification；当前显式命题通常是转折后项 `pred.arg.2`。 |
| `because/if/although/when/unless` 等引导从句 | `LogicalPredicator` | 主句作 `pred.arg.1`；连接词直接引导的从句作 `pred.arg.2`。 |
| 句首或句中 `So/Therefore/Thus` 表示结果、结论、推论 | `LogicalPredicator` | `A so/therefore/thus B` 中前项 `A` 是未标记命题，作 `pred.arg.1`；连接词标记的后项 / 当前命题 `B` 作 `pred.arg.2`。句首单边 `So/Therefore/Thus B` 不臆造前项，显式 `B` 仍作 `pred.arg.2`。 |
| 句首 `And/Or` 表示延续并列或选择 | `ConjunctionalFunctor` | 保留并列/选择本意与 `func.arg` 结构；另一分支在样本外时，可只保留可见分支。 |
| `well`、`anyway`、无事实关系的 `so/but/and` 话语填充 | discourse / modification 类 | 仅在没有并列、选择、转折、因果、条件、结论等事实关系时使用。 |
| `if` 表示“如果”，连接条件从句与主句 | `LogicalPredicator` | 如 `if it rains, the match is canceled`。 |
| `if` 表示“是否”，引导名词性从句 | `GeneralFunctor` | 如 `ask if he came`，整体从句作为被询问内容。 |
| `that` 引导名词性从句 | `GeneralFunctor` | 如 `said that he left`、`think that it is true`。 |
| `that` 独立指代实体、事件或命题 | `NominalConstant` | 如 `I want that`、`That is true`。 |
| `that` 限定名词 | `DeterminerConstant` | 如 `that book`、`that environment`；不应标为 `ModificationalConstant`。 |
| `that` 作为关系从句标记 | `GeneralFunctor` | 如 `the place that sells food`，并应通过 `Ref` 或 `ref/as:ref` 与 head entity 建立指代关系。 |
| possessive determiner 修饰名词 | `DeterminerConstant` | `my book`、`their work`、`its quality` 中的 `my/their/its` 不应在同一语义功能下漂移为 `NominalConstant` 或 `ModificationalConstant`。 |

关系从句中的 `that` 不应仅因“指代名词”而标为 `NominalConstant`。推荐结构与 `who/which/where` 类似：`that` 作为从句中的论元参与内部谓词结构，head entity 与该关系结构通过 `Ref` 或 `ref/as:ref` 连接。

错误：

| 错误 | 严重程度 |
|---|---|
| 将转折 `but` 标为普通并列，导致转折语义丢失 | major |
| 将条件 `if` 与“是否” `if` 混淆 | major |
| 将限定名词的 `that` 标为 `ModificationalConstant` | minor / major，取决于是否影响实体结构 |
| 将关系从句 `that` 标为普通实体而没有 `Ref` 结构 | major |
| 将名词性从句 complementizer `that` 建成 `Ref` 指代结构 | major |
| possessive determiner 类型在同一功能下漂移 | consistency error |

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
6. `spans: "Parataxis"` symbol node 不得只有一个 `func.arg`。若只有一个并置项，应删除该 wrapper，或用实际 close punctuation / discourse / list / expression 结构表示可恢复范围。

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
5. 当枚举中有显式 list operator（`,`、`;`、`/`、`...`、可作列表分隔的 `-` 等）时，该 operator 节点本身就是 `ListFunctor` root，直接以 `func.arg` 指向各 list items；不要再额外创建 `spans: "List"` aux 只为了包住这个 operator。
6. 一个 `ListFunctor` 可以覆盖多个分隔符 token，表示同一个 n-ary list；`spans: "List"` aux 只用于没有显式 operator 的并列/表格字段，或需要组合多个子列表/异质字段且无单一 operator token 承载时。
7. `spans: "List"` symbol node 不得只有一个 `func.arg`。只有一个 item 时，不构成 list；外层关系应直接指向该 item 或其 punctuation close wrapper。
8. `List.func.arg` 的 item 粒度是逻辑字段，不是每个表面 token。表格行、排名行、赛果行中，队名/姓名、排名、成绩、比分、日期、年龄、角色、注释等可以作为 sibling fields；但字段内部若有可恢复结构，应先建成一个字段再进入外层 `List`：`name (country/team/role)` 先做 `Attribute`，比分/日期/范围/货币先做 `ExpressionFunctor` / `Expression`，`vs.`、cricket scorecard 的 `b Harris` 等领域记号应作为字段内部 functor / expression，而不是把 `vs.`、`b`、逗号等 marker 当成普通 sibling item。
9. `OtherConstant` 是低信息 fallback，不应作为 `List.func.arg` 的 sibling item。若该 token 是可恢复字段标签，如 `See also` 或脚注字母，应给出具体常量类型；若是 delimiter，应作为 `ListFunctor`、`PunctuationalConstant` close wrapper，或进入字段内部结构。

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

若 `Ref` 用于显式相对从句 / 定语从句，argument 型关系词（如 `who/which/that` 在从句中充当 subject/object）应通过 `Ref.pred.arg.2 -> Modification` 表示，关系词进入 `Modification.variable`，从句谓词或从句 root 进入 `Modification.body`。`when/where` 等 relative adverb 在从句中充当状语成分时，`Ref.pred.arg.2` 直接指向该 relative adverb，且该 relative adverb 以 `func.arg` 指向从句 root。

若 `that` 表达同位语从句，且本身不在其引导从句中承担任何论元或修饰成分，则 `that` 节点本身应为 lexical `AppositionalPredicator`，用 `pred.arg.1` 指向 head / host，用 `pred.arg.2` 指向从句内容；不要标为 `ReferentialPredicator`，也不要另造 `Ref` aux node 再把 `that` 当作 `Modification.variable`。

评估原则：

1. 指代词与先行词必须连接。
2. 相对从句中显式出现的 wh-word、`that` 或其他指代性关系标记也应通过 `ref/as:ref` 或等价 `Ref` 节点连接 head entity；零关系从句不应凭空添加 `Ref`。
3. 指代关系不应代替谓词论元关系；二者都应存在。
4. Coreference 的语义对象错误是严重错误。

---

### 6.13 Missing Node Policy

当前标准不允许 `Missing` node。`Missing` 不再用于处理省略、椭圆、省略主语、field/key-value 缺失 host，或任何无法由文本 token 承载的隐式实体。

处理原则：

1. 若论元、controller、host、attribute bearer 或 implicit participant 可由句内显式 token、控制关系、并列共享、被动/提升关系、相对从句 head，或某个 wrapper 的 `DESC(v)` 负责任恢复，应直接连接到该显式节点或 descendant subgraph。
2. 若无法负责任恢复，应让对应论元缺省；不得为了满足形式完整性而创建 `Missing`。
3. field/key-value fragment 中，field label 仍是 lexical `AttributePredicator`，value 是 `pred.arg.2`；缺失的外部 host 不标注。
4. 祈使句、标题、短 fragment、泛指不定式、目的状语、问候/签名/感谢等若没有句内可恢复的 subject-like participant，可以缺少 `pred.arg.1`。
5. 任何 `spans: "Missing"`、symbol label `Missing`，或以 `Missing` 作为边目标的结构都应判为非法标注并删除。

判错：

| 情况 | 严重程度 |
|---|---|
| 可恢复 controller / host / antecedent 却未连接 | major |
| 无法恢复的论元保持缺省 | 正确 |
| 添加 `Missing` node 表示省略、host 或隐式实体 | major |
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
  pred.arg.2 -> the door
  vocative -> John
  discourse -> please
```

说明：祈使句的 subject-like participant 未在句内显式出现，当前标准不创建 `Missing/you`；若上下文中有显式 controller，才连接到该 controller。

评估原则：

1. 呼语不是主语，除非句子语义确实如此。
2. `please` 等 discourse marker 不应作为事实核心论元。
3. 话语关系影响语气但不改变核心事实时，错误可判 minor。

---

### 6.15 Document Metadata / Header Fragments

训练数据中可能出现 section header、speaker label、field label、key-value label、UI fragment 等文档元数据片段与正文内容拼接在同一个样本中。此类片段应保留 token coverage，但不得强行解释为正文命题的普通 predicate、argument 或 modifier。

基本原则：

1. 若片段只是栏目名、页面标题、字段名、角色标签或界面控件文本，且不参与正文事实语义，应作为低语义权重的 discourse metadata 处理。
2. 这类 metadata token 可以建成 `NominalConstant` 或必要时的轻量 wrapper，并通过 `as:discourse` 连接到正文子图，以保持 single-root DAG 与 token coverage。compact JSONL 中优先使用 `context --as:discourse--> core`，其中 `context` 是 topic/header/source/speaker/discourse marker 本身或其 punctuation close wrapper；外层若需要“带语境的正文”，应指向该 `context` wrapper，使 `DESC(context)` 恢复完整话语。避免新标注写成 `core --discourse--> context`。
3. metadata 不得作为正文事实谓词的主语、宾语、补语、普通修饰体或逻辑连接词，除非文本明确表达它参与该事实。
4. `Overview Since 1992, P`、`Biography X was born...`、`History P` 等结构中，`Overview` / `Biography` / `History` 是 header，不应修饰 `Since`、`X` 或 `P` 的核心谓词；正文命题应独立恢复。
5. 冒号、破折号等分隔符若只是 header 与正文的边界，应作为局部 punctuation / discourse boundary，而不是事实谓词、逻辑谓词或同位谓词。
6. 若样本本身是纯 key-value / field-value 记录，没有独立正文谓词，应优先用覆盖 key / field label token 的 lexical `AttributePredicator` 表示该元数据关系；只有属性名未显式出现时才使用 symbol node `Attribute`。value 对应属性值，冒号、破折号仅作为 boundary punctuation。
7. 若 key-value 中缺少被赋予属性的外部实体，不得把字段值提升为 `pred.arg.1` host；字段值始终是 `pred.arg.2`。无法从句内恢复的上下文 host 保持缺省，不添加 `Missing` node。

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

或等价地由 `Overview` 或关闭 `Overview` 的 punctuation wrapper 通过 `as:discourse` 挂接正文 root；只要 `DESC(root)` 能恢复正文命题且 `Overview` 不进入核心 PFA。新标注不使用裸 `discourse` 边表示这类 topic/header metadata。

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
2. 页码、编号、比分、战绩、日期时间、时间范围、地址、电话号码、email、URL、文件名、百分比、货币数值、模型号、化学式、公式、专名缩写等常规符号串可用 `ExpressionFunctor` / `Expression` 打包为一个可恢复 expression unit。format 本身也是一种 functor：`HH:MM`、`HH:MM:SS`、`M:SS.xx`、日期、地址、姓名 / 专名、代码、计算式等表达式内部的分隔符或操作符，若负责把两侧部件组合成一个格式表达，应作为独立 `ExpressionFunctor` 连接两侧部件；这个 operator 节点本身就是该 expression 的 root，不得把整个格式串无结构地合并为一个 `NominalConstant`，也不要再额外创建 `Expression` aux 只为了包住它。
3. 列表中的显式分隔符同理：若 `,`、`;`、`/`、`...`、可作列表分隔的 `-` 等负责连接 list items，该分隔符节点应作为 `ListFunctor` root 并直接以 `func.arg` 指向各项。一个 `ListFunctor` 可以覆盖多个分隔符 token；不要把分隔符作为 `List` aux 的普通子项，也不要误当成 close punctuation。
4. 表格/赛果/排名行的外层 `List` item 应按 field 粒度切分：纯数字列可保持 flat list；`name (country/team/role)`、`artist (note)`、`name, role, age`、`player vs. player`、`b bowler runs` 等复合字段应先做 `Attribute`、`Expression`、`ConjunctionalFunctor` / `ListFunctor` 或 close wrapper，再作为一个 item。不要把 marker token 本身作为孤立 item。
5. `key: value`、`field - value`、`Game 11: Tuesday...`、`Preseason Game B: Thursday...`、`Coach: Doug Sewell`、`Email: ...`、`Location: ...`、`Attendance : 15,150`、`Pin #: ...` 等字段记录应以字段名 token 本身作为 lexical `AttributePredicator`；字段值为 `pred.arg.2`；若外部 host entity 缺失，`pred.arg.1` 保持缺省，不添加 `Missing` node。字段名不是普通事实谓词时，不应为了连接 key/value 而把它标成 `FactualPredicator`，也不得用 `Attribute` symbol node 加 `attribute -> field label` 重复表示字段名。
6. 冒号、破折号、逗号、斜杠、括号、句点若只是 fragment 内部 boundary 或 close marker，应作为局部 punctuation / expression boundary，不得成为事实谓词、逻辑谓词、同位谓词、属性谓词或其他 `*Predicator`。
7. 纯装饰符号、乱码、截断严重、问候/签名、媒体边界、版权声明等低信息片段仍有片段语义，应优先保留为低语义信息 fragment；除非上游数据集有明确的非语义清洗策略，不应仅因信息量低而过滤。
8. 对问候、签名、感谢、残句等仍能看出普通语法或省略句结构的片段，应先按省略结构解析；若省略论元无法由句内结构恢复，则保持缺省，不添加 `Missing` node。不得因为它是邮件结束语、寒暄语或低信息文本就直接平铺为 `Expression`。
9. `Expression`、`List`、`Attribute`、局部 punctuation root 或 `nonsense` 只应作为 fallback：当片段内部没有可负责任恢复的谓词、修饰、呼语、介词短语、字段、并列或省略关系时，才把它作为低语义信息 fragment 打包。不得把 `nonsense` 用于可按上述规则解释的正常字段或表达式。
10. 纯 fragment 的 root 可以是普通语法结构、省略结构、`ListFunctor`、`ExpressionFunctor`、`AttributePredicator`、对应 aux node，或能关闭该 fragment 的 `PunctuationalConstant`；只要 `DESC(root)` 恢复完整 fragment 即可。
11. 纯日期、时间、日期时间和时间范围片段，如 `02/27/2001 08:23 AM`、`Feb 2005`、`September 21, 2004`、`Tuesday, March 20th.`、`September 20, 1888?`、`4:00 pm - 6:00 pm`，应表示为一个 `Expression`，其子项可按 date / time / meridiem / range boundary 或更细粒度符号部件恢复；其中 `HH:MM` / `HH:MM:SS` 的 `:`、日期中的 `/` / `-`、小数或时长中的 `.`, 数字范围中的 `-` / `to` 等都是格式 `ExpressionFunctor`，两侧部件是其 `func.arg`。不要把 date-time 内部格式关系标成普通修饰，也不要把时间范围分隔符误标为事实谓词。
12. 新闻 dateline 或来源行，如 `DUBLIN 1996-08-27`、`LOS ANGELES 1996-12-05`，应表示为 `List(location-expression, date-expression)` 或等价可恢复结构：location 和 date 是同一元数据行的两个字段，不是彼此的修饰体，也不是正文事实谓词的论元；若中间有显式字段分隔符，优先用该分隔符作为 `ListFunctor` root。
13. 纯页码/页引用片段，如 `p. 36`、`pp. 12-15`，应作为 page-reference `Expression` 或等价可恢复表达式；不要把 `p.` 和页码误当成并列 `List` 项。

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

核心区分：`Expression` 表示“这个名字 / 代码 / 标题作为一个表达式 unit 如何由表面部件组成”；`Modification` 表示“这个专名内部仍有可恢复的 head、type 和限定 scope”。判断点不是“它是不是专名”，而是“把内部部件拆开后是否能负责任地恢复 head-modifier 关系”。专名不是一律摊平成 `Expression`，但也不能为了拆开专名而把名、姓、型号片段、品牌词或官方组织名中的普通名词强行做成 predicate-argument / modifier。审查时按以下顺序判断：

1. 先判定该片段是否只是一个 entity/title/name fragment。若缺少显式谓词，不得为了满足普通句子模板而补 `is`、`be named`、`means` 等事实谓词；root 只需通过 `DESC(root)` 恢复完整专名或标题。
2. 再判定内部关系是“格式 / 身份组合”还是“head + qualifier”。若内部部件只是共同组成一个名字、代码、标题、缩写、官方名称或符号串，且没有可负责任恢复的中心词和修饰 scope，用 `ExpressionFunctor` / `Expression`。若内部有仍然承担类型/类别/中心词功能的 head，或地点、机构、人物、领域、等级等限定的 scope 可恢复，则用层次化 `Modification` / `as:modification`。
3. 简单单词专名或不可分实体可以保持单个 `NominalConstant`，不必为了形式统一强行包成 `Expression`。多词但不透明的专名、标题、品牌、作品名、人名、组织名、型号、缩写等，可以作为一个 `Expression` unit。

`Expression` 的典型使用：

1. **人名和不透明名称**：`George Vukmanovich`、`Donna Sells`、`Marquis de Condorcet` 这类人名内部通常不是 head-modifier；名、姓、贵族小品词等只是共同组成姓名，应作为 `Expression` / name unit，而不是把 first name 标成 last name 的 `body`。
2. **标题、品牌、口号、作品名和不透明官方名称**：`Great Sandwiches, Great Prices`、`PHOTOS DONE WELL`、`World Bank`、`World Hats Mart`、`Clean Ocean Action` 这类标题式或官方名称，如果上下文没有把它的内部词作为普通命题或可恢复 head-scope 使用，可作为 title/name expression；不要仅因其中含有 `Bank`、`Mart`、`Action` 等普通名词就强行拆成 `Modification`。
3. **代码、型号、缩写、URL/email/path、化学/生物标记等符号串**：内部的 `/`、`.`、`@`、`?`、`=`、`&`、`#`、`+`、括号、hyphen 等若负责把两侧部件组合为一个格式表达，应以该 operator 为 `ExpressionFunctor` root，`func.arg` 指向左右部件或内层 expression。典型包括 `ATF-1`、`Epstein-Barr`、`NF-kappa`、URL/path/email、型号中的 slash/dot/hyphen。这个 operator 节点本身就是 expression root；不要额外再加一层只包住 operator 的无词面 `Expression` aux。
4. **纯格式地名 / 地址 / 数字化名称**：当逗号、斜杠、点号只是地址或编号格式的一部分，并没有更具体的实体-限定关系需要表达时，可用 `ExpressionFunctor`。例如纯地址字段里的 `Houston, Texas` 可以按地址 expression 处理。

`Modification` 的典型使用：

1. **有可恢复中心词的专名**：若专名内部的 airport、school、university、hospital、river、mountain、province、department、district、league、final、promoter、virus 等仍承担普通名词中心词或类型词功能，应保留 head。中心词或已经带近层修饰的中心词作 `Modification.variable`；限定它的地名、机构名、人物名、领域名、等级名、类型名等作 `Modification.body` 或叶子 `as:modification` source。注意：词面像普通名词并不自动触发 `Modification`；若该词只是官方名称的一部分且没有可恢复 head-scope，则仍可用 `Expression`。
2. **有远近亲疏的多层限定**：多个限定不是平行 token 时，应按 scope 分层。`Los Angeles International Airport` 不应四词平铺成一个无层级 `Expression`：可先让 `International` 近层限定 `Airport`，再让 `Los Angeles` 这个地点 name unit 限定 `International Airport`。`NF-kappa B` 也不应把 `NF`、`kappa`、`B` 一概打平：hyphen/format 先组成 `NF-kappa`，再由该内层 name expression 限定外层的 `B`。同理，`World Cup quarter-final` 应保留 `final/quarter-final` 与 `World Cup` 的层级，不能把所有词打平成一个 list/expression。
3. **表达类型或所属领域的后项**：`Epstein-Barr virus` 中 `Epstein-Barr` 可先由 hyphen 构成 name expression，但 `virus` 是类型 head，整体应是 `Modification(variable=virus, body=Epstein-Barr-expression)`，而不是把 `virus` 也并入同一个无层级 expression。`Al-Anbar province`、`Mojave Airport`、`BamHI W promoter` 等也应优先检查是否有 recoverable head。
4. **专名中的格式 operator 不排斥外层修饰**：hyphen、slash、dot 等可以在内层作为 `ExpressionFunctor`，但若整个 formatted name 再限定一个 head，应在外层用 `Modification`。也就是说，`Expression` 解决“部件如何组成名字片段”，`Modification` 解决“这个名字片段如何限定另一个 head”。

边界和限定说明：

1. 含有职称、角色、括号说明、地理限定或消歧义说明时，可用 `Attribute`、`Apposition`、`Modification` 或 discourse metadata 表示其对核心实体的限定；选择取决于该说明是身份属性、同位描述、地理限定、类型限定还是数据源消歧义。
2. 逗号连接的地名/机构名如 `Jordan, New York`、`Beetham Tower, Manchester` 通常不是并列列表，而是 entity + location / qualifier；不能简单当作两个无关 list items。若该逗号只是地址格式 operator 且没有更具体的 head/qualifier 关系，可作为 `ExpressionFunctor`；若表达实体的地理限定，应保留限定关系。
3. 表格/赛果/名单行中出现 `name ( country/team/role ) score`、`rank. name ( country ) result` 这类结构时，括号项是前一实体的限定属性，应按 6.16 的 `A(B)` parenthetical pattern 建模：外层保留 `Modification(variable=name, body=paren-wrapper)` 表达括号整体修饰 name，括号 wrapper close 内部的 `Attribute` / `Apposition` / lexical relation。外层 `List` 应把这个能恢复 `name + 括号限定项` 的 wrapper/field 作为一个 field，而不是把 `name` 与括号项拆成两个 sibling `func.arg`。只有括号项出现在行首、句内没有可恢复宿主，或括号内容是整行 note（如 `( halftime 2-0 )`、`( home team in CAPS )`）时，才不要强行挂到前一实体；整行 note 应包装整行结果或 header。
4. 如果专名或标题内部边界无法可靠判断，只要 token coverage 完整、root 可恢复整个名词性片段、没有添加未出现事实，可判为 acceptable variant 或 minor，不应直接判 major。但当内部中心词和修饰层级清楚时，误打平成无层级 `Expression` 会丢失 head/scope，应判为 `EntityStructure` / `Modification` 问题；当内部只是姓名、代码或标题组成时，误拆成 `Modification` 同样是结构错误。

判错：

| 情况 | 严重程度 |
|---|---|
| 多词专名作为 `Expression` / entity fragment 可恢复 | 正确 |
| 为专名/标题虚构普通事实谓词 | major |
| 将有清楚中心词和修饰层级的专名无条件摊平成 `Expression` | minor / major |
| 将人名、代码、标题等无 recoverable head 的内部部件强行标成 `Modification` | minor / major |
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

核心限制：标点 / 符号节点可以是结构边界、局部 root、表达式部件或连接符号，但不能自己作为自然语言谓词。只要 `SELF(v)` 只包含标点或符号字符，原则上不得标为 `FactualPredicator`、`LogicalPredicator`、`AppositionalPredicator`、`AttributePredicator` 或 `ReferentialPredicator`。

具体处理：

1. `field label: value`、`header - body`、`speaker: utterance` 中的冒号、破折号是 delimiter punctuation。字段名、标题或 speaker label 若需要建模，应由 lexical token 自身承担 `AttributePredicator`、metadata 或 discourse 角色；分隔符只挂 `punctuation` / `as:punctuation`。
2. `A - B`、`A -- B`、`A: B`、`A ... B` 若表达解释、承接、并置或话语推进，不得把 `-`、`--`、`:`、`...` 标成 `LogicalPredicator` 或 `AppositionalPredicator`。可恢复的关系应由 `Discourse`、`Parataxis`、`Apposition` 或 lexical connector 承载；若关系不可稳定判断，只保留 punctuation boundary。
3. 括号、方括号、引号包住同位、属性、别名、插入语或从句时，括号 / 引号本身是 `PunctuationalConstant` wrapper；内部关系按 `Attribute`、`Apposition`、`Modification`、`Discourse` 或普通谓词结构建模。
4. 日期、时间、比分、范围、编号、地址、人名 / 专名、URL、email、货币、百分比、文件名、代码、化学式、公式等格式化表达中的 `:`, `-`, `/`, `.`, `,`, `@`, `#`, `%`, `$`, `+`, `=`, `*`, 括号等不得为了连接格式两侧而标成 `FactualPredicator`。若符号本身承担可恢复的格式组合功能，应标为 `ExpressionFunctor`：例如数字时间 / 时长 / 比分中的 `:`，日期中的 `/` / `-`，小数、版本号或姓名缩写中的 `.`, 数字千分位或日期 / 地址中的 `,`, hyphenated proper name / code 中的 `-`, URL / email / file path 中的分隔符，计算式和化学 / 生物标记中的操作符或 compact parentheses。列表分隔符若连接 list items，应同理作为 `ListFunctor` root。若符号只是字段 delimiter、话语 close marker、普通句法括号 / 引号，或不可拆分符号串的一部分，则按 `PunctuationalConstant` / `SymbolConstant` 或 expression/list 内部成分处理。
5. `&` 若在句中等同 `and` 连接两个并列项或命题，应标为 `ConjunctionalFunctor`；若只是品牌、代码、标题或符号串的一部分，则进入 `Expression` / `SymbolConstant`。不得标为 `LogicalPredicator`。
6. 形式逻辑、数学或编程表达里的符号算子也应优先作为 `ExpressionFunctor`、`Expression` 部件或 `SymbolConstant` 处理；除非后续标准专门定义，不把这些符号提升为自然语言 `*Predicator`。

标点 / 符号不是一律作为外层关系的 `punctuation` child。必须先判断它是否 close 某个局部成分：

1. 若 `A: B`、`A - B`、`A -- B`、`A ... B` 中的标点能自然视为 `A` 的 close marker，则该标点应作为 `A` 的局部 wrapper：`p/PunctuationalConstant --as:punctuation--> A`，外层 `Parataxis` / `Apposition` / `Attribute` / `Discourse` 等关系再把 `p` 作为前项或对应论元。不得写成外层关系 `--punctuation--> p`，否则 `DESC(p)` 不能恢复被 close 的前项。
2. 若括号、方括号、引号等 punctuation wrapper 的实际 close target 是 `B`，则 `p --as:punctuation--> B`，外层同位、属性、并置或修饰关系以 `p` 作为被 close 的括号项 / 引号项。不要仅凭表面标点数量臆造 scope；以实际 wrapper 节点和 `as:punctuation` 目标为准。
3. 只有当标点确实属于整个外层关系、无法可靠归入某个局部成分，或只是非结构性装饰时，才可作为外层关系的普通 `punctuation` child。
4. 一般疑问 / 极性问句的句末问号应在 `Whether` 之下、内部命题之上：使用 `Whether --func.arg--> ?` 与 `? --as:punctuation--> 内部命题`。不要使用 `? --as:punctuation--> Whether` 让问号成为整句 root，也不要使用 `Whether --punctuation--> ?` 让问号退化为 leaf；包含问号的复合句末标点同理保留 punctuation wrapper 链。

核心判定：一旦存在 `p/PunctuationalConstant --as:punctuation--> child`，就表示 `p` 是 `child` 的 close wrapper，`DESC(p)` 才是“带标点关闭的 child”。若外层谓词、functor 或 aux relation 需要这个带标点关闭的局部成分，应让外层语义边（如 `pred.arg.*`、`func.arg`、`variable`、`body`、`as:*` 展开后的对应边）指向 `p`，而不是只写成 `parent --punctuation--> p`。同一个 `p` 同时作为外层普通 `punctuation` child、又用 `as:punctuation` close 内部 `child`，通常是错误的双重身份；应改为外层关系指向 `p`，或在 `p` 已经有正确语义父边时删除冗余的 `parent --punctuation--> p`。

若某个标点可以确定为 close 某个结构，该 `PunctuationalConstant` 应作为该结构的局部 root，并用 `as:punctuation` 指向被关闭的内容。close 的范围必须尽量精确：句尾标点若只关闭第二个分句，就应成为第二分句的 root，而不应上提为整句或外层 discourse relation 的 root。只有无法确定 close 范围、或标点只是非结构性装饰时，才可作为普通附着项处理。

`A(B)` parenthetical pattern 的标准写法如下。若括号内容 `B` 是对左侧宿主 `A` 的补充、限定、说明、属性、同位、别名、分词/关系从句或短插入语，应保留一个外层 `Modification` 表达“括号结构整体修饰 A”：

```text
M/Modification
  variable -> A
  body -> P

P/PunctuationalConstant("(" ")")
  as:punctuation -> C
```

其中 `P` 是左右括号组成的 punctuation wrapper，`C` 是括号内完整语义关系的 root，`DESC(P)` 应恢复带括号关闭的 `C`。不得在同一个 `M` 下再添加 `M --modification--> C`；这会把“外层修饰包装”和“括号内语义关系”表达两次，并造成同一 token set 在层级学习中出现重复解码目标。

`C` 的选择按以下顺序判定：

1. 若 `B` 内部已有 lexical predicate / functor / relation node 能把 `A` 作为论元、变量或共享参数恢复出来，则 `C` 就是这个内部 root，不再额外插入 `Attribute`、`Apposition`、`Parataxis` 或第二层 `Modification`。例如 `A ( born 1976 )`、`A ( opened 2003 )`、`A ( completed 1997 )` 中，`born/opened/completed` 应直接以 `A` 为 `pred.arg.1`，日期/年份为 `pred.arg.2`，括号 wrapper `P --as:punctuation--> born/opened/completed`；不得把 `born 1976` 打成 `Expression`，也不得再用 `Attribute` 包一层。`a negative (representing positive liquidation)` 中，`representing` 已有 `pred.arg.1 -> a negative`，因此 `P --as:punctuation--> representing`，不再插 `Parataxis`。
2. 若 `B` 本身只是属性值、别名、类型、国家/队伍/角色等，内部没有节点能表达它与 `A` 的关系，则创建合适的关系 root `C`，如 `Attribute`、`Apposition`、`Parataxis`、`Expression` 或 `List`。此时 `C` 负责连接 `A` 与 `B`，例如 `C.pred.arg.1 -> A`、`C.pred.arg.2 -> B_root`，再由 `P --as:punctuation--> C` close 这个完整关系。
3. 若括号内容是整句 note、前置说明、表格 header 说明或没有明确左侧宿主的补充，不强行套到最近的 `A`；应按整行/整句的 discourse、list、attribute、expression 或普通谓词结构建模。

因此，`A(B)` 的外层 `Modification` 与括号内 `C` 不是二选一：`M` 表达“这个带括号的补充结构修饰 A”，`C` 表达括号内容内部如何解析。层级学习或解码时，第一次查询某个 token/node set 使用 `default` mode，输出上层 `Modification` / `Whether` 等 wrapper；若后续需要展开某个 `DESC(child)` 与当前输入 token/node set 相同的 child，则统一使用 `content` mode，不再区分 `content:2` 等深度编号。该 content 查询必须复用父层预测边体系作为输入上下文：child 覆盖的 token 填入其父边角色（如 `body`、`func.arg`），与 sibling 共享的 token 另标其父层共享角色（如 `variable`、`pred.arg.1`）。这样同一 token set 的 wrapper view 与 content view 由父层 query context 区分，不应被视为标注冲突，也不需要在正式边体系中新增 `*_REF` 边。

当一个节点用 `as:*` 反向边压缩表达某个关系，同时该节点还带有局部标点时，应区分“关系整体的 root”和“被标点 close 的局部成分”。若标点只关闭该节点本身所代表的局部成分，而不是关闭整个外层关系，则应优先恢复对应 aux node，并让标点作为局部 wrapper 进入 aux 的相应论元位置：

```text
x
  as:modification -> y
  punctuation -> p
```

应规范化为：

```text
Modification
  variable -> y
  body -> p

p/PunctuationalConstant
  as:punctuation -> x
```

其中 `p` 的 `DESC(p)` 恢复被标点关闭的局部修饰体（如 `Now,`），`Modification` 才恢复“带标点的修饰体修饰 y”的外层关系。`as:discourse`、`as:vocative` 等同理：先恢复 `Discourse` / `Vocative` 等 aux node，再把局部标点 wrapper 放入 `body` 或对应局部成分位置。不得把局部逗号、引号、括号等直接作为整个外层关系的 root，除非该标点确实 close 整个外层关系。

`as:*` 是 auxiliary relation 的压缩边。凡是出现 `A --as:*--> B`，`A` 的 outgoing edge 必须且只能是这一条 `as:*` 边；`A` 不得同时还有 `func.arg`、`pred.arg.*`、`variable`、`body`、`repeat`、`punctuation` 或第二条 `as:*` 子边。若 `A` 自身需要内部结构（如 `ExpressionFunctor` 还要连接 `func.arg`，标点 wrapper 还要 `as:punctuation` close 局部成分，或谓词仍有论元），不得使用压缩 `as:*`，必须显式恢复对应 aux node：例如用 `Modification(variable=B, body=A)` 表示带内部结构的 modifier，用 `Discourse(variable=core, body=context-wrapper)` 表示带局部 close punctuation 的 discourse/topic/header。压缩 `as:*` 只用于 source `A` 是无其他子节点的原子/叶子 wrapper 的情况；对应到 `Modification` 时，也就是只有 `body` 是单个叶子节点时，`Modification(variable=B, body=A)` 才可等价压成 `A --as:modification--> B`。

判错：

| 情况 | 严重程度 |
|---|---|
| 普通句号未标 | minor / 可忽略 |
| 括号属性未建模 | major |
| 逗号同位误解析 | major |
| 列表标点导致枚举项遗漏 | major |
| `as:*` 压缩关系中局部标点直接附着在 lexical node 上，导致 close 范围不清 | minor / major |
| 纯标点 / 符号节点被标成任何 `*Predicator` | major |

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
wooden
  as:modification -> chair
```

代替完整：

```text
Modification
  variable -> chair
  body -> wooden
```

但这种等价只适用于 `body` 是单个叶子节点的情况。复杂修饰、多个 scope、介词嵌套、相对从句、分词短语、带局部标点 wrapper 的修饰体、`Expression`/`List` 子图、谓词论元结构、或 `body` 本身已有 outgoing edge 的场景不能过度简化。

### 7.2.1 冗余 `Modification` 的判定与去除

`Modification` 的作用是显式打包“被修饰对象 + 修饰体”，帮助恢复 head、scope 和 analytic unit。若一个 `Modification` 节点没有提供新的 head、scope 或 recoverability 信息，只是把已有关系再包一层，则应视为冗余结构。

可判为冗余的情况：

1. `Modification` 只有 `variable` 和 `body` 两条核心边，且二者关系已经由 `body` 节点自身的语义边完整表达。
2. `body` 是 `LogicalPredicator`、`ConjunctionalFunctor`、`AttributePredicator`、`ReferentialPredicator` 等非修饰关系节点，而 `Modification` 只是把该关系重新包装成普通修饰。
3. 删除该 `Modification` 后，把原来指向它的 parent 改指向真正的关系 root，图仍保持 single-root DAG，且所有核心事实、实体、scope 都可通过 `DESC` 恢复。
4. `Modification` 的存在只是为了让某个标点、wrapper 或 root 连接到图中，而不是表达真实修饰关系。
5. `Modification.variable` 和 `Modification.body` 的 `DESC` 恢复同一组 token，说明外层 aux 没有新增修饰体或 scope，应删除该空包裹。
6. `Modification.body` 本身是 `ModificationalFunctor` / `Modification`，且该 body 已经用自己的 `variable` 边指向与外层 `Modification.variable` 相同的可恢复单位时，外层 `Modification` 是重复包裹；外层 parent 应直接指向该 body。典型例子是 `according to`、`prior to`、`close to`、`following`、`so as to` 等 anchored modifier 已经表达了 `variable/body`，外面又套一个无 token `Modification`。
7. 若 `Modification.body` 已经可以从 `Modification.variable` 的内部边到达，说明 body 已经是 variable 所恢复结构的一部分（常见于 relative clause 已通过 `Ref` 接入，或 complement/adjunct 已作为 predicate argument 接入），外层 `Modification` 又重复挂同一个 body。此时外层 parent 应直接指向 `variable`，再删除该 aux wrapper。

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
7. **特殊结构检查**：检查 `Modification`、`Expression`、`Attribute`、`Apposition`、`Ref`、`LogicalPredicator`、`Parataxis`、`List`、`Discourse`、`Vocative`，并确认不存在 `Missing` node。
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
| 相对从句 | 使用 `Ref` / `as:ref` 等价结构 | 保留；argument 型 wh/that 通过 `Ref.pred.arg.2 -> Modification` 包装，`when/where` 状语型 relative adverb 可直接作 `Ref.pred.arg.2` 并以 `func.arg` 指向从句 root。 |
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
7. 核心 predicate arguments 不得缺失；predicate node 原则上必须补足可由句内结构恢复的 `pred.arg.1`。若原句确实无对应成分、该谓词语义上不需要 `arg1`，或 subject-like participant 无法负责任恢复，不得添加 `Missing` node；field/key-value fragment 的缺失 host 也保持缺省。
8. `be + adjective/state complement` 中 `be` 保持为 `FactualPredicator`，补语作为 `pred.arg.2`；副词、程度词和否定应拆成 `ModificationalConstant` 并保持 scope 可恢复。`be` 直接接介词谓词时合并；`be + scope modifier + PP` 中 modifier 应修饰 PP relation，PP 的 `variable` 指向 `be`；`there be` 必须拆分并以 `there` 作 `pred.arg.1`。
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
