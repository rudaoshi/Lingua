# 审查记录与标准定位

本契约属于 `$lingua-graph` 的在线 Mode B。入口是 `scripts/review_online.py`；最终结果使用 `schema_version: lingua-parse-validator.online/1`。既有 Mode A/C 的报告继续使用 `output-schema.json`。

Skill 名称和路径已统一为 `lingua-graph`；已发布的 JSON schema 标识保持不变，使既有回执、结果和重复生成检查仍兼容。标识里的旧名称不是另一个 skill 入口。

## 标准读取路线

按 [SKILL.md](../SKILL.md#required-references) 选择仓库规范或随 skill 分发的快照。运行目录的 `standard.md` 冻结本次实际使用的版本。

| 检查对象 | 标准章节 |
|---|---|
| SELF、DESC、入边需要的完整 analytic unit | 0.1、1 |
| compact serialization、symbol/type、边标签 | 2.1、3.1–3.4 |
| token partition、单根 DAG、不可增删原义 | 4.1–4.3、5.1–5.2 |
| 谓词/论元、copula、否定、控制、adjunct | 6.1、6.1.1 |
| 实体、Modification、语义亲疏、介词归属 | 6.2–6.4 |
| relative、Expression、Attribute、Apposition | 6.5–6.8 |
| 逻辑、协调、列表、指代、禁止 Missing | 6.9–6.13 |
| fragment、专名、标点 close、压缩 as:* | 6.14–6.16 |
| 合法变体、冗余、最小修正 | 7、17 |

注意：`as:*` 表达反向/压缩关系，不能只反转遍历方向计算 DESC。DESC 始终沿实际存储的有向边；语义解释再依 `as:*` 规则恢复。当前 6.16 要求 `as:*` 的 source 不得同时有其他 outgoing edge。
可达 token 只是诊断材料，不能替代局部 head、论元、内部关系与 scope 的审查。

## review.json

从**最终候选**的 inspect 模板开始；再次编辑候选后需重新 inspect 并复核受影响内容。顶层：

- `standard_sha256`、`original_sha256`、`candidate_sha256`：不得手动伪造，必须与 packet 一致。
- `reviewer`：实际执行语言学判断的审查者；`summary`：本次整体判断。
- `reviews.original`、`reviews.candidate`：每个节点、每条边、七个语义维度、每条程序 finding。
- `issues`：原图发现的问题及处理，不要修掉后从报告中删掉。
- `edit_reasons`：每个 `/graph/nodes/N` 或 `/graph/edges/N` 改动一条，连到 `issues`。
- `unresolved`：明确写出语义待定、规则冲突或尚未修复的问题；无问题时为空数组。

每个 node/edge record：

```json
{
  "index": 0,
  "status": "correct_variant",
  "assessment": "SELF 是限定词；DESC 恢复完整的 the building 名词短语，作为 is 的主体论元。",
  "reason": "入边要求主体实体，限定词 wrapper 的 DESC 包含中心名词且未越入谓词或补语。",
  "rules": ["0.1", "6.1", "6.2"]
}
```

该例只说明记录格式，不能机械套用到节点 0。node assessment 写 SELF/DESC 解释；edge assessment 写该边所需单位、实际目标子图与语义角色。编号是对应原图或候选图的数组下标。

status 允许 `correct`、`correct_variant`、`minor`、`major`、`blocker`、`needs_human_decision`。仅前两个可用于无缺口的候选。维度不适用时 status=`correct`，reason 写原句为何没有该结构；不是跳过检查。
每条 record 都需非空 reason 与存在于标准的章节号 rules；对程序 finding 也必须说明接受警告、修正或仍未解决的原因。

问题与改动格式：

```json
{
  "issues": [{
    "id": "I1",
    "severity": "major",
    "location": "original.nodes[7]",
    "explanation": "is suggestive 合成单谓词，形容词补语无法独立恢复。",
    "resolution": "拆成 is 与 suggestive，由 pred.arg.2 选择完整形容词补语。",
    "reason": "当前是系词加形容词结构，不是被动或进行谓词。",
    "rules": ["6.1.1"]
  }],
  "edit_reasons": [{
    "path": "/graph/nodes/7",
    "issue_ids": ["I1"],
    "reason": "保留原谓词下标并将 spans 收窄到 is，形容词另行追加。",
    "rules": ["6.1.1", "17"]
  }]
}
```

所有实际变化都会由工具重新计算；额外变化或缺少理由会阻止 finalize。保留 words/meta 是本版工程边界，不是宣称标准禁止纠正 tokenization。

## 与线上服务和自动检查的关系

接口发现：HF Space metadata → `/config` → `/gradio_api/info`。
调用：`POST /gradio_api/call/process_sentence`，`{"data":[sentence]}`；再 GET event_id 对应 SSE，等待 `complete`。
参考：[Gradio 官方 curl 指南](https://www.gradio.app/guides/querying-gradio-apps-with-curl)。

网络失败、API 漂移和 Word-Lingua fallback 不允许回填伪造在线结果。
预览渲染失败但完整 JSON 存在时，仍可审查 JSON；保留服务状态。
程序检查器包含部分语言结构启发式，它不是完整语言学证明。候选仍有 blocking finding 时返回 needs_human_decision；若判断是检查器与标准冲突，应写出冲突、标准依据和候选，不能偷偷改全局检查器以取得通过。
图太大（本版 512 节点、2048 token）或格式无法展开时明确阻断，不裁剪成部分图。
