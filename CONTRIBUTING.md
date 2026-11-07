# 贡献指南

## 开发环境

```bash
# 需要 Python 3.12+（开发机 3.14 也可）
git clone <仓库地址> llm-relay && cd llm-relay

# venv + 依赖（5 个运行期依赖 + pytest）
uv venv
uv pip install -r requirements-dev.txt     # 含 requirements.txt + pytest
# 没有 uv 就：python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

跑一次全量测试确认环境没问题（**完全离线**，不需要任何真实 API Key）：

```bash
python -m pytest -q          # 应全绿
```

测试在本机起一个真 HTTP 假上游（`tests/fake_upstream.py`，同时提供 OpenAI 和
Anthropic 两套接口）+ 真 relay 实例，走真实请求链路验证——所以测试绿 = 端到端绿。

## 改完代码的验证流程（每次都要过）

1. **后端改动**：在 `tests/test_relay.py` 追加回归测试（编号规则见下），全量 pytest 绿。
   - 测试夹具在 `tests/conftest.py`：`client`（TestClient）、`hdr`（Bearer 头）、
     假上游随机端口，临时 `RELAY_CONFIG_DIR` / `RELAY_DATA_DIR` 隔离数据。
   - 造数走 `store.record(platform=..., model=..., status=..., user_id=..., client_ip=...)`
     关键字形状，别手编位置参数。
   - 时间桶类断言别钉死「当前小时桶」——测试跨整点会 flake，用后缀断言。
   - 造「去年/上月」数据显式算年份（`now.year - 1`），别用「上月所在年」。
2. **前端改动**（`app/static/index.html`）：
   - 先过一遍 `<script>` 块语法（node --check），再全量 pytest；
   - 纯前端逻辑（过滤/日历/格式化）pytest 覆盖不到 → 写 node 冒烟：
     按花括号匹配抽出目标函数源码，eval 进 stub 上下文，对渲染出的 HTML 字符串断言。
   - 浏览器实测：起假上游 + relay，控制台逐页检查渲染，截图核对。

## 测试编号规则

回归测试按 `R1…Rn` 顺序编号（函数名 `test_R63_xxx` 形式），**只增不改号**：
已有编号对应的断言形状被端点语义变更时，在同一 PR 里把旧测试一并迁移到新流程，
别只加新测试。加新测试前先看文件尾部的最大 R 编号。

## 新增上游平台预设

预设清单在 `app/admin.py` 的 `PRESETS`，前端下拉框自动跟上。加新厂商固定五步：

1. **先查重**：`grep '"platform":' app/admin.py` 列全现有 slug，并对一遍
   现有条目的 `base_url`——每个平台只保留一份模板（slug 不同不代表不重复，
   如 `qwen` 与 `dashscope` 是同一平台）。已有条目就合并进它（已有条目的
   `thinking_off` 等实测字段优先，新条目只贡献增量）。
2. `PRESETS` 追加一个 dict，字段：`platform`（小写 slug）、`name`（中文显示名）、
   `type`（`openai` / `anthropic`，按协议）、`base_url`、`models`（预置该家已知模型）、
   `extra`（可选）、`thinking_off`（可选）、`note`。
   `note` 全中文写清四件事：**key 在哪创建、模型名什么形式、无额度/欠费测试会 402
   属计费不是配置错、国内直连不通时走平台「代理」栏**——不确定的不写。
3. 追加一条回归测试：锁死预设存在 + `type` + `base_url` + `models` 预置项
   （照抄最近邻的 preset 测试形状，编号取当前最大 +1）。
4. 保留全局防重断言（所有预设的 slug 与 base_url 各自不得重复）。
5. 全量 pytest 绿。

> 没有真实 key 也欢迎提 PR：填好 note 里「key 在哪创建」即可，
> `models` 预置该家公开文档里确认过的模型名。

## 其他约定

- 后端代码风格：函数短小、注释写「为什么」；中文注释 OK。
- 新增管理端点：`app/admin.py` 路由 + `tests/test_relay.py` 回归；
  **给 api_keys 等对象加字段时，所有嵌入该对象的端点都要补**（改前
  `grep list_api_keys app/admin.py` 列全嵌入点）。
- 文档：行为变更同步改 `README.md` / `DEPLOY.md` / `docs/DEEP_DIVE.md` 对应章节；
  新功能在 `CHANGELOG.md` 记一行。
- 提交信息：一句中文说清改了什么，必要时附 R 编号。

## 分支 / PR

- 小改直接提 main 的 PR 也可以；功能分支名随意。
- PR 描述里贴：改了什么、怎么验证的（pytest 输出摘要 / 截图）、影响哪些页面。
- CI（`python -m pytest`）绿了才合并。
