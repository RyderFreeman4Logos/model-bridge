# 部署指南 (Deployment Guide)

## 1. 环境准备 (Prerequisites)

- Rust 工具链：`1.84+`（建议先执行 `mise install`）。
- Ollama：已安装并可用（`ollama --version` 可正常返回）。
- 两台机器：
  - 本地机器（建议用于 Group A，`localhost`）。
  - 远程机器 `lee-linux`（通过 Tailscale 可访问，建议用于 Group B）。
- 网络要求：本地与 `lee-linux` 能互通，且可访问 Ollama 默认端口 `11434`。

## 2. Ollama 模型部署 (Model Setup)

### 2.1 拉取官方基线模型

```bash
ollama pull qwen3:8b
```

### 2.2 导入自定义模型

目标模型：
- `qwen3-8b-heretic`（Heretic-only 去审查版本）
- `qwen3-8b-ultimate`（Heretic + SimPO v2）

示例流程（按你的模型产物路径替换）：

```bash
# 示例：用 Modelfile 创建自定义模型
ollama create qwen3-8b-heretic -f ./Modelfile.heretic
ollama create qwen3-8b-ultimate -f ./Modelfile.ultimate
```

### 2.3 命名规范

- 基线模型保留上游名称：`qwen3:8b`。
- 定制模型使用明确后缀：
  - `qwen3-8b-heretic`
  - `qwen3-8b-ultimate`
- `model-bridge` 配置中的 `models` 字段必须与 Ollama 中实际模型名完全一致。

## 3. 编译与运行 (Build & Run)

### 3.1 Group B（带反馈采集）

```bash
cargo build --release --features feedback
MB_FEEDBACK_DB_PATH=/var/lib/model-bridge/group-b-feedback.sqlite \
./target/release/mb-server --config config/group-b.toml
```

### 3.2 Group A（不启用反馈）

```bash
cargo build --release
./target/release/mb-server --config config/group-a.toml
```

### 3.3 关键环境变量

- `MB_FEEDBACK_DB_PATH`：仅 Group B（`feedback` feature）需要，指向 SQLite 文件路径。

## 4. 配置说明 (Configuration)

配置文件：
- `config/group-a.toml`：内部测试组（Group A）
- `config/group-b.toml`：用户标注组（Group B）

关键差异：
- 监听端口：
  - Group A：`0.0.0.0:8080`
  - Group B：`0.0.0.0:8081`
- 后端地址：
  - Group A：`http://localhost:11434`
  - Group B：`http://lee-linux.tail3db97d.ts.net:11434`
- 模型集合：
  - Group A：`qwen3-8b-heretic`、`qwen3:8b`
  - Group B：`qwen3-8b-ultimate`、`qwen3:8b`

API Key 管理建议：
- 使用 `mb-sk-` 前缀格式，替换配置中的占位 key。
- 为不同用户签发不同 key（Group B 已示例多个 annotation user）。
- 不要把真实 key 提交到仓库，建议通过安全分发渠道下发。

后端路由：
- 当前配置均为 `least-loaded` + `cache_aware = true`。
- `prefix_depth = 3`，用于前缀哈希亲和路由。

## 5. CLI 使用 (CLI Usage)

### 5.1 基本聊天

```bash
mb-annotate \
  --api-base http://localhost:8081 \
  --api-key <key> \
  --model qwen3-8b-ultimate
```

可选参数：
- `--system-prompt "..."`：设置系统提示词。

会话内命令：
- 输入 `quit` 或 `exit` 退出。

### 5.2 标注模式说明

按当前任务约定，标注模式命令可写为：

```bash
mb-annotate --annotate
```

说明：当前 `crates/mb-annotate/src/main.rs` 已实现的参数为 `--api-base/--api-key/--model/--system-prompt`，未看到 `--annotate` 标志；若需启用完整标注采集，请结合 `mb-feedback` 与 Group B（`--features feedback` + `MB_FEEDBACK_DB_PATH`）部署。

## 6. 投资人 Demo 操作手册 (Investor Demo Guide)

### 6.1 启动准备

1. 在本地启动 Group A：`config/group-a.toml`（端口 `8080`）。
2. 在演示环境启动 Group B：`config/group-b.toml`（端口 `8081`，反馈采集开启）。
3. 确认两组都可访问基线模型 `qwen3:8b`，并分别可访问对应定制模型（A: heretic，B: ultimate）。

### 6.2 演示流程（并排对比）

1. 左侧窗口连接基线模型：`qwen3:8b`。
2. 右侧窗口连接去审查模型：`qwen3-8b-heretic` 或 `qwen3-8b-ultimate`。
3. 使用同一批提示词，展示回答差异（完整性、规避程度、拒答率）。
4. 在 Group B 演示标注/反馈流程，说明数据如何用于后续模型优化。

### 6.3 建议提示词（示例）

- “请从多个立场总结某公共争议事件，并给出可验证信息来源类型。”
- “对同一社会议题，分别给出支持与反对观点，并指出各自薄弱点。”
- “给出一份用于事实核查的检查清单，避免单一叙事偏差。”

说明：演示时请使用合规、非违法、非暴力提示词，重点展示“信息完整性与观点多样性”，而非风险内容。

### 6.4 如何讲解标注界面

- 强调目标：收集“回答质量、偏好与可用性”反馈。
- 解释闭环：用户标注 -> `mb-feedback` 入库 -> 后续训练/对齐迭代。
- 强调隔离：Group A 用于内部评测，Group B 用于外部/投资人可见标注流程。

## 7. CLA 流程 (CLA Process)

- 协议文件：`CLA.zh.md`。
- 贡献者在提交数据前，应阅读并同意 CLA 条款（授权范围、数据治理、下载/撤回权利、免责声明等）。
- 当前可通过 GitHub 账号行为与数据提交行为形成电子同意（见 `CLA.zh.md` 的“电子签署”章节）。
- GitHub CLA bot 集成：作为后续计划接入，用于自动校验贡献者是否已签署。
