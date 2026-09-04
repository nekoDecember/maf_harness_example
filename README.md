# Microsoft Agent Framework 1.17.0 — 安定版Harnessサンプル

Microsoft Agent FrameworkのHarness Agentを、experimentalなAgent Loop機能なしで動かす
実行可能なPython CLIです。通常のFramework tool loopと、短いホスト側supervisorを組み合わせています。
LiteLLMを含むOpenAI互換Chat Completions、またはOpenAI Responses APIに接続できます。

## 方針

1回の `agent.run()` が通常のテキスト応答で終わっても、`task_finish` が呼ばれていなければ
ホストが同じ `AgentSession` で再実行します。これにより、experimentalなAgent Loop middlewareに
依存せず「途中でモデルが文章を返しただけでタスク終了」という問題を防ぎます。

ホストsupervisorが担当するのは次の最小限です。

- 未完了なら次のFramework turnを呼ぶ
- `ask_user` とtool approvalで停止し、回答後に再開する
- 毎turn checkpointを保存する
- 一時的なAPIエラーを再試行する
- 長時間未完了のときだけユーザーへ続行確認する

## バージョン

```text
agent-framework-core==1.17.0
agent-framework-openai==1.14.2
```

`agent-framework-openai==1.17.0` は公開されていません。providerはcoreとは独立して
リリースされており、1.14.2が `agent-framework-core>=1.17.0,<2` を要求する組合せです。

## ファイル構成

```text
harness_cli.py    # CLI、入力/承認、ホストsupervisor
harness_agent.py  # Harness設定、completion provider、workspace tools
harness_state.py  # 設定、バージョン検査、checkpoint
tests/            # ローカルOpenAI互換サーバーを使う回帰テスト
```

責務を分けていますが、外部DB・ジョブキュー・独自抽象化などは追加していません。

## セットアップ

Python 3.10以上と、`uv` または`venv`を使います。

### uv

```bash
cp .env.example .env
# .envを編集
uv sync --extra dev
uv run python harness_cli.py
```

### venv / pip

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
cp .env.example .env               # Windowsはcopy .env.example .env
# .envを編集
python harness_cli.py
```

## LiteLLM設定例

```dotenv
OPENAI_API_KEY=your-litellm-key
OPENAI_MODEL=your-model-alias
OPENAI_BASE_URL=http://localhost:4000/v1
MAF_CLIENT=chat_completions
```

LiteLLM側で認証を無効化していても、SDKの入力検査を通すため任意の非空文字列を
`OPENAI_API_KEY` に設定してください。

## OpenAI設定例

```dotenv
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.4
OPENAI_BASE_URL=
MAF_CLIENT=chat_completions
```

接続先がResponses APIを実装している場合は `MAF_CLIENT=responses` も選択できます。

## 操作

```text
user> workspaceにある仕様を確認し、実装案をdesign.mdとして作成して
```

ファイル変更時は承認を求めます。

```text
[承認が必要] workspace_write_text
{ ... }
実行を承認しますか？ [y/N]: y
```

情報が必要な場合は `ask_user` を通して質問し、回答後に同じsessionで続行します。

コマンド:

- `/mode plan` — 計画を対話的に作る
- `/mode execute` — 自律実行（初期値）
- `/todos` — 現在のtodoを表示
- `/new` — 新しいsessionを開始
- `/exit` — checkpointを保存して終了

未完了checkpointの再開:

```bash
python harness_cli.py --resume
```

## 継続の上限

Framework内部のfunction invocationは1turnあたり最大80回です。通常のテキスト応答でturnが終わった
場合は、host supervisorが同じsessionで次turnを開始します。既定では12turnごとにユーザーへ
「続行・追加指示・保存して中断」を確認します。未完了のまま黙って完了扱いにはしません。

これは無限実行を保証する仕組みではありません。モデル停止、API障害、ユーザーによる中断は起こり得るため、
checkpointと `--resume` を用意しています。

## テスト

```bash
pytest -q
```

テストでは、次を実パッケージで確認します。

- 通常応答後にhost supervisorが再実行する
- `ask_user` の回答後に再開する
- 書込approval後に実ファイルを作る
- `task_finish` で終了する
- session checkpointを復元する

## セキュリティ上の注意

- `.env`、checkpoint、仮想環境、runtime workspaceは `.gitignore` 済みです。
- ファイル操作は設定したworkspace配下に限定しています。
- 書込と削除は常にユーザー承認が必要です。
- このサンプルは本番銀行システム向けの耐障害基盤ではありません。必要以上に複雑化せず、
  挙動を読みやすく保つことを優先しています。
