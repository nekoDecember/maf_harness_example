# Microsoft Agent Framework 1.17.0 — 継続実行Harnessサンプル

Microsoft Agent Frameworkの公式Harness loopを使う、実行可能なPython CLIです。
LiteLLMを含むOpenAI互換Chat Completions、またはOpenAI Responses APIに接続できます。

## バージョンについて

このサンプルは次の公開パッケージを厳密に固定します。

```text
agent-framework-core==1.17.0
agent-framework-openai==1.14.2
```

`agent-framework-openai==1.17.0` は公開されていません。Microsoft Agent Frameworkは
coreとproviderを独立してリリースしており、OpenAI provider 1.14.2は
`agent-framework-core>=1.17.0,<2` を要求します。起動時にもこの組合せを検査します。

## 1.8.1版との主な違い

1.17.0では `create_harness_agent()` に次の公式APIがあります。

```python
loop_should_continue=...
loop_next_message=...
loop_max_iterations=20
```

そのため、通常の継続処理はFramework自身が同一session内で行います。このサンプルでは
`todos_remaining()` だけに頼らず、`task_finish` が未実行なら継続する独自predicateと組み合わせています。
モデルがtodoを作り忘れたり、途中の説明文を返しただけでタスクが終わるのを防ぐためです。

一方、`loop_max_iterations` は安全のため必ず有限にしています。上限に達しても、ホスト側supervisorが
未完了を検出し、同一sessionで次の公式loop batchを開始します。80 iteration（既定値）ごとにだけ、
続行・追加指示・保存中断をユーザーへ確認します。未完了のまま黙って終了しません。

## 実装している継続性

- 公式 `loop_should_continue` / `loop_next_message` による自律再実行
- `task_finish` と未完了todoの両方を確認して完了判定
- approvalや `ask_user` ではFramework loopが安全にcallerへ制御を返す
- 回答後は同じ `AgentSession` にtool resultを戻して自動再開
- loop batch上限後はhost supervisorが次batchを開始
- sessionをatomic checkpointし、`Ctrl+C`・再起動後に `--resume`
- API一時エラーの自動再試行と、失敗継続時のユーザー選択
- ファイル書込・削除は毎回Human-in-the-loop承認
- workspace外へのパストラバーサル防止

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

LiteLLM proxyが `http://localhost:4000` で動作している場合:

```dotenv
OPENAI_API_KEY=your-litellm-key
OPENAI_MODEL=your-model-alias
OPENAI_BASE_URL=http://localhost:4000/v1
MAF_CLIENT=chat_completions
```

LiteLLM側で認証を無効化している場合も、SDKの入力検査を通すため任意の非空文字列を
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

起動後、そのままタスクを入力します。

```text
user> workspaceにある仕様を確認し、実装案をdesign.mdとして作成して
```

ファイル変更時は、ツール名と引数が表示されます。

```text
[承認が必要] workspace_write_text
{ ... }
実行を承認しますか？ [y/N]: y
```

エージェントが情報を必要とすると、declaration-onlyの `ask_user` を通して質問が表示されます。
回答は同じsessionへfunction resultとして戻るため、そこから公式loopが再開します。

コマンド:

- `/mode plan` — 次タスクを対話的に計画する
- `/mode execute` — 自律実行（初期値）
- `/todos` — 現在のtodoを表示
- `/new` — 新しいsessionを開始
- `/exit` — checkpointを保存して終了

未完了checkpointの再開:

```bash
python harness_cli.py --resume
```

## テスト

ローカルのOpenAI互換HTTPサーバーを使い、実パッケージに対して次を検証します。

1. 最初の通常テキスト応答後に公式Harness loopが自動再実行する
2. `ask_user` がcallerへ返り、回答後に再開する
3. 書込approvalの承認後に実ファイルを作る
4. `task_finish` 後にloopが停止する
5. session checkpointを復元できる

```bash
pytest -q
```

## 重要な制約

- Agent loopingは1.17.0でもexperimental APIです。バージョン固定と回帰テストを維持してください。
- 「途中で止まらない」は、未完了を黙って完了扱いしない・入力待ちを明示する・再開可能にする、という
  制御です。モデルや外部APIが必ず成功する保証ではありません。
- checkpointは複数プロセスから同時更新しないでください。
- 書込・削除のapproval内容は必ず確認してください。
- 自律loopは必ず有限にしてください。`MAF_LOOP_ITERATIONS_PER_BATCH` を無制限にはしない設計です。

## 公式資料

- [Agent looping](https://learn.microsoft.com/en-us/agent-framework/agents/looping)
- [Planning and todos](https://learn.microsoft.com/en-us/agent-framework/agents/planning-and-todos)
- [Agent sessions](https://learn.microsoft.com/en-us/agent-framework/concepts/agents/conversations/session)
- [Python 1.17.0 release](https://github.com/microsoft/agent-framework/releases/tag/python-1.17.0)
