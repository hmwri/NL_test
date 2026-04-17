# NL_test

## 日本語

自然言語処理の練習用 Python リポジトリです。LSTM 関連の実験、共通関数、pickle の確認用スクリプト、試作用コードが含まれています。

### 構成

- `main.py`: 実験の入口
- `showpickle.py`: pickle データ確認用ヘルパー
- `common/`: 共通処理
- `lstm/`: LSTM 関連の実験
- `playground/`: 試作用コード

### 実行

固定の package や CLI は定義されていません。まず `main.py` または対象の実験スクリプトを確認して実行します。

```bash
python main.py
```

### 注意

依存関係は実験ごとに異なる可能性があります。実行時の import error に応じて必要な package を追加してください。

## English

NL_test is a Python natural-language-processing practice repository.

### Structure

- `main.py`: experiment entry point
- `showpickle.py`: helper for inspecting pickle data
- `common/`: shared utility code
- `lstm/`: LSTM-related experiments
- `playground/`: scratch code

### Run

```bash
python main.py
```

Dependencies may vary by experiment.
