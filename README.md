# 概要
X線検出器hydra-TESの解析ツールです。X線信号がパルス信号としてhdf5ファイルに保存されています（このリポジトリにはありません）。パルス信号は横軸時間、縦軸電圧のオシロスコープの波形をそのまま保存しています。`run045/libs/readhdf5.py`モジュールで生の波形を読み出し、`run045/libs/feature.py`で波形の特徴量（パルスの立ち上がり時間、立ち下がり時間、波高値など）を抽出します。特徴量を取り出したら、X線イベント1つが特徴量空間の中で1プロットに対応するので、特徴量空間でクラスタリングしてX線イベントを分類します（波形弁別）。

# `run045/hydra.py`
メインの解析モジュール。Readhdf5, Featureクラスを継承。
- `open_raw()`
  生データを読み出す。return: pulse, noise, vres, hres, time
- `close_raw()`
  生データのファイルを閉じる。
- `calc_features(pulse, noise, time)`
  特徴量(立ち上がり時間、立ち下がり時間、波高値、ベースライン)を計算する。return: raw_rise, raw_fall, raw_ph, raw_bsl
- `import_features(features_path)`
  既に計算されて保存してある特徴量をファイルから読み込む。return: raw_rise, raw_fall, raw_ph, raw_bsl
- `add_new_feature(features_array, new_feature)`
  特徴量をまとめた配列を作る。`new_feature`に追加したい特徴量の1次元配列を指定する。return: result
- `generate_mask(pix_define)`
  特徴量空間において外れ値を除外するためのマスクを作る。
- `clustering(features, mask)`
  特徴量空間でクラスタリング。return: pred, pixmask
- `calc_pha()`
  PHA（補正波高値）を計算する。
