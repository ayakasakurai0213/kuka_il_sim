# Kuka-Il-sim

## 概要
pybulletの環境で模倣学習を実装しました．kuka iiwaロボットのシミュレータを使用して，マニピュレーション動作のデータ収集，学習，推論を行うことができます．シミュレーションできる模倣学習の手法はALOHAの研究でも使われた，ACT(Action Chunking with Transformer)があります．ACTの詳細は以下の論文を参照してください．<br>
論文：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)


## インストール
このシミュレータを実行するためには以下の手順に従って環境を構築してください．

### ワークスペースの作成
好きな場所に新しいディレクトリを作成し，その中にこのシミュレータのリポジトリをcloneしてください．
```
mkdir ~/il_ws
cd ~/il_ws
git clone git@github.com:ayakasakurai0213/kuka_il_sim.git
```

### 仮想環境の構築

#### [conda]
以下のコマンドを実行して仮想環境を構築してください．
```
conda create --name il_sim python=3.10
```
以下のコマンドを実行し，仮想環境の中で必要なパッケージをインストールしてください．
```
conda activate il_sim
cd ~/il_ws/kuka_il_sim
./build/install.sh
```

#### [docker]
- Linux
- Windows

## 使い方
以下の手順に従って、データ収集・学習・推論を行うことができます．

### データ収集
引数を指定して以下のコマンドを実行することで，データ収集を開始してください．
```
python collect_data/collect_data.py --max_timesteps 500 --task_name "project_test" --episode_idx 0 --control "gamepad"
```

#### [引数]
- ```max_timesteps```：タスク開始から終了までにかかるタイムステップ
- ```task_name```：タスク名（保存するデータセットの名前）
- ```episode_idx```：何個目に保存するエピソードかを指定
- ```control```：シミュレータ上のロボットを操作するのに使用するコントローラ（"keyboard"か"gamepad"を指定）

上記のコマンドを実行すると以下のようなシミュレータが表示されます．

### 学習
引数を指定して以下のコマンドを実行することで，学習を開始してください．

```
python model_train/act/train.py --ckpt_dir ./ckpt/project_test --task_name "project_test" --num_episodes 100 --num_epochs 3000
```

#### [引数]
- ```ckpt_dir```：モデルを保存するディレクトリ
- ```task_name```：タスク名（学習させるデータセットの名前）
- ```num_episodes```：学習するエピソードの数
- ```num_epochs```：エポック数
- ```batch_size```：バッチサイズ（default：8）
- ```chunk_size```：チャンクサイズ（default：32）

### 推論
引数を指定して以下のコマンドを実行することで，推論を開始してください．
```
python model_train/act/inference.py --ckpt_dir ./ckpt/project_test --task_name "project_test"
```

#### [引数]
- ```ckpt_dir```：モデルが保存されているディレクトリ
- ```task_name```：タスク名（実行するモデル名）