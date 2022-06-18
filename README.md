# 書籍"ゼロから作る Deep Learning" 学習用のテスト実行環境

## 環境
Laravel+Vue.jsでUIを構築。
バックエンドで、python3 + numpyを使っています。

## install

### Laravel

```
composer install 
```
### Vueのビルド

```
yarn dev
```
or
```
yarn dev
```
### AI学習データ

学習データ作成
```
 python3 create_weight.py
 ```

## 実行
枠内に手書きの0-9の一文字の数値を書いて、「判定」でどの文字かを判定します。

<p align="center"><img src="https://raw.githubusercontent.com/kysaeed/number-predictor/main/np-screen.png" width="400"></p>


