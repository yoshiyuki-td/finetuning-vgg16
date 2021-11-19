# Packages and versions that are likely to be relevant.
# Please note that some items may not be relevant.

# (1) PC pattern with GPU installed

# Package                 Version
# ----------------------- -------------------
# Python                  3.7.9
# h5py                    3.1.0
# Keras                   2.4.3
# matplotlib              3.4.1
# numpy                   1.19.5
# pandas                  1.2.3
# scikit-learn            0.24.1
# tensorflow-gpu          2.5.0

# (2) PC pattern without GPU

# Package                 Version
# ----------------------- -------------------
# Python                  3.7.9
# h5py                    2.10.0
# Keras                   2.3.1
# matplotlib              3.3.2
# numpy                   1.19.2
# pandas                  1.1.3
# scikit-learn            0.23.2
# tensorflow-gpu          2.1.0

# The code is as follows
# Sorry to trouble you, but please translate Google when you refer to the Japanese comment.
# ----------------必要なライブラリをインストール----------------
import tkinter as tk
import tkinter.ttk as ttk
#from tkinter import messagebox

import sys# 処理中断用ライブラリ

import os# OSを扱うライブラリ

import numpy as np# 配列を扱うライブラリ
                  # 画像が配列で表現できるイメージ 参考：https://www.infiniteloop.co.jp/blog/2018/02/learning-keras-06/

import pickle# 複数のオブジェクトを1つのまとまりに保存したり、保存したオブジェクトを読み込むライブラリ

import matplotlib.pyplot as plt# 学習結果を可視化するためのライブラリ

from keras.applications.vgg16 import VGG16# 出力層1000ユニット、1000クラスを分類するニューラルネット
                                          # 参考：https://aidiary.hatenablog.com/entry/20170104/1483535144
    
from keras.preprocessing.image import ImageDataGenerator# 画像を水増しする際のライブラリ

from keras.models import Sequential, Model# kerasのモデル構成を作成するためのライブラリ

from keras.layers import Input, Activation, Dropout, Flatten, Dense#モデル作成で使うメソッドのライブラリ

from keras import optimizers

# --------------学習精度を確認する関数---------------
def plot_history(history,# 学習結果
                save_graph_img_path,# 学習結果の保存先パス
                fig_size_width,# 学習結果保存時の横幅サイズ
                fig_size_height,# 学習結果保存時の高さサイズ
                lim_font_size):# 学習結果保存時の文字サイズ

    acc = history.history['accuracy']# 学習結果から学習時の正解率を辞書型で取得する
    val_acc = history.history['val_accuracy']# 学習結果から検証時の正解率を辞書型で取得する
    loss = history.history['loss']# 学習結果から学習時の損失を辞書型で取得する
    val_loss = history.history['val_loss']# 学習結果から検証時の損失を辞書型で取得する
   
    epochs = range(len(acc))# エポック数を取得

    # グラフ表示
    plt.figure(figsize=(fig_size_width, fig_size_height))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = lim_font_size # 全体のフォント
    # plt.subplot(121)

    # plot accuracy values
    plt.plot(epochs, acc, color = "blue", linestyle = "solid", label = 'train acc')
    plt.plot(epochs, val_acc, color = "green", linestyle = "solid", label= 'valid acc')
    # plt.title('Training and Validation acc')
    # plt.grid()
    # plt.legend()
 
    # plot loss values
    # plt.subplot(122)
    plt.plot(epochs, loss, color = "red", linestyle = "solid" ,label = 'train loss')
    plt.plot(epochs, val_loss, color = "orange", linestyle = "solid" , label= 'valid loss')
    #plt.title('Training and Validation loss')
    plt.legend()
    plt.grid()

    plt.savefig(save_graph_img_path)
    plt.close()# バッファ解放
    
# ---------------学習モデルを構築する関数----------------
# ここではVGG16をファインチューニングするモデルを構成
def build_model(num_classes,img_width,img_height,color):# 引数内容(判別するクラス数，学習画像の横幅，学習画像の縦幅)
    
    # 学習画像の情報を取得
    input_tensor = Input(shape=(img_width, img_height, color))# kerasのInputメソッドに情報(カテゴリ数,学習画像の横幅，学習画像の縦幅)入力。
                                                              # 引数内容(学習画像の横幅，学習画像の縦幅，カラー情報※カラー:3 グレー:1)

    # VGG16の読み込み
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    # 引数内容(1000クラス分類フル結合層使用有無，ImageNetで学習した重みの使用有無，習画像情報)
    # 重みは1000クラス加味されたのもを使用するが、全結合についはVGG16を特徴抽出器としてのみ使うためinclude_topはFalse
    # 参考：https://lp-tech.net/articles/ks8F9
     
    # モデル作成
    model = Sequential()# 構造作成宣言。参考：https://child-programmer.com/ai/keras/dense/
    model.add(Flatten(input_shape=vgg16.output_shape[1:]))# 1次元にスライス。0を除くのは恐らく重回帰切片確保のため?

    model.add(Dense(256, activation='relu'))# Relu関数を用いて256の出力を算出。参考：https://tutorials.chainer.org/ja/13_Basics_of_Neural_Networks.html
    model.add(Dropout(0.5))# ドロップアウト50％。過学習防止
    model.add(Dense(num_classes, activation='softmax'))# クラスを0～1の幅で算出する。参考：https://atmarkit.itmedia.co.jp/ait/articles/2004/08/news016.html

    # VGG16のモデルをインプットし、アウトプットもVGG16から得る。以上のモデルを格納する。
    new_model = Model(inputs = vgg16.input, outputs=model(vgg16.output))

    return new_model

# ---------------訓練用画像と検証用画像をロード---------------
def img_generator(classes, 
                train_path, 
                validation_path, 
                batch_size,#=16, 
                img_width,# =32, 
                img_height):# =32):#なせ参考コードでサイズが32なのか不明。精度に影響する？

    # ディレクトリ内の画像を読み込んでトレーニングデータとバリデーションデータの作成
    train_gen = ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.2, horizontal_flip=True)

    validation_gen = ImageDataGenerator(rescale=1.0 / 255)#標準化

    # 学習用データと検証用データを取得
    train_datas = train_gen.flow_from_directory(train_path,
                target_size=(img_width, img_height),
                color_mode='rgb',# BGRの場合もあるため注意する
                classes=classes,
                class_mode='categorical',
                batch_size=batch_size,
                shuffle=True)

    valid_datas = validation_gen.flow_from_directory(
                validation_path,
                target_size=(img_width, img_height),
                color_mode='rgb',# BGRの場合もあるため注意する
                classes=classes,
                class_mode='categorical',
                batch_size=batch_size,
                shuffle=True)

    return train_datas, valid_datas
    
# ---------------メイン関数---------------
def main():
 
    # モデルを作成
    model = build_model(num_classes,# クラス数
                        img_width,#学習画像の横幅
                        img_height,# 学習画像の高さ
                        color)# 学習画像のカラー

    # 最後の畳み込み層の直前までの層が学習しない（重みを学習せずにVGG16のまま固定：frozen）
    for layer in model.layers[:15]:# スライスで後ろから15層を指定
                                   # スライスの概念 参考：https://qiita.com/kaeruair/items/e7f1c08915839ce3c9b4
            
        layer.trainable = False# 後ろから15層を無効化

    # 最適化アルゴリズムでモデルをコンパイルする際に必要となるパラメータを設定する
    model.compile(loss = 'categorical_crossentropy',# 引数内容(損失関数)今回は多クラス分類
              optimizer = optimizers.SGD(lr=1e-3, momentum=0.9),# 引数内容(学習率,モーメンタム)
              metrics=['accuracy'])# 引数内容(評価関数)

    # 学習と検証で画像を水増しする
    train_datas, valid_datas = img_generator(classes = classes,# 引数内容(分類数)
                            train_path = SAVE_DATA_DIR_PATH + "train",# 引数内容(学習データパス)
                            validation_path = SAVE_DATA_DIR_PATH + "validation",# 引数内容(検証データパス)
                            batch_size = batch_size,# 引数内容(バッチサイズ)
                            img_width = img_width,# 引数内容(画像横幅)
                            img_height = img_height)# 引数内容(画像高さ)

    # Fine-tuning
    history = model.fit(train_datas,# 引数内容(水増し済みの学習データ)
                        epochs = num_epoch,# 引数内容(エポック数)
                        validation_data=valid_datas)# 引数内容(水増し済み検証データ)

 
    # モデル構造の保存
    open(SAVE_DATA_DIR_PATH  + "model.json","w").write(model.to_json())  

    # 学習済みの重みを保存
    model.save_weights(SAVE_DATA_DIR_PATH + "weight.hdf5")

    # 学習履歴を保存
    with open(SAVE_DATA_DIR_PATH + "history.json", 'wb') as f:# 指定フォルダにファイル名を付け0‐1のバイナリ形式(wb)で保存するため一旦開いて変数(f)に格納
                                                              # Windows上のPythonはtextとbinaryを区別する。historyはjson拡張子でないとエラーが出るらしく
                                                              # jsonはbinary形式のためw(write)b(binary)で保存する。
            pickle.dump(history.history, f)# 指定したファイルにオブジェクトを保存 引数内容(オブジェクト, ファイル)
            # model.fitしたhistoryをそのまま保存はできないのでhistory.history(dictionary型)を保存
            # History.historyは実行に成功したエポック、訓練の損失値と評価関数値、適用可能なら検証における損失値と評価関数値も記録
            # openしたあとclose処理をする必要があるが、行頭のwithがあるためにclose処理は自動的に行わる。
    
    # 学習過程をプロット
    plot_history(history,# 引数内容(学習結果)
                save_graph_img_path = SAVE_DATA_DIR_PATH + "graph.png",# 引数内容(学習結果の保存先+ファイル名)
                fig_size_width = FIG_SIZE_WIDTH,# 引数内容(グラフ横幅)
                fig_size_height = FIG_SIZE_HEIGHT,# 引数内容(グラフ高さ)
                lim_font_size = FIG_FONT_SIZE)# 引数内容(水増し済みの学習データ)

# ---------------学習データや検証データが準備されていない場合---------------
# ---------------ファルダを作成する関数---------------
def mkdir():
    
    for i in range(1):# 学習と検証の2つのフォルダを作成
            
        os.mkdir(SAVE_DATA_DIR_PATH)
        os.mkdir(SAVE_DATA_DIR_PATH+"/train")
        os.mkdir(SAVE_DATA_DIR_PATH+"/validation")
                
    for i in range(len(classes)):# クラス別にフォルダを作成

        os.mkdir(SAVE_DATA_DIR_PATH+"/train/"+str(classes[i]))
        os.mkdir(SAVE_DATA_DIR_PATH+"/validation/"+str(classes[i]))

# ---------------エラー画面を中央寄せにする関数--------------- 参考：https://teratail.com/questions/318881
def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()
     
# ---------------エラー画面を表示する関数---------------
def error():
    
    root = tk.Tk()# root.mainloop()との間にメッセージ内容を記述する
                  # 参考：https://imagingsolution.net/program/python/tkinter/widget_layout_pack/
                  # 参考：https://pg-chain.com/python-pack-grid-place
    
    root.title("エラー")# 画面のタイトル
    root.geometry("700x150")# 画面の横x縦サイズ
    
    # 画面内の内容
    label = tk.Label(root, text="学習に必要なフォルダや画像データが準備できていません。",anchor="w")
    label.pack(side="top", fill="both", expand=True, padx=20, pady=0)
                     
    label = tk.Label(root, text="下記フォルダにカテゴリ別のフォルダを作成したので縦224横224の画像データを入れてください。",anchor="w")
    label.pack(side="top", fill="both", expand=True, padx=20, pady=0)
                     
    label = tk.Label(root, text=os.path.abspath(SAVE_DATA_DIR_PATH),anchor="w")
    label.pack(side="top", fill="both", expand=True, padx=20, pady=0)                    
                     
    # 画面の表示位置設定
    root.attributes('-alpha', 0.0)# 透明度0
    center(root)
    root.attributes('-alpha', 1.0)# 透明度MAX
    # 透明度で表示する理由は不明？この方が自然に描写される？？無くても動作は可能
    
    # 画面を最前面にする
    root.attributes("-topmost", True)
    
    # ボタンの配置
    button = tk.Button(root, text="  OK  ", command=lambda: root.destroy())
    # ボタン配置設定
    button.pack(side="bottom", fill="none", expand=True)

    root.mainloop()

    sys.exit()# 処理を強制終了

# ---------------”VGG16_ファインチューニング.py”の実行---------------
if __name__ == '__main__':# バックグラウンドでこのpyファイルが実行された時だけ処理されるように常套文
                          # この記述がないと、pyファイルをインポートしただけでも実行されてしまう。参考：https://blog.pyq.jp/entry/Python_kaiketsu_180207
                          # このメイン処理を関数(正確に言うとクラスやパッケージのイメージ)のように、別のpyファイルで実行する際は、このような記述が重要となる。

    
    # ファインチューニング参考　参考：https://algorithm.joho.info/machine-learning/python-keras-cnn-fine-tuning/
    # NNの概要 参考：https://www.youtube.com/watch?v=FwuBbj8F6cI    
    
    # 以下で各処理で必要となる設定を行う
    
    # エポック数
    num_epoch = 1# 初期20。トライした結果を見て5に調整。

    # バッチサイズ
    batch_size = 16
    
    # バッチサイズとエポックの詳細 参考：https://qiita.com/kenta1984/items/bad75a37d552510e4682
    
    # クラス数(≒分類したい数)
    num_classes = 2

    # 分類するクラス名(≒分類するフォルダ名)
    classes = ['with_grases', 'without_grases']

    # 学習する画像の幅と高さ(学習結果を使う際も同じサイズにリサイズすることが好ましい)
    img_width = 224
    img_height = 224
    
    # 学習する画像のカラー(カラー：3　白黒：1)
    color = 3

    # グラフ画像のサイズ
    FIG_SIZE_WIDTH = 12
    FIG_SIZE_HEIGHT = 10
    FIG_FONT_SIZE = 25            
            
    # トレーニング用とバリデーション用の画像格納先
    # Pythonでは￥が/になるため注意。
    # このファイルを基準に相対パスを指定するのがシンプルで分かりやすい
    # SAVE_DATA_DIR_PATH = 'Data_rareplanes/ts/'
    SAVE_DATA_DIR_PATH = 'Data/'
    
    # ディレクトリがなかった場合
    if os.path.exists(SAVE_DATA_DIR_PATH) == False:
        
        mkdir() # ディレクトリを作成
        error() # エラー画面を表示

    # 基本設定が終わったら処理を実行する
    main()


# [reference]
# I have also implemented the learning model created in (2) on raspberrypi.
# The main packages and versions of raspberry pi at that time are as follows.

 Package                 Version
----------------------- -------------------
Python                  3.7.9?
opencv-contrib-python   4.5.3.56?
opencv-python           4.5.3.56?
matplotlib              3.4.1?
Keras                   2.4.3?
numpy                   1.19.5?
pandas                  1.2.3?
scikit-learn            0.24.1?
tensorflow              2.5.0?
h5py                    3.1.0?
rpi.gpio
python-spidev
