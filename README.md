#DRNNで人間行動認識  

###動作確認環境  
Ubuntu14.04LTS  
Python2.7.12 (anaconda2-4.1.1)  
Chainer1.18.0  
CUDA7.5  
cudnn v4  

####データセットの取得  
`python download.py`  

[Human Activity Sensing Consortium(HASC)](http://hasc.jp/)のデータを使用します  
####学習  
`python HAR.py`  

#####主なオプション  
GPUの使用`-g 0`  
ミニバッチサイズ指定`-b hoge`  
中間層ユニット数`-u hoge`  
エポック数`-e hoge`  
ドロップアウトレート`-dr hoge`  