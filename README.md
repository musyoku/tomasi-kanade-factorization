## Tomasi-Kanade Factorization

**Tomasi-Kanadeの因子分解法**の特異値分解による計算手法のPython実装例です。

Chainerの自動微分を利用しています。

## 実行例

分かりづらいですが立方体の上にピラミッドが乗ったような形を真の形状とします。

![image](https://qiita-image-store.s3.amazonaws.com/0/109322/29fcc31b-df4d-7dcc-6f08-4803906cef41.png)

これを様々な角度から2次元平面に正射影した点データのみから元の3D形状を復元します。

形状の推定結果は以下のようになります。（回転しています）

![image](https://qiita-image-store.s3.amazonaws.com/0/109322/db3af4cf-7d35-8cbf-1ebc-e1d846d83e68.png)

## 参考文献
 - [Shape and Motion from Image Streams under Orthography: a Factorization Method](https://people.eecs.berkeley.edu/~yang/courses/cs294-6/papers/TomasiC_Shape%20and%20motion%20from%20image%20streams%20under%20orthography.pdf)