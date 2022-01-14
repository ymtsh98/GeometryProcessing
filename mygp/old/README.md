# Geometry-Processing

幾何形状処理関連の実装まとめ
  
* chaos.hpp
  * ロジスティク写像
  * 池田写像
  * ティンカーベル写像
  * ボグダノフ写像
  * Thomas' cyclically symmetric attractor
  * ジンジャーブレッドマン写像
  * ![chaos](/images/chaos.gif)
  
* distance.hpp
  * Dijkstra最短経路
  * A*
  * 熱拡散による測地距離計算 (The Heat Method for Distance Computation. Crane, K. et al. 2017)
  * ![dist](/images/ketten_dist.png)

* feature.hpp
  * 平均曲率
  * ガウス曲率
  * Heat Kernel Signature ( A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion. Sun, J. et al. 2009)
  * Wave Kernel Signature (The Wave Kernel Signature: A Quantum Mechanical Approach to Shape Analysis. Aubry, M. et al. 2011)
  * ![wks](/images/wks.gif)

* halfedge.hpp
  * ハーフエッジデータ構造
  
* icp.hpp
  * iterative closest point
 
* laplace.hpp
  * cotangent重みラプラシアン

* space_part.hpp
  * kd木
  * ![kdtree](/images/bunny_div.png)

* union_find.hpp
  * ユニオンファインド

