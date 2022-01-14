# Geometry-Processing

幾何形状処理関連の実装まとめ
  
* alg_topology.hpp
  * Tree-Cotree Decomposition
  * 最小サイクル基底
  * 最小ホモロジー基底
  * ![mincy](/images/minCycle.png)
  * ![homology](/images/homology.png)
  * ![cycles](/images/cycles.png)

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
  * Fast Marching Method
  * 熱拡散による測地距離計算 (The Heat Method for Distance Computation. Crane, K. et al. 2017)
  * ![dist](/images/ketten_dist.png)
  * ![dist2](/images/astar.png)

* intrinsic_feature.hpp
  * 平均曲率
  * ガウス曲率
  * Heat Kernel Signature ( A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion. Sun, J. et al. 2009)
  * Wave Kernel Signature (The Wave Kernel Signature: A Quantum Mechanical Approach to Shape Analysis. Aubry, M. et al. 2011)
  * ![wks](/images/wks.gif)
  * Salient Feature Extraction
  * ![sal](/images/saplient_feat.png)
  
* principal_curvature.hpp
  * 主曲率・方向 (Efficient Multi-scale Curvature and Crease Estimation. D. Panozzo, et al. 2010)
  * ![pc](/images/pc.png)
  
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
  
* simplification.hpp
  * Quadric Error Metrices
  * ![qem](/images/qem.png)
