# Geometry-Processing

幾何形状処理関連の実装まとめ
mygp(old version) -> kika(refactored version(unfinished))

* algebraic topology
  * Tree-Cotree Decomposition
  * Minimum Cycle Basis
  * Minimum Homology Basis
  * ![mincy](/images/minCycle.png)
  * ![homology](/images/homology.png)
  * ![cycles](/images/cycles.png)

* chaos maps
  * ロジスティク写像
  * 池田写像
  * ティンカーベル写像
  * ボグダノフ写像
  * Thomas' cyclically symmetric attractor
  * ジンジャーブレッドマン写像
  * ![chaos](/images/chaos.gif)
  
* geodesic distance
  * Dijkstra最短経路
  * A*
  * Fast Marching Method
  * 熱拡散による測地距離計算 (The Heat Method for Distance Computation. Crane, K. et al. 2017)
  * 仮想頂点を利用した測地距離計算（Geodesic Distance Computation via Virtual Source Propagation. Trettner, P. et al. 2021）
  * ![dist](/images/ketten_dist.png)
  * ![dist2](/images/astar.png)

* intrinsic features
  * mean curvature
  * gaussian curvature
  * Heat Kernel Signature ( A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion. Sun, J. et al. 2009)
  * Wave Kernel Signature (The Wave Kernel Signature: A Quantum Mechanical Approach to Shape Analysis. Aubry, M. et al. 2011)
  * Wavelet-based Heat Kernel (Wavelet-based Heat Kernel Derivatives: Towards Informative Localized Shape Analysis. Krigo, M. et al. 2020）
  * ![wks](/images/wks.gif)
  * Salient Feature Extraction
  * ![sal](/images/salient_feat.png)
  * principal curvature (Efficient Multi-scale Curvature and Crease Estimation. D. Panozzo, et al. 2010)
  * ![pc](/images/pc.png)
  
* halfedge data structure
  
* iterative closest point
* coherent point drift
 
* vector field
  * trivial connection (Trivial Connections on Discrete Surfaces. Crane, K. et al.2010.)
  * vector heat (The Vector Heat Method. Sharp, N. et al. 2019.)
  
* differential geometry
  * cotangent weight laplacian
  * polygonal laplacian（Discrete Differential Operators on Polygonal Meshes. Goes D. F., et al. 2020.)
  * polygonal laplacian (Polygon Laplacian Made Simple. Bunge A., et al. 2020.)
  * diamond laplacian (The Diamond Laplace for Polygonal and Polyhedral Meshes. Bunge A., et al. 2021.)

* signed distance function
  * marching cubes

* sspace partitioning
  * kd木
  * ![kdtree](/images/bunny_div.png)

* union find
  
* simplification
  * Quadric Error Metrices
  * ![qem](/images/qem.png)
