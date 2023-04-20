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
  * ![astar](/images/astar.png)
  * ![dist1](/images/geodist.gif)

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
* coherent point drift (
 Myronenko, Andriy and Song, Xubo. Point Set Registration: Coherent Point Drift. IEEE Transactions on Pattern Analysis and Machine Intelligence. 32. 2262-75, 2010.
O. Hirose. A Bayesian Formulation of Coherent Point Drift. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43, 2269-2286, 2021.
O. Hirose. Acceleration of Non-Rigid Point Set Registration With Downsampling and Gaussian Process Regression. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43, 2858-2865, 2021.)
  * ![cpd1](/images/affine.gif)
  * ![cpd2](/images/bcpd.gif)
  
* vector field
  * trivial connection (Trivial Connections on Discrete Surfaces. Crane, K. et al.2010.)
  * vector heat (The Vector Heat Method. Sharp, N. et al. 2019.)
  * hodge decomposition
  * ![field2](/images/trivialConnection.png)
  * ![field3](/images/vector_heat.png)
  * ![field1](/images/dir_field.gif)
  
* differential geometry
  * cotangent weight laplacian
  * polygonal laplacian（Discrete Differential Operators on Polygonal Meshes. Goes D. F., et al. 2020.)
  * polygonal laplacian (Polygon Laplacian Made Simple. Bunge A., et al. 2020.)
  * diamond laplacian (The Diamond Laplace for Polygonal and Polyhedral Meshes. Bunge A., et al. 2021.)

* signed distance function
  * ![sdf](/images/mario_cart.gif)
  * marching cubes
  
* sspace partitioning
  * kd木
  * ![kdtree](/images/bunny_div.png)

* spherization
  * ![sph](/images/sphere.gif)
  
* simplification
  * Quadric Error Metrices
  * ![qem](/images/qem.png)
  
* shape correspondences
  * functional maps
  * ![corres1](/images/shape_corres.png)
  * ![corres2](/images/cat_to.png)
  * ![corres3](/images/volume_uv.png)
  * ![corres4](/images/dog2dog.gif)
  * ![corres5](/images/rhino2wolf.gif)
  
https://user-images.githubusercontent.com/36565846/233379018-b9b529f0-c69a-4e1c-9494-cb1ed7cd2005.mp4

https://user-images.githubusercontent.com/36565846/233379185-faff3e0e-4373-4e22-bc96-7799f24836ca.mp4

