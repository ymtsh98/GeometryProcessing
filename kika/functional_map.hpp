#pragma once

#include <kika/eigutil.hpp>
#include <kika/surface_mesh.hpp>
#include <kika/diff_geometry.hpp>

namespace kika::fmap {

class FunctionalMap {
  struct MeshData {
    kika::heds::SurfaceMesh& mesh;
  };

};

}