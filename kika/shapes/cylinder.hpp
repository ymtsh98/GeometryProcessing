#pragma once

///          ,,.::::::::::::::::::.,,
///      .::::.,,,              ,,,.::::.,
///   ,::.,                            ,.::,
///  ,:,                                  ,:,
///  :.                                    .:
///  ::+.,                              ,.+::
///  :, .:::.,,                    ,,.:::. ,:
///  :,    ,..::::::::::::::::::::::..,    ,:
///  :,            ,,,,,,,,,,,,            ,:
///  :,                                    ,:
///  :,                                    ,:
///  :,                                    ,:
///  :,                                    ,:
///  :,                                    ,:
///  :,                                    ,:
///  :,                                    ,:
///  :,                                    ,:
///  :,                                    ,:
///  :,                                    ,:
///  :,               ,,,,,,               ,:
///  :,     ,.::::::::::::::::::::::.,     ,:
///  :, .::::,,                    ,,::::. ,:
///  :::.,                              ,.:::
///  ::                                    ::
///  ,:,                                  ,:,
///   ,::.,                            ,.::,
///     ,.::::.,,                ,,.::::.,
///         ,,.::::::::::::::::::::.,,
/// 
/// Parameters 
/// hReso, vReso :horizontally,  vertically resolutions >= 3, 1 (initial value = 32, 1)
/// radius : radius (initial value = 1.0)
/// height : height (initial value = 2.0)
/// 
#include <kika/shapes/shape.hpp>
namespace kika::shape {

template <typename Prec=double, typename Idx=uint32_t>
struct CylinderParams {
  Idx hReso, vReso;
  Prec radius, height;

  CylinderParams(const int& hReso, const int& vReso, const double& radius, const double& height) :
    hReso(static_cast<Idx>(std::max(hReso, 3))), vReso(static_cast<Idx>(std::max(vReso, 1))), 
    radius(static_cast<Prec>(radius)), height(static_cast<Prec>(height)) {}
  CylinderParams(const int& hReso, const int& vReso, const double& radius) : CylinderParams(hReso, vReso, radius, 2.0) {}
  CylinderParams(const int& hReso, const int& vReso) : CylinderParams(hReso, vReso, 1.0, 2.0) {}
  CylinderParams(const int& hReso) : CylinderParams(hReso, 1, 1.0, 2.0) {}
  CylinderParams() : CylinderParams(32, 1, 1.0, 2.0) {}
};

template <typename Prec=double, typename Idx=uint32_t>
class Cylinder : public Shape<Prec, Idx>{
  using Params = CylinderParams<Prec, Idx>;
  using XyzGeneratorBase = typename Shape<Prec, Idx>::template Generator<Params, Prec>;
  using TriGeneratorBase = typename Shape<Prec, Idx>::template Generator<Params, Idx>;
  Params params;

  class XyzGenerator : public XyzGeneratorBase {
    using XyzGeneratorBase::params;

  public:
    XyzGenerator(const Params& params) : XyzGeneratorBase(params) {}

    Idx size() const override {
      return (params.hReso * (1 + params.vReso)) * 3;
    };

    std::vector<Prec> map(const int storage = StorageOptions::RowMajor) const override {
      Idx n = this->size() / 3;
      Idx k1 = (storage == StorageOptions::ColMajor) ? 1 : 3, k2 = (storage == StorageOptions::ColMajor) ? n : 1;
      std::vector<Prec> pos(3 * n);

      for (Idx i = 0; i < params.vReso + 1; i++) {
        Prec h = static_cast<Prec>((params.height * i / params.vReso) - (params.height / 2.));

        for (Idx j = 0; j < params.hReso; j++) {
          double t = 2.0 * j * _PI / params.hReso;

          pos[(j + params.hReso * i) * k1 + 0] = static_cast<Prec>(params.radius * cos(t));
          pos[(j + params.hReso * i) * k1 +k2] = h;
          pos[(j + params.hReso * i) * k1+2*k2] = static_cast<Prec>(params.radius * sin(t));
        }
      }

      return pos;
    }
  };

  class TriGenerator : public TriGeneratorBase {
    using TriGeneratorBase::params;

  public:
    TriGenerator(const Params& params) : TriGeneratorBase(params) {}

    Idx size() const override {
      return (2 * params.hReso * (1 + params.vReso) - 4) * 3;
    };

    std::vector<Idx> map(const int storage = StorageOptions::RowMajor) const override {
      Idx n = params.hReso * (1 + params.vReso);
      Idx f = 2 * n - 4;
      Idx k1 = (storage == StorageOptions::ColMajor) ? 1 : 3, k2 = (storage == StorageOptions::ColMajor) ? f : 1;

      std::vector<Idx> idx(3*f);
      Idx c = 0;

      for (Idx i = 1; i < params.hReso-1; i++) {
        idx[c * k1 + 0] = 0;
        idx[c * k1 + k2] = i;
        idx[c++ * k1 + 2*k2] = i+1;
      }

      for (Idx i = 0; i < params.vReso; i++) {
        for (Idx j = 0; j < params.hReso; j++) {
          Idx i1 = j + i * params.hReso,
            i2 = j + params.hReso + i * params.hReso,
            i3 = (j + 1) % params.hReso + params.hReso + i * params.hReso,
            i4 = (j + 1) % params.hReso + i * params.hReso;
          idx[c * k1 + 0] = i1;
          idx[c * k1 + k2] = i2;
          idx[c++ * k1 + 2*k2] = i4;
          idx[c * k1 + 0] = i4;
          idx[c * k1 + k2] = i2;
          idx[c++ * k1 + 2*k2] = i3;
        }
      }

      for (Idx i = 1; i < params.hReso-1; i++) {
        idx[c * k1 + 0] = n - params.hReso;
        idx[c * k1 + k2] = i + 1 + n - params.hReso;
        idx[c++ * k1 + 2*k2] = i + n - params.hReso;
      }

      return idx;
    }
  };

public:

  XyzGenerator xyz;
  TriGenerator tri;

  Cylinder() : params(), xyz(params), tri(params) {}
  Cylinder(const CylinderParams<Prec, Idx>& params) : params(params), xyz(params), tri(params) {}
  Cylinder(const int& hReso) : params(hReso), xyz(params), tri(params) {}
  Cylinder(const int& hReso, const int& vReso) : params(hReso, vReso), xyz(params), tri(params) {}
  Cylinder(const int& hReso, const int& vReso, const double& radius) : 
    params(hReso, vReso, radius), xyz(params), tri(params) {}
  Cylinder(const int& hReso, const int& vReso, const double& radius, const double& height) : 
    params(hReso, vReso, radius, height), xyz(params), tri(params) {}

  Idx nv() const override { return params.hReso * (1 + params.vReso); }
  Idx nf() const override { return 2 * nv() - 4; }
  Idx ne() const override { return 3 * nv() + 6; }

  std::vector<std::vector<Idx>> poly() const override {
    Idx n = params.hReso * params.vReso;
    Idx f = n + 2;

    std::vector<std::vector<Idx>> idx(f);

    Idx c = 0;
    std::vector<Idx> ngon(params.hReso);
    for (Idx i = 0; i < params.hReso; i++) ngon[i] = i;
    idx[c++] = ngon;

    std::vector<Idx> q(4);
    for (Idx i = 0; i < params.vReso; i++) {
      for (Idx j = 0; j < params.hReso; j++) {
        Idx i1 = j + i * params.hReso,
          i2 = j + params.hReso + i * params.hReso,
          i3 = (j + 1) % params.hReso + params.hReso + i * params.hReso,
          i4 = (j + 1) % params.hReso + i * params.hReso;
    
        q[0] = i1;
        q[1] = i2;
        q[2] = i3;
        q[3] = i4;
        idx[c++] = q;
      }
    }

    for (Idx i = 0; i < params.hReso; i++) ngon[params.hReso - i - 1] = n + i;
    idx[c++] = ngon;

    return idx;
  }

  void setParams(const Params& params) { this->params = params; }
  void setHorzReso(const Idx& reso) { params.hReso = reso; }
  void setVertReso(const Idx& reso) { params.vReso = reso; }
  void setHeight(const Prec& height) { params.height = height; }
  void setRadius(const Prec& radius) { params.radius = radius; }

};
}
