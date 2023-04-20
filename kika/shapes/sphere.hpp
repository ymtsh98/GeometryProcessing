#pragma once

///              ,,.::, ,
///          .*++++++++++++,
///        ***+++++++++++++++:
///      ****++++++++++++++++++,
///     ****+++++++++++++++++++++
///    #****++++++++++++++++++++++
///   #*****++++++++++++++++++++++.
///  +#*****+++++++++++++++++++++++
///  ###*****++++++++++++++++++++++:
///  ###*****+*+++++++++++++++++++++
/// :####*****++++++++++++++++++++++
/// .####******+++++++++++++++++++++
/// ,####*******++++++++++++++++++++
///  ######******+*++++++++++++++++*
///  #######********+++++++++++++++
///   #######********++++++*++++***
///    ######**#******+***++*++***
///    ,#########****************
///      #########*************+,
///       .############**#****
///         ,##############*
///             ,+#####: ,
///
/// Parameters
/// hReso, vReso :horizontally,  vertically resolutions >= 3 (initial value = 32, 16)
/// rx, ry, rz : radius(x y z) (initial value = 1.0)
/// 
#include <kika/shapes/shape.hpp>
namespace kika::shape {

template <typename Prec=double, typename Idx=uint32_t>
struct SphereParams {
  Idx hReso, vReso;
  Prec rx, ry, rz;

  SphereParams(const int& hReso, const int& vReso, const double& rx, const double& ry, const double& rz) :
    hReso(static_cast<Idx>(std::max(hReso, 3))), vReso(static_cast<Idx>(std::max(vReso, 3))),
    rx(static_cast<Prec>(rx)), ry(static_cast<Prec>(ry)), rz(static_cast<Prec>(rz)) {}
  SphereParams(const int& hReso, const int& vReso, const double& radius) : 
    SphereParams(hReso, vReso, radius, radius, radius) {}
  SphereParams(const int& hReso, const int& vReso) : SphereParams(hReso, vReso, 1.0) {}
  SphereParams(const int& reso) : SphereParams(reso, reso) {}
  SphereParams() : SphereParams(32, 16, 1.0, 1.0, 1.0) {}
};

template <typename Prec=double, typename Idx=uint32_t>
class Sphere : public Shape<Prec, Idx> {
  using Params = SphereParams<Prec, Idx>;
  using XyzGeneratorBase = typename Shape<Prec, Idx>::template Generator<Params, Prec>;
  using TriGeneratorBase = typename Shape<Prec, Idx>::template Generator<Params, Idx>;
  Params params;

  class XyzGenerator : public XyzGeneratorBase {
    using XyzGeneratorBase::params;

  public:
    XyzGenerator(const Params& params) : XyzGeneratorBase(params) {}

    Idx size() const override {
      return ((params.vReso - 1) * params.hReso + 2) * 3;
    };

    std::vector<Prec> map(const int storage = StorageOptions::RowMajor) const override {
      Idx n = this->size() / 3;
      Idx k1 = (storage == StorageOptions::ColMajor) ? 1 : 3, k2 = (storage == StorageOptions::ColMajor) ? n : 1;
      std::vector<Prec> pos(3 * n);

      for (Idx i = 0; i < params.vReso - 1; i++) {
        double s = (i+1) * _PI / params.vReso;

        for (Idx j = 0; j < params.hReso; j++) {
          double t = 2.0 * j * _PI / params.hReso;

          pos[(1 + j + params.hReso * i) * k1 + 0] = static_cast<Prec>(params.rx * cos(t) * sin(s));
          pos[(1 + j + params.hReso * i) * k1 + k2] = static_cast<Prec>(params.ry * cos(s));
          pos[(1 + j + params.hReso * i) * k1 + 2*k2] = static_cast<Prec>(params.rz * sin(t) * sin(s));
        }
      }

      pos[(0) * k1 + 0] = static_cast<Prec>(0.);
      pos[(0) * k1 + k2] = params.ry;
      pos[(0) * k1 + 2*k2] = static_cast<Prec>(0.);
      pos[(n - 1) * k1 + 0] = static_cast<Prec>(0.);
      pos[(n - 1) * k1 + k2] = -params.ry;
      pos[(n - 1) * k1 + 2*k2] = static_cast<Prec>(0.);

      return pos;
    }
  };

  class TriGenerator : public TriGeneratorBase {
    using TriGeneratorBase::params;

  public:
    TriGenerator(const Params& params) : TriGeneratorBase(params) {}

    Idx size() const override {
      return (2 * ((params.vReso - 1) * params.hReso + 2) - 4) * 3;
    };

    std::vector<Idx> map(const int storage = StorageOptions::RowMajor) const override {
      Idx n = (params.vReso - 1) * params.hReso + 2;
      Idx f = 2 * n - 4;
      Idx k1 = (storage == StorageOptions::ColMajor) ? 1 : 3, k2 = (storage == StorageOptions::ColMajor) ? f : 1;
      std::vector<Idx> idx(f * 3);

      Idx c = 0;
      for (Idx i = 0, i3 = params.hReso * (params.vReso - 2); i < params.hReso; i++) {
        Idx i1 = (i + 1) % params.hReso + 1, i2 = i + 1;
        idx[c   * k1 + 0] = 0;
        idx[c   * k1 + k2] = i1;
        idx[c++ * k1 + 2*k2] = i2;
      }

      for (Idx i = 0; i < params.vReso - 2; i++) {
        for (Idx j = 0; j < params.hReso; j++) {
          Idx i1 = params.hReso*i+1+j,
            i2 = params.hReso*i+1+(j+1)%params.hReso,
            i3 = params.hReso*i+1+(j+1)%params.hReso+params.hReso,
            i4 = params.hReso*i+1+j+params.hReso;
          idx[c   * k1 + 0] = i1;
          idx[c   * k1 + k2] = i2;
          idx[c++ * k1 + 2*k2] = i3;
          idx[c   * k1 + 0] = i1;
          idx[c   * k1 + k2] = i3;
          idx[c++ * k1 + 2*k2] = i4;
        }
      }

      for (Idx i = 0, i3 = params.hReso * (params.vReso - 2); i < params.hReso; i++) {
        Idx i1 = (i + 1) % params.hReso + 1, i2 = i + 1;
        idx[c   * k1 + 0] = n-1;
        idx[c   * k1 + k2] = i2 + i3;
        idx[c++ * k1 + 2*k2] = i1 + i3;
      }

      return idx;
    }
  };

public:

  XyzGenerator xyz;
  TriGenerator tri;

  Sphere() : params(), xyz(params), tri(params) {}
  Sphere(const Params& params) : params(params), xyz(params), tri(params) {}
  Sphere(const int& reso) : params(reso), xyz(params), tri(params) {}
  Sphere(const int& hReso, const int& vReso) : params(hReso, vReso), xyz(params), tri(params) {}
  Sphere(const int& hReso, const int& vReso, const double& radius) : 
    params(hReso, vReso, radius), xyz(params), tri(params) {}
  Sphere(const int& hReso, const int& vReso, const double& rx, const double& ry, const double& rz) :
    params(hReso, vReso, rx, ry, rz), xyz(params), tri(params) {}

  Idx nv() const override { return (params.vReso - 1) * params.hReso + 2; }
  Idx nf() const override { return 2 * nv() - 4; }
  Idx ne() const override { return 3 * nv() + 6; }

  std::vector<std::vector<Idx>> poly() const override {
    Idx n = (params.vReso - 1) * params.hReso + 2;
    Idx f = 2 * params.hReso + (params.vReso - 2) * params.hReso;

    std::vector<std::vector<Idx>> idx(f);

    Idx c = 0;
    std::vector<Idx> t(3);
    for (Idx i = 0, i3 = params.hReso * (params.vReso - 2); i < params.hReso; i++) {
      Idx i1 = (i + 1) % params.hReso + 1, i2 = i + 1;
      t[0] = 0;
      t[1] = i1;
      t[2] = i2;
      idx[c++] = t;
    }

    std::vector<Idx> q(4);
    for (Idx i = 0; i < params.vReso - 2; i++) {
      for (Idx j = 0; j < params.hReso; j++) {
        Idx i1 = params.hReso*i+1+j,
          i2 = params.hReso*i+1+(j+1)%params.hReso,
          i3 = params.hReso*i+1+(j+1)%params.hReso+params.hReso,
          i4 = params.hReso*i+1+j+params.hReso;
        q[0] = i1;
        q[1] = i2;
        q[2] = i3;
        q[3] = i4;
        idx[c++] = q;
      }
    }

    for (Idx i = 0, i3 = params.hReso * (params.vReso - 2); i < params.hReso; i++) {
      Idx i1 = (i + 1) % params.hReso + 1, i2 = i + 1;
      t[0] = n-1;
      t[1] = i2 + i3;
      t[2] = i1 + i3;
      idx[c++] = t;
    }

    return idx;
  }

  void setParams(const Params& params) { this->params = params; }
  void setHorzReso(const Idx& reso) { params.hReso = reso; }
  void setVertReso(const Idx& reso) { params.vReso = reso; }
  void setRadius(const Prec& rx = 1.0, const Prec& ry = 1.0, const Prec& rz = 1.0) {
    params.rx = rx;
    params.ry = ry;
    params.rz = rz;
  }

};
}
