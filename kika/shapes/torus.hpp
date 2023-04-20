#pragma once

///               $@@@@$$$$@@@@$
///           $$$$$############$$$$$
///         $$###****!***!!!!****###$$
///       ####***!!!!!!====!!!!*!!**####
///     *##*****!!==;;::::::;;;=!!!!***##*
///    ******!!==;::~--,,,,--~::;==!!!*****
///   =*****!!==;:~-,........,-~:;==!!!****=
///   !****!!==;:~,............,~:;==!!!***!
///  ;!!***!!==;:-,..        ..,~:;==!!!**!!
///  ;!!!**!!!==;~-.          .-~;==!!!**!!=:
///  :=!!****!!!==;~          ~;==!!!***!!!=:
///  ~==!!!*******!!;        =!!********!!=;
///   ;=!!!****#######**!!*########****!!!=;
///   ~;=!!!***####$$$$$$$@$$$$$###***!!==;,
///    ~;=!!!!**###$$$$@@@@$$$$###**!!!==;-
///     -:;=!!!!**####$$$$$$$###**!!!!=;:,
///       ~:;==!!!!****####****!!!!!=;:~ 
///        .-:;;=!!!!!!!!!!!!!!!==;;~-  
///           .-~::;;;======;;;::~-.   
///               ..,-~~~~~~-,.. 
/// 
/// Parameters
/// mjrReso, mnrReso : major, minor resolutions >= 3 (initial value = 48, 12)
/// mjrRad, mnrRad : major, minor radius (initial value = 1.0, 0.25)
/// 
#include <kika/shapes/shape.hpp>
namespace kika::shape {

template <typename Prec=double, typename Idx=uint32_t>
struct TorusParams {
  Idx mjrReso, mnrReso;
  Prec mjrRad, mnrRad;

  TorusParams(const int& mjrReso, const int& mnrReso, const double& mjrRad, const double& mnrRad) :
    mjrReso(static_cast<Idx>(std::max(mjrReso, 3))), mnrReso(static_cast<Idx>(std::max(mnrReso, 3))), 
    mjrRad(static_cast<Prec>(mjrRad)), mnrRad(static_cast<Prec>(mnrRad)) {}
  TorusParams(const int& reso) : TorusParams(reso, reso, 1.0, 0.25) {}
  TorusParams(const int& mjrReso, const int& mnrReso) : TorusParams(mjrReso, mnrReso, 1.0, 0.25) {}
  TorusParams() : TorusParams(48, 12) {}
};

template <typename Prec=double, typename Idx=uint32_t>
class Torus : public Shape<Prec, Idx> {
  using Params = TorusParams<Prec, Idx>;
  using XyzGeneratorBase = typename Shape<Prec, Idx>::template Generator<Params, Prec>;
  using TriGeneratorBase = typename Shape<Prec, Idx>::template Generator<Params, Idx>;
  Params params;

  class XyzGenerator : public XyzGeneratorBase {
    using XyzGeneratorBase::params;

  public:
    XyzGenerator(const Params& params) : XyzGeneratorBase(params) {}

    Idx size() const override {
      return (params.mjrReso * params.mnrReso) * 3;
    };

    std::vector<Prec> map(const int storage = StorageOptions::RowMajor) const override {
      Idx n = this->size() / 3;
      Idx k1 = (storage == StorageOptions::ColMajor) ? 1 : 3, k2 = (storage == StorageOptions::ColMajor) ? n : 1;
      std::vector<Prec> pos(3 * n);

      for (Idx i = 0; i < params.mnrReso; i++) {
        double s = 2.0 * i * _PI / params.mnrReso;

        for (Idx j = 0; j < params.mjrReso; j++) {
          double t = 2.0 * j * _PI / params.mjrReso;

          pos[(j + params.mjrReso * i) * k1 + 0] = static_cast<Prec>((params.mjrRad + params.mnrRad * cos(s)) * cos(t));
          pos[(j + params.mjrReso * i) * k1 + k2] = static_cast<Prec>(params.mnrRad * sin(s));
          pos[(j + params.mjrReso * i) * k1 + 2*k2] = static_cast<Prec>((params.mjrRad + params.mnrRad * cos(s)) * sin(t));
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
      return (2 * params.mjrReso * params.mnrReso) * 3;
    };

    std::vector<Idx> map(const int storage = StorageOptions::RowMajor) const override {
      Idx n = params.mjrReso * params.mnrReso;
      Idx f = 2 * n;
      Idx k1 = (storage == StorageOptions::ColMajor) ? 1 : 3, k2 = (storage == StorageOptions::ColMajor) ? f : 1;
      std::vector<Idx> idx(f * 3);

      Idx c = 0;
      for (Idx i = 0; i < params.mnrReso; i++) {
        for (Idx j = 0; j < params.mjrReso; j++) {
          Idx i1 = j + i * params.mjrReso,
            i2 = (j + params.mjrReso + i * params.mjrReso) % n,
            i3 = ((j+1) % params.mjrReso + params.mjrReso + i * params.mjrReso) % n,
            i4 = ((j+1) % params.mjrReso + i * params.mjrReso) % n;
          idx[c   * k1 + 0] = i1;
          idx[c   * k1 + k2] = i2;
          idx[c++ * k1 + 2*k2] = i4;
          idx[c   * k1 + 0] = i4;
          idx[c   * k1 + k2] = i2;
          idx[c++ * k1 + 2*k2] = i3;
        }
      }

      return idx;
    }
  };

public:

  XyzGenerator xyz;
  TriGenerator tri;

  Torus() : params(), xyz(params), tri(params) {}
  Torus(const Params& params) : params(params), xyz(params), tri(params) {}
  Torus(const int& reso) : params(reso), xyz(params), tri(params) {}
  Torus(const int& mjrReso, const int& mnrReso) : 
    params(mjrReso, mnrReso), xyz(params), tri(params) {}
  Torus(const int& mjrReso, const int& mnrReso, const double& mjrRad, const double& mnrRad) : 
    params(mjrReso, mnrReso, mjrRad, mnrRad), xyz(params), tri(params) {}

  Idx nv() const override {
    return params.mjrReso * params.mnrReso;
  }

  Idx nf() const override {
    return 2 * nv();
  }

  Idx ne() const override {
    return 3 * nv();
  }

  std::vector<std::vector<Idx>> poly() const override {
    Idx n = params.mjrReso * params.mnrReso;
    Idx f = n;

    std::vector<std::vector<Idx>> idx(f, std::vector<Idx>(4));

    Idx c = 0;
    for (Idx i = 0; i < params.mnrReso; i++) {
      for (Idx j = 0; j < params.mjrReso; j++) {
        Idx i1 = j + i * params.mjrReso,
          i2 = (j + params.mjrReso + i * params.mjrReso) % n,
          i3 = ((j+1) % params.mjrReso + params.mjrReso + i * params.mjrReso) % n,
          i4 = ((j+1) % params.mjrReso + i * params.mjrReso) % n;
        idx[c  ][0] = i1;
        idx[c  ][1] = i2;
        idx[c  ][2] = i3;
        idx[c++][3] = i4;
      }
    }

    return idx;
  }

  void setParams(const Params& params) { this->params = params; }
  void setMajorRadius(const Prec& rad) { params.mjrRad = rad; }
  void setMajorReso(const Idx& reso) { params.mjrReso = reso; }
  void setMinorRadius(const Prec& rad) { params.mnrRad = rad; }
  void setMinorReso(const Idx& reso) { params.mnrReso = reso; }

};
}