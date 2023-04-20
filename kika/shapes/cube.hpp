#pragma once
                                        
///             ,.++.
///          ,.#****+*+,
///       ,:*****@**#*+++:,
///    ,:*@#*****#*##+++++*#.
/// ,:*******@@**++++*@*#*+++++,
/// :#@****@#**@#+++*#*+##+++++#@
/// .**#@@*******#@#+++++++#*W*+:
/// .***W#@#***##*+*@*++++@#@+++:
/// ,***@**##@**++++++##@*++@+++:
/// ,***@***#@#@*+++*@#W++++#+++:
/// ,#**@****#*####@+++@++++#+++:
/// ,@@#@****#***##++++@++++#++@.
/// ,*##@#***@***#*++++@++++@@#+,
///  ***@@@#*@***#*++++@++#W#+++,
///  ***@*##@@***#*++++W@@+:*+++,
///  ***#***#@@#*#*++*@@+++:++++,
///  #**#****@##@###@*+@++++++++,
///  #@##****@**##@*+++@++++++#@ 
///  **#@#***@***#*++++#+++*#@++ 
///  ***#@@#*@***#*++++#++#W++++ 
///  +***#*#@@***#*++++##@+*++++ 
///  +***#**#@@**#*+++@@+++#++++ 
///   ***#***@####*+@@+#+++#++++ 
///    ,#@***@**##W#:++*+++#++,
///      :***W***##++++*+++*,
///        **W***#*++++*++,
///         ,W***#*++++#,
///           :**#*+++,
///             +#*+.
///              ,.
/// 
/// Parameters
/// reso : resolutions >= 1 (initial value = 1)
/// size : size (initial value = 2.0)
/// 
#include <kika/shapes/shape.hpp>

namespace kika::shape {

template <typename Prec=double, typename Idx=uint32_t>
struct CubeParams {
  Idx reso;
  Prec size;

  CubeParams(const int& reso, const double& size) :
    reso(static_cast<Idx>(std::max(reso, 1))), size(static_cast<Prec>(size)) {}
  CubeParams(const int& reso) : CubeParams(reso, 2.0) {}
  CubeParams() : CubeParams(1, 1.0) {}
};

//TODO
template <typename Prec=double, typename Idx=uint32_t>
class Cube : public Shape<Prec, Idx> {
  using Params = CubeParams<Prec, Idx>;
  using XyzGeneratorBase = typename Shape<Prec, Idx>::template Generator<Params, Prec>;
  using TriGeneratorBase = typename Shape<Prec, Idx>::template Generator<Params, Idx>;
  Params params;

  class XyzGenerator : public XyzGeneratorBase {
    using XyzGeneratorBase::params;

  public:
    XyzGenerator(const Params& params) : XyzGeneratorBase(params) {}

    Idx size() const override {
      return (6 * params.reso * params.reso + 2) * 3;
    };

    std::vector<Prec> map(const int storage = StorageOptions::RowMajor) const override {
      Idx n = size() / 3;
      Idx k1 = (storage == StorageOptions::ColMajor) ? 1 : 3, k2 = (storage == StorageOptions::ColMajor) ? n : 1;
      std::vector<Prec> pos(3 * n);

      double x, y, z, hs = params.size / 2., sr = params.size / params.reso;
      Idx c = 0;
      
      auto set = [&k1, &k2, &pos, &c](double& x, double& y, double& z) {
        pos[(c)*k1 + 0] = static_cast<Prec>(x);
        pos[(c)*k1 + k2] = static_cast<Prec>(y);
        pos[(c++) * k1 + 2*k2] = static_cast<Prec>(z);
      };

      for (Idx i = 0; i < 2; i++) {
        x = i ? -hs : hs;
        for (Idx j = 0; j < params.reso + 1; j++) {
          y = sr * j - hs;
          for (Idx r = 0; r < params.reso + 1; r++) {
            z = sr * r - hs;
            set(x, y, z);
          }
        }
      }
      
      for (Idx i = 0; i < 2; i++) {
        z = i ? -hs : hs;
        for (Idx j = 1; j < params.reso; j++) {
          x = sr * j - hs;
          for (Idx r = 0; r < params.reso + 1; r++) {
            y = sr * r - hs;
            set(x, y, z);
          }
        }
      }
      
      for (Idx i = 0; i < 2; i++) {
        y = i ? -hs : hs;
        for (Idx j = 1; j < params.reso; j++) {
          z = sr * j - hs;
          for (Idx r = 1; r < params.reso; r++) {
            x = sr * r - hs;
            set(x, y, z);
          }
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
      return (2 * (6 * params.reso * params.reso + 2) - 4) * 3;
    };

    std::vector<Idx> map(const int storage = StorageOptions::RowMajor) const override {
      Idx n = 6 * params.reso * params.reso + 2;
      Idx f = 2 * n - 4;
      Idx k1 = (storage == StorageOptions::ColMajor) ? 1 : 3, k2 = (storage == StorageOptions::ColMajor) ? f : 1;
      std::vector<Idx> idx(f * 3);

      Idx c = 0, r1 = params.reso + 1, r2 = params.reso, r3 = params.reso - 1;
      auto set = [&c, &k1, &k2, &idx](Idx i1, Idx i2, Idx i3, Idx i4) {
        idx[(c)*k1 + 0] = i1; idx[(c)*k1 + k2] = i2; idx[(c++)*k1 + 2*k2] = i4;
        idx[(c)*k1 + 0] = i4; idx[(c)*k1 + k2] = i2; idx[(c++)*k1 + 2*k2] = i3;
      };

      Idx oxp = 0, oxm = r1 * r1;
      Idx ozp = oxm + r1 * r1, ozm = ozp + r2 * r2 - 1;
      Idx oyp = ozm + r2 * r2 - 1, oym = oyp + r3 * r3;
      for (Idx i = 0; i < params.reso; i++) {
        Idx o = oxp + i * r1;
        for (Idx j = 0; j < params.reso; j++)
          set(j + o, r1 + j + o, r1 + j + 1 + o, j + 1 + o);
      }
      for (Idx i = 0; i < params.reso; i++) {
        Idx o = oxm + i * r1;
        for (Idx j = 0; j < params.reso; j++)
          set(j + o, j + 1 + o, r1 + j + 1 + o, r1 + j + o);
      }

      if (params.reso == 1) {
        set(0, 1, 5, 4);
        set(0, 4, 6, 2);
        set(2, 6, 7, 3);
        set(3, 7, 5, 1);
      } else {
        for (Idx i = 1; i < params.reso-1; i++) {
          Idx o = ozp + (i-1) * r1;
          for (Idx j = 0; j < params.reso; j++)
            set(j + o, r1 + j + o, r1 + j + 1 + o, j + 1 + o);
        }
        for (Idx i = 0; i < params.reso; i++) {
          set((i+2) * r1 - 1, i + r2 * r3 - 1 + ozp, i + r2 * r3 - 2 + ozp, (i+1) * r1 - 1);
          set(i + 1 + ozp, (i+2) * r1 - 1 + oxm, (i+1) * r1 - 1 + oxm, i + ozp);
        }

        for (Idx i = 1; i < params.reso - 1; i++) {
          Idx o = ozm + (i-1) * r1;
          for (Idx j = 0; j < params.reso; j++)
            set(j + o, j + 1 + o, r1 + j + 1 + o, r1 + j + o);
        }
        for (Idx i = 0; i < params.reso; i++) {
          set((i+1) * r1, i * r1, i + r2 * r3 - 2 + ozm, i + r2 * r3 - 1 + ozm);
          set(i * r1 + oxm, (i+1) * r1 + oxm, i + 1 + ozm, i + ozm);
        }

        for (Idx i = 1; i < params.reso-1; i++) {
          Idx o = oyp + (i - 1) * r3;
          for (Idx j = 1; j < params.reso - 1; j++) {
            set(j + o, j - 1 + o, r3 - 1 + j + o, r3 + j + o);
          }
        }
        set(oxm - 2, oym - 1, ozm - 1, oxm - 1);
        set(2*oxm + 1 - r1, oyp, ozm+r1-1, 2*oxm - r1);
        set(oyp + r3 - 1, oxm + 1 - r1, oxm - r1, oyp - 1);
        set(2*oxm - 1, ozp + r1 - 1, oym - r3, 2*oxm - 2);
        for (Idx i = 1; i < params.reso - 1; i++) {
          set(oxm-i-2, oym-i*r3-1, oym-(i-1)*r3-1, oxm-i-1);
          set(ozp-i-1, oym-i*r3, oym-(i+1)*r3, ozp-i-2);
          set(ozm+(i+1)*r1-1, ozm+i*r1-1, oyp-1+i, oyp+i);
          set(ozp+i*r1-1, ozp+(i+1)*r1-1, oym-r3+i, oym-r3+i-1);
        }

        for (Idx i = 1; i < params.reso-1; i++) {
          Idx o = oym + (i - 1) * r3;
          for (Idx j = 1; j < params.reso - 1; j++) {
            set(j + o, r3 + j + o, r3 - 1 + j + o, j - 1 + o);
          }
        }
        set(0, 1, oym + r3 - 1, oyp-r3-2);
        set(n - r3, ozp, 1+r1-2+oxm, 0+r1-2+oxm);
        set(0+r1-2, 1+r1-2, ozm - r1, n - 1);
        set(1+oxm, oxm, ozm, oym);
        for (Idx i = 1; i < params.reso - 1; i++) {
          set(i, i+1, oym+(i+1)*r3-1, oym+i*r3-1);
          set(i+1+oxm, i+oxm, oym+(i-1)*r3, oym+i*r3);
          set(oym-1+i, ozm + (i-1) * r1, ozm + i * r1, oym+i);
          set(n-r3+i, ozp+i*r1, ozp+(i-1)*r1, n-r3-1+i);
        }
      }

      return idx;
    }
  };

public:

  XyzGenerator xyz;
  TriGenerator tri;

  Cube() : params(), xyz(params), tri(params) {}
  Cube(const CubeParams<Prec, Idx>& params) : params(params), xyz(params), tri(params) {}
  Cube(const int& reso) : params(reso), xyz(params), tri(params) {}
  Cube(const int& reso, const double& size) : params(reso, size), xyz(params), tri(params) {}

  Idx nv() const override { return 6 * params.reso * params.reso + 2; }
  Idx nf() const override { return 2 * nv() - 4; }
  Idx ne() const override { return 3 * nv() + 6; }

  std::vector<std::vector<Idx>> poly() const override {
    Idx n = 6 * params.reso * params.reso + 2;
    Idx f = 6 * params.reso * params.reso;

    std::vector<std::vector<Idx>> idx(f, std::vector<Idx>(4));

    Idx c = 0, r1 = params.reso + 1, r2 = params.reso, r3 = params.reso - 1;
    auto set = [&c, &idx](Idx i1, Idx i2, Idx i3, Idx i4) {
      idx[c][0] = i1; idx[c][1] = i2; idx[c][2] = i3; idx[c++][3] = i4;
    };

    Idx oxp = 0, oxm = r1 * r1;
    Idx ozp = oxm + r1 * r1, ozm = ozp + r2 * r2 - 1;
    Idx oyp = ozm + r2 * r2 - 1, oym = oyp + r3 * r3;
    for (Idx i = 0; i < params.reso; i++) {
      Idx o = oxp + i * r1;
      for (Idx j = 0; j < params.reso; j++)
        set(j + o, r1 + j + o, r1 + j + 1 + o, j + 1 + o);
    }
    for (Idx i = 0; i < params.reso; i++) {
      Idx o = oxm + i * r1;
      for (Idx j = 0; j < params.reso; j++) 
        set(j + o, j + 1 + o, r1 + j + 1 + o, r1 + j + o);
    }

    if (params.reso == 1) {
      set(0, 1, 5, 4);
      set(0, 4, 6, 2);
      set(2, 6, 7, 3);
      set(3, 7, 5, 1);
    } else {
      for (Idx i = 1; i < params.reso-1; i++) {
        Idx o = ozp + (i-1) * r1;
        for (Idx j = 0; j < params.reso; j++)
          set(j + o, r1 + j + o, r1 + j + 1 + o, j + 1 + o);
      }
      for (Idx i = 0; i < params.reso; i++) {
        set((i+2) * r1 - 1, i + r2 * r3 - 1 + ozp, i + r2 * r3 - 2 + ozp, (i+1) * r1 - 1);
        set(i + 1 + ozp, (i+2) * r1 - 1 + oxm, (i+1) * r1 - 1 + oxm, i + ozp);
      }
      for (Idx i = 1; i < params.reso - 1; i++) {
        Idx o = ozm + (i-1) * r1;
        for (Idx j = 0; j < params.reso; j++) 
          set(j + o, j + 1 + o, r1 + j + 1 + o, r1 + j + o);
      }
      for (Idx i = 0; i < params.reso; i++) {
        set((i+1) * r1, i * r1, i + r2 * r3 - 2 + ozm, i + r2 * r3 - 1 + ozm);
        set(i * r1 + oxm, (i+1) * r1 + oxm, i + 1 + ozm, i + ozm);
      }
      for (Idx i = 1; i < params.reso-1; i++) {
        Idx o = oyp + (i - 1) * r3;
        for (Idx j = 1; j < params.reso - 1; j++) {
          set(j + o, j - 1 + o, r3 - 1 + j + o, r3 + j + o);
        }
      }
      set(oxm - 2, oym - 1, ozm - 1, oxm - 1);
      set(2*oxm + 1 - r1, oyp, ozm+r1-1, 2*oxm - r1);
      set(oyp + r3 - 1, oxm + 1 - r1, oxm - r1, oyp - 1);
      set(2*oxm - 1, ozp + r1 - 1, oym - r3, 2*oxm - 2);
      for (Idx i = 1; i < params.reso - 1; i++) {
        set(oxm-i-2, oym-i*r3-1, oym-(i-1)*r3-1, oxm-i-1);
        set(ozp-i-1, oym-i*r3, oym-(i+1)*r3, ozp-i-2);
        set(ozm+(i+1)*r1-1, ozm+i*r1-1, oyp-1+i, oyp+i);
        set(ozp+i*r1-1, ozp+(i+1)*r1-1, oym-r3+i, oym-r3+i-1);
      }
      for (Idx i = 1; i < params.reso-1; i++) {
        Idx o = oym + (i - 1) * r3;
        for (Idx j = 1; j < params.reso - 1; j++) {
          set(j + o, r3 + j + o, r3 - 1 + j + o, j - 1 + o);
        }
      }
      set(0, 1, oym + r3 - 1, oyp-r3-2);
      set(n - r3, ozp, 1+r1-2+oxm, 0+r1-2+oxm);
      set(0+r1-2, 1+r1-2, ozm - r1, n - 1);
      set(1+oxm, oxm, ozm, oym);
      for (Idx i = 1; i < params.reso - 1; i++) {
        set(i, i+1, oym+(i+1)*r3-1, oym+i*r3-1);
        set(i+1+oxm, i+oxm, oym+(i-1)*r3, oym+i*r3);
        set(oym-1+i, ozm + (i-1) * r1, ozm + i * r1, oym+i);
        set(n-r3+i, ozp+i*r1, ozp+(i-1)*r1, n-r3-1+i);
      }
    }

    return idx;
  }
};
}
