// sgmmbin/ubm-copy.cc

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/kaldi-io.h"
#include "gmm/full-gmm.h"

using kaldi::FullGmm;
using kaldi::int32;
using kaldi::BaseFloat;

void kaldi::FullGmm::WriteHTK(std::ostream &out_stream, bool binary) const {
  if (!valid_gconsts_)
    KALDI_ERR << "Must call ComputeGconsts() before writing the model.";

  if(binary)
    KALDI_WARN << "Binary mode HTK not tested. \
      Not writing in binary mode even though \
      the flag is binary";
      binary = false;

  WriteToken(out_stream, binary, "<NUMMIXES>");
  int32 num_gauss = NumGauss();
  WriteBasicType(out_stream, binary, num_gauss);
  if (!binary) out_stream << "\n";

  int32 feature_dim = Dim();
  Matrix<double> means(num_gauss, feature_dim);
  GetMeans(&means);

  for (int32 i = 0; i < num_gauss; i++) {
    WriteToken(out_stream, binary, "<MIXTURE>");
    WriteBasicType(out_stream, binary, i+1);
    WriteBasicType(out_stream, binary, weights_(i));
    if (!binary) out_stream << "\n";

    WriteToken(out_stream, binary, "<MEAN>");
    WriteBasicType(out_stream, binary, feature_dim);
    if (!binary) out_stream << "\n";
    out_stream << " ";
    for (int32 j = 0; j < feature_dim; j++)
      WriteBasicType(out_stream, binary, means(i,j));
    if (!binary) out_stream << "\n";

    WriteToken(out_stream, binary, "<INVCOVAR>");
    WriteBasicType(out_stream, binary, feature_dim);
    if (!binary) out_stream << "\n";
    for (int32 j = 0; j < feature_dim; j++) {
      out_stream << " ";
      for (int32 k = j; k < feature_dim; k++) {
        WriteBasicType(out_stream, binary, inv_covars_[i](j,k));
      }
      if (!binary) out_stream << "\n";
    }

    WriteToken(out_stream, binary, "<GCONST>");
    double gconst = feature_dim * M_LOG_2PI 
      - inv_covars_[i].LogPosDefDet(); 
    WriteBasicType(out_stream, binary, gconst);
    if (!binary) out_stream << "\n";
  }
}


void kaldi::FullGmm::ReadHTK(std::istream &in_stream, bool binary) {
//  ExpectToken(in_stream, binary, "<FullGMMBegin>");
  std::string token;
  ReadToken(in_stream, binary, &token);
  if (token != "<NUMMIXES>")
    KALDI_ERR << "Expected <NUMMIXES>, got " << token;
  int32 num_gauss;
  ReadBasicType(in_stream, binary, &num_gauss);
  inv_covars_.resize(num_gauss);
  weights_.Resize(num_gauss);

  int32 feature_dim = -1;
  Matrix<BaseFloat> means;

  for (int32 i = 0; i < num_gauss; i++) {
    ExpectToken(in_stream, binary, "<MIXTURE>");
    int32 m;
    ReadBasicType(in_stream, binary, &m);
    KALDI_ASSERT(m == i+1);
    BaseFloat w_i = -1;
    ReadBasicType(in_stream, binary, &w_i);
    KALDI_ASSERT(w_i > 0);
    weights_(i) = w_i;
    
    ExpectToken(in_stream, binary, "<MEAN>");

    int32 feat_dim_tmp; 
    ReadBasicType(in_stream, binary, &feat_dim_tmp);
    KALDI_ASSERT(feat_dim_tmp > 0);
    if (feature_dim == -1) {
      feature_dim = feat_dim_tmp;
      means.Resize(num_gauss, feature_dim);
      means_invcovars_.Resize(num_gauss, feature_dim);
      gconsts_.Resize(num_gauss);
    }
    inv_covars_[i].Resize(feature_dim);

    for (int32 d = 0; d < feature_dim; d++) {
      BaseFloat tmp;
      if (d == -1) {
        ReadBasicType(in_stream, binary, &tmp);
        continue;
      }
      ReadBasicType(in_stream, binary, &means(i,d));
    }

    ExpectToken(in_stream, binary, "<INVCOVAR>");
    ReadBasicType(in_stream, binary, &feat_dim_tmp);
    KALDI_ASSERT(feat_dim_tmp == feature_dim);
    
    for (int32 p = 0; p < feature_dim; p++) {
      for (int32 q = p; q < feature_dim; q++) {
        BaseFloat tmp;
        if (q == p-1) {
          ReadBasicType(in_stream, binary, &tmp);
          continue;
        }
        ReadBasicType(in_stream, binary, &inv_covars_[i](p,q));
      }
    }

    ExpectToken(in_stream, binary, "<GCONST>");
    double gconst;
    ReadBasicType(in_stream, binary, &gconst);
    double gconst2 = feature_dim * M_LOG_2PI - inv_covars_[i].LogPosDefDet();

    KALDI_ASSERT(ApproxEqual(gconst, gconst2));
  
    Vector<BaseFloat> means_times_invcovars;
    Vector<BaseFloat> mean_i(means.Row(i));
    means_times_invcovars.Resize(feature_dim);
    means_times_invcovars.AddSpVec(1.0, inv_covars_[i], mean_i, 0.0);
    means_invcovars_.Row(i).CopyFromVec(means_times_invcovars);
  }
 
  ComputeGconsts();  // safer option than trusting the read gconsts

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::BaseFloat BaseFloat;

    const char *usage =
      "Copy UBM based model (and possibly change binary/text format)\n"
      "Usage:  ubm-copy [options] <model-in> <model-out>\n"
      "e.g.:\n"
      " ubm-copy --binary=false 1.mdl 1_txt.mdl\n";

    bool binary_write = true;
    std::string target_kind = "kaldi";
    bool htk_in = false;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("target-kind", &target_kind, "Target kind [kaldi/htk]");
    po.Register("htk-in", &htk_in, "HTK MMF Input");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
      model_out_filename = po.GetArg(2);

    kaldi::FullGmm ubm;
    {
      bool binary_read;
      kaldi::Input ki(model_in_filename, &binary_read);
      if (htk_in == false)
        ubm.Read(ki.Stream(), binary_read);
      else
        ubm.ReadHTK(ki.Stream(), binary_read);
    }

    if (target_kind == "kaldi") {
      Output ko(model_out_filename, binary_write);
      ubm.Write(ko.Stream(), binary_write);
    }
    else if (target_kind == "htk") {
      Output ko(model_out_filename, binary_write);
      ubm.WriteHTK(ko.Stream(), binary_write);
    }
    else {
      KALDI_ERR << "Invalid target_kind string" << target_kind;
      return -1;
    }

    KALDI_LOG << "Written model to " << model_out_filename;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


