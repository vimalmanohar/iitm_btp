// phoneTxCATbin/phoneTxCAT-copy.cc

#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "phoneTxCAT/am-phoneTxCAT.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    const char *usage =
        "Copy Phone Transform CAT Model (possibly changing binary/text format)\n"
        "Usage: phoneTxCAT-copy [options] <model-in> <model-out>\n"
        "e.g.: phoneTxCAT-copy --binary=false 1.mdl 1_text.mdl\n";

    bool binary_write = true;
    bool debug_model= false;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("debug-model", &debug_model, "Write output in binary mode");

    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    AmPhoneTxCAT am_phoneTxCAT;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_phoneTxCAT.Read(ki.Stream(), binary);
    }

    if (!debug_model)
    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_phoneTxCAT.Write(ko.Stream(), binary_write, kPhoneTxCATWriteAll);
    }
    else {
      binary_write = false;
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_phoneTxCAT.Write(ko.Stream(), binary_write, kPhoneTxCATWriteAll);

      int32 num_gaussians = am_phoneTxCAT.NumGauss();
      int32 num_clusters = am_phoneTxCAT.NumClusters();
      int32 feature_dim = am_phoneTxCAT.FeatureDim();

      {
        std::vector< Matrix<BaseFloat> > M;
        M.resize(num_gaussians);
        if (!binary_write) ko.Stream() << "\n";
        WriteToken(ko.Stream(), binary_write, "<M>");
        for (int32 i = 0; i < num_gaussians; i++) {
          M[i].Resize(feature_dim, num_clusters);
          am_phoneTxCAT.GetModelSpaceProjection(i, &M[i]);
          M[i].Write(ko.Stream(), binary_write);
        }
      }
      {
        std::vector< Matrix<BaseFloat> > M_SigmaInv_i;
        M_SigmaInv_i.resize(num_gaussians);
        am_phoneTxCAT.ComputeM_SigmaInv(&M_SigmaInv_i);
        if (!binary_write) ko.Stream() << "\n";
        WriteToken(ko.Stream(), binary_write, "<M_SigmaInv>");
        for (int32 i = 0; i < num_gaussians; i++) {
          M_SigmaInv_i[i].Write(ko.Stream(), binary_write);
        }
      }
      {
        std::vector< SpMatrix<BaseFloat> > H;
        H.resize(num_gaussians);
        am_phoneTxCAT.ComputeH(&H);
        int32 dim = H[0].NumRows();
        if (!binary_write) ko.Stream() << "\n";
        WriteToken(ko.Stream(), binary_write, "<H>");
        for (int32 i = 0; i < num_gaussians; i++) {
          Matrix<BaseFloat> H_tmp(dim, dim);
          H_tmp.CopyFromSp(H[i]);
          H_tmp.Write(ko.Stream(), binary_write);
        }
      }
    }
    
    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



