#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "phoneTxCAT/am-phoneTxCAT.h"
#include "hmm/transition-model.h"
#include "phoneTxCAT/estimate-am-phoneTxCAT.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Accumulate stats for Phone Transform CAT training.\n"
        "Usage: phoneTxCAT-acc-stats [options] <model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> <stats-out>\n"
        "e.g.: phoneTxCAT-acc-stats 1.mdl 1.ali scp:train.scp 'ark:ali-to-post 1.ali ark:-|' 1.acc\n";
    
    ParseOptions po(usage);
    
    bool binary = true;
    std::string gselect_rspecifier, utt2spk_rspecifier;
    std::string update_flags_str = "vAwuSt";
    BaseFloat rand_prune = 1.0e-05;
    PhoneTxCATGselectConfig phoneTxCAT_opts;

    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("gselect", &gselect_rspecifier, "Precomputed Gaussian indices (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Register("rand-prune", &rand_prune, "Pruning threshold for posteriors");
    po.Register("update-flags", &update_flags_str, "Which Phone \
        Transform CAT parameters to accumulate "
                "stats for: subset of vAwuSt.");
    phoneTxCAT_opts.Register(&po);

    po.Read(argc, argv);

    kaldi::PhoneTxCATUpdateFlagsType acc_flags = 
      StringToPhoneTxCATUpdateFlags(update_flags_str);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetArg(4);
    
    using namespace kaldi;
    typedef kaldi::int32 int32;

    // Initialize the readers before the model, as the model can
    // be large, and we don't want to call fork() after reading it if
    // virtual memory may be low.
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    RandomAccessTokenReader utt2spk_reader(utt2spk_rspecifier);

    AmPhoneTxCAT am_phoneTxCAT;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_phoneTxCAT.Read(ki.Stream(), binary);
    }

    Vector<double> transition_accs;
    if (acc_flags & kaldi::kPhoneTxCATTransitions)
      trans_model.InitStats(&transition_accs);
    
    MleAmPhoneTxCATAccs phoneTxCAT_accs(rand_prune);
    phoneTxCAT_accs.ResizeAccumulators(am_phoneTxCAT, acc_flags);
    
    double tot_like = 0.0;
    double tot_t = 0;

    kaldi::PhoneTxCATPerFrameDerivedVars per_frame_vars;

    int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!posteriors_reader.HasKey(utt)) {
        num_no_posterior++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const Posterior &posterior = posteriors_reader.Value(utt);

        bool have_gselect  = !gselect_rspecifier.empty()
            && gselect_reader.HasKey(utt)
            && gselect_reader.Value(utt).size() == mat.NumRows();
        if (!gselect_rspecifier.empty() && !have_gselect)
          KALDI_WARN << "No Gaussian-selection info available for utterance "
                     << utt << " (or wrong size)";
        std::vector<std::vector<int32> > empty_gselect;
        const std::vector<std::vector<int32> > *gselect =
            (have_gselect ? &gselect_reader.Value(utt) : &empty_gselect);

        if (posterior.size() != mat.NumRows()) {
          KALDI_WARN << "Alignments has wrong size "<< (posterior.size()) <<
              " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }
        
        string utt_or_spk;
        if (utt2spk_rspecifier.empty())  utt_or_spk = utt;
        else {
          if (!utt2spk_reader.HasKey(utt)) {
            KALDI_WARN << "Utterance " << utt << " not present in utt2spk map; "
                       << "skipping this utterance.";
            num_other_error++;
            continue;
          } else {
            utt_or_spk = utt2spk_reader.Value(utt);
          }
        }
        
        num_done++;
        BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;
        for (size_t i = 0; i < posterior.size(); i++) {
          std::vector<int32> this_gselect;
          if (!gselect->empty()) this_gselect = (*gselect)[i];
          else am_phoneTxCAT.GaussianSelection(
              phoneTxCAT_opts, mat.Row(i), &this_gselect);

          am_phoneTxCAT.ComputePerFrameVars(
              mat.Row(i), this_gselect, 
              &per_frame_vars);

          for (size_t j = 0; j < posterior[i].size(); j++) {
            int32 tid = posterior[i][j].first,  // transition identifier.
                pdf_id = trans_model.TransitionIdToPdf(tid);
            BaseFloat weight = posterior[i][j].second;
            if (acc_flags & kaldi::kPhoneTxCATTransitions)
              trans_model.Accumulate(weight, tid, &transition_accs);
            tot_like_this_file += phoneTxCAT_accs.Accumulate(
                am_phoneTxCAT, per_frame_vars, pdf_id, 
                weight, acc_flags) * weight;
            tot_weight += weight;
          }
        }

        KALDI_VLOG(2) << "Average like for this file is "
                      << (tot_like_this_file/tot_weight) << " over "
                      << tot_weight <<" frames.";
        tot_like += tot_like_this_file;
        tot_t += tot_weight;
        if (num_done % 50 == 0) {
          KALDI_LOG << "Processed " << num_done << " utterances; for utterance "
                    << utt << " avg. like is "
                    << (tot_like_this_file/tot_weight)
                    << " over " << tot_weight <<" frames.";
        }
      }
    }
    KALDI_LOG << "Overall like per frame (Gaussian only) = "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";

    KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
              << " with no posteriors, " << num_other_error
              << " with other errors.";

    {
      Output ko(accs_wxfilename, binary);
      transition_accs.Write(ko.Stream(), binary);
      phoneTxCAT_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

