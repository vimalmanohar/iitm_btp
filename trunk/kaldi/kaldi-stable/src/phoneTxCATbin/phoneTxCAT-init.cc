// phoneTxCATbin/phoneTxCAT-init.cc

#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "phoneTxCAT/am-phoneTxCAT.h"
#include "hmm/transition-model.h"
#include "tree/event-map.h"
#include "tree/build-tree.h"
#include "tree/clusterable-classes.h"
#include "thread/kaldi-thread.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize a PhoneTxCAT model from a trained diag-covariance UBM and a specified"
        " CDHMM model. \n"
        "Usage: phoneTxCAT-init [options] <topology> <tree> <treeacc> <init-model> <phoneTxCAT-out>\n"
        "The <init-model> argument can be a UBM (the default case) or another\n"
        "PhoneTxCAT model (if the --init-from-phoneTxCAT flag is used).\n";

    bool binary = true, init_from_phoneTxCAT = false;
    bool copy_from_phoneTxCAT = false, recluster_gaussians_flag = false,
         reinitializeA_flag = false;
    bool tie_weights_to_cluster = true, use_state_dep_map = false;
    bool use_weight_projection = true;
    int32 num_transform_classes = 1;
    int32 num_cluster_weight_classes = 1;
    bool use_full_covar = false;
    std::string roots_filename;

    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("num-threads", &g_num_threads, "Number of threads to use in "
                "weight update and normalizer computation");
    po.Register("init-from-phoneTxCAT", &init_from_phoneTxCAT,
        "Initialize from another PhoneTxCAT (instead of a UBM).");
    po.Register("copy-from-phoneTxCAT", &copy_from_phoneTxCAT,
        "Copy from another PhoneTxCAT");
    po.Register("recluster-gaussians", &recluster_gaussians_flag,
        "Recluster Gaussians for Transform Class");
    po.Register("reinitializeA", &reinitializeA_flag,
        "Reinitialize Transforms after clustering");
    po.Register("tie-gaussian-weights-to-clusters", &tie_weights_to_cluster, 
        "Tie Gaussian weights w_ji to cluster (w_pi) [true])");
    po.Register("use-state-dep-map", &use_state_dep_map,
        "Create phone clusters based on the phone and the pdf class");
    po.Register("use-weight-projection", &use_weight_projection,
        "Use Weight Projection with Log Exponential Transform");
    po.Register("num-transform-classes", &num_transform_classes,
        "Number of Transform classes");
    po.Register("num-cluster-weight-classes", &num_cluster_weight_classes,
        "Number of Cluster Weight Classes");
    po.Register("use-full-covar", &use_full_covar,
        "Use Full Covariance matrix");
    po.Register("roots-file", &roots_filename,
        "Use roots file to initialize clusters");

    po.Read(argc, argv);
    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string topo_in_filename = po.GetArg(1),
        tree_in_filename = po.GetArg(2),
        tree_stats_in_filename = po.GetArg(3),
        init_model_filename = po.GetArg(4),
        phoneTxCAT_out_filename = po.GetArg(5);

    if (roots_filename != "") {
      use_state_dep_map = false;
    }

    ContextDependency ctx_dep;
    {
      bool binary_in;
      Input ki(tree_in_filename.c_str(), &binary_in);
      ctx_dep.Read(ki.Stream(), binary_in);
    }
  
    HmmTopology topo;
    ReadKaldiObject(topo_in_filename, &topo);
    
    TransitionModel trans_model(ctx_dep, topo);
    
    BuildTreeStatsType stats;
    {
      bool binary_in;
      GaussClusterable gc;
      Input ki(tree_stats_in_filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
    }
    KALDI_LOG << "Number of separate statistics is " << stats.size() << "\n";
    
    int32 num_states = trans_model.NumPdfs();
    std::vector<int32> pdf_id_to_cluster;
    pdf_id_to_cluster.resize(num_states);

    for (int32 j = 0; j < num_states; j++) 
      pdf_id_to_cluster[j] = -1;

    const EventMap &map = ctx_dep.ToPdfMap();
    int32 N = ctx_dep.ContextWidth();
    int32 P = ctx_dep.CentralPosition();

    int32 max_cluster= 0;
 
    // Assumes contiguous phones
    
    if (use_state_dep_map) {
      std::vector<int32> phone2num_pdf_classes;
      topo.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);
      int32 num_phones = phone2num_pdf_classes.size() - 1;  // 1-based indexing of phones
      std::vector<int32> cumulative_index;                  // 0-based indexing
      cumulative_index.resize(num_phones);
      cumulative_index[0] = 0;
      for (int32 i = 1; i < num_phones; i++)     
        cumulative_index[i] = cumulative_index[i-1] + phone2num_pdf_classes[i];

      for (size_t i = 0; i < stats.size(); i++) {
        EventAnswerType ans;
        EventType evec = stats[i].first;

        if (map.Map(evec, &ans)) {
          KALDI_ASSERT(ans < num_states);
          int32 phone;
          if (evec.size() == 2) {
            phone = evec[1].second - 1;
          } else {
            KALDI_ASSERT(evec.size() == 1+N);
            KALDI_ASSERT(evec[1+P].first == P);
            phone = evec[1+P].second - 1; // 1-based to 0-based
          }
          KALDI_ASSERT(phone >= 0 && phone < num_phones);
          KALDI_ASSERT(evec[0].first == kPdfClass);
          int32 pdf_class = evec[0].second;
          KALDI_ASSERT(pdf_class < phone2num_pdf_classes[phone+1]);   // phone2num_pdf_classes has 1-based index
          if (pdf_id_to_cluster[ans] == 0) {
            pdf_id_to_cluster[ans] = cumulative_index[phone] + pdf_class;
          }
        } else {
          KALDI_WARN << "Unable to get pdf-id for context " << EventTypeToString(evec);
        }
      }
    } else {
      if (roots_filename != "") { // Will work for non-contiguous phones as well
        std::vector< std::vector<int32> > phone_sets;
        std::vector<bool> is_shared_split;
        std::vector<bool> is_split_root;
        
        Input ki(roots_filename.c_str());
        ReadRootsFile(ki.Stream(), &phone_sets, &is_shared_split, &is_split_root);
        
        for (size_t i = 0; i < stats.size(); i++) {
          EventAnswerType ans;
          EventType evec = stats[i].first;

          if (map.Map(evec, &ans)) {
            KALDI_ASSERT(ans < num_states);
            int32 phone;
            if (evec.size() == 2) {
              phone = evec[1].second - 1;
            } else {
              KALDI_ASSERT(evec.size() == 1+N);
              KALDI_ASSERT(evec[1+P].first == P);
              phone = evec[1+P].second - 1; // 1-based to 0-based
            }

            if(pdf_id_to_cluster[ans] == -1) {
              std::vector< std::vector<int32> >::iterator set_it;
              std::vector<int32>::iterator phone_it;
              int32 cluster = 0;
              for (cluster = 0, set_it = phone_sets.begin(); 
                  set_it != phone_sets.end(); set_it++, cluster++) {
                for (phone_it = (*set_it).begin(); phone_it != (*set_it).end(); phone_it++) {
                  if ( (*phone_it)-1 != phone ) // using 0-based phones
                    continue;
                  else
                    pdf_id_to_cluster[ans] = cluster;
                }
              }
              KALDI_ASSERT(pdf_id_to_cluster[ans] != -1);
            }
          } else {
            KALDI_WARN << "Unable to get pdf-id for context " << EventTypeToString(evec);
          }
        }
      } else {
        for (size_t i = 0; i < stats.size(); i++) {
          EventAnswerType ans;
          EventType evec = stats[i].first;

          if (map.Map(evec, &ans)) {
            KALDI_ASSERT(ans < num_states);
            int32 phone;
            if (evec.size() == 2) {
              phone = evec[1].second - 1;
            } else {
              KALDI_ASSERT(evec.size() == 1+N);
              KALDI_ASSERT(evec[1+P].first == P);
              phone = evec[1+P].second - 1; // 1-based to 0-based
            }

            if(pdf_id_to_cluster[ans] == -1) {
              pdf_id_to_cluster[ans] = phone;
              if(phone>max_cluster) max_cluster = phone;
            }
          } else {
            KALDI_WARN << "Unable to get pdf-id for context " << EventTypeToString(evec);
          }
        }
      }
    }

    std::vector<int32> weights_map;
    weights_map.resize(num_states);
    
    if (tie_weights_to_cluster) {
      for (int32 j = 0; j < num_states; j++) {
        weights_map[j] = pdf_id_to_cluster[j];
      }
    } else
    {
      for (int32 j = 0; j < num_states; j++) {
        weights_map[j] = j;
      }
    }

    kaldi::AmPhoneTxCAT phoneTxCAT;
    if (init_from_phoneTxCAT) {
      // Initialize from another PhoneTxCAT model
      // Copy the global parameters from the 
      // previous phoneTxCAT model to this.
      // i.e. the Cannonical model, Covariances and Clusters Transforms
      kaldi::AmPhoneTxCAT init_phoneTxCAT;
      {
        bool binary_read;
        TransitionModel tmp_trans;
        kaldi::Input ki(init_model_filename, &binary_read);
        tmp_trans.Read(ki.Stream(), binary_read);
        init_phoneTxCAT.Read(ki.Stream(), binary_read);
      }
      phoneTxCAT.CopyGlobalsInitVecs(init_phoneTxCAT, 
          pdf_id_to_cluster, weights_map);
    } else if (copy_from_phoneTxCAT) {
      kaldi::AmPhoneTxCAT init_phoneTxCAT;
      {
        bool binary_read;
        TransitionModel tmp_trans;
        kaldi::Input ki(init_model_filename, &binary_read);
        tmp_trans.Read(ki.Stream(), binary_read);
        init_phoneTxCAT.Read(ki.Stream(), binary_read);
      }
      phoneTxCAT.CopyFromPhoneTxCAT(init_phoneTxCAT, false,
          recluster_gaussians_flag, reinitializeA_flag);
      
    } else {
      // Initialize from Full GMM and a decision tree
      // The UBM parameters are copied after diagonalization
      // One cluster is associated to each real phone in the decision tree
      kaldi::FullGmm ubm;
      {
        bool binary_read;
        kaldi::Input ki(init_model_filename, &binary_read);
        ubm.Read(ki.Stream(), binary_read);
      }

      phoneTxCAT.InitializeFromFullGmm(ubm, pdf_id_to_cluster, 
          weights_map, use_weight_projection, num_transform_classes, use_full_covar, num_cluster_weight_classes); 
    }
    phoneTxCAT.ComputeNormalizers();

    {
      kaldi::Output ko(phoneTxCAT_out_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      phoneTxCAT.Write(ko.Stream(), binary, kaldi::kPhoneTxCATWriteAll);
    }

    KALDI_LOG << "Written model to " << phoneTxCAT_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

