# Rust vs C parity over the UCR archive

- root: `/home/irazza/DATA/ucr`
- files: 256
- series compared: 191158
- rows skipped (<4 samples): 0
- degenerate (constant) rows, audited separately: 0 (unexpected C output on 0)
- mode: compute_all
- tolerance: exact for [6, 11, 13, 14, 16, 17, 21], otherwise |rust-c| <= 1e-12 + 1e-9*|c|

**21 / 25 features match.**

| # | feature | mismatches | % | worst rel err | example |
|---|---------|-----------|---|---------------|---------|
| 0 | DN_OutlierInclude_n_001_mdrmd | 0 | 0.00% | 0.000e0 | - |
| 1 | DN_OutlierInclude_p_001_mdrmd | 0 | 0.00% | 0.000e0 | - |
| 2 | DN_HistogramMode_5 | 0 | 0.00% | 0.000e0 | - |
| 3 | DN_HistogramMode_10 | 0 | 0.00% | 0.000e0 | - |
| 4 | CO_Embed2_Dist_tau_d_expfit_meandiff | 1 | 0.00% | 2.059e-1 | ElectricDevices_TRAIN.tsv:row4859 rust=5.26122664974227675e-2 c=4.36290636146738284e-2 |
| 5 | CO_f1ecac | 0 | 0.00% | 0.000e0 | - |
| 6 | CO_FirstMin_ac | 1 | 0.00% | 3.333e-1 | ElectricDevices_TRAIN.tsv:row4859 rust=2.00000000000000000e0 c=3.00000000000000000e0 |
| 7 | CO_HistogramAMI_even_2_5 | 0 | 0.00% | 0.000e0 | - |
| 8 | CO_trev_1_num | 0 | 0.00% | 0.000e0 | - |
| 9 | FC_LocalSimple_mean1_tauresrat | 36 | 0.02% | 2.500e0 | ScreenType_TRAIN.tsv:row179 rust=2.59259259259259245e-1 c=7.40740740740740700e-2 |
| 10 | FC_LocalSimple_mean3_stderr | 0 | 0.00% | 0.000e0 | - |
| 11 | IN_AutoMutualInfoStats_40_gaussian_fmmi | 0 | 0.00% | 0.000e0 | - |
| 12 | MD_hrv_classic_pnn40 | 0 | 0.00% | 0.000e0 | - |
| 13 | SB_BinaryStats_diff_longstretch0 | 0 | 0.00% | 0.000e0 | - |
| 14 | SB_BinaryStats_mean_longstretch1 | 0 | 0.00% | 0.000e0 | - |
| 15 | SB_MotifThree_quantile_hh | 0 | 0.00% | 0.000e0 | - |
| 16 | SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1 | 0 | 0.00% | 0.000e0 | - |
| 17 | SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1 | 0 | 0.00% | 0.000e0 | - |
| 18 | SP_Summaries_welch_rect_area_5_1 | 0 | 0.00% | 0.000e0 | - |
| 19 | SP_Summaries_welch_rect_centroid | 0 | 0.00% | 0.000e0 | - |
| 20 | SB_TransitionMatrix_3ac_sumdiagcov | 1 | 0.00% | 6.142e-2 | ElectricDevices_TRAIN.tsv:row4859 rust=1.81459566074950729e-1 c=1.93333333333333302e-1 |
| 21 | PD_PeriodicityWang_th0_01 | 0 | 0.00% | 0.000e0 | - |
| 22 | DN_Mean | 0 | 0.00% | 0.000e0 | - |
| 23 | DN_Spread_Std | 0 | 0.00% | 0.000e0 | - |
| 24 | SlopeOfLinearFit | 0 | 0.00% | 0.000e0 | - |
