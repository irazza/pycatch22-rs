/*
 * Reference driver for the original catch22 C implementation.
 *
 * The upstream C/main.c is interactive (it scanf's the catch24 flag) and reads a
 * single series per file, so it cannot drive a sweep over the UCR archive. This
 * driver reads a UCR .tsv (column 0 = class label, remaining columns = series)
 * and emits, for every row, the 25 features *in pycatch22-rs index order* as raw
 * little-endian f64 so the comparison never round-trips through decimal text.
 *
 * Feature 0..21 are computed on the z-scored series (this is what upstream
 * main.c does via zscore_norm2); 22..24 (mean, std, slope) are computed on the
 * raw series, matching catch24 semantics.
 *
 * Usage:
 *   driver <input.tsv> <output.bin>            write reference feature values
 *   driver <input.tsv> --bench <timing.csv>    time each feature over the file
 *
 * In bench mode every row is loaded and z-scored up front, then each feature is
 * timed across the whole file in one go. Timing individual calls instead would
 * put a ~25ns clock_gettime pair around features that take less than that.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "DN_HistogramMode_5.h"
#include "DN_HistogramMode_10.h"
#include "DN_OutlierInclude.h"
#include "DN_Mean.h"
#include "DN_Spread_Std.h"
#include "CO_AutoCorr.h"
#include "FC_LocalSimple.h"
#include "IN_AutoMutualInfoStats.h"
#include "MD_hrv.h"
#include "SB_BinaryStats.h"
#include "SB_MotifThree.h"
#include "SB_TransitionMatrix.h"
#include "SC_FluctAnal.h"
#include "SP_Summaries.h"
#include "PD_PeriodicityWang.h"
#include "stats.h"

#define N_FEATURES 25

static const char *FEATURE_NAMES[N_FEATURES] = {
    "DN_OutlierInclude_n_001_mdrmd",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "FC_LocalSimple_mean1_tauresrat",
    "FC_LocalSimple_mean3_stderr",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_diff_longstretch0",
    "SB_BinaryStats_mean_longstretch1",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_area_5_1",
    "SP_Summaries_welch_rect_centroid",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "DN_Mean",
    "DN_Spread_Std",
    "SlopeOfLinearFit",
};

/* Mirrors catch22::statistics::slope on the Rust side: least-squares slope
 * against x = 1..n, using the closed-form sums for x and x^2. */
static double slope_of_linear_fit(const double a[], const int n)
{
    if (n == 0) {
        return 0.0;
    }

    double n_f = (double)n;
    double x_mean = (n_f + 1.0) / 2.0;
    double x2_mean = (n_f + 1.0) * (2.0 * n_f + 1.0) / 6.0;

    double y_sum = 0.0;
    double xy_sum = 0.0;
    for (int i = 0; i < n; i++) {
        double x = (double)(i + 1);
        y_sum += a[i];
        xy_sum += x * a[i];
    }

    double y_mean = y_sum / n_f;
    double xy_mean = xy_sum / n_f;
    return (xy_mean - x_mean * y_mean) / (x2_mean - x_mean * x_mean);
}

static double elapsed_ns(struct timespec start, struct timespec end)
{
    return (double)(end.tv_sec - start.tv_sec) * 1e9 +
           (double)(end.tv_nsec - start.tv_nsec);
}

/* Evaluate one feature by index. `z` is the z-scored series, `raw` the original. */
static double eval_feature(int f, const double *z, const double *raw, int n)
{
    switch (f) {
        case 0:  return DN_OutlierInclude_n_001_mdrmd(z, n);
        case 1:  return DN_OutlierInclude_p_001_mdrmd(z, n);
        case 2:  return DN_HistogramMode_5(z, n);
        case 3:  return DN_HistogramMode_10(z, n);
        case 4:  return CO_Embed2_Dist_tau_d_expfit_meandiff(z, n);
        case 5:  return CO_f1ecac(z, n);
        case 6:  return (double)CO_FirstMin_ac(z, n);
        case 7:  return CO_HistogramAMI_even_2_5(z, n);
        case 8:  return CO_trev_1_num(z, n);
        case 9:  return FC_LocalSimple_mean1_tauresrat(z, n);
        case 10: return FC_LocalSimple_mean3_stderr(z, n);
        case 11: return IN_AutoMutualInfoStats_40_gaussian_fmmi(z, n);
        case 12: return MD_hrv_classic_pnn40(z, n);
        case 13: return SB_BinaryStats_diff_longstretch0(z, n);
        case 14: return SB_BinaryStats_mean_longstretch1(z, n);
        case 15: return SB_MotifThree_quantile_hh(z, n);
        case 16: return SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(z, n);
        case 17: return SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(z, n);
        case 18: return SP_Summaries_welch_rect_area_5_1(z, n);
        case 19: return SP_Summaries_welch_rect_centroid(z, n);
        case 20: return SB_TransitionMatrix_3ac_sumdiagcov(z, n);
        case 21: return (double)PD_PeriodicityWang_th0_01(z, n);
        case 22: return DN_Mean(raw, n);
        case 23: return DN_Spread_Std(raw, n);
        case 24: return slope_of_linear_fit(raw, n);
        default: return NAN;
    }
}

/* One loaded series: the raw samples and their z-scored counterpart. */
typedef struct {
    double *raw;
    double *z;
    int n;
} Series;

static int load_series(const char *path, Series **out_series, long *out_rows)
{
    FILE *in = fopen(path, "r");
    if (in == NULL) {
        fprintf(stderr, "cannot open input %s\n", path);
        return 1;
    }

    long rows = 0;
    long rows_cap = 256;
    Series *series = malloc(rows_cap * sizeof *series);

    char *line = NULL;
    size_t line_cap = 0;
    ssize_t line_len;

    while ((line_len = getline(&line, &line_cap, in)) != -1) {
        if (line_len == 0 || line[0] == '\n') {
            continue;
        }

        int cap = 256;
        int n = 0;
        double *raw = malloc(cap * sizeof *raw);

        char *cursor = line;
        char *end;
        int column = 0;
        for (;;) {
            double value = strtod(cursor, &end);
            if (end == cursor) {
                break;
            }
            cursor = end;
            /* column 0 is the class label */
            if (column > 0) {
                if (n == cap) {
                    cap *= 2;
                    raw = realloc(raw, cap * sizeof *raw);
                }
                raw[n++] = value;
            }
            column++;
        }

        if (n < 4) {
            fprintf(stderr, "row %ld: only %d samples, skipping\n", rows, n);
            free(raw);
            continue;
        }

        if (rows == rows_cap) {
            rows_cap *= 2;
            series = realloc(series, rows_cap * sizeof *series);
        }
        series[rows].raw = raw;
        series[rows].z = malloc(n * sizeof(double));
        series[rows].n = n;
        zscore_norm2(raw, n, series[rows].z);
        rows++;
    }

    free(line);
    fclose(in);

    *out_series = series;
    *out_rows = rows;
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <input.tsv> <output.bin>\n"
                "       %s <input.tsv> --bench <timing.csv>\n",
                argv[0], argv[0]);
        return 2;
    }

    int bench = strcmp(argv[2], "--bench") == 0;
    if (bench && argc < 4) {
        fprintf(stderr, "--bench needs an output path\n");
        return 2;
    }

    Series *series;
    long rows;
    if (load_series(argv[1], &series, &rows) != 0) {
        return 1;
    }

    if (bench) {
        double feature_ns[N_FEATURES] = {0};

        for (int f = 0; f < N_FEATURES; f++) {
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (long r = 0; r < rows; r++) {
                volatile double sink =
                    eval_feature(f, series[r].z, series[r].raw, series[r].n);
                (void)sink;
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            feature_ns[f] = elapsed_ns(t0, t1);
        }

        /* Whole-pipeline cost: z-score plus all 25 features, one timer. */
        struct timespec p0, p1;
        clock_gettime(CLOCK_MONOTONIC, &p0);
        for (long r = 0; r < rows; r++) {
            int n = series[r].n;
            double *z = malloc(n * sizeof(double));
            zscore_norm2(series[r].raw, n, z);
            for (int f = 0; f < N_FEATURES; f++) {
                volatile double sink = eval_feature(f, z, series[r].raw, n);
                (void)sink;
            }
            free(z);
        }
        clock_gettime(CLOCK_MONOTONIC, &p1);

        FILE *tf = fopen(argv[3], "w");
        if (tf == NULL) {
            fprintf(stderr, "cannot open timing output %s\n", argv[3]);
            return 1;
        }
        fprintf(tf, "index,name,total_ns,rows\n");
        for (int f = 0; f < N_FEATURES; f++) {
            fprintf(tf, "%d,%s,%.0f,%ld\n", f, FEATURE_NAMES[f], feature_ns[f], rows);
        }
        fprintf(tf, "-1,PIPELINE,%.0f,%ld\n", elapsed_ns(p0, p1), rows);
        fclose(tf);
    } else {
        FILE *out = fopen(argv[2], "wb");
        if (out == NULL) {
            fprintf(stderr, "cannot open output %s\n", argv[2]);
            return 1;
        }
        for (long r = 0; r < rows; r++) {
            double values[N_FEATURES];
            for (int f = 0; f < N_FEATURES; f++) {
                values[f] = eval_feature(f, series[r].z, series[r].raw, series[r].n);
            }
            if (fwrite(values, sizeof(double), N_FEATURES, out) != N_FEATURES) {
                fprintf(stderr, "short write on row %ld\n", r);
                return 1;
            }
        }
        fclose(out);
    }

    for (long r = 0; r < rows; r++) {
        free(series[r].raw);
        free(series[r].z);
    }
    free(series);

    fprintf(stderr, "%s: %ld rows\n", argv[1], rows);
    return 0;
}
