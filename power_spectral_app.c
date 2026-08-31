#include <algorithm>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// 0.5-45 Hz zero-phase Butterworth bandpass, matching the offline pipeline's
// lilia/io.py::bandpass_filter (scipy.signal.butter(4, [0.5, 45], btype=
// 'bandpass', fs=500, output='sos') + sosfiltfilt). Coefficients below are
// precomputed for that fixed (order=4, 0.5-45 Hz, fs=500 Hz) configuration;
// regenerate them from scipy if any of those three parameters change.
// ---------------------------------------------------------------------------
struct SOSSection { double b0, b1, b2, a1, a2; };

static const SOSSection BANDPASS_SOS[4] = {
    {0.0032357120981354338, 0.0064714241962708675, 0.0032357120981354338, -1.1437629243598593, 0.346223946903943},
    {1.0,                   2.0,                   1.0,                   -1.4077978195668215, 0.6656297889988326},
    {1.0,                  -2.0,                   1.0,                   -1.9882392413999683, 0.9882799826069772},
    {1.0,                  -2.0,                   1.0,                   -1.9952369463218593, 0.9952765732294849},
};

// Unit initial conditions per section (scipy.signal.sosfilt_zi(sos)), scaled
// by the boundary sample before each filtering pass -- this is what makes the
// filter start "as if it had already reached steady state" and is what gives
// sosfiltfilt its (near-)zero transient behaviour.
static const double BANDPASS_ZI[4][2] = {
    {0.060691893472085215, -0.01889755581850485},
    {0.9278439844470985,   -0.5962251086280443},
    {-0.9917715900200642,  0.991771590020032},
    {-0.0,                 0.0},
};

// 3 * (2 * n_sections + 1), matching scipy.signal.sosfiltfilt's default padlen
// for this 4-section SOS cascade.
static const int BANDPASS_PADLEN = 27;

// Direct-Form-II-Transposed cascade of the 4 SOS sections (same structure
// scipy.signal.sosfilt uses internally). `zi_scale` multiplies the unit
// initial conditions above, matching sosfiltfilt's `zi * x_boundary` step.
static std::vector<double> sosfilt_cascade(const std::vector<double> &x, double zi_scale) {
  std::vector<double> cur = x;
  for (int s = 0; s < 4; s++) {
    const double b0 = BANDPASS_SOS[s].b0, b1 = BANDPASS_SOS[s].b1, b2 = BANDPASS_SOS[s].b2;
    const double a1 = BANDPASS_SOS[s].a1, a2 = BANDPASS_SOS[s].a2;
    double z1 = BANDPASS_ZI[s][0] * zi_scale;
    double z2 = BANDPASS_ZI[s][1] * zi_scale;

    std::vector<double> out(cur.size());
    for (size_t i = 0; i < cur.size(); i++) {
      double xn = cur[i];
      double yn = b0 * xn + z1;
      double z1n = b1 * xn - a1 * yn + z2;
      double z2n = b2 * xn - a2 * yn;
      z1 = z1n;
      z2 = z2n;
      out[i] = yn;
    }
    cur = std::move(out);
  }
  return cur;
}

// Odd (180-degree rotation) boundary extension, matching
// scipy.signal._arraytools.odd_ext -- required so the forward/backward
// passes below start from a signal-consistent edge instead of a hard zero.
static std::vector<double> odd_ext(const std::vector<double> &x, int n_pad) {
  const int N = (int)x.size();
  std::vector<double> ext(N + 2 * n_pad);
  const double x0 = x.front(), x_last = x.back();
  for (int k = 0; k < n_pad; k++) ext[k] = 2.0 * x0 - x[n_pad - k];
  for (int i = 0; i < N; i++) ext[n_pad + i] = x[i];
  for (int j = 0; j < n_pad; j++) ext[n_pad + N + j] = 2.0 * x_last - x[N - 2 - j];
  return ext;
}

// Zero-phase 0.5-45 Hz bandpass filter, equivalent to
// scipy.signal.sosfiltfilt(BANDPASS_SOS, raw). Intended to run once over the
// whole recording (not per Goertzel window), same as the Python pipeline:
// filter the full channel first, then slice out 5 s windows for
// goertzel_power(). Returns float to match lilia/io.py's bandpass_filter,
// which casts its sosfiltfilt output down to float32.
std::vector<float> bandpass_filter_0p5_45hz(const std::vector<float> &raw) {
  std::vector<double> x(raw.begin(), raw.end());

  auto ext = odd_ext(x, BANDPASS_PADLEN);
  const double x0 = ext.front();
  auto y = sosfilt_cascade(ext, x0);

  const double y0 = y.back();
  std::reverse(y.begin(), y.end());
  auto y2 = sosfilt_cascade(y, y0);
  std::reverse(y2.begin(), y2.end());

  std::vector<float> out(x.size());
  for (size_t i = 0; i < out.size(); i++) {
    out[i] = (float)y2[BANDPASS_PADLEN + i];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Single-frequency power via the Goertzel algorithm (unchanged).
// ---------------------------------------------------------------------------
double goertzel_power(const std::vector<float> &data, float target_freq,
                      float sample_rate) {
  if (data.empty())
    return 0.0;

  // 1. Calculate Mean (DC Offset)
  double sum = 0.0;
  for (float val : data) {
    sum += val;
  }
  double mean = sum / data.size();

  double w = 2.0 * M_PI * (target_freq / sample_rate);

  double cosine = cos(w);
  double coeff = 2.0 * cosine;
  double s_prev = 0.0;
  double s_prev2 = 0.0;

  size_t N = data.size();
  size_t n = 0;

  for (float x : data) {
    // 2. Remove DC
    double sample = (double)x - mean;

    // 3. Apply Hanning Window to reduce spectral leakage
    // w[n] = 0.5 * (1 - cos(2*PI*n / (N-1)))
    double window = 0.5 * (1.0 - cos(2.0 * M_PI * n / (N - 1)));
    sample *= window;

    double s = sample + (coeff * s_prev) - s_prev2;
    // double s = x + (coeff * s_prev) - s_prev2;
    s_prev2 = s_prev;
    s_prev = s;
    n++;
  }

  return s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2;
}
