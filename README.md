# 🧠 NeuroSTDP — Brain-Inspired Spike-Timing Plasticity for Reliable Label-Efficient Event-Camera Vision

> Reproducibility code for our submission to *Nature Communications Engineering*.

## 👥 Authors

**Mohamad Yazan Sadoun**, **Sarah Sharif**, **Yaser Mike Banad** *(corresponding author)*

School of Electrical and Computer Engineering, University of Oklahoma, Norman, OK, USA.
Contact: `bana@ou.edu`

## 📄 Abstract

Deploying event-camera object detectors on drones is gated by two costs: per-frame bounding-box label budgets at training and GPU compute at inference. We close both with brain-inspired spike-timing-dependent plasticity (STDP) at three local modules (sequence routing, candidate reliability, tube reliability), delivering **78.60 ± 0.42% mAP@30** on the FRED event-camera drone benchmark on a single CPU thread, with no GPU.

Under acquisition-order distribution shift on FRED, a sequence-level spiking cohort gate adapts by **+2.03 ± 0.58 pp** over streaming *k*-means (20/20 seeds positive); a matched no-drift control returns the advantage to noise (−0.44 ± 1.97 pp, 6/20 positive), isolating drift-tracking as the mechanism. STDP plasticity tightens single-model seed variance by **6.6×** — a single trained gate meets the analytic sample-mean variance bound of a **44-seed** random-init ensemble.

The cohort gate ports to the **Intel Lava neuromorphic simulator** at 89.4% (42 of 47) top-2 cohort preservation, stable under simulated 8-bit Loihi-2 fixed-point quantisation. On EV-UAV, a tube-level STDP reliability layer reduces the false-alarm rate from 454 to **331 × 10⁻⁴** at P_d ≥ 88%.

## 📊 Headline Results

**FRED canonical test (47 sequences):**

| Track | mAP@30 (%) | mAP@50 (%) |
|---|---:|---:|
| Strict zero-label five-channel | 53.81 | 31.43 |
| Six-channel support confidence (≈26 train-derived bits) | 76.87 | 51.67 |
| R-STDP candidate-reliability gate (≥3,072 stored bits, N=10) | **78.60 ± 0.42** | **52.57 ± 0.43** |

**EV-UAV test (24 sequences):**

| Method | mAP@30 (%) | F_a (×10⁻⁴) | P_d (%) |
|---|---:|---:|---:|
| Six-channel baseline | 50.97 | 662.34 | 91.17 |
| Fixed event-tube K=5 | 56.26 | 404.20 | 89.63 |
| STDP-Tube K=5 (ours) | **57.03** | **340.21** | 88.44 |

## 📁 Repository Layout

```
code/
├── neuro/         spiking cohort gate, R-STDP candidate gate, STDP-Tube,
│                  Lava port, drift protocol, TOST tests, energy proxy
├── discovery/     six classical event-detection channels
└── scripts/       evaluation entry point

manuscript.pdf     current build (30 pp)
requirements.txt   Python dependencies
LICENSE            MIT
```

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Python 3.10+. For the Lava port, install `lava-nc==0.10.0` separately following its own instructions.

## 🧪 Reproducibility

Source datasets (FRED, EV-UAV) are publicly available from their original publications. Rebuilt multi-drone FRED ground truth and full result manifests are available in the supplementary package accompanying the manuscript. Every headline number in the paper is traced to a specific script in `code/`.

## 📚 Citation

```bibtex
@article{sadoun2026neurostdp,
  title   = {Brain-inspired spike-timing plasticity for reliable label-efficient event-camera vision},
  author  = {Sadoun, Mohamad Yazan and Sharif, Sarah and Banad, Yaser Mike},
  journal = {Nature Communications Engineering},
  year    = {2026},
  note    = {under review}
}
```

## 📜 License

This project is released under the **MIT License**. See `LICENSE` for details. Source datasets remain under their original licence terms.
