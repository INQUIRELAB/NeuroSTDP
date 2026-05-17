#!/usr/bin/env python3
from dataclasses import dataclass
E_WITHIN_TILE_SPIKE_PJ = 1.7
E_SYNOP_MIN_PJ = 23.6
E_SYNAPTIC_UPDATE_PJ = 120.0
E_NEURON_ACTIVE_PJ = 81.0
E_NEURON_INACTIVE_PJ = 52.0
P_CHIP_IDLE_MW = 29.0
DT_PER_STEP_S = 0.001
N_INP = 17
N_HID = 32
N_OUT = 12
N_LIF = N_HID + N_OUT
W_IH_PROVISIONED = N_INP * N_HID
W_IH_EFFECTIVE = int(0.5 * W_IH_PROVISIONED)
W_HO_PLASTIC = N_HID * N_OUT
FANOUT_IH_PER_INPUT_SPIKE = W_IH_PROVISIONED // N_INP
FANIN_IH_PER_HIDDEN_SPIKE_AVG = W_IH_EFFECTIVE // N_INP
SYNOPS_PER_INPUT_SPIKE = FANOUT_IH_PER_INPUT_SPIKE
SYNOPS_PER_HIDDEN_SPIKE = N_OUT
SYNOPS_LATERAL_PER_HIDDEN_SPIKE = N_HID - 1
T = 350
INPUT_RATE_HZ = 60.0
INPUT_SPIKES_PER_NEURON_PER_CLASS = INPUT_RATE_HZ * T * DT_PER_STEP_S
TOTAL_INPUT_SPIKES = int(N_INP * INPUT_SPIKES_PER_NEURON_PER_CLASS)
HIDDEN_SPIKES_PER_STEP = 1.0
TOTAL_HIDDEN_SPIKES = int(HIDDEN_SPIKES_PER_STEP * T)
OUTPUT_SPIKES_PER_STEP = 2.0
TOTAL_OUTPUT_SPIKES = int(OUTPUT_SPIKES_PER_STEP * T)
STDP_UPDATES = TOTAL_OUTPUT_SPIKES * N_HID

@dataclass

class Component:
    name: str
    per_unit_pj: float
    count: float
    source: str
    @property
    def total_pj(self) -> float:
        return self.per_unit_pj * self.count
    @property
    def total_uj(self) -> float:
        return self.total_pj * 1e-06
components_single_draw = [Component(name='Input->hidden synaptic ops (Poisson fan-out)', per_unit_pj=E_SYNOP_MIN_PJ, count=TOTAL_INPUT_SPIKES * SYNOPS_PER_INPUT_SPIKE, source='Davies 2018 Table 2 (min syn-op). Poisson emitter fan-out R2-item-1.'), Component(name='Hidden->output synaptic ops (plastic pathway)', per_unit_pj=E_SYNOP_MIN_PJ, count=TOTAL_HIDDEN_SPIKES * SYNOPS_PER_HIDDEN_SPIKE, source='Davies 2018 Table 2.'), Component(name='Lateral-inhibition WTA synaptic ops', per_unit_pj=E_SYNOP_MIN_PJ, count=TOTAL_HIDDEN_SPIKES * SYNOPS_LATERAL_PER_HIDDEN_SPIKE, source='Davies 2018 Table 2. Covers R2-item-2 (recurrent WTA crosstalk).'), Component(name='STDP synaptic updates', per_unit_pj=E_SYNAPTIC_UPDATE_PJ, count=STDP_UPDATES, source='Davies 2018 Table 2 (pairwise STDP).'), Component(name='LIF neuron updates (active)', per_unit_pj=E_NEURON_ACTIVE_PJ, count=T * (HIDDEN_SPIKES_PER_STEP + OUTPUT_SPIKES_PER_STEP), source='Davies 2018 Table 2.'), Component(name='LIF neuron updates (inactive)', per_unit_pj=E_NEURON_INACTIVE_PJ, count=T * (N_LIF - HIDDEN_SPIKES_PER_STEP - OUTPUT_SPIKES_PER_STEP), source='Davies 2018 Table 2.'), Component(name='Within-tile spike routing', per_unit_pj=E_WITHIN_TILE_SPIKE_PJ, count=TOTAL_INPUT_SPIKES + TOTAL_HIDDEN_SPIKES + TOTAL_OUTPUT_SPIKES, source='Davies 2018 Table 2 (1.7 pJ within-tile spike).'), Component(name='Static/idle power over T*dt (per-core, 29/128 mW)', per_unit_pj=P_CHIP_IDLE_MW / 128.0 * 1000000000.0, count=T * DT_PER_STEP_S, source='Blouw 2019 Table 1 Loihi chip-idle 29 mW scaled to 1/128 for single-core deployment. R2-item-3.')]

def _print_block(components, label):
    total_pj = sum((c.total_pj for c in components))
    print(f'\n===== {label} =====')
    print(f'{'Component':55s}  {'per-unit':>12s}  {'count':>12s}  {'total (uJ)':>12s}  {'share':>7s}')
    for c in components:
        share = 100.0 * c.total_pj / total_pj
        print(f'{c.name:55s}  {c.per_unit_pj:10.2f} pJ  {c.count:12.0f}  {c.total_uj:12.6f}  {share:6.1f}%')
    total_uj = total_pj * 1e-06
    print(f'{'TOTAL':55s}  {'':12s}  {'':12s}  {total_uj:12.6f}  100.0%')
    return (total_pj, total_uj)
print('=' * 90)
print('T1.5 refined energy proxy for the 17->32->12 cohort gate')
print('All per-op energies: Davies et al. 2018 IEEE Micro Table 2, 0.75 V')
print('All idle power: Blouw et al. 2019 Table 1 (Loihi measured).')
print('=' * 90)
print(f'Network: {N_INP} Poisson inputs -> {N_HID} LIF hidden (WTA) -> {N_OUT} LIF output')
print(f'Simulation: T={T} steps at {DT_PER_STEP_S * 1000.0:.1f} ms/step -> {T * DT_PER_STEP_S * 1000:.1f} ms wall')
print(f'Input spikes/classification (60 Hz Poisson): ~{TOTAL_INPUT_SPIKES}')
print(f'Hidden spikes (WTA, 1/step): {TOTAL_HIDDEN_SPIKES}')
print(f'Output spikes (~2/step post-convergence): {TOTAL_OUTPUT_SPIKES}')
print(f'STDP updates (on output spikes, 32 synapses each): {STDP_UPDATES}')
single_total_pj, single_total_uj = _print_block(components_single_draw, 'SINGLE POISSON DRAW, T=350 steps')
components_plurality5 = [Component(name=c.name, per_unit_pj=c.per_unit_pj, count=c.count * 5 if 'static' not in c.name.lower() else c.count * 5, source=c.source) for c in components_single_draw]
plurality_total_pj, plurality_total_uj = _print_block(components_plurality5, 'PLURALITY-5 (5 independent Poisson draws)')
single_chip_upper_uj = single_total_uj + T * DT_PER_STEP_S * P_CHIP_IDLE_MW * 0.001 * 1000000.0 - T * DT_PER_STEP_S * (P_CHIP_IDLE_MW / 128.0) * 0.001 * 1000000.0
plur5_chip_upper_uj = plurality_total_uj + 5 * T * DT_PER_STEP_S * P_CHIP_IDLE_MW * 0.001 * 1000000.0 - 5 * T * DT_PER_STEP_S * (P_CHIP_IDLE_MW / 128.0) * 0.001 * 1000000.0
prior_proxy_uj = 2.3
prior_proxy_plur5_uj = 12.0
factor_single = single_total_uj / prior_proxy_uj
factor_plur5 = plurality_total_uj / prior_proxy_plur5_uj
factor_single_chip = single_chip_upper_uj / prior_proxy_uj
factor_plur5_chip = plur5_chip_upper_uj / prior_proxy_plur5_uj
print('\n===== Delta vs prior proxy =====')
print(f'Prior single-draw proxy: {prior_proxy_uj:.2f} uJ')
print(f'Refined single-draw (per-core): {single_total_uj:.2f} uJ (factor x{factor_single:.1f})')
print(f'Refined single-draw (whole-chip UB): {single_chip_upper_uj:.2f} uJ (factor x{factor_single_chip:.1f})')
print(f'Prior plurality-5 proxy: {prior_proxy_plur5_uj:.2f} uJ')
print(f'Refined plurality-5 (per-core): {plurality_total_uj:.2f} uJ (factor x{factor_plur5:.1f})')
print(f'Refined plurality-5 (whole-chip UB): {plur5_chip_upper_uj:.2f} uJ (factor x{factor_plur5_chip:.1f})')
cpu_w = 20.0
cpu_s_single = 0.02
cpu_s_plur5 = 0.1
cpu_uj_single = cpu_w * cpu_s_single * 1000000.0
cpu_uj_plur5 = cpu_w * cpu_s_plur5 * 1000000.0
print('\n===== vs CPU reference (20 W single core at wall-clock time) =====')
print(f'CPU single-draw: {cpu_uj_single:.0f} uJ')
print(f'CPU plurality-5: {cpu_uj_plur5:.0f} uJ')
print(f'Single-draw ratio Loihi(per-core):CPU = 1 : {cpu_uj_single / single_total_uj:,.0f} (log10: {int(cpu_uj_single / single_total_uj):,}, ~{len(str(int(cpu_uj_single / single_total_uj))) - 1} orders)')
print(f'Single-draw ratio Loihi(chip-UB):CPU = 1 : {cpu_uj_single / single_chip_upper_uj:.1f} (log10: ~{len(str(int(cpu_uj_single / single_chip_upper_uj))) - 1} orders)')
print(f'Plurality-5 ratio Loihi(per-core):CPU = 1 : {cpu_uj_plur5 / plurality_total_uj:,.0f} (log10: ~{len(str(int(cpu_uj_plur5 / plurality_total_uj))) - 1} orders)')
print(f'Plurality-5 ratio Loihi(chip-UB):CPU = 1 : {cpu_uj_plur5 / plur5_chip_upper_uj:.1f} (log10: ~{len(str(int(cpu_uj_plur5 / plur5_chip_upper_uj))) - 1} orders)')
import json

def to_row(c):
    return {'component': c.name, 'per_unit_pJ': c.per_unit_pj, 'count': int(c.count) if c.count == int(c.count) else c.count, 'total_pJ': c.total_pj, 'total_uJ': c.total_uj, 'source': c.source}
payload = {'network': {'name': 'cohort gate (17 -> 32 -> 12)', 'N_inp': N_INP, 'N_hid': N_HID, 'N_out': N_OUT, 'N_LIF': N_LIF, 'W_ih_provisioned': W_IH_PROVISIONED, 'W_ih_effective_nonzero': W_IH_EFFECTIVE, 'W_ho_plastic': W_HO_PLASTIC, 'timesteps_T': T, 'dt_per_step_s': DT_PER_STEP_S, 'input_rate_hz_assumed': INPUT_RATE_HZ, 'total_input_spikes_per_class': TOTAL_INPUT_SPIKES, 'total_hidden_spikes_per_class': TOTAL_HIDDEN_SPIKES, 'total_output_spikes_per_class': TOTAL_OUTPUT_SPIKES, 'stdp_updates_per_class': STDP_UPDATES}, 'per_op_energies_pJ': {'within_tile_spike': E_WITHIN_TILE_SPIKE_PJ, 'synaptic_op_min': E_SYNOP_MIN_PJ, 'synaptic_update_stdp': E_SYNAPTIC_UPDATE_PJ, 'neuron_update_active': E_NEURON_ACTIVE_PJ, 'neuron_update_inactive': E_NEURON_INACTIVE_PJ, 'chip_idle_mW': P_CHIP_IDLE_MW}, 'sources': {'davies_2018_micro': "Davies et al. 2018, 'Loihi: A Neuromorphic Manycore Processor with On-Chip Learning', IEEE Micro 38(1):82-99, Table 2 at 0.75 V. PDF: https://redwood.berkeley.edu/wp-content/uploads/2021/08/Davies2018.pdf", 'blouw_2019_nice': "Blouw, Choo, Hunsberger, Eliasmith 2019, 'Benchmarking Keyword Spotting Efficiency on Neuromorphic Hardware', Table 1. arXiv:1812.01739. Loihi (Wolf Mountain board) idle = 29 mW, running = 110 mW, dynamic = 81 mW.", 'loihi2_brief_2021': "Intel Labs, 'Taking Neuromorphic Computing with Loihi 2 to the Next Level', 2021 technology brief. Reports up to 10x speedup vs Loihi 1 but DOES NOT publish updated per-op energy numbers (silicon characterized but not released). PDF: https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf", 'davies_2021_procieee': "Davies et al. 2021, 'Advancing Neuromorphic Computing With Loihi: A Survey of Results and Outlook', Proceedings of the IEEE 109(5), doi:10.1109/JPROC.2021.3067593. Cites 100x dynamic-power reduction on SLAM vs CPU.", 'orchard_2021_loihi2': "Orchard, Frady, Rubin, Sanborn, Shrestha, Sommer, Davies 2021, 'Efficient Neuromorphic Signal Processing with Loihi 2', arXiv:2111.03746. Does not publish per-op energy."}, 'single_draw_per_core': {'components': [to_row(c) for c in components_single_draw], 'total_pJ': single_total_pj, 'total_uJ': single_total_uj}, 'plurality5_per_core': {'components': [to_row(c) for c in components_plurality5], 'total_pJ': plurality_total_pj, 'total_uJ': plurality_total_uj}, 'single_draw_chip_upper_bound_uJ': single_chip_upper_uj, 'plurality5_chip_upper_bound_uJ': plur5_chip_upper_uj, 'prior_proxy_for_comparison_uJ': {'single_draw': prior_proxy_uj, 'plurality5': prior_proxy_plur5_uj}, 'factor_refined_vs_prior_per_core': {'single_draw': factor_single, 'plurality5': factor_plur5}, 'factor_refined_vs_prior_chip_UB': {'single_draw': factor_single_chip, 'plurality5': factor_plur5_chip}, 'cpu_reference_uJ': {'single_draw_20W_20ms': cpu_uj_single, 'plurality5_20W_100ms': cpu_uj_plur5}, 'caveats': ['Intel has not published updated Loihi 2 per-op energy numbers; the technology brief documents only up-to-10x *speed* improvements. We use Davies 2018 Loihi 1 per-op values as the honest best-published proxy and label the refined number a proxy, not a silicon measurement.', 'Input Poisson-rate emitters are stateless and have no published per-op energy. We upper-bound the emitter cost by attributing the full input-side synaptic-op cost (32 synapses per input spike at 23.6 pJ/op) to the emitter stage, which is conservative.', 'The 29 mW chip-idle figure is whole-chip: our 44 LIF network fits in a single core, so this is an upper bound (actual per-core leakage for one core on a 128-core die would be ~29/128 = 0.23 mW, but Intel has not published a per-core leakage breakdown -- we retain the chip-level bound for honesty).', 'STDP is applied only to 384 hidden->output synapses; input->hidden weights are non-plastic so do not incur STDP update cost.', 'The classification fits within one Loihi core, so no inter-tile hops are accrued (tile-hop energy 3-4 pJ would apply if the network spanned tiles, which it does not for 44 neurons).', 'Output spikes fan into the off-chip readout (spike counting for plurality vote), which we do not attribute energy to because the readout is host-side.']}
with open('/tmp/energy_proxy_refined.json', 'w') as f:
    json.dump(payload, f, indent=2)
print('\nWrote /tmp/energy_proxy_refined.json')
