"""
PERC2: Power Electronic Reconnecting and Ceasing load model, version 2.

Positive-sequence model for aggregations of power-electronic loads (data
centers, EV chargers, crypto-mining, variable-frequency drives), reproducing
PowerWorld Simulator 24's ``Load Characteristic PERC2`` (July 28, 2026 patch,
with the six integer flags added in the September 25, 2026 patch) as
documented in the PowerWorld help page of October 2026.

PERC2 extends PERC1 with

- a ramp time defined at rated power: ``TrampUsed = Tramp * Lfm`` (also for
  ``TLrampL`` and ``TLrampH``);
- a total current limit ``Imax`` applied together with ``Ipmin/Ipmax`` and
  ``Iqmin/Iqmax`` (Ip is limited first, Iq gets the remaining current);
- a high voltage cease threshold ``VHcease`` (the former ``Vcease`` is
  ``VLcease``) and a reconnection window ``VLrecon <= Vfilt <= VHrecon``;
- four long-cease low-voltage threshold/timer pairs (``VLC1..4/TVLC1..4``,
  reconnection with ``TLreconL`` and ramp ``TLrampL*Lfm``) and four
  long-cease high-voltage pairs (``VHC1..4/TVHC1..4``, ``TLreconH``,
  ``TLrampH*Lfm``);
- low/high frequency threshold/timer pairs (``FLC1/TFLC1``, ``FHC1/TFHC1``)
  after which the load never reconnects.

The cease/reconnect pseudo-code, the current limit enforcement and the data
modifications of the help page are implemented literally; see
:py:func:`perc2_fracon_step`, the model equations and
:py:func:`perc2_data_modifications`.
"""

import logging
from collections import OrderedDict, namedtuple

import numpy as np

from andes.core import (ModelData, IdxParam, NumParam, Model, ExtParam,
                        ExtService, ExtAlgeb, ConstService, State, Algeb,
                        DeadBand, LessThan, Limiter)
from andes.core.service import DeviceFinder, VarService
from andes.utils.tab import Tab

logger = logging.getLogger(__name__)

# PowerWorld mode codes (``vtm_*``)
VTM_MONITOR, VTM_CEASED, VTM_RAMP = 0.0, 1.0, 2.0

# ``CeaseLevel`` / ``TimerDelayActiveLevel`` codes of the pseudo-code
CL_NONE, CL_CEASE, CL_LONG, CL_NEVER = -1.0, 0.0, 1.0, 2.0

# values written by the frequency (never reconnect) zone and the ignored thresholds
TRAMP_NEVER, TRECON_NEVER, V_IGNORED = 999999999.0, -1.0, 999.0

# bit positions of the timers in the ``Tact`` output (sum of 2**bit over the active timers)
TIMER_BITS = OrderedDict([('VLcease', 0), ('VHcease', 1),
                          ('LV1', 2), ('LV2', 3), ('LV3', 4), ('LV4', 5),
                          ('HV1', 6), ('HV2', 7), ('HV3', 8), ('HV4', 9),
                          ('LF1', 10), ('HF1', 11), ('Recon', 12)])

PERC2State = namedtuple('PERC2State', ['mode', 'cl', 'tdl', 'tdelay', 'tsr', 'trampu', 'treconu',
                                       'trec', 'treca', 'tvl', 'tvla', 'tvh', 'tvha',
                                       'tlv', 'tlva', 'thv', 'thva', 'tlf', 'tlfa', 'thf', 'thfa',
                                       'fon'])
PERC2State.__doc__ = """
State of the PERC2 cease/reconnect logic, one array per device (shape ``(n,)``,
or ``(n, 4)`` for the four long-cease pairs):

``mode`` Mode; ``cl`` CeaseLevel; ``tdl`` TimerDelayActiveLevel; ``tdelay``
TimerDelay; ``tsr`` TStartRamp; ``trampu`` TrampUsed; ``treconu`` TreconUsed;
``trec``/``treca`` TimerRecon and TimerReconActive; ``tvl``/``tvla`` and
``tvh``/``tvha`` the VLcease/VHcease timers; ``tlv``/``tlva`` the LV1..4
timers; ``thv``/``thva`` the HV1..4 timers; ``tlf``/``tlfa`` and ``thf``/``thfa``
the LF1/HF1 timers; ``fon`` FracOn. Active flags are 0/1 floats.
"""

PERC2Params = namedtuple('PERC2Params', ['VLMflag', 'VHMflag', 'VLLflag', 'VHLflag', 'FLflag', 'FHflag',
                                         'Lfm', 'Fcease', 'Frecon', 'Tdelay', 'Tcease',
                                         'Tramp', 'Trecon', 'TLrampL', 'TLreconL', 'TLrampH', 'TLreconH',
                                         'VLcease', 'VHcease', 'VLrecon', 'VHrecon',
                                         'VLC', 'TVLC', 'VHC', 'TVHC',
                                         'FLC1', 'TFLC1', 'FHC1', 'TFHC1'])
PERC2Params.__doc__ = """
Parameters of the PERC2 cease/reconnect logic *after* the data modifications
of the help page (:py:func:`perc2_data_modifications`), one array per
parameter (``VLC``, ``TVLC``, ``VHC`` and ``TVHC`` have shape ``(n, 4)``).
``Lfm`` is the effective loading factor; the ramp times are multiplied by it
inside :py:func:`perc2_fracon_step`.
"""

_PAIR_FIELDS = ('VLC', 'TVLC', 'VHC', 'TVHC')


def perc2_initial_state(n):
    """
    Return the pre-simulation state of the cease/reconnect logic for `n`
    devices, as initialized by the pseudo-code: ``FracOn = 1``, ``Mode =
    vtm_Monitor``, ``CeaseLevel = TimerDelayActiveLevel = -1``, all timers
    inactive and zero, ``TrampUsed`` and ``TreconUsed`` zero (the pseudo-code
    sets them to ``Tramp*Lfm`` and ``Trecon``; they are only read after a
    cease zone has written them, so the value is immaterial. The model sets
    them to ``Tramp*Lfm`` and ``Trecon`` at initialization).
    """
    z = np.zeros(n)
    z4 = np.zeros((n, 4))
    return PERC2State(mode=np.full(n, VTM_MONITOR), cl=np.full(n, CL_NONE), tdl=np.full(n, CL_NONE),
                      tdelay=z.copy(), tsr=z.copy(), trampu=z.copy(), treconu=z.copy(),
                      trec=z.copy(), treca=z.copy(), tvl=z.copy(), tvla=z.copy(), tvh=z.copy(), tvha=z.copy(),
                      tlv=z4.copy(), tlva=z4.copy(), thv=z4.copy(), thva=z4.copy(),
                      tlf=z.copy(), tlfa=z.copy(), thf=z.copy(), thfa=z.copy(),
                      fon=np.ones(n))


def perc2_data_modifications(prm):
    """
    Apply the "data modifications run at the beginning of a simulation" of
    the PERC2 help page to a dict of parameter arrays and return a new dict.

    - ``Lfm < 0.001`` is replaced by 1.0;
    - ``Fcease > 1`` is replaced by 1;
    - negative ``Tramp, TLrampL, TLrampH, Trecon, TLreconL, TLreconH`` by 0;
    - ``VLrecon`` is raised to ``max(VLrecon, VLcease, VLC1..4)``;
    - ``VHcease, VHC1..4 <= 0`` are replaced by 999 (ignored);
    - ``VHrecon`` is lowered to ``min(VHrecon, VHcease, VHC1..4)`` (modified
      values); if then ``VHrecon < VLrecon``, ``VHrecon = 999`` (ignored).

    The modifications are unconditional, i.e., they do not depend on the
    flags that disable zones (as in the pseudo-code).
    """
    p = {k: np.array(v, dtype=float) for k, v in prm.items()}
    p['Lfm'] = np.where(p['Lfm'] < 0.001, 1.0, p['Lfm'])
    p['Fcease'] = np.minimum(p['Fcease'], 1.0)
    for name in ('Tramp', 'TLrampL', 'TLrampH', 'Trecon', 'TLreconL', 'TLreconH'):
        p[name] = np.maximum(p[name], 0.0)
    for name in ('VLcease', 'VLC1', 'VLC2', 'VLC3', 'VLC4'):
        p['VLrecon'] = np.maximum(p['VLrecon'], p[name])
    for name in ('VHcease', 'VHC1', 'VHC2', 'VHC3', 'VHC4'):
        p[name] = np.where(p[name] <= 0.0, V_IGNORED, p[name])
        p['VHrecon'] = np.minimum(p['VHrecon'], p[name])
    p['VHrecon'] = np.where(p['VHrecon'] < p['VLrecon'], V_IGNORED, p['VHrecon'])
    return p


def perc2_params(n=1, modify=True, **kwargs):
    """
    Build a :py:class:`PERC2Params` for `n` devices from PowerWorld parameter
    names, using the PERC2 defaults for the parameters not given.

    With ``modify=True`` (default) the data modifications of the help page
    are applied first. Scalars are broadcast to all devices.
    """
    data = PERC2Data()
    names = list(PERC2Params._fields[:21]) + [f'{k}{i}' for k in _PAIR_FIELDS for i in range(1, 5)] + \
        list(PERC2Params._fields[25:])
    prm = {}
    for name in names:
        val = kwargs.pop(name, data.__dict__[name].default)
        prm[name] = np.broadcast_to(np.asarray(val, dtype=float), (n,)).copy()
    if kwargs:
        raise ValueError(f'unknown PERC2 parameters: {sorted(kwargs)}')
    if modify:
        prm = perc2_data_modifications(prm)
    return params_from_dict(prm)


def params_from_dict(prm):
    """
    Assemble a :py:class:`PERC2Params` from a dict of (already modified)
    parameter arrays with PowerWorld names (``VLC1..4`` etc. are stacked).
    """
    out = {}
    for field in PERC2Params._fields:
        if field in _PAIR_FIELDS:
            out[field] = np.column_stack([prm[f'{field}{i}'] for i in range(1, 5)])
        else:
            out[field] = np.asarray(prm[field], dtype=float)
    return PERC2Params(**out)


def perc2_timer_bitmask(state):
    """
    Return the ``Tact`` output: the sum of ``2**bit`` over the active timers
    (bits in :py:data:`TIMER_BITS`).
    """
    out = state.tvla * 2 ** TIMER_BITS['VLcease'] + state.tvha * 2 ** TIMER_BITS['VHcease']
    for k in range(4):
        out = out + state.tlva[:, k] * 2 ** TIMER_BITS[f'LV{k + 1}'] + \
            state.thva[:, k] * 2 ** TIMER_BITS[f'HV{k + 1}']
    out = out + state.tlfa * 2 ** TIMER_BITS['LF1'] + state.thfa * 2 ** TIMER_BITS['HF1'] + \
        state.treca * 2 ** TIMER_BITS['Recon']
    return out


def perc2_fracon_step(state, t, vfilt, ffilt, prm, active=None, ttol=1e-8):
    """
    Execute the PERC2 cease/reconnect pseudo-code ("Run the following at each
    time step") once, for all devices.

    This is a pure function: `state` is not modified and the new state is
    returned. The statements keep the order of the PowerWorld pseudo-code:
    ramp update (frozen while a Tdelay timer is active), the never-reconnect
    frequency timers (level 2), the long-cease high (purple) and low (red)
    voltage timers (level 1), the cease timers (blue, level 0), the
    transition to Ceased when the delay timer expires, and the reconnection
    timer with the ``VLrecon <= Vfilt <= VHrecon`` window. A device that
    enters Ceased in this execution does not enter the reconnection branch.

    Two typos of the published pseudo-code are corrected: the fourth low
    voltage timer compares ``Vfilt`` with ``VLC4`` (not ``VHC4``), and the
    high frequency timer is ``TimerHF1``.

    Parameters
    ----------
    state : PERC2State
        Present state arrays.
    t : float
        Present simulation time, ``PresentTime`` [s].
    vfilt : array-like
        Filtered voltage ``Vfilt`` [pu].
    ffilt : array-like
        Filtered frequency ``Ffilt = fnom * (1 + wfilt)`` [Hz].
    prm : PERC2Params
        Parameters after the data modifications (scalars are broadcast).
    active : array-like of bool, optional
        Devices on which the logic is executed. Others keep their state.
    ttol : float
        Tolerance [s] of the time comparisons ``a >= b``, evaluated as
        ``a >= b - ttol`` so that elapsed times accumulated with rounding
        errors expire on the intended step. ``ttol = 0`` is the literal
        comparison.

    Returns
    -------
    PERC2State
        The new state.
    """
    S = PERC2State(*(np.array(x, dtype=float) for x in state))
    n = S.mode.shape[0]

    def _b(x, shape=(n,)):
        return np.broadcast_to(np.asarray(x, dtype=float), shape)

    P = PERC2Params(*(_b(x, (n, 4)) if name in _PAIR_FIELDS else _b(x)
                      for name, x in zip(PERC2Params._fields, prm)))
    vfilt, ffilt = _b(vfilt), _b(ffilt)
    t = float(t)

    def ge(a, b):
        return a >= b - ttol

    # "If Fcease <= 0 Then Exit"
    run = P.Fcease > 0.0
    if active is not None:
        run = run & np.asarray(active, dtype=bool)

    tramp_u, trampl_u, tramph_u = P.Tramp * P.Lfm, P.TLrampL * P.Lfm, P.TLrampH * P.Lfm
    f_ceased = 1.0 - P.Fcease
    f_final = 1.0 - P.Fcease + P.Frecon * P.Fcease

    def handle_timer(mask, make_active, act, timer):
        """LOCAL_HandleTimer on the devices in `mask` (in place)."""
        off = mask & ~make_active & (act != 0)
        act[off] = 0.0
        on = mask & make_active & (act == 0)
        timer[on] = t
        act[on] = 1.0

    def zone_expired(mask, tramp_val, trecon_val, level):
        """A cease zone timer expired: record TrampUsed/TreconUsed, escalate the level."""
        if not np.any(mask):
            return
        S.trampu[mask] = _b(tramp_val)[mask]
        S.treconu[mask] = _b(trecon_val)[mask]
        ceased = S.mode == VTM_CEASED
        S.cl[mask & ceased] = level                 # already ceased: change CeaseLevel
        nc = mask & ~ceased
        start = nc & (S.tdl < 0)                    # only start the delay timer once
        S.tdelay[start] = t
        S.tdl[nc] = level

    def gate(level):
        """Devices whose timers of `level` are processed."""
        ceased = S.mode == VTM_CEASED
        return run & ((~ceased & (S.tdl < level)) | (ceased & (S.cl < level)))

    # --- update FracOn while ramping; frozen while a Tdelay timer is active ---
    ramp = run & (S.mode == VTM_RAMP) & (S.tdl < 0)
    done = ramp & ge(t, S.tsr + S.trampu)
    S.fon[done] = f_final[done]
    S.mode[done] = VTM_MONITOR
    S.trampu[done] = tramp_u[done]                  # reset back
    going = ramp & ~done
    if np.any(going):
        S.fon[going] = f_ceased[going] + P.Frecon[going] * P.Fcease[going] * \
            (t - S.tsr[going]) / S.trampu[going]

    # --- level 2: never-reconnect frequency timers ---
    g2 = gate(CL_NEVER)
    fl, fh = P.FLflag != 0, P.FHflag != 0
    handle_timer(g2 & fl, ffilt < P.FLC1, S.tlfa, S.tlf)
    handle_timer(g2 & fh, ffilt > P.FHC1, S.thfa, S.thf)
    exp2 = g2 & ((fl & (S.tlfa != 0) & ge(t - S.tlf, P.TFLC1)) |
                 (fh & (S.thfa != 0) & ge(t - S.thf, P.TFHC1)))
    zone_expired(exp2, TRAMP_NEVER, TRECON_NEVER, CL_NEVER)

    # --- level 1: long-cease high voltage timers (purple zone) ---
    g1h = gate(CL_LONG) & (P.VHLflag != 0)
    for k in range(4):
        handle_timer(g1h, vfilt > P.VHC[:, k], S.thva[:, k], S.thv[:, k])
    exp1h = g1h & np.any((S.thva != 0) & ge(t - S.thv, P.TVHC), axis=1)
    zone_expired(exp1h, tramph_u, P.TLreconH, CL_LONG)

    # --- level 1: long-cease low voltage timers (red zone) ---
    g1l = gate(CL_LONG) & (P.VLLflag != 0)
    for k in range(4):
        handle_timer(g1l, vfilt < P.VLC[:, k], S.tlva[:, k], S.tlv[:, k])
    exp1l = g1l & np.any((S.tlva != 0) & ge(t - S.tlv, P.TVLC), axis=1)
    zone_expired(exp1l, trampl_u, P.TLreconL, CL_LONG)

    # --- level 0: cease timers (blue zone) ---
    g0 = gate(CL_CEASE)
    vlm, vhm = P.VLMflag != 0, P.VHMflag != 0
    handle_timer(g0 & vlm, vfilt < P.VLcease, S.tvla, S.tvl)
    handle_timer(g0 & vhm, vfilt > P.VHcease, S.tvha, S.tvh)
    exp0 = g0 & ((vlm & (S.tvla != 0) & ge(t - S.tvl, P.Tcease)) |
                 (vhm & (S.tvha != 0) & ge(t - S.tvh, P.Tcease)))
    zone_expired(exp0, tramp_u, P.Trecon, CL_CEASE)

    # --- transition to Ceased after the delay timer expires ---
    not_ceased = run & (S.mode != VTM_CEASED)
    tc = not_ceased & (S.tdl >= 0) & ge(t - S.tdelay, P.Tdelay)
    S.cl[tc] = S.tdl[tc]
    S.mode[tc] = VTM_CEASED
    S.fon[tc] = f_ceased[tc]
    S.tdl[tc] = CL_NONE

    # --- Ceased (before this test): reconnection timer with the voltage window ---
    ceased = run & ~not_ceased
    handle_timer(ceased, (vfilt >= P.VLrecon) & (vfilt <= P.VHrecon), S.treca, S.trec)
    expr = ceased & (S.treca != 0) & (S.treconu >= 0) & ge(t - S.trec, S.treconu)
    if np.any(expr):
        for arr in (S.treca, S.tvla, S.tvha, S.tlfa, S.thfa):
            arr[expr] = 0.0
        S.tlva[expr, :] = 0.0
        S.thva[expr, :] = 0.0
        S.cl[expr] = CL_NONE
        S.treconu[expr] = P.Trecon[expr]            # reset back
        jump = expr & (S.trampu == 0.0)             # skip over Ramp mode
        S.mode[jump] = VTM_MONITOR
        S.fon[jump] = f_final[jump]
        rs = expr & (S.trampu != 0.0)
        S.mode[rs] = VTM_RAMP
        S.tsr[rs] = t

    return S


def perc2_limit_currents(Pctrl, Qctrl, Vfilt, Vinit, nP, nQ, Imax, Ipmin, Ipmax, Iqmin, Iqmax, vguard=0.001):
    """
    Independent NumPy transliteration of the help-page procedure
    ``CalculateIpvIqv`` (limit enforcement), for testing the model equations.

    Returns ``(Ipv, Iqv, Ipcmd, Iqcmd, Iqlim)`` where ``Ipcmd, Iqcmd`` are the
    unlimited currents and ``Iqlim`` the current left for Iq (``nan`` when
    ``Imax <= 0``).
    """
    Pctrl, Qctrl, Vfilt = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (Pctrl, Qctrl, Vfilt)))
    Ipv = Pctrl * (Vfilt / Vinit) ** nP
    Iqv = Qctrl * (Vfilt / Vinit) ** nQ
    div = np.where(Vfilt > vguard, Vfilt, vguard)
    Ipv, Iqv = Ipv / div, Iqv / div
    Ipcmd, Iqcmd = Ipv.copy(), Iqv.copy()
    lo_p, hi_p, lo_q, hi_q = Ipmin, Ipmax, Iqmin, Iqmax
    if Imax > 0:
        hi_p, lo_p = min(hi_p, Imax), max(lo_p, -Imax)
    Ipv = np.where(Ipv > hi_p, hi_p, Ipv)
    Ipv = np.where(Ipv < lo_p, lo_p, Ipv)
    Iqlim = np.full(Ipv.shape, np.nan)
    if Imax > 0:
        temp = Imax ** 2 - Ipv ** 2
        noleft = temp < 1e-12                        # no current left for Iq: both bounds are zero
        Iqlim = np.where(noleft, 0.0, np.sqrt(np.maximum(temp, 0.0)))
        hi_q = np.where(noleft, 0.0, np.where(hi_q > Iqlim, Iqlim, hi_q))
        lo_q = np.where(noleft, 0.0, np.where(lo_q < -Iqlim, -Iqlim, lo_q))
    Iqv = np.where(Iqv > hi_q, hi_q, Iqv)
    Iqv = np.where(Iqv < lo_q, lo_q, Iqv)
    return Ipv, Iqv, Ipcmd, Iqcmd, Iqlim


class PERC2Data(ModelData):
    """
    Data for the PERC2 load model (PowerWorld parameter names and defaults).
    """

    def __init__(self):
        ModelData.__init__(self)

        self.pq = IdxParam(model='StaticLoad', mandatory=True,
                           info='idx of the PQ to replace',
                           replaces=True,
                           )
        self.busf = IdxParam(model='BusFreq',
                             info='optional idx of the BusFreq device to use',
                             )
        self.comp = NumParam(default=0,
                             info='1 if used as a CompLoad component (Iqextra = 0); '
                                  '0 for stand-alone',
                             tex_name='z_{comp}',
                             vrange=(0, 1),
                             )

        # --- integer flags (September 25, 2026 patch) ---
        self.VLMflag = NumParam(default=1, info='low voltage cease check (0 off; else on)',
                                tex_name='z_{VLM}', vrange=(0, 1))
        self.VHMflag = NumParam(default=1, info='high voltage cease check (0 off; else on)',
                                tex_name='z_{VHM}', vrange=(0, 1))
        self.VLLflag = NumParam(default=1, info='long low voltage cease check (0 off; else on)',
                                tex_name='z_{VLL}', vrange=(0, 1))
        self.VHLflag = NumParam(default=1, info='long high voltage cease check (0 off; else on); '
                                                'VLHflag in the pseudo-code',
                                tex_name='z_{VHL}', vrange=(0, 1))
        self.FLflag = NumParam(default=1, info='low frequency cease check (0 off; else on)',
                               tex_name='z_{FL}', vrange=(0, 1))
        self.FHflag = NumParam(default=1, info='high frequency cease check (0 off; else on)',
                               tex_name='z_{FH}', vrange=(0, 1))

        # --- load and dynamic blocks ---
        self.Lfm = NumParam(default=0.8,
                            info='loading factor, MBase = MWinit / Lfm (1.0 if Lfm < 0.001)',
                            tex_name='L_{fm}',
                            )
        self.QPratio = NumParam(default=0.66,
                                info='Q/P ratio, MvarInit = QPratio * MWinit',
                                tex_name='QP_{ratio}',
                                )
        self.Tt = NumParam(default=0.02,
                           info='time delay for outputs (treated as 4*TimeStep if smaller)',
                           tex_name='T_t',
                           unit='s',
                           )
        self.Tv = NumParam(default=0.02,
                           info='voltage measurement time constant',
                           tex_name='T_v',
                           unit='s',
                           )
        self.Tf = NumParam(default=0.02,
                           info='frequency measurement time constant',
                           tex_name='T_f',
                           unit='s',
                           )
        self.Dbdl = NumParam(default=0.0,
                             info='frequency deadband low (<= 0)',
                             tex_name='d_{bdl}',
                             unit='p.u.',
                             )
        self.Dbdh = NumParam(default=0.0,
                             info='frequency deadband high (>= 0)',
                             tex_name='d_{bdh}',
                             unit='p.u.',
                             )
        self.Kdroop = NumParam(default=10.0,
                               info='frequency droop',
                               tex_name='K_{droop}',
                               unit='p.u.',
                               )
        self.Kvp = NumParam(default=0.02,
                            info='gain of active power washout',
                            tex_name='K_{vp}',
                            )
        self.Tvp = NumParam(default=0.02,
                            info='time constant of active power washout',
                            tex_name='T_{vp}',
                            unit='s',
                            )
        self.Kvq = NumParam(default=0.5,
                            info='gain of reactive power washout',
                            tex_name='K_{vq}',
                            )
        self.Tvq = NumParam(default=0.1,
                            info='time constant of reactive power washout',
                            tex_name='T_{vq}',
                            unit='s',
                            )
        self.Tap = NumParam(default=0.0,
                            info='lead time constant, active power path',
                            tex_name='T_{ap}',
                            unit='s',
                            )
        self.Tbp = NumParam(default=0.0,
                            info='lag time constant, active power path',
                            tex_name='T_{bp}',
                            unit='s',
                            )
        self.Taq = NumParam(default=0.0,
                            info='lead time constant, reactive power path',
                            tex_name='T_{aq}',
                            unit='s',
                            )
        self.Tbq = NumParam(default=0.0,
                            info='lag time constant, reactive power path',
                            tex_name='T_{bq}',
                            unit='s',
                            )
        self.nP = NumParam(default=0.0,
                           info='active power voltage exponent',
                           tex_name='n_P',
                           )
        self.nQ = NumParam(default=1.0,
                           info='reactive power voltage exponent',
                           tex_name='n_Q',
                           )
        self.Ipmax = NumParam(default=1.0,
                              info='maximum Ip on MBase',
                              tex_name='I_{pmax}',
                              unit='p.u.',
                              )
        self.Ipmin = NumParam(default=0.0,
                              info='minimum Ip on MBase',
                              tex_name='I_{pmin}',
                              unit='p.u.',
                              )
        self.Iqmax = NumParam(default=0.66,
                              info='maximum Iq on MBase',
                              tex_name='I_{qmax}',
                              unit='p.u.',
                              )
        self.Iqmin = NumParam(default=-0.66,
                              info='minimum Iq on MBase',
                              tex_name='I_{qmin}',
                              unit='p.u.',
                              )
        self.Imax = NumParam(default=2.0,
                             info='maximum total current on MBase (<= 0: not used)',
                             tex_name='I_{max}',
                             unit='p.u.',
                             )

        # --- cease / reconnect ---
        self.Fcease = NumParam(default=1.0,
                               info='fraction that will cease (between 0 and 1)',
                               tex_name='F_{cease}',
                               )
        self.Frecon = NumParam(default=1.0,
                               info='fraction of the ceased load that reconnects (>= 0; > 1 allowed)',
                               tex_name='F_{recon}',
                               )
        self.Tdelay = NumParam(default=0.0,
                               info='time delay to cease after detection, all cease zones (>= 0)',
                               tex_name='T_{delay}',
                               unit='s',
                               )
        self.VLrecon = NumParam(default=0.8,
                                info='low voltage threshold of the reconnection window',
                                tex_name='V_{Lrecon}',
                                unit='p.u.',
                                )
        self.VHrecon = NumParam(default=2.0,
                                info='high voltage threshold of the reconnection window',
                                tex_name='V_{Hrecon}',
                                unit='p.u.',
                                )
        self.Tramp = NumParam(default=0.15,
                              info='ramp time from 0 to 100% at rated power (>= 0); TrampUsed = Tramp*Lfm',
                              tex_name='T_{ramp}',
                              unit='s',
                              )
        self.Trecon = NumParam(default=0.05,
                               info='reconnection time delay of the cease zone (>= 0)',
                               tex_name='T_{recon}',
                               unit='s',
                               )
        self.VLcease = NumParam(default=0.8,
                                info='low voltage threshold of the cease zone',
                                tex_name='V_{Lcease}',
                                unit='p.u.',
                                )
        self.VHcease = NumParam(default=2.0,
                                info='high voltage threshold of the cease zone (<= 0: ignored)',
                                tex_name='V_{Hcease}',
                                unit='p.u.',
                                )
        self.Tcease = NumParam(default=0.01,
                               info='time delay of the cease zone (>= 0)',
                               tex_name='T_{cease}',
                               unit='s',
                               )
        self.TLrampL = NumParam(default=5.0,
                                info='long low cease ramp time (>= 0); TrampUsed = TLrampL*Lfm',
                                tex_name='T_{LrampL}',
                                unit='s',
                                )
        self.TLreconL = NumParam(default=0.05,
                                 info='long low cease reconnection time delay (>= 0)',
                                 tex_name='T_{LreconL}',
                                 unit='s',
                                 )
        for i, (v_def, t_def) in enumerate(((0.90, 2.00), (0.80, 0.50), (0.50, 0.25), (0.35, 0.15)), 1):
            setattr(self, f'VLC{i}',
                    NumParam(default=v_def, info=f'low voltage threshold {i} for long cease',
                             tex_name=f'V_{{LC{i}}}', unit='p.u.'))
            setattr(self, f'TVLC{i}',
                    NumParam(default=t_def, info=f'time delay {i} for long low cease (>= 0)',
                             tex_name=f'T_{{VLC{i}}}', unit='s'))
        self.TLrampH = NumParam(default=5.0,
                                info='long high cease ramp time (>= 0); TrampUsed = TLrampH*Lfm',
                                tex_name='T_{LrampH}',
                                unit='s',
                                )
        self.TLreconH = NumParam(default=0.05,
                                 info='long high cease reconnection time delay (>= 0)',
                                 tex_name='T_{LreconH}',
                                 unit='s',
                                 )
        for i, (v_def, t_def) in enumerate(((1.10, 1.00), (2.00, 99.0), (2.00, 99.0), (2.00, 99.0)), 1):
            setattr(self, f'VHC{i}',
                    NumParam(default=v_def, info=f'high voltage threshold {i} for long cease (<= 0: ignored)',
                             tex_name=f'V_{{HC{i}}}', unit='p.u.'))
            setattr(self, f'TVHC{i}',
                    NumParam(default=t_def, info=f'time delay {i} for long high cease (>= 0)',
                             tex_name=f'T_{{VHC{i}}}', unit='s'))
        self.FLC1 = NumParam(default=57.0,
                             info='low frequency threshold for forever cease',
                             tex_name='F_{LC1}',
                             unit='Hz',
                             )
        self.TFLC1 = NumParam(default=999.0,
                              info='low frequency time delay for forever cease (>= 0)',
                              tex_name='T_{FLC1}',
                              unit='s',
                              )
        self.FHC1 = NumParam(default=63.0,
                             info='high frequency threshold for forever cease',
                             tex_name='F_{HC1}',
                             unit='Hz',
                             )
        self.TFHC1 = NumParam(default=999.0,
                              info='high frequency time delay for forever cease (>= 0)',
                              tex_name='T_{FHC1}',
                              unit='s',
                              )


# held state of the logic: (attribute, initial value string, tex, info)
_HELD = [('hMode', '0', 'M_h', 'held Mode'),
         ('hCL', '-1', 'CL_h', 'held CeaseLevel'),
         ('hTDL', '-1', 'TDL_h', 'held TimerDelayActiveLevel'),
         ('hTdelay', '0', 'T_{delay,h}', 'held TimerDelay'),
         ('hTsr', '0', 'T_{sr,h}', 'held TStartRamp'),
         ('hTrampU', '0', 'T_{ramp,u,h}', 'held TrampUsed'),
         ('hTreconU', '0', 'T_{recon,u,h}', 'held TreconUsed'),
         ('hTrec', '0', 'T_{rec,h}', 'held TimerRecon'),
         ('hTrecA', '0', 'T_{rec,a,h}', 'held TimerReconActive'),
         ('hTvl', '0', 'T_{vl,h}', 'held TimerVLCease'),
         ('hTvlA', '0', 'T_{vl,a,h}', 'held TimerVLCeaseActive'),
         ('hTvh', '0', 'T_{vh,h}', 'held TimerVHCease'),
         ('hTvhA', '0', 'T_{vh,a,h}', 'held TimerVHCeaseActive'),
         ('hTlf', '0', 'T_{lf,h}', 'held TimerLF1'),
         ('hTlfA', '0', 'T_{lf,a,h}', 'held TimerLF1Active'),
         ('hThf', '0', 'T_{hf,h}', 'held TimerHF1'),
         ('hThfA', '0', 'T_{hf,a,h}', 'held TimerHF1Active'),
         ('hFon', '1', 'F_{on,h}', 'held FracOn'),
         ]
for _i in range(1, 5):
    _HELD += [(f'hTlv{_i}', '0', f'T_{{lv{_i},h}}', f'held TimerLV{_i}'),
              (f'hTlv{_i}A', '0', f'T_{{lv{_i},a,h}}', f'held TimerLV{_i}Active'),
              (f'hThv{_i}', '0', f'T_{{hv{_i},h}}', f'held TimerHV{_i}'),
              (f'hThv{_i}A', '0', f'T_{{hv{_i},a,h}}', f'held TimerHV{_i}Active')]


class PERC2Model(Model):
    """
    Implementation of the PERC2 load model.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'DynLoad'
        self.flags.tds = True

        self.config.add(OrderedDict((('qinit', 1),
                                     ('minvi', 0.5),
                                     ('ilow', 0),
                                     ('pneg', 1),
                                     ('smallload', 1),
                                     ('pmin', 0.001),
                                     ('tstep', 0.0),
                                     ('tstol', 5e-4),
                                     ('tsrule', 0),
                                     ('vfmin', 0.7),
                                     ('vguard', 1e-3),
                                     ('vfloor', 1e-9),
                                     ('ttol', 1e-8),
                                     )))
        self.config.add_extra("_help",
                              qinit='1: MvarInit = QPratio*MWinit, Iqextra for stand-alone (block diagram); '
                                    '0: MvarInit = initial Mvar for stand-alone (parameter table text)',
                              minvi='MinVoltILoad [pu]; use constant impedance if V0 is lower',
                              ilow='1 to reduce the network current by sin(pi/2 V/minvi) '
                                   'below minvi',
                              pneg='for MWinit < 0: 0 error, 1 constant impedance, '
                                   '2 run with a negative MBase',
                              smallload='1 to leave loads with |P0| < pmin to the PQ model',
                              pmin='small-load threshold [pu, system base]',
                              tstep='TimeStep [s] for Tt >= 4*TimeStep; 0 uses TDS.config.tstep',
                              tstol='relative tolerance of the Tt >= 4*TimeStep check',
                              tsrule='LDEV1-type time-step rules: 0 off; 1 for Tv, Tvp, Tvq, Tbp, '
                                     'Tbq; 2 also for Tf',
                              vfmin='frequency input is ignored when V < vfmin [pu]',
                              vguard='divisor guard of Vfilt in the current commands [pu] (0.001 in '
                                     'the pseudo-code)',
                              vfloor='numerical floor of Vfilt in the voltage-exponent numerator [pu] '
                                     '(0 reproduces the pseudo-code literally; keeps the Jacobian finite)',
                              ttol='tolerance of time comparisons in cease/reconnect logic [s]',
                              )
        self.config.add_extra("_alt",
                              qinit=(0, 1),
                              minvi='float',
                              ilow=(0, 1),
                              pneg=(0, 1, 2),
                              smallload=(0, 1),
                              pmin='>=0',
                              tstep='>=0',
                              tstol='>=0',
                              tsrule=(0, 1, 2),
                              vfmin='float',
                              vguard='>0',
                              vfloor='>=0',
                              ttol='>=0',
                              )
        self.config.add_extra("_tex",
                              minvi='V_{min,I}',
                              vfmin='V_{f,min}',
                              )

        # --- external data ---
        self.bus = ExtParam(model='PQ', src='bus', indexer=self.pq,
                            info='retrieved bus idx',
                            export=False,
                            )
        self.p0 = ExtService(model='PQ', src='Ppf', indexer=self.pq,
                             tex_name='P_0',
                             info='initial active power in system base',
                             )
        self.q0 = ExtService(model='PQ', src='Qpf', indexer=self.pq,
                             tex_name='Q_0',
                             info='initial reactive power in system base',
                             )
        self.v0 = ExtService(model='Bus', src='v', indexer=self.bus,
                             tex_name='V_0',
                             info='initial voltage magnitude',
                             )

        self.busfreq = DeviceFinder(u=self.busf, link=self.bus, idx_name='bus',
                                    default_model='BusFreq',
                                    info='found idx of BusFreq',
                                    )
        self.f = ExtAlgeb(model='FreqMeasurement', src='f', indexer=self.busfreq,
                          tex_name='f',
                          info='bus frequency',
                          )

        # --- initialization (computed once when TDS is initialized) ---
        self._init = ConstService(v_numeric=self._init_calc,
                                  info='initialization flag (1 dynamic, 0 fallback)',
                                  tex_name='z_{init}',
                                  )

        def _ic(key, tex, info):
            return ConstService(v_numeric=(lambda k: (lambda **kw: self._icd[k]))(key),
                                tex_name=tex, info=info)

        self.Lfe = _ic('Lfe', 'L_{fe}', 'effective loading factor')
        self.zfb = _ic('zfb', 'z_{fb}', '1 if replaced by constant impedance')
        self.zdy = _ic('zdy', 'z_{dy}', '1 if the dynamic model is used')
        self.kb = _ic('kb', 'k_b', 'MBase / system base')
        self.MB = _ic('MB', 'M_{Base}', 'model base [MVA]')
        self.Pi0 = _ic('Pi0', 'P_{i0}', 'MWinit / MBase')
        self.Qi0 = _ic('Qi0', 'Q_{i0}', 'MvarInit / MBase')
        self.Iqx = _ic('Iqx', 'I_{qextra}', 'Iqextra on MBase')
        self.Tte = _ic('Tte', 'T_{t,e}', 'effective Tt')
        self.Tve = _ic('Tve', 'T_{v,e}', 'effective Tv')
        self.Tfe = _ic('Tfe', 'T_{f,e}', 'effective Tf')
        self.Tvpe = _ic('Tvpe', 'T_{vp,e}', 'effective Tvp')
        self.Tvqe = _ic('Tvqe', 'T_{vq,e}', 'effective Tvq')
        self.Tbpe = _ic('Tbpe', 'T_{bp,e}', 'effective Tbp')
        self.Tbqe = _ic('Tbqe', 'T_{bq,e}', 'effective Tbq')
        self.zwp = _ic('zwp', 'z_{wp}', '1 if the active power washout is in use')
        self.zwq = _ic('zwq', 'z_{wq}', '1 if the reactive power washout is in use')
        self.zlp = _ic('zlp', 'z_{lp}', '1 if the active power lead-lag is in use')
        self.zlq = _ic('zlq', 'z_{lq}', '1 if the reactive power lead-lag is in use')
        self.zdb0 = _ic('zdb0', 'z_{db0}', '1 if the frequency deadband has zero width')

        # effective cease/reconnect parameters after the data modifications
        self.Fce = _ic('Fce', 'F_{cease,e}', 'effective Fcease (<= 1)')
        self.Trampe = _ic('Trampe', 'T_{ramp,e}', 'effective Tramp (>= 0)')
        self.Trecone = _ic('Trecone', 'T_{recon,e}', 'effective Trecon (>= 0)')
        self.TLrampLe = _ic('TLrampLe', 'T_{LrampL,e}', 'effective TLrampL (>= 0)')
        self.TLreconLe = _ic('TLreconLe', 'T_{LreconL,e}', 'effective TLreconL (>= 0)')
        self.TLrampHe = _ic('TLrampHe', 'T_{LrampH,e}', 'effective TLrampH (>= 0)')
        self.TLreconHe = _ic('TLreconHe', 'T_{LreconH,e}', 'effective TLreconH (>= 0)')
        self.VLrecone = _ic('VLrecone', 'V_{Lrecon,e}', 'effective VLrecon (>= VLcease, VLC1..4)')
        self.VHrecone = _ic('VHrecone', 'V_{Hrecon,e}', 'effective VHrecon (<= VHcease, VHC1..4; 999 if ignored)')
        self.VHceasee = _ic('VHceasee', 'V_{Hcease,e}', 'effective VHcease (999 if ignored)')
        for i in range(1, 5):
            setattr(self, f'VHC{i}e', _ic(f'VHC{i}e', f'V_{{HC{i},e}}', f'effective VHC{i} (999 if ignored)'))
        self.TrampU0 = _ic('TrampU0', 'T_{ramp,u0}', 'initial TrampUsed = Tramp*Lfm')

        # effective Ip limits with Imax
        self.zim = _ic('zim', 'z_{Imax}', '1 if Imax > 0 is used')
        self.Ipmaxu = _ic('Ipmaxu', 'I_{pmax,u}', 'IpmaxUsed = min(Ipmax, Imax)')
        self.Ipminu = _ic('Ipminu', 'I_{pmin,u}', 'IpminUsed = max(Ipmin, -Imax)')

        self.Iqminn = ConstService(v_str='-Iqmin', tex_name='-I_{qmin}', info='negative of Iqmin')
        self.fnom = ConstService(v_str='sys_f', tex_name='f_{nom}', info='nominal frequency [Hz]')
        self.vfm = ConstService(v_str='vfmin', tex_name='V_{f,min}',
                                info='voltage below which the frequency input is ignored')
        self.vgd = ConstService(v_str='vguard', tex_name='V_{guard}',
                                info='divisor guard of Vfilt in the current commands')
        self.vfl = ConstService(v_str='vfloor', tex_name='V_{floor}',
                                info='numerical floor of Vfilt in the voltage-exponent numerator')
        self.vmi = ConstService(v_str='ilow * minvi', tex_name='V_{min,I}',
                                info='voltage below which the network current falls off')
        self.vmis = ConstService(v_str='vmi + Indicator(vmi <= 0)', tex_name='V_{min,I}^s',
                                 info='nonzero divisor for the current fall-off')

        # --- cease/reconnect logic state, committed once per accepted TDS step ---
        for name, v_str, tex, info in _HELD:
            setattr(self, name, ConstService(v_str=v_str, tex_name=tex, info=info))
        self.hTlast = ConstService(v_str='-1', tex_name='t_{last}',
                                   info='time of the last execution of the logic')

        # external frequency deviation input (0 unless set by the user)
        self.dwext = ConstService(v_str='0', tex_name=r'\Delta\omega_{ext}',
                                  info='external frequency deviation added to f - 1')

        # --- network interface: current injection (consumption positive) ---
        _gv = '(LTi_z0 + LTi_z1 * sin(pi / 2 * v / vmis))'
        _pmw = f'(zdy * kb * v * Ip * {_gv} + zfb * p0 * (v / v0) ** 2)'
        _qmvar = f'(zdy * kb * v * Iq * {_gv} + zfb * q0 * (v / v0) ** 2)'

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus,
                          tex_name=r'\theta',
                          e_str=f'ue * {_pmw}',
                          ename='P',
                          tex_ename='P',
                          )
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus,
                          tex_name='V',
                          e_str=f'ue * {_qmvar}',
                          ename='Q',
                          tex_ename='Q',
                          )

        # --- discrete ---
        self.LTf = LessThan(u=self.v, bound=self.vfm, equal=False,
                            info='V < vfmin: ignore frequency input',
                            )
        self.LTi = LessThan(u=self.v, bound=self.vmi, equal=False,
                            info='V < MinVoltILoad with the current fall-off enabled',
                            )

        # --- states 5, 6: frequency and voltage measurement ---
        self.wfilt = State(info='filtered frequency deviation (state 5)',
                           tex_name=r'\omega_{filt}',
                           t_const=self.Tfe,
                           v_str='0',
                           e_str='LTf_z0 * (f - 1 + dwext) - wfilt',
                           unit='p.u.',
                           )
        self.Vfilt = State(info='filtered voltage (state 6)',
                           tex_name='V_{filt}',
                           t_const=self.Tve,
                           v_str='v',
                           e_str='v - Vfilt',
                           unit='p.u.',
                           )
        self.Ffilt = Algeb(info='filtered frequency used by the cease logic',
                           tex_name='F_{filt}',
                           unit='Hz',
                           v_str='fnom * (1 + wfilt)',
                           e_str='fnom * (1 + wfilt) - Ffilt',
                           )

        self.dbw = DeadBand(u=self.wfilt, center=0.0, lower=self.Dbdl, upper=self.Dbdh,
                            tex_name='db_w',
                            info='frequency deadband',
                            )

        # --- states 3, 4: voltage washouts (input is the raw voltage) ---
        self.xdP = State(info='state of the active power washout (state 3)',
                         tex_name='x_{dP}',
                         t_const=self.Tvpe,
                         v_str='v',
                         e_str='v - xdP',
                         )
        self.xdQ = State(info='state of the reactive power washout (state 4)',
                         tex_name='x_{dQ}',
                         t_const=self.Tvqe,
                         v_str='v',
                         e_str='v - xdQ',
                         )
        self.dP = Algeb(info='active power washout output',
                        tex_name='dP',
                        v_str='0',
                        e_str='zwp * (Kvp * (v - xdP) - Tvpe * dP) - (1 - zwp) * dP',
                        )
        self.dQ = Algeb(info='reactive power washout output',
                        tex_name='dQ',
                        v_str='0',
                        e_str='zwq * (Kvq * (v - xdQ) - Tvqe * dQ) - (1 - zwq) * dQ',
                        )

        # continuous deadband; the `zdb0` term (zero at its only point) gives the slope 1 of a
        # zero-width band, whose flags are "inside" exactly at the steady state wfilt = 0
        _fdb = '(dbw_zu * (wfilt - Dbdh) + dbw_zl * (wfilt - Dbdl) + zdb0 * dbw_zi * (wfilt - Dbdh))'
        self.Pin = Algeb(info='active power request before lead-lag, on MBase',
                         tex_name='P_{in}',
                         v_str=f'Pi0 + dP + Kdroop * {_fdb}',
                         e_str=f'Pi0 + dP + Kdroop * {_fdb} - Pin',
                         discrete=self.dbw,
                         )
        self.Qin = Algeb(info='reactive power request before lead-lag, on MBase',
                         tex_name='Q_{in}',
                         v_str='Qi0 + dQ',
                         e_str='Qi0 + dQ - Qin',
                         )

        # --- states 1, 2: lead-lags ---
        self.xPc = State(info='state of the active power lead-lag (state 1)',
                         tex_name='x_{Pc}',
                         t_const=self.Tbpe,
                         v_str='Pin',
                         e_str='Pin - xPc',
                         )
        self.xQc = State(info='state of the reactive power lead-lag (state 2)',
                         tex_name='x_{Qc}',
                         t_const=self.Tbqe,
                         v_str='Qin',
                         e_str='Qin - xQc',
                         )
        self.Pctrl = Algeb(info='active power request, on MBase',
                           tex_name='P_{ctrl}',
                           v_str='Pin',
                           e_str='zlp * (Tbpe * xPc + Tap * (Pin - xPc) - Tbpe * Pctrl) + '
                                 '(1 - zlp) * (Pin - Pctrl)',
                           )
        self.Qctrl = Algeb(info='reactive power request, on MBase',
                           tex_name='Q_{ctrl}',
                           v_str='Qin',
                           e_str='zlq * (Tbqe * xQc + Taq * (Qin - xQc) - Tbqe * Qctrl) + '
                                 '(1 - zlq) * (Qin - Qctrl)',
                           )

        # --- voltage exponent and 1/Vfilt (CalculateIpvIqv): the exponent uses Vfilt itself, only the
        #     division is guarded at 0.001 pu. The exponent base is floored at `vfloor` (1e-9 pu by
        #     default) so that the Jacobian of (Vfilt/V0)**n stays finite at Vfilt = 0.
        self.vgl = Limiter(u=self.Vfilt, lower=self.vgd, upper=self.vgd,
                           no_upper=True, allow_adjust=False, no_warn=True,
                           info='divisor guard of Vfilt in the current commands',
                           )
        self.vgn = Limiter(u=self.Vfilt, lower=self.vfl, upper=self.vfl,
                           no_upper=True, allow_adjust=False, no_warn=True,
                           info='numerical floor of Vfilt in the voltage-exponent numerator',
                           )
        _vfg = '(Vfilt * (1 - vgl_zl) + vgd * vgl_zl)'
        _vfn = '(Vfilt * (1 - vgn_zl) + vfl * vgn_zl)'
        self.Ipcmd = Algeb(info='active current before limits (Ipvnolimit)',
                           tex_name='I_{pcmd}',
                           v_str=f'Pctrl * ({_vfn} / v0) ** nP / {_vfg}',
                           e_str=f'Pctrl * ({_vfn} / v0) ** nP / {_vfg} - Ipcmd',
                           discrete=(self.vgl, self.vgn),
                           )
        self.Iqcmd = Algeb(info='reactive current before limits (Iqvnolimit)',
                           tex_name='I_{qcmd}',
                           v_str=f'Qctrl * ({_vfn} / v0) ** nQ / {_vfg}',
                           e_str=f'Qctrl * ({_vfn} / v0) ** nQ / {_vfg} - Iqcmd',
                           discrete=(self.vgl, self.vgn),
                           )

        # --- limit enforcement in the order of the pseudo-code: Ip first, Iq gets the leftover ---
        self.LTpu = LessThan(u=self.Ipcmd, bound=self.Ipmaxu, info='Ipcmd < IpmaxUsed')
        self.Ipv1 = Algeb(info='active current after the upper limit',
                          tex_name='I_{pv1}',
                          v_str='LTpu_z1 * Ipcmd + LTpu_z0 * Ipmaxu',
                          e_str='LTpu_z1 * Ipcmd + LTpu_z0 * Ipmaxu - Ipv1',
                          discrete=self.LTpu,
                          )
        self.LTpl = LessThan(u=self.Ipv1, bound=self.Ipminu, info='Ipv1 < IpminUsed')
        self.Ipv = Algeb(info='limited active current',
                         tex_name='I_{pv}',
                         v_str='LTpl_z0 * Ipv1 + LTpl_z1 * Ipminu',
                         e_str='LTpl_z0 * Ipv1 + LTpl_z1 * Ipminu - Ipv',
                         discrete=self.LTpl,
                         )
        # Imax^2 - Ipv^2 is evaluated from the limited current *as selected by the flags* (identical to Ipv,
        # but exactly equal to the bound when a limit is active) so that the zero test `< 1e-12` of the
        # pseudo-code is not defeated by the Newton floor of a solved algebraic variable
        _ipvx = '(LTpl_z0 * (LTpu_z1 * Ipcmd + LTpu_z0 * Ipmaxu) + LTpl_z1 * Ipminu)'
        _dq = f'(Imax ** 2 - {_ipvx} ** 2)'
        self.dqs = VarService(v_str=_dq, tex_name='d_{q}',
                              info='Imax^2 - Ipv^2 from the flag-selected limited current (exact zero at a limit)')
        self.LTq = LessThan(u=self.dqs, bound=1e-12, info='Imax^2 - Ipv^2 < 1e-12: no current left for Iq')
        self.Iqlim = Algeb(info='current left for Iq, sqrt(Imax^2 - Ipv^2) (0 if less than 1e-12)',
                           tex_name='I_{qlim}',
                           v_str=f'LTq_z0 * sqrt(LTq_z0 * {_dq} + LTq_z1)',
                           e_str=f'LTq_z0 * sqrt(LTq_z0 * {_dq} + LTq_z1) - Iqlim',
                           discrete=(self.LTq, self.LTpu, self.LTpl),
                           )
        self.LTqx = LessThan(u=self.Iqlim, bound=self.Iqmax, info='Iqlim < Iqmax')
        self.LTqn = LessThan(u=self.Iqlim, bound=self.Iqminn, info='Iqlim < -Iqmin')
        # with Imax > 0: IqmaxUsed = min(Iqmax, Iqlim) and IqminUsed = max(Iqmin, -Iqlim), but both are
        # zero whenever no current is left (Imax^2 - Ipv^2 < 1e-12), whatever Iqmax and Iqmin are
        self.Iqmaxu = Algeb(info='IqmaxUsed = min(Iqmax, Iqlim) if Imax > 0 (0 if no current is left)',
                            tex_name='I_{qmax,u}',
                            v_str='zim * LTq_z0 * (LTqx_z1 * Iqlim + LTqx_z0 * Iqmax) + (1 - zim) * Iqmax',
                            e_str='zim * LTq_z0 * (LTqx_z1 * Iqlim + LTqx_z0 * Iqmax) + '
                                  '(1 - zim) * Iqmax - Iqmaxu',
                            discrete=(self.LTq, self.LTqx),
                            )
        self.Iqminu = Algeb(info='IqminUsed = max(Iqmin, -Iqlim) if Imax > 0 (0 if no current is left)',
                            tex_name='I_{qmin,u}',
                            v_str='zim * LTq_z0 * (LTqn_z0 * Iqmin - LTqn_z1 * Iqlim) + (1 - zim) * Iqmin',
                            e_str='zim * LTq_z0 * (LTqn_z0 * Iqmin - LTqn_z1 * Iqlim) + '
                                  '(1 - zim) * Iqmin - Iqminu',
                            discrete=(self.LTq, self.LTqn),
                            )
        self.LTqu = LessThan(u=self.Iqcmd, bound=self.Iqmaxu, info='Iqcmd < IqmaxUsed')
        self.Iqv1 = Algeb(info='reactive current after the upper limit',
                          tex_name='I_{qv1}',
                          v_str='LTqu_z1 * Iqcmd + LTqu_z0 * Iqmaxu',
                          e_str='LTqu_z1 * Iqcmd + LTqu_z0 * Iqmaxu - Iqv1',
                          discrete=self.LTqu,
                          )
        self.LTql = LessThan(u=self.Iqv1, bound=self.Iqminu, info='Iqv1 < IqminUsed')
        self.Iqv = Algeb(info='limited reactive current',
                         tex_name='I_{qv}',
                         v_str='LTql_z0 * Iqv1 + LTql_z1 * Iqminu',
                         e_str='LTql_z0 * Iqv1 + LTql_z1 * Iqminu - Iqv',
                         discrete=self.LTql,
                         )

        # --- states 7, 8: output lags; Iqextra enters after the Q limiter ---
        self.Ip = State(info='active current on MBase (state 7)',
                        tex_name='I_p',
                        t_const=self.Tte,
                        v_str='hFon * Ipv',
                        e_str='hFon * Ipv - Ip',
                        )
        self.Iq = State(info='reactive current on MBase (state 8)',
                        tex_name='I_q',
                        t_const=self.Tte,
                        v_str='hFon * (Iqv + Iqx)',
                        e_str='hFon * (Iqv + Iqx) - Iq',
                        )

        # --- recorded outputs of the cease/reconnect logic ---
        self.FracOn = Algeb(info='fraction of the load that remains connected',
                            tex_name='F_{on}',
                            v_str='hFon',
                            e_str='hFon - FracOn',
                            )
        self.Mode = Algeb(info='mode: 0 Monitor, 1 Ceased, 2 Ramp',
                          tex_name='M',
                          v_str='hMode',
                          e_str='hMode - Mode',
                          )
        self.CeaseLevel = Algeb(info='CeaseLevel: -1 none, 0 cease, 1 long cease, 2 never reconnect',
                                tex_name='CL',
                                v_str='hCL',
                                e_str='hCL - CeaseLevel',
                                )
        self.TdLevel = Algeb(info='TimerDelayActiveLevel: -1 inactive, else the level of the delay timer',
                             tex_name='TDL',
                             v_str='hTDL',
                             e_str='hTDL - TdLevel',
                             )
        self.TreconUsed = Algeb(info='TreconUsed (-1: never reconnect)',
                                tex_name='T_{recon,u}',
                                unit='s',
                                v_str='hTreconU',
                                e_str='hTreconU - TreconUsed',
                                )
        self.TrampUsed = Algeb(info='TrampUsed (already multiplied by Lfm)',
                               tex_name='T_{ramp,u}',
                               unit='s',
                               v_str='hTrampU',
                               e_str='hTrampU - TrampUsed',
                               )
        self.Tact = Algeb(info='active timers as a bitmask (see TIMER_BITS)',
                          tex_name='T_{act}',
                          v_str='0',
                          e_str='-Tact',
                          )
        self.Pmw = Algeb(info='active power consumed',
                         tex_name='P_{MW}',
                         unit='MW',
                         v_str=f'ue * sys_mva * {_pmw}',
                         e_str=f'ue * sys_mva * {_pmw} - Pmw',
                         discrete=self.LTi,
                         )
        self.Qmvar = Algeb(info='reactive power consumed',
                           tex_name='Q_{Mvar}',
                           unit='Mvar',
                           v_str=f'ue * sys_mva * {_qmvar}',
                           e_str=f'ue * sys_mva * {_qmvar} - Qmvar',
                           discrete=self.LTi,
                           )

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------

    def _timestep(self):
        """
        Return the ``TimeStep`` of the time-step rules, fixed at initialization.
        """
        tds = self.system.TDS
        if self.config.tstep > 0:
            return float(self.config.tstep)

        if (not tds.config.fixt) or tds.method.requires_variable_step:
            raise ValueError(f'{self.class_name}: variable-step integration requires an explicit '
                             f'TimeStep in {self.class_name}.config.tstep.')
        if tds.config.tstep <= 0:
            raise ValueError(f'{self.class_name}: TDS.config.tstep must be positive.')
        return float(tds.config.tstep)

    def _check_data(self):
        """
        Check parameters against the documented ranges.

        Data that makes the equations meaningless raises an error. Other
        violations are reported; the data modifications of the help page are
        applied afterwards in :py:meth:`_init_calc`.
        """
        cn = self.class_name
        idx = np.array(self.idx.v, dtype=object)
        online = self.u.v != 0

        def report(mask, text, error=False):
            mask = mask & online
            if not np.any(mask):
                return
            msg = f'{cn} {text}: {list(idx[mask])}'
            if error:
                raise ValueError(msg)
            logger.warning(msg)

        for name in ('Tt', 'Tv', 'Tf', 'Tvp', 'Tvq', 'Tbp', 'Tbq'):
            report(self.__dict__[name].v < 0, f'{name} < 0 (unstable lag)', error=True)
        report(self.Ipmin.v > self.Ipmax.v, 'Ipmin > Ipmax', error=True)
        report(self.Iqmin.v > self.Iqmax.v, 'Iqmin > Iqmax', error=True)

        report(self.Fcease.v < 0, 'Fcease < 0 (logic disabled)')
        report(self.Fcease.v > 1, 'Fcease > 1 (treated as 1)')
        report(self.Frecon.v < 0, 'Frecon < 0')
        report(self.Dbdl.v > 0, 'Dbdl > 0')
        report(self.Dbdh.v < 0, 'Dbdh < 0')
        report(self.Imax.v < 0, 'Imax < 0 (treated as not used)')
        report((self.Imax.v > 0) & (self.Ipmin.v > self.Imax.v), 'Ipmin > Imax (Ipv fixed at Ipmin)')
        report(self.VHrecon.v < self.VLrecon.v, 'VHrecon < VLrecon (VHrecon ignored)')
        for name in ('Tap', 'Taq', 'Tcease', 'Tdelay', 'Trecon', 'Tramp', 'TLrampL', 'TLreconL',
                     'TLrampH', 'TLreconH', 'TFLC1', 'TFHC1') + \
                tuple(f'{k}{i}' for k in ('TVLC', 'TVHC') for i in range(1, 5)):
            report(self.__dict__[name].v < 0, f'{name} < 0')

        # only one dynamic load may replace a PQ
        refs = OrderedDict()
        for mdl in self.system.exist.tds.values():
            for p in mdl.idx_params.values():
                if not p.replaces or p.model != 'StaticLoad':
                    continue
                for i in range(mdl.n):
                    if mdl.u.v[i] != 0 and p.v[i] is not None:
                        refs.setdefault(p.v[i], []).append(f'{mdl.class_name} {mdl.idx.v[i]}')
        dup = {pq: refs[pq] for pq in self.pq.v if len(refs.get(pq, [])) > 1}
        if len(dup):
            raise ValueError(f'{cn}: PQ replaced by more than one dynamic load: {dup}')

    def _logic_param_names(self):
        """PowerWorld names of the cease/reconnect parameters subject to the data modifications."""
        return ['Lfm', 'Fcease', 'Tramp', 'TLrampL', 'TLrampH', 'Trecon', 'TLreconL', 'TLreconH',
                'VLrecon', 'VHrecon', 'VLcease', 'VHcease'] + \
               [f'{k}{i}' for k in ('VLC', 'VHC') for i in range(1, 5)]

    def _init_calc(self, **kwargs):
        """
        Compute the initialization constants of all devices.

        Returns a flag array (1 dynamic model, 0 otherwise) and stores the
        constants in ``self._icd``.
        """
        cn = self.class_name
        cfg = self.config
        n = self.n
        ue = self.u.v.astype(float)

        self._check_data()

        p0 = np.array(self.p0.v, dtype=float)
        q0 = np.array(self.q0.v, dtype=float)
        v0 = np.array(self.v0.v, dtype=float)
        idx = np.array(self.idx.v, dtype=object)

        if np.any((v0 <= 0) & (ue != 0)):
            raise ValueError(f'{cn}: non-positive initial voltage at {list(idx[v0 <= 0])}')

        # host rule: loads below `pmin` are left to the static PQ model
        if cfg.smallload:
            small = (np.abs(p0) < cfg.pmin) & (ue != 0)
            if np.any(small):
                self.u.v[small] = 0
                self.ue.v[small] = 0
                ue = self.u.v.astype(float)
                logger.info('%s small loads (|P0| < %g pu) left to the PQ model: %s',
                            cn, cfg.pmin, list(idx[small]))

        # data modifications of the help page
        raw = {name: self.__dict__[name].v for name in self._logic_param_names()}
        mod = perc2_data_modifications(raw)
        changed = {}
        for name in ('VLrecon', 'VHrecon', 'VHcease', 'VHC1', 'VHC2', 'VHC3', 'VHC4', 'Fcease', 'Lfm'):
            diff = (mod[name] != raw[name]) & (ue != 0)
            if np.any(diff):
                changed[name] = {str(i): (float(a), float(b)) for i, a, b in
                                 zip(idx[diff], np.asarray(raw[name])[diff], mod[name][diff])}
        if changed:
            logger.info('%s data modifications (input -> used): %s', cn, changed)

        # constant impedance fallback
        neg = p0 < 0
        if np.any(neg & (ue != 0)):
            if cfg.pneg == 0:
                raise ValueError(f'{cn}: MWinit < 0 at {list(idx[neg & (ue != 0)])}')
            logger.warning('%s MWinit < 0 at %s: %s', cn, list(idx[neg & (ue != 0)]),
                           'constant impedance used' if cfg.pneg == 1 else 'negative MBase used')
        zfb = (v0 < self.VLcease.v) | (v0 < cfg.minvi) | (p0 == 0)
        if cfg.pneg == 1:
            zfb = zfb | neg
        if np.any(zfb & (ue != 0)):
            logger.info('%s replaced by constant impedance (V0 < VLcease, V0 < MinVoltILoad '
                        'or MWinit = 0): %s', cn, list(idx[zfb & (ue != 0)]))
        zdy = (~zfb).astype(float)
        zfb = zfb.astype(float)

        # loading factor, model base, MvarInit and Iqextra
        # (the unused dynamic part of constant-impedance devices is kept at zero)
        Lfe = mod['Lfm']
        kb = zdy * p0 / Lfe
        Pi0 = zdy * Lfe
        kb_safe = np.where(kb != 0, kb, 1.0)
        comp = self.comp.v != 0
        if cfg.qinit == 1:
            Qi0 = zdy * self.QPratio.v * Lfe
            Iqx = zdy * (~comp) * (q0 / kb_safe - Qi0) / v0
        else:
            Qi0 = zdy * np.where(comp, self.QPratio.v * Lfe, q0 / kb_safe)
            Iqx = np.zeros(n)

        comp_dyn = comp & (zdy != 0) & (ue != 0)
        if np.any(comp_dyn):
            dq = (q0 - self.QPratio.v * p0)[comp_dyn] * self.system.config.mva
            logger.warning('%s CompLoad components: Mvar mismatch Q0 - QPratio*MWinit to be '
                           'supplied by other devices: %s', cn,
                           dict(zip(idx[comp_dyn], np.round(dq, 6))))

        # time constants
        ts = self._timestep()
        Tte = _ts_minimum(self.Tt.v, 4 * ts, cfg.tstol)
        Tve, Tfe = self.Tv.v.copy(), self.Tf.v.copy()
        Tvpe, Tvqe = self.Tvp.v.copy(), self.Tvq.v.copy()
        Tbpe, Tbqe = self.Tbp.v.copy(), self.Tbq.v.copy()
        if cfg.tsrule in (1, 2):
            Tve = _ts_measurement(Tve, ts, cfg.tstol)
            if cfg.tsrule == 2:
                Tfe = _ts_measurement(Tfe, ts, cfg.tstol)
            Tvpe = np.where(Tvpe <= 2 * ts, 0.0, Tvpe)
            Tvqe = np.where(Tvqe <= 2 * ts, 0.0, Tvqe)
            Tbpe = np.where(Tbpe <= 2 * ts, 0.0, Tbpe)
            Tbqe = np.where(Tbqe <= 2 * ts, 0.0, Tbqe)

        # current limits with Imax
        Imax = np.array(self.Imax.v, dtype=float)
        zim = (Imax > 0).astype(float)
        Ipmaxu = np.where(zim != 0, np.minimum(self.Ipmax.v, Imax), self.Ipmax.v)
        Ipminu = np.where(zim != 0, np.maximum(self.Ipmin.v, -Imax), self.Ipmin.v)

        self._icd = OrderedDict(
            Lfe=Lfe, zfb=zfb, zdy=zdy, kb=kb, MB=kb * self.system.config.mva,
            Pi0=Pi0, Qi0=Qi0, Iqx=Iqx, Tte=Tte, Tve=Tve, Tfe=Tfe,
            Tvpe=Tvpe, Tvqe=Tvqe, Tbpe=Tbpe, Tbqe=Tbqe,
            zwp=(Tvpe > 0).astype(float), zwq=(Tvqe > 0).astype(float),
            zlp=(Tbpe > 0).astype(float), zlq=(Tbqe > 0).astype(float),
            zdb0=(self.Dbdh.v == self.Dbdl.v).astype(float),
            Fce=mod['Fcease'], Trampe=mod['Tramp'], Trecone=mod['Trecon'],
            TLrampLe=mod['TLrampL'], TLreconLe=mod['TLreconL'],
            TLrampHe=mod['TLrampH'], TLreconHe=mod['TLreconH'],
            VLrecone=mod['VLrecon'], VHrecone=mod['VHrecon'], VHceasee=mod['VHcease'],
            VHC1e=mod['VHC1'], VHC2e=mod['VHC2'], VHC3e=mod['VHC3'], VHC4e=mod['VHC4'],
            TrampU0=mod['Tramp'] * Lfe,
            zim=zim, Ipmaxu=Ipmaxu, Ipminu=Ipminu,
        )
        self._timestep_used = ts

        self._check_init_limits(zdy, ue, Pi0, Qi0, v0, Imax, Ipmaxu, Ipminu)

        return zdy * ue

    def _check_init_limits(self, zdy, ue, Pi0, Qi0, v0, Imax, Ipmaxu, Ipminu):
        """
        Report devices whose initial current commands violate the limits
        (including the total current limit Imax).

        Limits are not adjusted; the initialization test of TDS will fail.
        """
        Ip0 = Pi0 / v0
        Iq0 = Qi0 / v0
        Ipv0 = np.minimum(np.maximum(Ip0, Ipminu), Ipmaxu)
        Iqmaxu, Iqminu = np.array(self.Iqmax.v, dtype=float), np.array(self.Iqmin.v, dtype=float)
        use = Imax > 0
        noleft = use & (Imax ** 2 - Ipv0 ** 2 < 1e-12)
        left = np.sqrt(np.maximum(Imax ** 2 - Ipv0 ** 2, 0.0))
        Iqmaxu = np.where(noleft, 0.0, np.where(use, np.minimum(Iqmaxu, left), Iqmaxu))
        Iqminu = np.where(noleft, 0.0, np.where(use, np.maximum(Iqminu, -left), Iqminu))
        bad = (zdy != 0) & (ue != 0) & ((Ip0 > Ipmaxu) | (Ip0 < Ipminu) | (Iq0 > Iqmaxu) | (Iq0 < Iqminu))
        if not np.any(bad):
            return

        pos = np.where(bad)[0]
        tab = Tab(title=f'{self.class_name}: initial current commands out of the limits used '
                        '(limits NOT adjusted)',
                  header=['idx', 'V0', 'Ipcmd0', 'IpminUsed', 'IpmaxUsed', 'Iqcmd0', 'IqminUsed', 'IqmaxUsed'],
                  data=[[self.idx.v[i], v0[i], Ip0[i], Ipminu[i], Ipmaxu[i], Iq0[i], Iqminu[i], Iqmaxu[i]]
                        for i in pos],
                  )
        logger.warning(tab.draw())

    # ------------------------------------------------------------------
    # cease/reconnect logic
    # ------------------------------------------------------------------

    def logic_params(self):
        """
        Return the :py:class:`PERC2Params` of all devices as used by the
        cease/reconnect logic (effective values after the data modifications;
        read live, so an ``Alter`` of a raw time-delay or threshold parameter
        that is not modified takes effect, and of the effective services
        ``VLrecone`` etc. for the modified ones).
        """
        d = {}
        for name in ('VLMflag', 'VHMflag', 'VLLflag', 'VHLflag', 'FLflag', 'FHflag',
                     'Frecon', 'Tdelay', 'Tcease', 'VLcease', 'FLC1', 'TFLC1', 'FHC1', 'TFHC1'):
            d[name] = self.__dict__[name].v
        for name, src in (('Lfm', 'Lfe'), ('Fcease', 'Fce'), ('Tramp', 'Trampe'), ('Trecon', 'Trecone'),
                          ('TLrampL', 'TLrampLe'), ('TLreconL', 'TLreconLe'),
                          ('TLrampH', 'TLrampHe'), ('TLreconH', 'TLreconHe'),
                          ('VHcease', 'VHceasee'), ('VLrecon', 'VLrecone'), ('VHrecon', 'VHrecone')):
            d[name] = self.__dict__[src].v
        for i in range(1, 5):
            d[f'VLC{i}'] = self.__dict__[f'VLC{i}'].v
            d[f'TVLC{i}'] = self.__dict__[f'TVLC{i}'].v
            d[f'VHC{i}'] = self.__dict__[f'VHC{i}e'].v
            d[f'TVHC{i}'] = self.__dict__[f'TVHC{i}'].v
        return params_from_dict(d)

    def held_state(self):
        """
        Return the held :py:class:`PERC2State` (copies of the held services).
        """
        g = lambda name: np.array(self.__dict__[name].v, dtype=float)  # NOQA
        stack = lambda prefix, suffix='': np.column_stack([g(f'{prefix}{i}{suffix}') for i in range(1, 5)])  # NOQA
        return PERC2State(mode=g('hMode'), cl=g('hCL'), tdl=g('hTDL'), tdelay=g('hTdelay'), tsr=g('hTsr'),
                          trampu=g('hTrampU'), treconu=g('hTreconU'), trec=g('hTrec'), treca=g('hTrecA'),
                          tvl=g('hTvl'), tvla=g('hTvlA'), tvh=g('hTvh'), tvha=g('hTvhA'),
                          tlv=stack('hTlv'), tlva=stack('hTlv', 'A'), thv=stack('hThv'), thva=stack('hThv', 'A'),
                          tlf=g('hTlf'), tlfa=g('hTlfA'), thf=g('hThf'), thfa=g('hThfA'), fon=g('hFon'))

    def _store_state(self, S):
        """Write a :py:class:`PERC2State` into the held services."""
        for name, val in (('hMode', S.mode), ('hCL', S.cl), ('hTDL', S.tdl), ('hTdelay', S.tdelay),
                          ('hTsr', S.tsr), ('hTrampU', S.trampu), ('hTreconU', S.treconu),
                          ('hTrec', S.trec), ('hTrecA', S.treca), ('hTvl', S.tvl), ('hTvlA', S.tvla),
                          ('hTvh', S.tvh), ('hTvhA', S.tvha), ('hTlf', S.tlf), ('hTlfA', S.tlfa),
                          ('hThf', S.thf), ('hThfA', S.thfa), ('hFon', S.fon)):
            self.__dict__[name].v[:] = val
        for i in range(4):
            self.__dict__[f'hTlv{i + 1}'].v[:] = S.tlv[:, i]
            self.__dict__[f'hTlv{i + 1}A'].v[:] = S.tlva[:, i]
            self.__dict__[f'hThv{i + 1}'].v[:] = S.thv[:, i]
            self.__dict__[f'hThv{i + 1}A'].v[:] = S.thva[:, i]

    def init(self, routine, **kwargs):
        """
        Initialize the model; the held ``TrampUsed`` and ``TreconUsed`` are
        set to ``Tramp*Lfm`` and ``Trecon`` as in the pseudo-code.
        """
        ret = super().init(routine, **kwargs)
        if self.n and routine == 'tds':
            self.hTrampU.v[:] = self.TrampU0.v
            self.hTreconU.v[:] = self.Trecone.v
            for var, val in ((self.TrampUsed, self.TrampU0.v), (self.TreconUsed, self.Trecone.v)):
                var.v[:] = val
                self.system.dae.y[var.a] = val
        return ret

    def b_update(self):
        """
        Evaluate the observables, then advance the cease/reconnect logic.

        ``TDS.run()`` calls ``b_update`` of every TDS model exactly once after
        each accepted step, with the converged solution at ``dae.t`` and
        before the step is stored; ``TDS.init()`` and ``TDS.reinit()`` call it
        once at t = 0. Newton iterations, line search trials and rejected
        steps do not call it, so the pseudo-code is executed once per
        accepted step.
        """
        ret = super().b_update()
        self._commit_step()
        return ret

    def _commit_step(self):
        """
        Execute the cease/reconnect pseudo-code for the accepted step at
        ``dae.t`` with the converged ``Vfilt`` and ``Ffilt``, and hold the new
        ``FracOn`` for the next step.

        The recorded outputs and the derivatives of ``Ip`` and ``Iq`` (``f0``
        of the next trapezoidal step) are refreshed, and a jump of ``FracOn``
        is flagged to TDS as an event. A repeated call at the same time does
        nothing, and nothing is done when TDS replays results from a CSV file.

        Returns
        -------
        bool
            True if the logic was executed.
        """
        system = self.system
        dae = system.dae

        if system.TDS.data_csv is not None:
            return False
        t = float(dae.t)
        if self.hTlast.v[0] == t:
            return False
        self.hTlast.v[:] = t

        old = self.held_state()
        active = (self.ue.v != 0) & (self.zdy.v != 0)
        ffilt = self.fnom.v * (1.0 + self.wfilt.v)

        new = perc2_fracon_step(old, t, self.Vfilt.v, ffilt, self.logic_params(),
                                active=active, ttol=self.config.ttol)

        fon_changed = new.fon != old.fon
        jumped = fon_changed & (new.mode != old.mode)

        self._store_state(new)

        for var, val in ((self.FracOn, new.fon), (self.Mode, new.mode), (self.CeaseLevel, new.cl),
                         (self.TdLevel, new.tdl), (self.TreconUsed, new.treconu),
                         (self.TrampUsed, new.trampu), (self.Tact, perc2_timer_bitmask(new))):
            var.v[:] = val
            dae.y[var.a] = val

        if np.any(fon_changed):
            dae.f[self.Ip.a] = self.hFon.v * self.Ipv.v - self.Ip.v
            dae.f[self.Iq.a] = self.hFon.v * (self.Iqv.v + self.Iqx.v) - self.Iq.v

        if np.any(jumped):
            system.TDS.custom_event = True

        return True


def _ts_minimum(T, tmin, rtol):
    """
    Return `T`, or `tmin` if `T` is below `tmin` by more than ``rtol * tmin``.

    PowerWorld checks minimum limits of time constants with a tolerance of
    0.05% of the limit (Simulator 23 release notes).
    """
    return np.where(T < tmin * (1 - rtol), tmin, T)


def _ts_measurement(T, ts, rtol):
    """
    LDEV1-type rule for a measurement time constant: treated as 0 if smaller
    than ``2 * TimeStep``, and as ``4 * TimeStep`` if smaller than that.
    """
    return np.where(T < 2 * ts, 0.0, _ts_minimum(T, 4 * ts, rtol))


class PERC2(PERC2Data, PERC2Model):
    r"""
    PERC2 (Power Electronic Reconnecting and Ceasing, version 2) load model.

    PERC2 replaces the linked PQ load and models an aggregation of power
    electronic loads as a current injection that passes through voltage
    washouts, a frequency droop with deadband, lead-lags, a voltage exponent,
    current limits (including the total current limit ``Imax``), the
    cease/reconnect multiplier ``FracOn`` and an output lag. It reproduces
    the PowerWorld Simulator 24 model of the same name (July 28, 2026 patch
    with the September 25, 2026 flags).

    Per-unit quantities are on the model base ``MBase = MWinit / Lfm`` [MVA]
    (``Lfm < 0.001`` is taken as 1.0), where ``MWinit`` and ``Mvarinit`` are
    the actual PQ powers at the power flow solution (``PQ.Ppf``, ``PQ.Qpf``).
    The network sees ``P = MBase * V * Ip`` and ``Q = MBase * V * Iq``.

    Correspondence to the PowerWorld block diagram:

    ========= ============== =========================================
    PW state  ANDES          Notes
    ========= ============== =========================================
    1 Pctrl   ``xPc``        lag state; lead-lag output is ``Pctrl``
    2 Qctrl   ``xQc``        lag state; lead-lag output is ``Qctrl``
    3 dP      ``xdP``        washout state; washout output is ``dP``
    4 dQ      ``xdQ``        washout state; washout output is ``dQ``
    5 wfilt   ``wfilt``      ``Ffilt = fnom (1 + wfilt)`` [Hz]
    6 Vfilt   ``Vfilt``
    7 Ip      ``Ip``
    8 Iq      ``Iq``
    ========= ============== =========================================

    Others: ``Pmw``, ``Qmvar``, ``Ipv``, ``Iqv``, ``FracOn``, and the logic
    outputs ``Mode``, ``CeaseLevel``, ``TdLevel`` (TimerDelayActiveLevel),
    ``TreconUsed``, ``TrampUsed`` and ``Tact`` (bitmask of the active timers,
    see :py:data:`TIMER_BITS`).

    Current limit enforcement (help-page procedure ``CalculateIpvIqv``):
    ``Ipcmd = Pctrl (Vfilt/V0)^nP / Vfilt`` and ``Iqcmd = Qctrl (Vfilt/V0)^nQ
    / Vfilt``, where only the divisor is guarded at ``config.vguard`` (0.001
    pu) as in the pseudo-code; the exponent uses ``Vfilt`` itself, floored at
    the numerical ``config.vfloor`` (1e-9 pu) so that the Jacobian stays
    finite at zero voltage. Ip is limited to ``[max(Ipmin, -Imax),
    min(Ipmax, Imax)]`` first, then Iq to ``[max(Iqmin, -Iqlim), min(Iqmax,
    Iqlim)]`` with ``Iqlim = sqrt(Imax^2 - Ipv^2)``; if ``Imax^2 - Ipv^2 <
    1e-12`` no current is left and both Iq bounds are zero whatever ``Iqmax``
    and ``Iqmin`` are. ``Imax <= 0`` disables the total limit. The upper
    check is applied before the lower one, as in the pseudo-code.
    ``Iqextra`` is added after the limiter.

    Cease/reconnect logic (:py:func:`perc2_fracon_step`): the pseudo-code of
    the help page runs once after each accepted time step with the converged
    ``Vfilt`` and ``Ffilt`` from :py:meth:`b_update`; the new ``FracOn`` is
    held during the next step. Four cease zones with their own reconnection
    delay and ramp time: the cease zone (``Vfilt < VLcease`` or ``Vfilt >
    VHcease`` for ``Tcease``; ``Trecon``, ``Tramp*Lfm``), the long low
    (``VLC1..4/TVLC1..4``; ``TLreconL``, ``TLrampL*Lfm``) and long high
    (``VHC1..4/TVHC1..4``; ``TLreconH``, ``TLrampH*Lfm``) zones, and the
    frequency zone (``Ffilt < FLC1`` for ``TFLC1`` or ``Ffilt > FHC1`` for
    ``TFHC1``; never reconnects). Higher-level timers keep counting after a
    lower-level cease and escalate ``CeaseLevel`` and the times used.
    Reconnection requires ``VLrecon <= Vfilt <= VHrecon`` for ``TreconUsed``.
    All ramp times are multiplied by ``Lfm`` (``TrampUsed = Tramp*Lfm``).

    Data modifications of the help page are applied at initialization
    (:py:func:`perc2_data_modifications`): ``Fcease <= 1``; negative ramp and
    reconnection times are zero; ``VLrecon`` is raised to the largest low
    cease threshold; non-positive high thresholds are ignored (999);
    ``VHrecon`` is lowered to the smallest high cease threshold and ignored
    if below ``VLrecon``. With the default data, the effective reconnection
    window is therefore ``[0.90, 1.10]`` pu. The effective values are the
    services ``VLrecone``, ``VHrecone``, ``VHceasee``, ``VHC1e..VHC4e``,
    ``Fce``, ``Trampe`` etc.

    Initialization and host rules (as PERC1):

    - If ``V0 < VLcease``, ``V0 < MinVoltILoad`` (``config.minvi``) or
      ``MWinit = 0``, the model is ignored and the load is a constant
      impedance. ``config.pneg`` selects the handling of ``MWinit < 0``.
    - ``config.qinit = 1`` (default, block diagram): ``MvarInit = QPratio *
      MWinit`` and stand-alone devices (``comp = 0``) add ``Iqextra = (Q0 -
      MvarInit) / (MBase V0)`` after the reactive current limiter.
      ``config.qinit = 0`` (parameter-table text): stand-alone devices use
      ``MvarInit = Q0`` and ``Iqextra = 0``. CompLoad components (``comp =
      1``) always use ``QPratio`` with ``Iqextra = 0``.
    - ``Tt`` is treated as ``4 * TimeStep`` if smaller (by more than
      ``config.tstol`` of the limit). ``TimeStep`` is ``config.tstep`` or,
      if zero, ``TDS.config.tstep`` at initialization.
    - With ``config.smallload = 1``, loads with ``|P0|`` below
      ``config.pmin`` (system base) are not replaced.
    - Current limits are never adjusted to the initial values. Initial
      current commands outside the limits are reported, and the TDS
      initialization test fails.

    The frequency input is ``f - 1`` of the linked ``BusFreq`` plus the
    service ``dwext`` (zero by default), ignored when ``V < config.vfmin``
    (0.7 pu). ``Ffilt = sys_f (1 + wfilt)`` enters the frequency zone.
    """

    def __init__(self, system, config):
        PERC2Data.__init__(self)
        PERC2Model.__init__(self, system, config)
