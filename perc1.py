"""
PERC1: Power Electronic Reconnecting and Ceasing load model.

Positive-sequence model for aggregations of power-electronic loads (data
centers, EV chargers, crypto-mining, variable-frequency drives), reproducing
PowerWorld Simulator 24's ``Load Characteristic PERC1`` as documented in the
PowerWorld help page of September 2026, i.e., including the 2026 patches of
February 3 (a CompLoad with a single PERC1 component is a stand-alone model),
February 11 (``MvarInit = QPratio * MWInit`` with ``Iqextra``), February 12
(default ``Kvp = Kvq = Kdroop = 0``), July 17 (``Frecon > 1`` allowed; the
help page says July 14) and August 6 (frequency input ignored below 0.7 pu
terminal voltage).
"""

import logging
from collections import OrderedDict, namedtuple

import numpy as np

from andes.core import (ModelData, IdxParam, NumParam, Model, ExtParam,
                        ExtService, ExtAlgeb, ConstService, State, Algeb,
                        DeadBand, LessThan, Limiter)
from andes.core.service import DeviceFinder
from andes.utils.tab import Tab

logger = logging.getLogger(__name__)

# PowerWorld mode codes (``vtm_*``) and timer codes (``vtt_*``)
VTM_MONITOR, VTM_CEASED, VTM_RAMP = 0.0, 1.0, 2.0
VTT_NONE, VTT_TCEASE, VTT_TDELAY, VTT_TRECON = 0.0, 1.0, 2.0, 3.0

PERC1State = namedtuple('PERC1State', ['mode', 'tact', 'timer', 'tsr', 'fon'])
PERC1State.__doc__ = """
State of the PERC1 cease/reconnect logic: ``Mode``, ``TimerActive``,
``Timer``, ``TStartRamp`` and ``FracOn`` of the PowerWorld pseudo-code.
"""


def perc1_initial_state(n):
    """
    Return the pre-simulation state of the cease/reconnect logic for `n`
    devices: ``FracOn = 1``, ``Mode = vtm_Monitor``, ``TimerActive =
    vtt_None``, ``Timer = 0`` and ``TStartRamp = 0``.
    """
    return PERC1State(np.full(n, VTM_MONITOR), np.full(n, VTT_NONE),
                      np.zeros(n), np.zeros(n), np.ones(n))


def perc1_fracon_step(state, t, vfilt, Fcease, Vcease, Tcease, Tdelay,
                      Frecon, Vrecon, Trecon, Tramp, active=None, ttol=1e-8):
    """
    Execute the PERC1 cease/reconnect pseudo-code once, for all devices.

    This is a pure function: `state` is not modified, and the new state is
    returned. The statements keep the order of the PowerWorld pseudo-code:
    the ramp is updated first (unless frozen by an active Tdelay timer), then
    the timers of the present mode are processed. A device found in Ceased
    mode by the mode test enters the reconnect branch; a device that has just
    entered Ceased in the same execution does not.

    Parameters
    ----------
    state : PERC1State
        Present ``(mode, tact, timer, tsr, fon)`` arrays.
    t : float
        Present simulation time, ``PresentTime`` [s].
    vfilt : array-like
        Filtered voltage ``Vfilt`` [pu].
    Fcease, Vcease, Tcease, Tdelay, Frecon, Vrecon, Trecon, Tramp : array-like
        PERC1 parameters (scalars are broadcast).
    active : array-like of bool, optional
        Devices on which the logic is executed. Others keep their state.
    ttol : float
        Tolerance [s] of the time comparisons ``a >= b``, evaluated as
        ``a >= b - ttol`` so that elapsed times accumulated with rounding
        errors (e.g., ``t += h``) expire on the intended step.
        ``ttol = 0`` gives the literal comparison.

    Returns
    -------
    PERC1State
        The new state.
    """
    mode = np.array(state.mode, dtype=float)
    tact = np.array(state.tact, dtype=float)
    timer = np.array(state.timer, dtype=float)
    tsr = np.array(state.tsr, dtype=float)
    fon = np.array(state.fon, dtype=float)
    n = mode.shape[0]

    def _b(x):
        return np.broadcast_to(np.asarray(x, dtype=float), (n,))

    vfilt, Fc, Vc, Tc, Td, Fr, Vr, Tr, Tp = map(
        _b, (vfilt, Fcease, Vcease, Tcease, Tdelay, Frecon, Vrecon, Trecon, Tramp))
    t = float(t)

    def _ge(a, b):
        return a >= b - ttol

    # "If Fcease > 0 Then": ignore all the logic if Fcease <= 0
    run = Fc > 0.0
    if active is not None:
        run = run & np.asarray(active, dtype=bool)

    f_ceased = 1.0 - Fc
    f_final = 1.0 - Fc + Fr * Fc

    # --- update FracOn while ramping; frozen during an active Tdelay timer ---
    ramp = run & (mode == VTM_RAMP) & (tact != VTT_TDELAY)
    done = ramp & _ge(t, tsr + Tp)
    going = ramp & ~done
    fon[done] = f_final[done]
    mode[done] = VTM_MONITOR
    if np.any(going):
        frac = (t - tsr[going]) / Tp[going]
        fon[going] = 1.0 - Fc[going] + Fr[going] * Fc[going] * frac

    # --- the mode test: Monitor and Ramp count Tcease/Tdelay the same way ---
    not_ceased = run & (mode != VTM_CEASED)
    ceased = run & (mode == VTM_CEASED)

    check = not_ceased & (tact != VTT_TDELAY)
    v_ok = vfilt >= Vc
    tact[check & v_ok] = VTT_NONE
    start = check & ~v_ok & (tact != VTT_TCEASE)
    timer[start] = t
    tact[start] = VTT_TCEASE

    expired = not_ceased & (tact == VTT_TCEASE) & _ge(t - timer, Tc)
    timer[expired] = t
    tact[expired] = VTT_TDELAY

    expired = not_ceased & (tact == VTT_TDELAY) & _ge(t - timer, Td)
    tact[expired] = VTT_NONE
    mode[expired] = VTM_CEASED
    fon[expired] = f_ceased[expired]

    # --- Ceased: Trecon timer, then ramp (or jump) back ---
    v_low = vfilt < Vr
    tact[ceased & v_low] = VTT_NONE
    start = ceased & ~v_low & (tact != VTT_TRECON)
    timer[start] = t
    tact[start] = VTT_TRECON

    expired = ceased & (tact == VTT_TRECON) & _ge(t - timer, Tr)
    tact[expired] = VTT_NONE
    jump = expired & (Tp == 0.0)
    mode[jump] = VTM_MONITOR
    fon[jump] = f_final[jump]
    ramp_start = expired & (Tp != 0.0)
    mode[ramp_start] = VTM_RAMP
    tsr[ramp_start] = t

    return PERC1State(mode, tact, timer, tsr, fon)


class PERC1Data(ModelData):
    """
    Data for the PERC1 load model.
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

        self.Lfm = NumParam(default=0.8,
                            info='loading factor, MBase = MWinit / Lfm (0.8 if Lfm < 0.001)',
                            tex_name='L_{fm}',
                            )
        self.QPratio = NumParam(default=0.66,
                                info='Q/P ratio, MvarInit = QPratio * MWinit',
                                tex_name='QP_{ratio}',
                                )
        self.Dbfl = NumParam(default=0.0,
                             info='frequency deadband low (<= 0)',
                             tex_name='d_{bfl}',
                             unit='p.u.',
                             )
        self.Dbfh = NumParam(default=0.0,
                             info='frequency deadband high (>= 0)',
                             tex_name='d_{bfh}',
                             unit='p.u.',
                             )
        self.Kdroop = NumParam(default=0.0,
                               info='frequency droop',
                               tex_name='K_{droop}',
                               unit='p.u.',
                               )
        self.Kvp = NumParam(default=0.0,
                            info='gain of active power washout',
                            tex_name='K_{vp}',
                            )
        self.Tvp = NumParam(default=0.1,
                            info='time constant of active power washout',
                            tex_name='T_{vp}',
                            unit='s',
                            )
        self.Kvq = NumParam(default=0.0,
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
        self.Fcease = NumParam(default=1.0,
                               info='fraction that will cease (between 0 and 1)',
                               tex_name='F_{cease}',
                               )
        self.Vcease = NumParam(default=0.5,
                               info='voltage threshold for cease logic',
                               tex_name='V_{cease}',
                               unit='p.u.',
                               )
        self.Tcease = NumParam(default=0.01,
                               info='time delay for cease logic to be initiated (>= 0)',
                               tex_name='T_{cease}',
                               unit='s',
                               )
        self.Tdelay = NumParam(default=0.0,
                               info='time delay to cease after detection (>= 0)',
                               tex_name='T_{delay}',
                               unit='s',
                               )
        self.Vrecon = NumParam(default=0.6,
                               info='voltage threshold to initiate reconnection (>= Vcease)',
                               tex_name='V_{recon}',
                               unit='p.u.',
                               )
        self.Trecon = NumParam(default=0.05,
                               info='time delay for ramp up logic to be initiated (>= 0)',
                               tex_name='T_{recon}',
                               unit='s',
                               )
        self.Tramp = NumParam(default=1.0,
                              info='ramp up time (>= 0)',
                              tex_name='T_{ramp}',
                              unit='s',
                              )
        self.Frecon = NumParam(default=1.0,
                               info='fraction of the ceased load that reconnects (>= 0; > 1 allowed)',
                               tex_name='F_{recon}',
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


class PERC1Model(Model):
    """
    Implementation of the PERC1 load model.
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
                                     ('ttol', 1e-8),
                                     )))
        self.config.add_extra("_help",
                              qinit='1: MvarInit = QPratio*MWinit, Iqextra for stand-alone '
                                    '(Feb-11-2026 patch); 0: legacy, MvarInit = initial Mvar',
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
                              vguard='lower guard of Vfilt in the current commands [pu]',
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
        self.vfm = ConstService(v_str='vfmin', tex_name='V_{f,min}',
                                info='voltage below which the frequency input is ignored')
        self.vgd = ConstService(v_str='vguard', tex_name='V_{guard}',
                                info='lower guard of Vfilt in the current commands')
        self.vmi = ConstService(v_str='ilow * minvi', tex_name='V_{min,I}',
                                info='voltage below which the network current falls off')
        self.vmis = ConstService(v_str='vmi + Indicator(vmi <= 0)', tex_name='V_{min,I}^s',
                                 info='nonzero divisor for the current fall-off')

        # --- cease/reconnect logic, committed once per accepted TDS step ---
        self.hMode = ConstService(v_str='0', tex_name='M_h', info='held Mode')
        self.hTact = ConstService(v_str='0', tex_name='T_{act,h}', info='held TimerActive')
        self.hTimer = ConstService(v_str='0', tex_name='T_{mr,h}', info='held Timer')
        self.hTsr = ConstService(v_str='0', tex_name='T_{sr,h}', info='held TStartRamp')
        self.hFon = ConstService(v_str='1', tex_name='F_{on,h}', info='held FracOn')
        self.hTlast = ConstService(v_str='-1', tex_name='t_{last}',
                                   info='time of the last execution of the logic')

        # external frequency deviation input (0 unless set by the user)
        self.dwext = ConstService(v_str='0', tex_name=r'\Delta\omega_{ext}',
                                  info='external frequency deviation added to f - 1')

        # --- network interface: current injection (consumption positive) ---
        # `_gv` is 1 unless the optional fall-off below MinVoltILoad is enabled
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

        self.dbw = DeadBand(u=self.wfilt, center=0.0, lower=self.Dbfl, upper=self.Dbfh,
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
        _fdb = '(dbw_zu * (wfilt - Dbfh) + dbw_zl * (wfilt - Dbfl) + zdb0 * dbw_zi * (wfilt - Dbfh))'
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
                           )   # Q: When Tap≠0, but Tbpe is zero or is set to zero by a rule, the code will also bypass the entire process.
        self.Qctrl = Algeb(info='reactive power request, on MBase',
                           tex_name='Q_{ctrl}',
                           v_str='Qin',
                           e_str='zlq * (Tbqe * xQc + Taq * (Qin - xQc) - Tbqe * Qctrl) + '
                                 '(1 - zlq) * (Qin - Qctrl)',
                           )

        # --- voltage exponent and 1/Vfilt, with a lower guard on Vfilt ---
        self.vgl = Limiter(u=self.Vfilt, lower=self.vgd, upper=self.vgd,
                           no_upper=True, allow_adjust=False, no_warn=True,
                           info='lower guard of Vfilt in the current commands',
                           )
        _vfg = '(Vfilt * (1 - vgl_zl) + vgd * vgl_zl)'
        self.Ipcmd = Algeb(info='active current command before limits',
                           tex_name='I_{pcmd}',
                           v_str=f'Pctrl * {_vfg} ** (nP - 1) / v0 ** nP',
                           e_str=f'Pctrl * {_vfg} ** (nP - 1) / v0 ** nP - Ipcmd',
                           discrete=self.vgl,
                           )
        self.Iqcmd = Algeb(info='reactive current command before limits',
                           tex_name='I_{qcmd}',
                           v_str=f'Qctrl * {_vfg} ** (nQ - 1) / v0 ** nQ',
                           e_str=f'Qctrl * {_vfg} ** (nQ - 1) / v0 ** nQ - Iqcmd',
                           discrete=self.vgl,
                           )

        # the limits are never adjusted to the initial values
        self.ipl = Limiter(u=self.Ipcmd, lower=self.Ipmin, upper=self.Ipmax,
                           allow_adjust=False,
                           info='active current limiter',
                           )
        self.iql = Limiter(u=self.Iqcmd, lower=self.Iqmin, upper=self.Iqmax,
                           allow_adjust=False,
                           info='reactive current limiter',
                           )
        self.Ipv = Algeb(info='limited active current',
                         tex_name='I_{pv}',
                         v_str='ipl_zi * Ipcmd + ipl_zu * Ipmax + ipl_zl * Ipmin',
                         e_str='ipl_zi * Ipcmd + ipl_zu * Ipmax + ipl_zl * Ipmin - Ipv',
                         discrete=self.ipl,
                         )
        self.Iqv = Algeb(info='limited reactive current',
                         tex_name='I_{qv}',
                         v_str='iql_zi * Iqcmd + iql_zu * Iqmax + iql_zl * Iqmin',
                         e_str='iql_zi * Iqcmd + iql_zu * Iqmax + iql_zl * Iqmin - Iqv',
                         discrete=self.iql,
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

        # --- recorded outputs ---
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
        self.TimerActive = Algeb(info='active timer: 0 None, 1 Tcease, 2 Tdelay, 3 Trecon',
                                 tex_name='T_{act}',
                                 v_str='hTact',
                                 e_str='hTact - TimerActive',
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
        violations are reported and used as given.
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

        report((self.Fcease.v < 0) | (self.Fcease.v > 1), 'Fcease out of [0, 1]')
        report(self.Frecon.v < 0, 'Frecon < 0')
        report(self.Vrecon.v < self.Vcease.v, 'Vrecon < Vcease')
        report(self.Dbfl.v > 0, 'Dbfl > 0')
        report(self.Dbfh.v < 0, 'Dbfh < 0')
        for name in ('Tap', 'Taq', 'Tcease', 'Tdelay', 'Trecon', 'Tramp'):
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

        # constant impedance fallback
        neg = p0 < 0
        if np.any(neg & (ue != 0)):
            if cfg.pneg == 0:
                raise ValueError(f'{cn}: MWinit < 0 at {list(idx[neg & (ue != 0)])}')
            logger.warning('%s MWinit < 0 at %s: %s', cn, list(idx[neg & (ue != 0)]),
                           'constant impedance used' if cfg.pneg == 1 else 'negative MBase used')
        zfb = (v0 < self.Vcease.v) | (v0 < cfg.minvi) | (p0 == 0)
        if cfg.pneg == 1:
            zfb = zfb | neg
        if np.any(zfb & (ue != 0)):
            logger.info('%s replaced by constant impedance (V0 < Vcease, V0 < MinVoltILoad '
                        'or MWinit = 0): %s', cn, list(idx[zfb & (ue != 0)]))
        zdy = (~zfb).astype(float)
        zfb = zfb.astype(float)

        # loading factor, model base, MvarInit and Iqextra
        # (the unused dynamic part of constant-impedance devices is kept at zero)
        Lfe = np.where(self.Lfm.v < 0.001, 0.8, self.Lfm.v)
        kb = zdy * p0 / Lfe
        Pi0 = zdy * Lfe
        kb_safe = np.where(kb != 0, kb, 1.0)
        if cfg.qinit == 1:
            Qi0 = zdy * self.QPratio.v * Lfe
            Iqx = zdy * (1 - self.comp.v) * (q0 / kb_safe - Qi0) / v0
        else:
            Qi0 = zdy * q0 / kb_safe
            Iqx = np.zeros(n)
            if np.any(self.comp.v != 0):
                logger.warning('%s: `comp` is ignored with the legacy `qinit = 0`.', cn)

        comp = (self.comp.v != 0) & (zdy != 0) & (ue != 0)
        if cfg.qinit == 1 and np.any(comp):
            dq = (q0 - self.QPratio.v * p0)[comp] * self.system.config.mva
            logger.warning('%s CompLoad components: Mvar mismatch Q0 - QPratio*MWinit to be '
                           'supplied by other devices: %s', cn,
                           dict(zip(idx[comp], np.round(dq, 6))))

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

        self._icd = OrderedDict(
            Lfe=Lfe, zfb=zfb, zdy=zdy, kb=kb, MB=kb * self.system.config.mva,
            Pi0=Pi0, Qi0=Qi0, Iqx=Iqx, Tte=Tte, Tve=Tve, Tfe=Tfe,
            Tvpe=Tvpe, Tvqe=Tvqe, Tbpe=Tbpe, Tbqe=Tbqe,
            zwp=(Tvpe > 0).astype(float), zwq=(Tvqe > 0).astype(float),
            zlp=(Tbpe > 0).astype(float), zlq=(Tbqe > 0).astype(float),
            zdb0=(self.Dbfh.v == self.Dbfl.v).astype(float),
        )
        self._timestep_used = ts

        self._check_init_limits(zdy, ue, Pi0, Qi0, v0)

        return zdy * ue

    def _check_init_limits(self, zdy, ue, Pi0, Qi0, v0):
        """
        Report devices whose initial current commands violate the limits.

        Limits are not adjusted; the initialization test of TDS will fail.
        """
        Ip0 = Pi0 / v0
        Iq0 = Qi0 / v0
        bad = (zdy != 0) & (ue != 0) & ((Ip0 > self.Ipmax.v) | (Ip0 < self.Ipmin.v) |
                                        (Iq0 > self.Iqmax.v) | (Iq0 < self.Iqmin.v))
        if not np.any(bad):
            return

        pos = np.where(bad)[0]
        tab = Tab(title=f'{self.class_name}: initial current commands out of limits '
                        '(limits NOT adjusted)',
                  header=['idx', 'V0', 'Ipcmd0', 'Ipmin', 'Ipmax', 'Iqcmd0', 'Iqmin', 'Iqmax'],
                  data=[[self.idx.v[i], v0[i], Ip0[i], self.Ipmin.v[i], self.Ipmax.v[i],
                         Iq0[i], self.Iqmin.v[i], self.Iqmax.v[i]] for i in pos],
                  )
        logger.warning(tab.draw())

    # ------------------------------------------------------------------
    # cease/reconnect logic
    # ------------------------------------------------------------------

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
        ``dae.t`` with the converged ``Vfilt``, and hold the new ``FracOn``
        for the next step.

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

        held = (self.hMode, self.hTact, self.hTimer, self.hTsr, self.hFon)
        old = PERC1State(*(item.v for item in held))
        active = (self.ue.v != 0) & (self.zdy.v != 0)

        new = perc1_fracon_step(old, t, self.Vfilt.v,
                                self.Fcease.v, self.Vcease.v, self.Tcease.v, self.Tdelay.v,
                                self.Frecon.v, self.Vrecon.v, self.Trecon.v, self.Tramp.v,
                                active=active, ttol=self.config.ttol,
                                )

        fon_changed = new.fon != old.fon
        jumped = fon_changed & (new.mode != old.mode)

        for item, val in zip(held, new):
            item.v[:] = val

        for var, val in ((self.FracOn, new.fon), (self.Mode, new.mode),
                         (self.TimerActive, new.tact)):
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


class PERC1(PERC1Data, PERC1Model):
    r"""
    PERC1 (Power Electronic Reconnecting and Ceasing) load model.

    PERC1 replaces the linked PQ load and models an aggregation of power
    electronic loads as a current injection that passes through voltage
    washouts, a frequency droop with deadband, lead-lags, a voltage exponent,
    current limits, the cease/reconnect multiplier ``FracOn`` and an output
    lag. It reproduces the PowerWorld Simulator 24 model of the same name.

    Per-unit quantities are on the model base ``MBase = MWinit / Lfm`` [MVA],
    where ``MWinit`` and ``Mvarinit`` are the actual PQ powers at the power
    flow solution (``PQ.Ppf``, ``PQ.Qpf``). The network sees
    ``P = MBase * V * Ip`` and ``Q = MBase * V * Iq``.

    Correspondence to the PowerWorld block diagram:

    ========= ============== =========================================
    PW state  ANDES          Notes
    ========= ============== =========================================
    1 Pctrl   ``xPc``        lag state; lead-lag output is ``Pctrl``
    2 Qctrl   ``xQc``        lag state; lead-lag output is ``Qctrl``
    3 dP      ``xdP``        washout state; washout output is ``dP``
    4 dQ      ``xdQ``        washout state; washout output is ``dQ``
    5 wfilt   ``wfilt``
    6 Vfilt   ``Vfilt``
    7 Ip      ``Ip``
    8 Iq      ``Iq``
    ========= ============== =========================================

    Others: ``Pmw``, ``Qmvar``, ``Ipv``, ``Iqv``, ``FracOn``, plus ``Mode``
    and ``TimerActive`` of the cease/reconnect logic.

    Initialization and host rules:

    - If ``V0 < Vcease``, ``V0 < MinVoltILoad`` (``config.minvi``) or
      ``MWinit = 0``, the model is ignored and the load is a constant
      impedance ``P0 (V/V0)^2``, ``Q0 (V/V0)^2``. ``config.pneg`` selects
      the handling of ``MWinit < 0``.
    - ``MvarInit = QPratio * MWinit``; stand-alone devices (``comp = 0``)
      add ``Iqextra = (Q0 - MvarInit) / (MBase V0)`` after the reactive
      current limiter. ``config.qinit = 0`` restores the behavior before
      the February 11, 2026 patch (``MvarInit = Q0``, ``Iqextra = 0``).
    - ``Tt`` is treated as ``4 * TimeStep`` if smaller (by more than
      ``config.tstol`` of the limit). ``TimeStep`` is ``config.tstep`` or,
      if zero, ``TDS.config.tstep`` at initialization. Set the TDS step size
      before initializing TDS.
    - With ``config.smallload = 1``, loads with ``|P0|`` below
      ``config.pmin`` (system base) are not replaced.
    - With ``config.ilow = 1``, the network current falls off as
      ``sin(pi/2 V / MinVoltILoad)`` below ``MinVoltILoad`` (off by default).
    - Current limits are never adjusted to the initial values. Initial
      current commands outside the limits are reported, and the TDS
      initialization test fails.

    The cease/reconnect pseudo-code (:py:func:`perc1_fracon_step`) runs once
    after each accepted time step with the converged ``Vfilt``, from
    :py:meth:`b_update`, which TDS calls once per accepted step; the new
    ``FracOn`` is held during the next step. Newton iterations, line search
    trials and rejected steps do not advance the logic.

    The frequency input is ``f - 1`` of the linked ``BusFreq`` plus the
    service ``dwext`` (zero by default), and is ignored when ``V <
    config.vfmin``. BusFreq has its own measurement filters in addition to
    ``Tf``.
    """

    def __init__(self, system, config):
        PERC1Data.__init__(self)
        PERC1Model.__init__(self, system, config)
