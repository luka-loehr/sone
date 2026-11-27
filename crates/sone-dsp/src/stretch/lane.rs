//! One band of the stretcher: a read position that plays its band verbatim and
//! a planner that decides where to repeat or skip a little, and by how much.

use std::collections::VecDeque;

use super::detect::Onset;
use super::{LOOKAHEAD, POST, PRE};

/// How a band is spliced.
pub(super) struct LaneParams {
    /// Crossfade of an ordinary splice.
    pub xf: i64,
    /// Largest jump.
    pub jmax: i64,
    /// Jumps are searched within this distance of the wanted shift (coarse splices).
    pub span: i64,
    /// Steer onto every attack (else only onto strong ones).
    pub all_targets: bool,
    /// Keep weak attacks out of splices (else only strong ones).
    pub protect_weak: bool,
    /// Timing tolerance at a strong and at a weak attack (samples).
    pub tol_strong: i64,
    pub tol_weak: i64,
    /// Two-step steering (coarse in the gaps, fine right before the attack).
    pub fine: bool,
    /// Drift allowed where no target is in sight, and how far ahead it is judged.
    pub soft: f64,
    pub ahead: i64,
    /// Decimation of the coarse similarity search (1: full rate).
    pub dec: i64,
}

/// Fine splice: crossfade, largest jump, and the shift below which it works alone.
const XF_FINE: i64 = 256;
const FINE_MAX: i64 = 441;
const FINE_ONLY: i64 = 88;

#[derive(Debug, Clone, Copy)]
struct Splice {
    /// Read position where the crossfade starts.
    at: i64,
    jump: i64,
    xf: i64,
    rho: f64,
}

#[derive(Debug, Clone, Copy)]
enum Plan {
    None,
    Splice(Splice),
    /// Nothing to do until the read position reaches this point.
    Wait(i64),
}

#[derive(Debug, Clone, Copy)]
struct Xfade {
    old: i64,
    done: i64,
    len: i64,
    rho: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct LaneStats {
    pub coarse: usize,
    pub fine: usize,
    pub missed: usize,
    pub relaxed: usize,
    pub rho_sum: f64,
    pub min_rho: f64,
}

/// What a lane reads: its band from absolute input position `base`, the attacks,
/// the ideal input position of the next output sample and the ratio.
pub(super) struct Ctx<'a> {
    pub buf: &'a VecDeque<[f32; 2]>,
    pub base: i64,
    pub onsets: &'a VecDeque<Onset>,
    pub p: f64,
    pub ratio: f64,
    /// How far the input (band and attacks) is known.
    pub known: i64,
}

impl Ctx<'_> {
    #[inline]
    fn at(&self, i: i64) -> [f32; 2] {
        if i < self.base || i >= self.base + self.buf.len() as i64 {
            [0.0; 2]
        } else {
            self.buf[(i - self.base) as usize]
        }
    }
}

pub(super) struct Lane {
    pub q: i64,
    plan: Plan,
    xfade: Option<Xfade>,
    prm: &'static LaneParams,
    pub stats: LaneStats,
}

impl Lane {
    pub(super) fn new(prm: &'static LaneParams) -> Self {
        Lane { q: 0, plan: Plan::None, xfade: None, prm, stats: LaneStats { min_rho: 1.0, ..Default::default() } }
    }

    pub(super) fn in_crossfade(&self) -> bool {
        self.xfade.is_some()
    }

    /// Earliest input position this lane may still read.
    pub(super) fn oldest_needed(&self) -> i64 {
        self.q - self.prm.jmax.max(FINE_MAX) - self.prm.xf - 64
    }

    /// The next output sample of this band.
    pub(super) fn next(&mut self, ctx: &Ctx) -> [f32; 2] {
        if self.xfade.is_none() {
            self.maybe_plan(ctx);
            if let Plan::Splice(s) = self.plan
                && self.q >= s.at
            {
                self.xfade = Some(Xfade { old: self.q, done: 0, len: s.xf, rho: s.rho });
                self.q += s.jump;
                self.plan = Plan::None;
                if s.xf == XF_FINE && self.prm.fine {
                    self.stats.fine += 1;
                } else {
                    self.stats.coarse += 1;
                    self.stats.min_rho = self.stats.min_rho.min(s.rho);
                }
                self.stats.rho_sum += s.rho;
            }
        }
        let y = if let Some(xf) = self.xfade {
            // Raised-cosine crossfade, normalised for the correlation of the two sides:
            // amplitude-complementary when they match, power-complementary when not.
            let w = 0.5 - 0.5 * (std::f64::consts::PI * (xf.done as f64 + 0.5) / xf.len as f64).cos();
            let rho = xf.rho.clamp(0.0, 1.0);
            let norm = 1.0 / ((1.0 - w).powi(2) + w * w + 2.0 * rho * w * (1.0 - w)).sqrt();
            let (ga, gb) = ((1.0 - w) * norm, w * norm);
            let a = ctx.at(xf.old + xf.done);
            let b = ctx.at(self.q);
            self.xfade = if xf.done + 1 < xf.len { Some(Xfade { done: xf.done + 1, ..xf }) } else { None };
            [
                (ga * f64::from(a[0]) + gb * f64::from(b[0])) as f32,
                (ga * f64::from(a[1]) + gb * f64::from(b[1])) as f32,
            ]
        } else {
            ctx.at(self.q)
        };
        self.q += 1;
        y
    }

    fn maybe_plan(&mut self, ctx: &Ctx) {
        match self.plan {
            Plan::Splice(_) => {}
            Plan::Wait(t) if self.q < t => {}
            _ => self.plan = self.make_plan(ctx),
        }
    }

    /// Timing error (input samples; positive: the input is read ahead of time, the
    /// event comes early) at input position `t` if played verbatim from here.
    fn error_at(&self, ctx: &Ctx, t: i64) -> f64 {
        (self.q as f64 - ctx.p) + (t - self.q) as f64 * (1.0 - 1.0 / ctx.ratio)
    }

    fn protects(&self, o: &Onset, relaxed: bool) -> bool {
        o.strong || (self.prm.protect_weak && !relaxed)
    }

    fn make_plan(&mut self, ctx: &Ctx) -> Plan {
        let q = self.q;
        let seen = ctx.known.min(q + LOOKAHEAD as i64);
        // Start of the free region: after any zone the read position is in.
        let mut f0 = q;
        for o in ctx.onsets {
            if self.protects(o, false) && o.pos - PRE <= f0 && f0 < o.pos + POST {
                f0 = o.pos + POST;
            }
        }
        let target = ctx
            .onsets
            .iter()
            .find(|o| (o.strong || self.prm.all_targets) && o.pos - PRE >= f0 && o.pos < seen)
            .copied();
        let Some(o) = target else {
            return self.plan_open(ctx, f0, seen);
        };
        let t = o.pos;
        let tol = if o.strong { self.prm.tol_strong } else { self.prm.tol_weak };
        let need = -self.error_at(ctx, t);
        if need.abs() <= tol as f64 {
            return Plan::Wait(t + POST);
        }
        let need = need.round() as i64;
        if !self.prm.fine {
            // One splice somewhere in the gaps before the attack, landing it within tolerance;
            // if the shift is too big for one jump, the biggest useful part.
            let (lo, hi) = if need.abs() <= self.prm.jmax {
                (need - tol, need + tol)
            } else if need < 0 {
                (-self.prm.jmax, -self.prm.jmax + 2 * tol)
            } else {
                (self.prm.jmax - 2 * tol, self.prm.jmax)
            };
            for relaxed in [false, true] {
                if let Some(s) = self.coarse_splice(ctx, f0, t - PRE, lo, hi, need, relaxed, 0.5 / tol as f64) {
                    self.stats.relaxed += usize::from(relaxed);
                    return Plan::Splice(s);
                }
            }
            self.stats.missed += 1;
            return Plan::Wait(t + POST);
        }
        // Two steps: a coarse splice with a good match takes most of the shift, a fine
        // splice right before the attack (masked by it) the exact rest.
        let coarse_end = t - PRE - XF_FINE - FINE_MAX;
        let coarse = if need.abs() > FINE_ONLY {
            let reach = FINE_ONLY.max(need.abs() / 2).min(self.prm.span);
            let (lo, hi) = ((need - reach).max(-self.prm.jmax), (need + reach).min(self.prm.jmax));
            let weight = 0.3 / self.prm.span as f64;
            [false, true].into_iter().find_map(|relaxed| {
                self.coarse_splice(ctx, f0, coarse_end, lo, hi, need, relaxed, weight).map(|s| (s, relaxed))
            })
        } else {
            None
        };
        let fine = if need.abs() <= FINE_MAX { self.fine_splice(ctx, f0, t, need, tol) } else { None };
        match (coarse, fine) {
            (Some((c, relaxed)), f) if c.rho >= 0.9 || f.is_none() => {
                self.stats.relaxed += usize::from(relaxed);
                Plan::Splice(c)
            }
            (_, Some(f)) => Plan::Splice(f),
            _ => {
                self.stats.missed += 1;
                Plan::Wait(t + POST)
            }
        }
    }

    /// No target in sight: keep the drift within `soft` with ordinary splices, each aimed
    /// at the error a short way ahead so it does not overshoot.
    fn plan_open(&mut self, ctx: &Ctx, f0: i64, seen: i64) -> Plan {
        let q = self.q;
        let prm = self.prm;
        let end = (f0 + prm.ahead).min(seen - prm.xf - prm.jmax);
        if end <= f0 + prm.xf {
            return Plan::Wait(f0.max(q + 256));
        }
        let e = self.error_at(ctx, end);
        if e.abs() <= prm.soft / 2.0 {
            return Plan::Wait(q + prm.ahead / 2);
        }
        let need = (-e).round() as i64;
        // Wide enough to contain a whole period of a low bass note.
        let reach = ((prm.soft / 2.0) as i64).max(prm.span);
        let (lo, hi) = ((need - reach).max(-prm.jmax), (need + reach).min(prm.jmax));
        for relaxed in [false, true] {
            if let Some(s) = self.coarse_splice(ctx, f0, end, lo, hi, need, relaxed, 0.3 / reach.max(1) as f64) {
                self.stats.relaxed += usize::from(relaxed);
                return Plan::Splice(s);
            }
        }
        Plan::Wait(q + prm.ahead / 2)
    }

    /// Free gaps of input in [from, to): outside every protected zone.
    fn gaps(&self, ctx: &Ctx, from: i64, to: i64, relaxed: bool) -> Vec<(i64, i64)> {
        let mut gaps = Vec::new();
        let mut g0 = from;
        for o in ctx.onsets {
            if !self.protects(o, relaxed) {
                continue;
            }
            let (z0, z1) = (o.pos - PRE, o.pos + POST);
            if z1 <= g0 {
                continue;
            }
            if z0 >= to {
                break;
            }
            if z0 > g0 {
                gaps.push((g0, z0));
            }
            g0 = g0.max(z1);
        }
        if g0 < to {
            gaps.push((g0, to));
        }
        gaps
    }

    /// Allowed crossfade starts for a jump `j` inside gap [g0, g1): both streams'
    /// crossfades and whatever is repeated or skipped stay inside the gap.
    fn start_range(&self, g0: i64, g1: i64, j: i64, xf: i64) -> Option<(i64, i64)> {
        let lo = (if j < 0 { g0 - j } else { g0 }).max(self.q);
        let hi = if j < 0 { g1 - xf } else { g1 - xf - j };
        (hi >= lo).then_some((lo, hi))
    }

    /// Jump in [jlo, jhi] and position in the gaps of [from, to) with the best waveform
    /// match, minus `weight` per sample of distance from `want`. With decimation the
    /// search runs on a decimated mono signal first and refines the best candidates.
    #[allow(clippy::too_many_arguments)]
    fn coarse_splice(
        &self,
        ctx: &Ctx,
        from: i64,
        to: i64,
        jlo: i64,
        jhi: i64,
        want: i64,
        relaxed: bool,
        weight: f64,
    ) -> Option<Splice> {
        let xf = self.prm.xf;
        let gaps = self.gaps(ctx, from.max(self.q), to, relaxed);
        if gaps.is_empty() || jlo > jhi {
            return None;
        }
        let penalty = |j: i64| weight * (want - j).abs() as f64;
        let dec = self.prm.dec;
        // Candidates (start, jump) to evaluate at full rate.
        let mut cands: Vec<(i64, i64, i64)> = Vec::new(); // (start lo, start hi, jump)
        if dec > 1 {
            let d0 = (gaps[0].0 - self.prm.jmax).div_euclid(dec);
            let d1 = (gaps[gaps.len() - 1].1 + self.prm.jmax + xf).div_euclid(dec) + 1;
            let m: Vec<f64> = (d0..d1)
                .map(|k| {
                    (0..dec)
                        .map(|i| {
                            let v = ctx.at(k * dec + i);
                            f64::from(v[0]) + f64::from(v[1])
                        })
                        .sum::<f64>()
                })
                .collect();
            let xd = (xf / dec) as usize;
            let mut top: Vec<(f64, i64, i64)> = Vec::new();
            let mut jd = jlo.div_euclid(dec);
            while jd * dec <= jhi {
                let j = jd * dec;
                if j != 0 && j >= jlo {
                    for &(g0, g1) in &gaps {
                        let Some((lo, hi)) = self.start_range(g0, g1, j, xf) else { continue };
                        let (klo, khi) = (lo.div_euclid(dec) + 1 - d0, hi.div_euclid(dec) - d0);
                        if khi < klo {
                            continue;
                        }
                        let span = (khi - klo) as usize + xd;
                        let (mut pp, mut ea, mut eb) = (vec![0.0; span + 1], vec![0.0; span + 1], vec![0.0; span + 1]);
                        for k in 0..span {
                            let a = m[klo as usize + k];
                            let b = m[(klo + jd) as usize + k];
                            pp[k + 1] = pp[k] + a * b;
                            ea[k + 1] = ea[k] + a * a;
                            eb[k + 1] = eb[k] + b * b;
                        }
                        let mut k = 0;
                        while k + xd <= span {
                            let rho = corr(pp[k + xd] - pp[k], ea[k + xd] - ea[k], eb[k + xd] - eb[k], xd as f64);
                            let score = rho - penalty(j);
                            if top.len() < 4 || score > top[top.len() - 1].0 {
                                top.push((score, (klo + d0 + k as i64) * dec, j));
                                top.sort_by(|a, b| b.0.total_cmp(&a.0));
                                top.truncate(4);
                            }
                            k += (32 / dec).max(1) as usize;
                        }
                    }
                }
                jd += 1;
            }
            for &(_, s0, j0) in &top {
                for j in (j0 - dec)..=(j0 + dec) {
                    if j == 0 || j < jlo || j > jhi {
                        continue;
                    }
                    let Some(&(g0, g1)) = gaps.iter().find(|g| g.0 <= s0 && s0 < g.1) else { continue };
                    let Some((lo, hi)) = self.start_range(g0, g1, j, xf) else { continue };
                    let (lo, hi) = ((s0 - 32).max(lo), (s0 + 32).min(hi));
                    if hi >= lo {
                        cands.push((lo, hi, j));
                    }
                }
            }
        } else {
            for j in jlo..=jhi {
                if j == 0 {
                    continue;
                }
                for &(g0, g1) in &gaps {
                    if let Some((lo, hi)) = self.start_range(g0, g1, j, xf) {
                        cands.push((lo, hi, j));
                    }
                }
            }
        }
        let mut best: Option<(f64, Splice)> = None;
        for (lo, hi, j) in cands {
            let (s, rho) = best_position(ctx, lo, hi, j, xf, 4);
            let score = rho - penalty(j);
            if best.as_ref().is_none_or(|(b, _)| score > *b) {
                best = Some((score, Splice { at: s, jump: j, xf, rho }));
            }
        }
        best.map(|(_, s)| s)
    }

    /// Fine splice right before the attack at `t`: short crossfade ending where the
    /// attack's protected zone begins, jump within `tol` of `need`.
    fn fine_splice(&self, ctx: &Ctx, f0: i64, t: i64, need: i64, tol: i64) -> Option<Splice> {
        let mut best: Option<(f64, Splice)> = None;
        for relaxed in [false, true] {
            // Whatever the splice repeats must lie outside earlier zones.
            let floor = self.gaps(ctx, f0.max(self.q) - FINE_MAX - XF_FINE, t - PRE, relaxed).last().map_or(t, |g| g.0);
            for j in (need - tol)..=(need + tol) {
                if j == 0 || j.abs() > FINE_MAX {
                    continue;
                }
                let hi = t - PRE - XF_FINE - j.max(0);
                let lo = (hi - 64).max(self.q).max(floor - j.min(0));
                if hi < lo {
                    continue;
                }
                let (s, rho) = best_position(ctx, lo, hi, j, XF_FINE, 4);
                // Among equal matches the jump closest to the exact shift.
                let score = rho - 0.5 * (j - need).abs() as f64 / tol.max(1) as f64;
                if best.as_ref().is_none_or(|b| score > b.0) {
                    best = Some((score, Splice { at: s, jump: j, xf: XF_FINE, rho }));
                }
            }
            if let Some((_, s)) = best {
                return Some(s);
            }
        }
        None
    }
}

/// Start in [s_lo, s_hi] (every `step` samples) where the band matches itself `j`
/// samples away best over a crossfade of `xf`, in stereo.
fn best_position(ctx: &Ctx, s_lo: i64, s_hi: i64, j: i64, xf: i64, step: i64) -> (i64, f64) {
    let span = (s_hi - s_lo + xf) as usize;
    let xfu = xf as usize;
    let mut pp = vec![0.0f64; span + 1];
    let mut ea = vec![0.0f64; span + 1];
    let mut eb = vec![0.0f64; span + 1];
    for k in 0..span {
        let a = ctx.at(s_lo + k as i64);
        let b = ctx.at(s_lo + k as i64 + j);
        let (a0, a1, b0, b1) = (f64::from(a[0]), f64::from(a[1]), f64::from(b[0]), f64::from(b[1]));
        pp[k + 1] = pp[k] + a0 * b0 + a1 * b1;
        ea[k + 1] = ea[k] + a0 * a0 + a1 * a1;
        eb[k + 1] = eb[k] + b0 * b0 + b1 * b1;
    }
    let mut best = (s_lo, f64::MIN);
    let mut s = s_lo;
    while s <= s_hi {
        let k = (s - s_lo) as usize;
        let rho = corr(pp[k + xfu] - pp[k], ea[k + xfu] - ea[k], eb[k + xfu] - eb[k], xf as f64);
        if rho > best.1 {
            best = (s, rho);
        }
        s += step;
    }
    best
}

/// Normalised correlation; two silent windows match perfectly.
fn corr(num: f64, ea: f64, eb: f64, len: f64) -> f64 {
    if ea < 1e-10 * len && eb < 1e-10 * len { 1.0 } else { num / (ea * eb).sqrt().max(1e-30) }
}
