#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Time the GT 1030 corpus with snx-shaderperf and fit the ucas cost model.

Each test runs as one snx-shaderperf invocation of four shaders, interleaved:
fadd.a, fadd.b (the reference), X.a, X.b, all with the same push constants.  The
difference b - a is the cost of A*C extra steps a lane, so the loop, the prologue and the
dispatch cancel, and X's cost is its difference over fadd's, from the same run (the clock
drifts alike for both).  Tests with a latency build also run their .la/.lb builds in one
warp and read the SM clock the shader stored (KHR_shader_clock), giving cycles a step.

The fit: every class the primary tests (fit: true) use is an unknown, fadd is 1, and a
least-squares solve of  sum(ops[c] * cost[c]) = measured  over the primary tests gives
the costs.  Every other test is then priced by the fitted costs and by the prelude's
gt1030 model and compared with what it measured.

Usage: run.py [--sp PATH] [--out DIR] [--only a,b,...] [--runs N] [--no-latency] [--tries N]
              [--analyse DIR]   (re-analyse a finished run without the GPU)
"""
import argparse, json, os, re, struct, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# the prelude's gt1030 (src/parse.c), set from the 2026-11-11 run
PRELUDE = dict(iadd=1, logic=1, shift=2, imul=3, idiv=28.5, idivk=8.75, cmp=2.375, select=1.625, minmax=1,
               bitcnt=3, conv=3, bitcast=0, fadd=1, fmul=1, fdiv=3, fsqrt=4, ftrans=4, fpow=8, f64=32.375,
               load=6.875, branch=6.25, diverge=14.375, barrier=9.125, atomic_shared=6.625, atomic_global=50,
               store=15)
CLASSES = list(PRELUDE)

PUSH = ['seed=u32:1', 'k=u32:2654435769', 'k2=u32:5', 'kf=f32:1.0000001', 'kf2=f32:0.5']


COMPUTE_KINDS = ('alu', 'mem', 'branch')


def ref_name(test):
    """the fadd every test of a stage is measured against"""
    return 'fadd' if test['kind'] in COMPUTE_KINDS else test['kind'] + '_fadd'


def sp_cmd(sp, shaders, iters, extra, compute=True):
    cmd = [sp] + shaders
    cmd += ['--push', 'iters=u32:%d' % iters]
    for p in PUSH:
        cmd += ['--push', p]
    if compute:
        cmd += ['--bind', '0.0=fill=noise', '--buffer-size', '32M']
    return cmd + extra


def run_sp(cmd, log):
    with open(log, 'w') as fh:
        fh.write(' '.join(cmd) + '\n')
        fh.flush()
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=fh, text=True, cwd=HERE)
    if p.returncode:
        return None
    return json.loads(p.stdout)


def med(sh):
    t = sh['time_ns_per_rep']
    return t['median'], (t['median_hi'] - t['median_lo']) / 2


def bin_size(sh):
    for e in sh.get('executables', []):
        return e.get('statistics', {}).get('Binary Size')
    return None


def sm_clock(d):
    m = re.match(r'\s*(\d+) MHz', (d.get('nvidia_smi') or {}).get('after', ''))
    return int(m.group(1)) if m else None


def throughput(test, d, reft):
    """per-step ns of the reference and of the test, from one run's JSON"""
    sh = d['shaders']
    C, A = test['chains'], test['unroll'][0]
    out = {}
    for key, i0 in (('ref', 0), ('x', 2)):
        ta, ea = med(sh[i0])
        tb, eb = med(sh[i0 + 1])
        inv = sh[i0]['invocations_per_rep']
        c, a = (reft['chains'], reft['unroll'][0]) if key == 'ref' else (C, A)
        steps = inv * test['iters'] * a * c
        ba, bb = bin_size(sh[i0]), bin_size(sh[i0 + 1])
        out[key] = dict(ns=(tb - ta) / steps, err=(ea * ea + eb * eb) ** 0.5 / steps, ta=ta, tb=tb, inv=inv,
                        bytes_per_step=(bb - ba) / (a * c) if ba is not None and bb is not None else None,
                        regs=[(e.get('statistics') or {}).get('Register Count') for e in
                              sh[i0 + 1].get('executables', [])][:1])
    return out


def latency(test, out_dir, sp, extra_log):
    L = test['latency']
    files = [test['files']['la'], test['files']['lb']]
    dump = os.path.join(out_dir, 'lat', test['name'] + '.bin')
    os.makedirs(os.path.dirname(dump), exist_ok=True)
    cmd = sp_cmd(sp, files, L['iters'], ['--size', '32', '--runs', '3', '--warmup', '2', '--warmup-ms', '20',
                                         '--check', '--dump', '0.1=' + dump, '--json', '--quiet', '--no-smi',
                                         '--compile-runs', '1'] + test['args'])
    for v in test['vkext']:
        cmd += ['--ext', v]
    d = run_sp(cmd, extra_log)
    if d is None:
        return None
    clk = []
    for n in range(2):
        fn = dump.replace('.bin', '-%d.bin' % n)
        if not os.path.exists(fn):
            return None
        with open(fn, 'rb') as fh:
            w = struct.unpack('<2I', fh.read(8))
        clk.append(w[1])
    steps = L['iters'] * (L['unroll'][1] - L['unroll'][0]) * L['chains']
    return dict(cycles_per_step=(clk[1] - clk[0]) / steps, clocks=clk)


def run_all(args, man):
    sp = args.sp
    out_dir = args.out
    os.makedirs(os.path.join(out_dir, 'raw'), exist_ok=True)
    by = {t['name']: t for t in man['tests']}
    only = set(args.only.split(',')) if args.only else None
    res = {}
    # the probe first: SM count and warp slots
    pdump = os.path.join(out_dir, 'probe.bin')
    d = run_sp(sp_cmd(sp, [man['probe']], 2000, ['--size', '65536', '--runs', '3', '--check', '--dump',
                                                 '0.1=' + pdump, '--json', '--quiet']),
               os.path.join(out_dir, 'raw', 'probe.log'))
    if d is not None and os.path.exists(pdump):
        res['_probe'] = probe_summary(pdump)
    for test in man['tests']:
        name = test['name']
        if only and name not in only:
            continue
        r = res.setdefault(name, {})
        rf = by[ref_name(test)]
        shaders = [rf['files']['a'], rf['files']['b'], test['files']['a'], test['files']['b']]
        extra = ['--runs', str(args.runs), '--json', '--quiet'] + test['args']
        compute = test['kind'] in COMPUTE_KINDS
        if compute:
            extra += ['--size', '65536']
        for v in test['vkext']:
            extra += ['--ext', v]
        t0 = time.time()
        # the compositor shares the GPU: a run whose interval is wide is run again, and
        # the tightest of up to args.tries runs is kept
        best = None
        for attempt in range(args.tries):
            d = run_sp(sp_cmd(sp, shaders, test['iters'], extra, compute), os.path.join(out_dir, 'raw', name + '.log'))
            if d is None:
                break
            tp = throughput(test, d, rf)
            if name == rf['name']:   # its own reference: 1 by definition
                cost, err = 1.0, tp['x']['err'] / tp['x']['ns']
            else:
                cost = tp['x']['ns'] / tp['ref']['ns']
                err = cost * ((tp['x']['err'] / tp['x']['ns']) ** 2 + (tp['ref']['err'] / tp['ref']['ns']) ** 2) ** 0.5
            if best is None or err / abs(cost) < best[2] / abs(best[1]):
                best = (d, cost, err, tp)
            if err <= args.max_rel_err * abs(cost):
                break
        if best is None:
            r['error'] = open(os.path.join(out_dir, 'raw', name + '.log')).read()[-2000:]
            print('%-24s FAILED (raw/%s.log)' % (name, name), flush=True)
            continue
        d, r['cost'], r['err'], tp = best
        r['tries'] = attempt + 1
        with open(os.path.join(out_dir, 'raw', name + '.json'), 'w') as fh:
            json.dump(d, fh)
        r['tp'] = tp
        r['clock_mhz'] = sm_clock(d)
        if 'latency' in test and not args.no_latency:
            lat = latency(test, out_dir, sp, os.path.join(out_dir, 'raw', name + '.lat.log'))
            if lat:
                r['latency'] = lat
        print('%-24s %8.3f +- %.3f  %s  (%.1fs)' % (name, r['cost'], r['err'],
              ('lat %.1f clk' % r['latency']['cycles_per_step']) if 'latency' in r else '', time.time() - t0),
              flush=True)
        with open(os.path.join(out_dir, 'results.json'), 'w') as fh:
            json.dump(res, fh, indent=1)
    return res


def per_unit(sh, unit):
    """ns a vertex, input primitive or pixel of one shader of a run, and its half interval"""
    t, e = med(sh)
    n = sh['invocations_per_rep']
    if unit == 'primitive':
        n = (sh.get('pipeline_statistics_per_rep') or {}).get('ia_primitives') or n
    return t / n, e / n


def run_io(args, man, res):
    out = res.setdefault('_io', {})
    only = set(args.only.split(',')) if args.only else None
    for g in man['io']:
        if only and g['group'] not in only:
            continue
        rows = []
        log = os.path.join(args.out, 'raw', 'io_' + g['group'] + '.log')
        base = ['--runs', str(args.runs), '--json', '--quiet'] + g['args']
        if g['mode'] == 'run':
            files = [x['file'] for x in g['items']]
            d = run_sp([args.sp] + files + base, log)
            if d is not None:
                for x, sh in zip(g['items'], d['shaders']):
                    ns, err = per_unit(sh, g['unit'])
                    rows.append(dict(name=x['name'], ns=ns, err=err, stats=sh.get('pipeline_statistics_per_rep')))
            else:
                rows.append(dict(name=g['group'], error=open(log).read()[-600:]))
        else:
            for x in g['items']:
                d = run_sp([args.sp, x['file']] + base + x['args'], log + '.' + x['name'])
                if d is None:
                    rows.append(dict(name=x['name'], error=open(log + '.' + x['name']).read()[-600:]))
                    continue
                ns, err = per_unit(d['shaders'][0], g['unit'])
                rows.append(dict(name=x['name'], ns=ns, err=err, clock_mhz=sm_clock(d),
                                 stats=d['shaders'][0].get('pipeline_statistics_per_rep')))
        out[g['group']] = rows
        print('%-16s %s' % (g['group'], '  '.join('%s %.3g' % (r['name'], r['ns']) if 'ns' in r else
                                                  '%s FAILED' % r['name'] for r in rows)), flush=True)
        with open(os.path.join(args.out, 'results.json'), 'w') as fh:
            json.dump(res, fh, indent=1)


def probe_summary(path):
    data = open(path, 'rb').read()
    n = len(data) // 32
    recs = [struct.unpack('<8I', data[i * 32:(i + 1) * 32]) for i in range(n)]
    recs = [r for r in recs if r[2]]          # warps that wrote (SM count is never 0)
    if not recs:
        return None
    sms = sorted(set(r[0] for r in recs))
    per_sm = {s: sum(1 for r in recs if r[0] == s) for s in sms}
    return dict(warps=len(recs), sm_count=recs[0][2], warps_per_sm=recs[0][3], sms_used=sms,
                warps_on_sm=per_sm, max_warp_slot=max(r[1] for r in recs),
                loop_clocks_median=sorted(r[7] for r in recs)[len(recs) // 2])


def fit(man, res):
    """Solve the classes one at a time: a primary test whose classes are all known but one
    gives that one, (measured - the known part) / its count.  fadd is 1.  Each cost keeps
    the test it came from (costs['_from'])."""
    costs, src = {'fadd': 1.0}, {'fadd': 'fadd (the reference)'}
    prim = [t for t in man['tests'] if t['fit'] and t['name'] in res and 'cost' in res[t['name']]]
    left = [t for t in prim if t['name'] != 'fadd']
    while left:
        done = None
        for t in left:
            unk = [c for c in t['ops'] if c not in costs]
            if len(unk) == 1:
                known = sum(n * costs[c] for c, n in t['ops'].items() if c in costs)
                costs[unk[0]] = (res[t['name']]['cost'] - known) / t['ops'][unk[0]]
                src[unk[0]] = t['name']
                done = t
                break
            if not unk:
                left.remove(t)
                done = t
                break
        if done is None:
            sys.stderr.write('fit: no order solves %s\n' % ', '.join(t['name'] for t in left))
            break
        if done in left:
            left.remove(done)
    costs['_from'] = src
    return costs


def price(ops, costs):
    if any(c not in costs or c == '_from' for c in ops):
        return None
    return sum(n * costs[c] for c, n in ops.items())


def report(man, res, costs, out_dir):
    L = []
    L.append('# GT 1030 calibration run\n')
    first = next((r for k, r in res.items() if not k.startswith('_') and r.get('clock_mhz')), {})
    L.append('SM clock after the runs: %s MHz (nvidia-smi). Costs are throughput ratios to fadd in the same run.\n'
             % first.get('clock_mhz'))
    pr = res.get('_probe')
    if pr:
        L.append('Probe: %d warps on SMs %s (gl_SMCountNV %d, gl_WarpsPerSMNV %d, highest warp slot %d).\n'
                 % (pr['warps'], pr['sms_used'], pr['sm_count'], pr['warps_per_sm'], pr['max_warp_slot']))
    rr = res.get('fadd', {}).get('tp')
    if rr and pr and first.get('clock_mhz'):
        ref = rr['ref']
        ipc = 1 / (ref['ns'] * 32 * pr['sm_count'] * first['clock_mhz'] * 1e-3)
        L.append('fadd: %.3f ns a lane-step, %.2f warp instructions a clock an SM (sm_61 peak 4).\n' % (ref['ns'], ipc))
    src = costs.get('_from', {})
    L.append('\n## Fitted classes\n\n| class | prelude | measured | from test |\n|---|---:|---:|---|\n')
    for c in CLASSES:
        L.append('| %s | %g | %s | %s |\n' % (c, PRELUDE[c], ('%.2f' % costs[c]) if c in costs else '-',
                                              src.get(c, '')))
    for c in sorted(k for k in costs if k not in PRELUDE and k != '_from'):
        L.append('| %s | - | %.2f | %s |\n' % (c, costs[c], src.get(c, '')))
    L.append('\n## Every test\n\nmeasured: cost of one step / fadd. fitted, prelude: the step priced by the fitted '
             'costs and by the prelude model. B/step: driver binary bytes a step (sm_61 SASS: ~10.7 B an '
             'instruction). lat: SM clocks a step in one warp.\n\n')
    L.append('| test | ops | measured | fitted | prelude | B/step | lat | note |\n|---|---|---:|---:|---:|---:|---:|---|\n')
    for t in man['tests']:
        r = res.get(t['name'])
        if r is None:
            continue
        if 'error' in r:
            L.append('| %s | %s | FAILED | | | | | %s |\n' % (t['name'], fmt_ops(t['ops']), t['note']))
            continue
        f, p = price(t['ops'], costs), price(t['ops'], PRELUDE)
        bps = r.get('tp', {}).get('x', {}).get('bytes_per_step')
        lat = r.get('latency', {}).get('cycles_per_step')
        L.append('| %s%s | %s | %.2f ± %.2f | %s | %s | %s | %s | %s |\n' % (
            t['name'], ' *' if t['fit'] else '', fmt_ops(t['ops']), r['cost'], r['err'],
            '%.2f' % f if f is not None else '-', '%g' % p if p is not None else '-',
            '%.0f' % bps if bps is not None else '-', '%.1f' % lat if lat is not None else '-', t['note']))
    L.append('\n`*` primary test of a class (used in the fit).\n')
    # the same steps in each stage, each against its own stage's fadd
    bases = [t['base'] for t in man['tests'] if t.get('base') and t['kind'] == 'vert']
    if bases:
        L.append('\n## Stages\n\nThe same steps in a compute, vertex, geometry and fragment shader, each a ratio to the '
                 'fadd of its own stage (the stage\'s I/O is fixed and cancels).\n\n'
                 '| test | compute | vertex | geometry | fragment |\n|---|---:|---:|---:|---:|\n')
        for bname in bases:
            cells = []
            for st in ('', 'vert_', 'geom_', 'frag_'):
                r = res.get(st + bname, {})
                cells.append('%.2f ± %.2f' % (r['cost'], r['err']) if 'cost' in r else '-')
            L.append('| %s | %s |\n' % (bname, ' | '.join(cells)))
    io = res.get('_io')
    if io:
        L.append('\n## Stage I/O\n\nns a unit (vertex, input primitive, pixel) over the whole GPU, the ratio to the '
                 'group\'s first item, and the same time in fadd steps of that stage (its fadd\'s ns a lane-step). '
                 '"run" groups share one run; "sweep" items are runs of their own (other options), so their ratios '
                 'carry the clock drift between runs.\n')
        for g in man['io']:
            rows = io.get(g['group'])
            if not rows:
                continue
            fr = res.get({'vert': 'vert_fadd', 'geom': 'geom_fadd', 'frag': 'frag_fadd'}[g['stage']], {})
            fns = fr.get('tp', {}).get('x', {}).get('ns')
            L.append('\n### %s (%s, a %s)\n\n%s\n\n| item | ns | ratio | fadd steps |\n|---|---:|---:|---:|\n'
                     % (g['group'], g['mode'], g['unit'], g['note']))
            first = next((r['ns'] for r in rows if 'ns' in r), None)
            for r in rows:
                if 'ns' not in r:
                    L.append('| %s | FAILED | | |\n' % r['name'])
                    continue
                L.append('| %s | %.4g ± %.2g | %.2f | %s |\n' % (r['name'], r['ns'], r['err'], r['ns'] / first,
                                                              '%.1f' % (r['ns'] / fns) if fns else '-'))
    L.append('\n## Model\n\n```\n%s```\n' % model_text(costs))
    with open(os.path.join(out_dir, 'report.md'), 'w') as fh:
        fh.write(''.join(L))
    with open(os.path.join(out_dir, 'model.ucas'), 'w') as fh:
        fh.write(model_text(costs))
    with open(os.path.join(out_dir, 'costs.json'), 'w') as fh:
        json.dump({k: (v if k == '_from' else float(v)) for k, v in costs.items()}, fh, indent=1)


def fmt_ops(o):
    return ' '.join('%s%s' % ('%d×' % n if n != 1 else '', c) for c, n in o.items())


def model_text(costs):
    parts = ['%s: %.2f' % (c, max(0.0, costs[c])) for c in CLASSES if c in costs]
    lines, cur = [], '  '
    for p in parts:
        if len(cur) + len(p) > 76:
            lines.append(cur.rstrip())
            cur = '  '
        cur += p + ', '
    lines.append(cur.rstrip().rstrip(','))
    return '# measured on the GT 1030 by tests/gt1030/run.py\nmodel gt1030_measured {\n%s\n}\n' % '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sp', default=os.path.expanduser('~/ScarletNX/build/rel/snx-shaderperf'))
    ap.add_argument('--out', default=os.path.expanduser('~/ucas/work/gt1030/run-' + time.strftime('%Y%m%d-%H%M%S')))
    ap.add_argument('--only')
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--no-latency', action='store_true')
    ap.add_argument('--tries', type=int, default=3, help='runs of a test whose interval is wide')
    ap.add_argument('--max-rel-err', type=float, default=0.03, help='the interval a test aims at')
    ap.add_argument('--analyse')
    ap.add_argument('--no-io', action='store_true', help='skip the graphics I/O groups')
    ap.add_argument('--io-only', action='store_true', help='only the graphics I/O groups')
    args = ap.parse_args()
    man = json.load(open(os.path.join(HERE, 'manifest.json')))
    if args.analyse:
        args.out = args.analyse
        res = json.load(open(os.path.join(args.out, 'results.json')))
    else:
        args.out = os.path.abspath(args.out)
        os.makedirs(os.path.join(args.out, 'raw'), exist_ok=True)
        res = run_all(args, man) if not args.io_only else {}
        if not args.no_io:
            run_io(args, man, res)
        with open(os.path.join(args.out, 'results.json'), 'w') as fh:
            json.dump(res, fh, indent=1)
    costs = fit(man, res)
    report(man, res, costs, args.out)
    print('\n' + model_text(costs))
    print('report: %s' % os.path.join(args.out, 'report.md'))


if __name__ == '__main__':
    main()
