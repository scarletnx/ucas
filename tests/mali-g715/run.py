#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Time the GT 1030 corpus on a Mali GPU of an Android device, over adb, and fit ucas costs.

The corpus, the shader sources and the manifest are tests/gt1030's: the same four-shader
interleaved runs (fadd.a, fadd.b, X.a, X.b) and the same difference, so only the transport
changes.  The device runs snx-shaderperf built for its ABI with the NDK; the SPIR-V and the
binary live in DEV and each run happens there, so paths are relative.

  adb push <spv> ...      once, with --push
  run.py [--out DIR] [--only a,b] [--runs N] [--tries N] [--device-ser DEV] [--push]

Latency builds are skipped: they dump a buffer and read the SM clock from it, which needs a
pull per dump and KHR_shader_clock on the device.
"""
import argparse, importlib.util, json, os, shlex, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = '/data/local/tmp'          # cwd of every run: the corpus's paths are spv/<name>.spv
TOOL = '/data/local/tmp/snx-shaderperf'


def gt_module():
    """tests/gt1030/run.py as a module: its corpus maths, imported, not copied"""
    path = os.path.join(HERE, '..', 'gt1030', 'run.py')
    spec = importlib.util.spec_from_file_location('gt1030_run', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def adb(args, **kw):
    return subprocess.run(['adb'] + args, check=True, **kw)


def dev_run(cmd, log):
    """gt.run_sp's device twin: cmd[0] is the tool, run in DEV with relative paths"""
    argv = ' '.join(shlex.quote(a) for a in cmd[1:])
    with open(log, 'w') as fh:
        fh.write('adb shell %s\n' % argv)
        fh.flush()
        p = subprocess.run(['adb', 'shell', 'cd %s && %s %s' % (DEV, TOOL, argv)],
                           stdout=subprocess.PIPE, stderr=fh, text=True)
    if p.returncode:
        return None
    i = p.stdout.find('{')
    if i < 0:
        return None
    return json.loads(p.stdout[i:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', default=None)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--tries', type=int, default=2)
    ap.add_argument('--max-rel-err', type=float, default=0.02)
    ap.add_argument('--ser', default=None, help='adb device serial, when more than one is attached')
    ap.add_argument('--push', action='store_true', help='push spv/ to the device first')
    ap.add_argument('--model-name', default='mali_g715')
    args = ap.parse_args()
    args.sp = TOOL
    args.no_latency = True

    if args.push:
        adb(['push', os.path.join(HERE, '..', 'gt1030', 'spv'), DEV + '/'], stdout=subprocess.DEVNULL)
        print('pushed the corpus to %s/spv' % DEV, flush=True)

    gt = gt_module()
    gt.run_sp = dev_run
    man = json.load(open(os.path.join(HERE, 'manifest.json')))
    man['device'] = 'Mali-G715 (Pixel 9, GrapheneOS)'
    res = gt.run_all(args, man)
    costs = gt.fit(man, res)

    with open(os.path.join(args.out, 'results.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    body = gt.model_text(costs).replace('model gt1030_measured', 'model ' + args.model_name)
    body = body.replace('# measured on the GT 1030 by tests/gt1030/run.py',
                        '# measured on the Mali-G715 of a Pixel 9 by tests/mali-g715/run.py')
    with open(os.path.join(args.out, 'mali.model.ucas'), 'w') as fh:
        fh.write(body)
    print(body)
    sys.stdout.flush()
    gt.report(man, res, costs, args.out)


if __name__ == '__main__':
    main()
