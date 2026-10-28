#!/usr/bin/env python3
"""
PRISM - research image codec (pure Python/numpy, bit-exact encode/decode round trip).

Tools combined (each one is a switch so it can be ablated):
  * YCbCr 4:2:0, 16x16 super-blocks
  * adaptive transform size per super-block: one 16x16 DCT or four 8x8 DCTs (RD choice)
  * closed-loop intra prediction (DC / V / H / gradient / planar) -> DCT of the residual
  * perceptual quant matrix + per-super-block adaptive quantisation (activity masking)
  * RDO: mode/size choice by  J = bits + lambda * distortion, dead-zone quantiser,
    decoder-side reconstruction bias (centroid) for free PSNR
  * entropy coding: binary range coder + context mixing (3 adaptive models per decision
    blended by an online-trained logistic mixer), "last significant" signalling
  * decoder-side deblocking on real block edges (optional)

usage:  python prism.py in.png out.prm --qb 12     |   python prism.py out.prm back.png
"""
import math, struct, sys
import numpy as np
import cv2

P = dict(wl=4.5, wc=2.2, cs=1.2, dz0=0.05, dz1=0.20, delta=0.0, lam=5.0,
         aq=1, intra=1, adapt=1, mix=1, deblock=0.5)
QTAB = [0.7, 0.85, 1.0, 1.2, 1.45]


# ----------------------------------------------------------------------------- transforms
def dct_mat(N):
    k = np.arange(N)[:, None]; n = np.arange(N)[None, :]
    C = np.cos(np.pi * (2 * n + 1) * k / (2 * N)) * math.sqrt(2.0 / N)
    C[0] /= math.sqrt(2)
    return C

CM = {8: dct_mat(8), 16: dct_mat(16)}


def zigzag(N):
    idx = [(u, v) for u in range(N) for v in range(N)]
    idx.sort(key=lambda t: (t[0] + t[1], t[0] if (t[0] + t[1]) % 2 else t[1]))
    return idx

SCAN = {N: zigzag(N) for N in (8, 16)}
SCANF = {N: np.array([u * N + v for u, v in SCAN[N]]) for N in (8, 16)}
BAND = {N: [min(11, int(1.5 * math.log2(1 + k))) for k in range(N * N)] for N in (8, 16)}
_QM = {}


def qmats(N, pl, qb):
    key = (N, pl, qb)
    if key not in _QM:
        u = np.arange(N)[:, None]; v = np.arange(N)[None, :]
        f = (u + v) / (2.0 * (N - 1))
        W = (1 + P['wl'] * f) if pl == 0 else P['cs'] * (1 + P['wc'] * f)
        _QM[key] = (qb * W, P['dz0'] + P['dz1'] * f)
    return _QM[key]


# ----------------------------------------------------------------------------- prediction
def modes_for(y0, x0):
    if not P['intra']:
        return [0]
    m = [0]
    if y0 > 0: m.append(1)
    if x0 > 0: m.append(2)
    if y0 > 0 and x0 > 0: m += [3, 4]
    return m


def predict(rec, y0, x0, N, mode):
    top = rec[y0 - 1, x0:x0 + N] if y0 > 0 else None
    left = rec[y0:y0 + N, x0 - 1] if x0 > 0 else None
    if mode == 1 and top is not None:
        return np.tile(top, (N, 1))
    if mode == 2 and left is not None:
        return np.tile(left[:, None], (1, N))
    if mode == 3 and top is not None and left is not None:
        return np.clip(left[:, None] + top[None, :] - rec[y0 - 1, x0 - 1], -128, 127)
    if mode == 4 and top is not None and left is not None:
        i = np.arange(N)[:, None]; j = np.arange(N)[None, :]
        return ((N - 1 - j) * left[:, None] + (j + 1) * top[-1] +
                (N - 1 - i) * top[None, :] + (i + 1) * left[-1]) / (2.0 * N)
    s, c = 0.0, 0
    if top is not None: s += top.sum(); c += N
    if left is not None: s += left.sum(); c += N
    return np.full((N, N), s / c if c else 0.0)


def dequant(q, Qm):
    a = np.abs(q)
    return np.sign(q) * np.where(a > 0, a - P['delta'], 0.0) * Qm


def apply_block(rec, y0, x0, N, mode, q, Qm):
    Pd = predict(rec, y0, x0, N, mode)
    rec[y0:y0 + N, x0:x0 + N] = np.clip(Pd + CM[N].T @ dequant(q, Qm) @ CM[N], -128, 127)


def bits_est(q, N):
    qs = q.ravel()[SCANF[N]]
    nz = np.flatnonzero(qs)
    if nz.size == 0:
        return 0.6
    m = np.abs(qs[nz])
    return 2.0 + float(np.sum(3.0 + 2.0 * np.log2(m))) + 0.3 * (nz[-1] + 1 - nz.size)


def try_block(rec, src, y0, x0, N, qb, pl):
    Qm, dz = qmats(N, pl, qb)
    S = src[y0:y0 + N, x0:x0 + N]
    best = None
    for mode in modes_for(y0, x0):
        C = CM[N] @ (S - predict(rec, y0, x0, N, mode)) @ CM[N].T
        a = np.abs(C) / Qm
        q = np.maximum(np.floor(a + 0.5 - dz), 0.0)
        dq = np.where(q > 0, q - P['delta'], 0.0) * Qm
        sse = float(((np.abs(C) - dq) ** 2).sum())
        J = bits_est(q, N) + (0.5 if mode == 0 else 1.5) + P['lam'] * sse / (qb * qb)
        if best is None or J < best[0]:
            best = (J, mode, (np.sign(C) * q).astype(np.int32))
    return best


# ----------------------------------------------------------------------------- range coder
class RCEnc:
    def __init__(s):
        s.low = 0; s.range = 0xFFFFFFFF; s.cache = 0; s.cs = 1; s.out = bytearray()

    def _shift(s):
        if s.low < 0xFF000000 or s.low >= (1 << 32):
            carry = s.low >> 32; t = s.cache
            while True:
                s.out.append((t + carry) & 0xFF); t = 0xFF; s.cs -= 1
                if s.cs == 0: break
            s.cache = (s.low >> 24) & 0xFF
        s.cs += 1
        s.low = (s.low & 0x00FFFFFF) << 8

    def encode(s, bit, p1):
        bound = (s.range >> 12) * p1
        if bit: s.range = bound
        else: s.low += bound; s.range -= bound
        while s.range < (1 << 24):
            s.range = (s.range << 8) & 0xFFFFFFFF; s._shift()

    def finish(s):
        for _ in range(5): s._shift()
        return bytes(s.out[1:])           # first byte is always 0


class RCDec:
    def __init__(s, data):
        s.d = data; s.i = 0; s.range = 0xFFFFFFFF; s.code = 0
        for _ in range(4): s.code = (s.code << 8) | s._b()

    def _b(s):
        b = s.d[s.i] if s.i < len(s.d) else 0
        s.i += 1
        return b

    def decode(s, p1):
        bound = (s.range >> 12) * p1
        if s.code < bound: s.range = bound; bit = 1
        else: s.code -= bound; s.range -= bound; bit = 0
        while s.range < (1 << 24):
            s.range = (s.range << 8) & 0xFFFFFFFF
            s.code = ((s.code << 8) | s._b()) & 0xFFFFFFFF
        return bit


class Coder:
    """Context-mixing binary coder. Same code path for encode and decode."""
    LIM = 80; LR = 0.02

    def __init__(s, enc, data=None):
        s.enc = enc; s.rc = RCEnc() if enc else RCDec(data)
        s.T = {}; s.W = {}

    def raw(s, b):
        if s.enc: s.rc.encode(b, 2048); return b
        return s.rc.decode(2048)

    def bit(s, ctxs, mix, b=None):
        T = s.T; sl = []; st = []
        for key in ctxs:
            e = T.get(key)
            if e is None: e = T[key] = [0.5, 0]
            sl.append(e); p = e[0]; st.append(math.log(p / (1 - p)))
        n = len(st)
        if P['mix']:
            st.append(0.3)
            w = s.W.get(mix)
            if w is None: w = s.W[mix] = [0.8 / n * 1.4] * n + [0.0]
            x = sum(a * c for a, c in zip(w, st))
            x = max(-12.0, min(12.0, x))
        else:
            x = sum(st) / n
        pm = 1.0 / (1.0 + math.exp(-x))
        p1 = min(4095, max(1, int(pm * 4096)))
        if s.enc: s.rc.encode(b, p1)
        else: b = s.rc.decode(p1)
        if P['mix']:
            err = (b - pm) * s.LR
            for i in range(n + 1): w[i] += err * st[i]
        for e in sl:
            c = e[1]; r = max(1.0 / (c + 1.6), 1.0 / s.LIM)
            e[0] = min(0.9995, max(0.0005, e[0] + (b - e[0]) * r)); e[1] = c + 1
        return b


def code_eg(C, v, key):
    enc = C.enc; x = v + 1 if enc else 0; nb = x.bit_length() if enc else 0
    t = 1
    while C.bit([(key, t)], ('egm',), (nb > t) if enc else None): t += 1
    val = 1
    for j in range(t - 2, -1, -1):
        val = (val << 1) | C.raw((x >> j) & 1 if enc else None)
    return val - 1


def code_coefs(C, q, N, pc, nb):
    enc = C.enc; scan = SCAN[N]; n = N * N; band = BAND[N]
    qs = [int(q[u, v]) for u, v in scan] if enc else None
    last = 0
    if enc:
        for i in range(n - 1, -1, -1):
            if qs[i]: last = i + 1; break
    maxcat = n.bit_length(); cat = last.bit_length() if enc else 0
    t = 0
    while t < maxcat and C.bit([('lc', pc, t, nb), ('lc2', pc, t)], ('lm', pc),
                               (cat > t) if enc else None):
        t += 1
    cat = t
    if cat >= 2:
        v = 1
        for j in range(cat - 2, -1, -1):
            bt = (last >> j) & 1 if enc else None
            b = C.bit([('lmant', pc, cat)], ('lmm', pc), bt) if j == cat - 2 else C.raw(bt)
            v = (v << 1) | b
        last = v
    else:
        last = cat
    A = [0] * n; p1 = p2 = 0
    for k in range(last):
        u, v = scan[k]
        a = abs(A[(u - 1) * N + v]) if u else 0
        b = abs(A[u * N + v - 1]) if v else 0
        nctx = min(a, 2) * 3 + min(b, 2); pctx = min(abs(p1) + abs(p2), 4); bd = band[k]
        if k < last - 1:
            nz = C.bit([('z1', pc, bd, nctx), ('z2', pc, bd, pctx, nb),
                        ('z3', pc, min(u, 7), min(v, 7))], ('zm', pc, bd),
                       (qs[k] != 0) if enc else None)
        else:
            nz = 1
        val = 0
        if nz:
            mag = abs(qs[k]) if enc else 0
            if C.bit([('g1', pc, nctx, bd // 3), ('g1b', pc, pctx)], ('g1m', pc), (mag > 1) if enc else None):
                if C.bit([('g2', pc, nctx, bd // 3), ('g2b', pc, pctx)], ('g2m', pc), (mag > 2) if enc else None):
                    mag = code_eg(C, mag - 3 if enc else 0, ('eg', pc)) + 3
                else: mag = 2
            else: mag = 1
            sg = C.raw(1 if enc and qs[k] < 0 else 0)
            val = -mag if sg else mag
        A[u * N + v] = val; p2 = p1; p1 = val
    out = np.zeros((N, N), np.int32)
    for k, (u, v) in enumerate(scan): out[u, v] = A[u * N + v]
    return out, sum(1 for x in A if x)


def nbctx(nnz, cy, cx):
    s = (nnz[cy, cx - 1] if cx else 0) + (nnz[cy - 1, cx] if cy else 0)
    return 0 if s == 0 else 1 if s <= 4 else 2 if s <= 16 else 3


def code_mode(C, mode, pc, leftmode):
    enc = C.enc; t = 0
    while t < 4 and C.bit([('m', pc, leftmode, t)], ('mm', pc), (mode > t) if enc else None): t += 1
    return t


def code_stream(C, sbs, nsy, nsx):
    """sbs: list of super-block dicts (encode) or None (decode, returns the list)."""
    enc = C.enc; out = []
    nnz = [np.zeros((nsy * 2, nsx * 2), int), np.zeros((nsy, nsx), int), np.zeros((nsy, nsx), int)]
    mp = [np.zeros((nsy * 2, nsx * 2), int), np.zeros((nsy, nsx), int), np.zeros((nsy, nsx), int)]
    sizes = np.zeros((nsy, nsx), int); prevq = 2; i = 0
    for sy in range(nsy):
        for sx in range(nsx):
            sb = sbs[i] if enc else {}; i += 1
            q = sb['qidx'] if enc else 0; t = 0
            while t < 4 and C.bit([('q', prevq, t)], ('qm',), (q > t) if enc else None): t += 1
            q = prevq = t
            if P['adapt']:
                sz = C.bit([('sz', int(sizes[sy, sx - 1]) if sx else 0, int(sizes[sy - 1, sx]) if sy else 0)],
                           ('szm',), sb['size'] if enc else None)
            else:
                sz = 0
            sizes[sy, sx] = sz
            res = []; bi = 0
            layout = [(0, 0, 16)] if sz == 0 else [(0, 0, 8), (0, 8, 8), (8, 0, 8), (8, 8, 8)]
            plan = [(0, dy, dx, N) for dy, dx, N in layout] + [(1, 0, 0, 8), (2, 0, 0, 8)]
            for pl, dy, dx, N in plan:
                if pl == 0:
                    cy = sy * 2 + dy // 8; cx = sx * 2 + dx // 8; pc = 0 if N == 16 else 1
                else:
                    cy, cx = sy, sx; pc = 1 + pl
                lm = int(mp[pl][cy, cx - 1]) if cx else 0
                blk = sb['blocks'][bi] if enc else None; bi += 1
                mode = code_mode(C, blk[3] if enc else 0, pc, lm)
                qa, nn = code_coefs(C, blk[4] if enc else None, N, pc, nbctx(nnz[pl], cy, cx))
                r = N // 8 if pl == 0 else 1
                nnz[pl][cy:cy + r, cx:cx + r] = nn; mp[pl][cy:cy + r, cx:cx + r] = mode
                res.append((pl, sy * 16 + dy if pl == 0 else sy * 8, sx * 16 + dx if pl == 0 else sx * 8, mode, qa, N))
            out.append(dict(qidx=q, size=sz, blocks=res))
    return out, sizes


# ----------------------------------------------------------------------------- image side
def rgb_to_planes(rgb):
    f = rgb.astype(np.float64)
    Y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    Cb = (f[..., 2] - Y) * 0.564; Cr = (f[..., 0] - Y) * 0.713
    h, w = Y.shape
    sub = lambda c: c.reshape(h // 2, 2, w // 2, 2).mean((1, 3))
    return Y - 128.0, sub(Cb), sub(Cr)


def planes_to_rgb(Y, Cb, Cr, H0, W0):
    up = lambda c: cv2.resize(c, (c.shape[1] * 2, c.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
    Y = Y + 128.0; B = Y + up(Cb) / 0.564; R = Y + up(Cr) / 0.713
    G = (Y - 0.299 * R - 0.114 * B) / 0.587
    out = np.stack([R, G, B], -1)[:H0, :W0]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def deblock(Y, sizes, qb):
    T = P['deblock'] * qb
    if T <= 0: return Y
    H, W = Y.shape; mv = np.zeros((H, W), bool); mh = np.zeros((H, W), bool)
    for sy in range(sizes.shape[0]):
        for sx in range(sizes.shape[1]):
            y, x = sy * 16, sx * 16
            if sx: mv[y:y + 16, x] = True
            if sy: mh[y, x:x + 16] = True
            if sizes[sy, sx]: mv[y:y + 16, x + 8] = True; mh[y + 8, x:x + 16] = True
    Y = Y.copy()
    ys, xs = np.nonzero(mv); d = Y[ys, xs] - Y[ys, xs - 1]; m = np.abs(d) < T
    Y[ys[m], xs[m] - 1] += 0.3 * d[m]; Y[ys[m], xs[m]] -= 0.3 * d[m]
    ys, xs = np.nonzero(mh); d = Y[ys, xs] - Y[ys - 1, xs]; m = np.abs(d) < T
    Y[ys[m] - 1, xs[m]] += 0.3 * d[m]; Y[ys[m], xs[m]] -= 0.3 * d[m]
    return Y


def reconstruct(sbs, sizes, H, W, qb):
    rec = [np.zeros((H, W)), np.zeros((H // 2, W // 2)), np.zeros((H // 2, W // 2))]
    for sb in sbs:
        for pl, y0, x0, mode, q, N in sb['blocks']:
            Qm, _ = qmats(N, 0 if pl == 0 else 1, qb * QTAB[sb['qidx']])
            apply_block(rec[pl], y0, x0, N, mode, q, Qm)
    reconstruct.pre = [r.copy() for r in rec]      # pre-filter planes (== encoder's closed loop)
    rec[0] = deblock(rec[0], sizes, qb)
    return rec


def aq_indices(Y, nsy, nsx):
    if not P['aq']: return np.full((nsy, nsx), 2)
    g = np.abs(np.diff(Y, axis=0, prepend=Y[:1])) + np.abs(np.diff(Y, axis=1, prepend=Y[:, :1]))
    a = g.reshape(nsy, 16, nsx, 16).mean((1, 3)); gm = a.mean()
    s = ((a + 4) / (gm + 4)) ** 0.25
    return np.abs(s[..., None] - np.array(QTAB)).argmin(-1)


def encode(rgb, qb):
    H0, W0 = rgb.shape[:2]; H = -(-H0 // 16) * 16; W = -(-W0 // 16) * 16
    img = np.pad(rgb, ((0, H - H0), (0, W - W0), (0, 0)), mode='edge')
    sY, sCb, sCr = rgb_to_planes(img); src = [sY, sCb, sCr]
    nsy, nsx = H // 16, W // 16; qi = aq_indices(sY, nsy, nsx)
    rec = [np.zeros((H, W)), np.zeros((H // 2, W // 2)), np.zeros((H // 2, W // 2))]
    sbs = []
    for sy in range(nsy):
        for sx in range(nsx):
            q = int(qi[sy, sx]); qbl = qb * QTAB[q]; y, x = sy * 16, sx * 16
            backup = rec[0][y:y + 16, x:x + 16].copy()
            b16 = try_block(rec[0], sY, y, x, 16, qbl, 0)
            if P['adapt']:
                subs = []; tot = 2.0
                for dy, dx in ((0, 0), (0, 8), (8, 0), (8, 8)):
                    r = try_block(rec[0], sY, y + dy, x + dx, 8, qbl, 0); tot += r[0]
                    Qm, _ = qmats(8, 0, qbl); apply_block(rec[0], y + dy, x + dx, 8, r[1], r[2], Qm)
                    subs.append((0, y + dy, x + dx, r[1], r[2], 8))
                rec[0][y:y + 16, x:x + 16] = backup
            if not P['adapt'] or b16[0] <= tot:
                size = 0; Qm, _ = qmats(16, 0, qbl); apply_block(rec[0], y, x, 16, b16[1], b16[2], Qm)
                blocks = [(0, y, x, b16[1], b16[2], 16)]
            else:
                size = 1; blocks = subs
                for _, yy, xx, m, qq, N in subs:
                    Qm, _ = qmats(8, 0, qbl); apply_block(rec[0], yy, xx, 8, m, qq, Qm)
            for pl in (1, 2):
                r = try_block(rec[pl], src[pl], sy * 8, sx * 8, 8, qbl, 1)
                Qm, _ = qmats(8, 1, qbl); apply_block(rec[pl], sy * 8, sx * 8, 8, r[1], r[2], Qm)
                blocks.append((pl, sy * 8, sx * 8, r[1], r[2], 8))
            sbs.append(dict(qidx=q, size=size, blocks=[(a, b, c, d, e) for (a, b, c, d, e, f) in blocks]))
    C = Coder(True); code_stream(C, [dict(qidx=s['qidx'], size=s['size'],
             blocks=s['blocks']) for s in sbs], nsy, nsx)
    # (blocks tuples: pl,y0,x0,mode,q) -> code_stream wants index 3=mode, 4=q
    payload = C.rc.finish(); encode.rec = rec
    return struct.pack('<HHH', W0, H0, int(qb * 64)) + payload


def decode(data):
    W0, H0, qb64 = struct.unpack('<HHH', data[:6]); qb = qb64 / 64.0
    H = -(-H0 // 16) * 16; W = -(-W0 // 16) * 16
    C = Coder(False, data[6:])
    sbs, sizes = code_stream(C, None, H // 16, W // 16)
    rec = reconstruct(sbs, sizes, H, W, qb); decode.rec = rec
    return planes_to_rgb(rec[0], rec[1], rec[2], H0, W0)


if __name__ == '__main__':
    from PIL import Image
    if sys.argv[1].endswith('.prm'):
        Image.fromarray(decode(open(sys.argv[1], 'rb').read())).save(sys.argv[2])
    else:
        qb = float(sys.argv[sys.argv.index('--qb') + 1]) if '--qb' in sys.argv else 12.0
        d = encode(np.array(Image.open(sys.argv[1]).convert('RGB')), round(qb * 64) / 64)
        open(sys.argv[2], 'wb').write(d); print(len(d), 'bytes')
