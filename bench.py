import sys, io, time, numpy as np, pillow_jxl, pillow_heif
sys.path.insert(0, '.')
import prism
from PIL import Image
from skimage import data
from skimage.metrics import structural_similarity as ssim
from skimage.color import deltaE_ciede2000, rgb2lab
pillow_heif.register_heif_opener()
try:
    from ssimulacra2 import compute_ssimulacra2
    import tempfile, os
    _d = tempfile.mkdtemp()
    def s2(a, b):
        pa, pb = os.path.join(_d, 'a.png'), os.path.join(_d, 'b.png')
        Image.fromarray(a).save(pa); Image.fromarray(b).save(pb)
        return float(compute_ssimulacra2(pa, pb))
    s2(data.astronaut()[:64,:64], data.astronaut()[:64,:64])
except Exception as e:
    print('ssimulacra2 unavailable:', e); s2 = None

def enc(img, fmt, q):
    b = io.BytesIO(); im = Image.fromarray(img)
    if fmt == 'jpeg': im.save(b, 'JPEG', quality=q, optimize=True, progressive=True)
    elif fmt == 'jxl': im.save(b, 'JXL', quality=q, effort=7)
    elif fmt == 'avif': im.save(b, 'AVIF', quality=q, speed=6)
    return b.getvalue()

def metrics(img, dec):
    r = [ssim(img, dec, channel_axis=-1), float(deltaE_ciede2000(rgb2lab(img), rgb2lab(dec)).mean())]
    if s2: r.append(s2(img, dec))
    return r

def curve(img, fmt):
    pts = []
    qs = {'jpeg': range(20, 98, 5), 'jxl': range(25, 98, 6), 'avif': range(15, 98, 7)}[fmt]
    for q in qs:
        d = enc(img, fmt, q); dec = np.array(Image.open(io.BytesIO(d)).convert('RGB'))
        pts.append([len(d)] + metrics(img, dec))
    return np.array(pts)

def prism_curve(img, qbs):
    pts = []
    for qb in qbs:
        d = prism.encode(img, qb); dec = prism.decode(d)
        assert all(np.array_equal(a, b) for a, b in zip(prism.encode.rec, prism.reconstruct.pre)), 'MISMATCH'
        pts.append([len(d)] + metrics(img, dec))
    return np.array(pts)

def size_at(c, col, t, lower_better):
    x = c[:, col]; o = np.argsort(x)
    if t < x.min() or t > x.max(): return float('nan')
    return float(np.exp(np.interp(t, x[o], np.log(c[o, 0]))))

if __name__ == '__main__':
    qbs = [round(v * 64) / 64 for v in (6, 9, 13, 18, 26, 38)]
    for name in sys.argv[1:] or ['astronaut', 'coffee']:
        img = getattr(data, name)(); print('\n==', name, img.shape, flush=True)
        cs = {f: curve(img, f) for f in ('jpeg', 'jxl', 'avif')}
        cs['prism'] = prism_curve(img, qbs)
        print('PRISM points (bytes, ssim, dE00, ssimulacra2):'); print(np.round(cs['prism'], 3))
        targets = [('dE00', 2, [1.5, 2.0, 2.5, 3.0])] + ([('SSIMULACRA2', 3, [50, 60, 70, 80])] if s2 else [])
        for lab, col, ts in targets:
            for t in ts:
                row = {f: size_at(c, col, t, True) for f, c in cs.items()}
                j = row['jpeg']
                print(f'{lab}={t}: ' + ' | '.join(f'{f} {v/1000:6.1f}KB' for f, v in row.items()) +
                      f'  -> prism vs jpeg {100*(1-row["prism"]/j):.0f}%  vs jxl {100*(1-row["prism"]/row["jxl"]):.0f}%  vs avif {100*(1-row["prism"]/row["avif"]):.0f}%', flush=True)
