#!/usr/bin/env python3
"""
gifcheck.py FILE.gif [FRAME...]

An independent GIF89a decoder used to check src/gif.c: parses every
block, LZW-decodes every image (codes must stay in range, every image must
fill its rectangle exactly), composites the frames with their disposal,
and reports the frame count, total duration and average fps.  Each FRAME
index given is written as FILE-FRAME.png (composited, full size).
"""
import struct
import sys
import zlib


def lzw_decode(data, min_size, npix):
    clear, eoi = 1 << min_size, (1 << min_size) + 1
    width = min_size + 1
    table = [bytes([i]) for i in range(clear)] + [b"", b""]
    out = bytearray()
    prev = None
    acc = nbits = pos = 0
    while True:
        while nbits < width:
            if pos >= len(data):
                raise ValueError("ran out of LZW data")
            acc |= data[pos] << nbits
            pos += 1
            nbits += 8
        code = acc & ((1 << width) - 1)
        acc >>= width
        nbits -= width
        if code == clear:
            table = table[:eoi + 1]
            width = min_size + 1
            prev = None
            continue
        if code == eoi:
            break
        if code < len(table):
            entry = table[code]
            if prev is not None:
                table.append(prev + entry[:1])
        elif code == len(table) and prev is not None:
            entry = prev + prev[:1]
            table.append(entry)
        else:
            raise ValueError("LZW code %d out of range (%d)" % (code, len(table)))
        out += entry
        prev = entry
        if len(table) == (1 << width) and width < 12:
            width += 1
    if len(out) != npix:
        raise ValueError("image has %d pixels, rectangle needs %d" %
                         (len(out), npix))
    return out


def png(path, w, h, rgb_rows):
    raw = b"".join(b"\x00" + bytes(r) for r in rgb_rows)

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def main():
    path = sys.argv[1]
    want = {int(a) for a in sys.argv[2:]}
    d = open(path, "rb").read()
    assert d[:6] == b"GIF89a", "not GIF89a"
    w, h, flags, bg, aspect = struct.unpack_from("<HHBBB", d, 6)
    p = 13
    assert flags & 0x80, "no global colour table"
    ncol = 2 << (flags & 7)
    pal = [tuple(d[p + 3 * i:p + 3 * i + 3]) for i in range(ncol)]
    p += 3 * ncol
    canvas = bytearray(w * h)
    frames = 0
    total_cs = 0
    delay = 0
    disposal = 0
    transp = -1
    loop = False
    while True:
        b = d[p]
        p += 1
        if b == 0x3b:
            break
        if b == 0x21:
            label = d[p]
            p += 1
            if label == 0xf9:
                assert d[p] == 4
                packed, delay, tidx = struct.unpack_from("<BHB", d, p + 1)
                disposal = (packed >> 2) & 7
                transp = tidx if packed & 1 else -1
                p += 6
                assert d[p - 1] == 0
                continue
            if label == 0xff and d[p + 1:p + 12] == b"NETSCAPE2.0":
                loop = True
            while d[p]:
                p += d[p] + 1
            p += 1
            continue
        assert b == 0x2c, "unknown block 0x%02x at %d" % (b, p - 1)
        x, y, iw, ih, f = struct.unpack_from("<HHHHB", d, p)
        p += 9
        assert not f & 0x80, "local palettes are not expected"
        assert x + iw <= w and y + ih <= h, "frame outside the screen"
        min_size = d[p]
        p += 1
        data = bytearray()
        while d[p]:
            data += d[p + 1:p + 1 + d[p]]
            p += d[p] + 1
        p += 1
        pix = lzw_decode(data, min_size, iw * ih)
        assert max(pix) < ncol
        for r in range(ih):
            if transp < 0:
                canvas[(y + r) * w + x:(y + r) * w + x + iw] = \
                    pix[r * iw:(r + 1) * iw]
                continue
            for c in range(iw):     # the transparent index keeps the canvas
                if pix[r * iw + c] != transp:
                    canvas[(y + r) * w + x + c] = pix[r * iw + c]
        if frames in want:
            rows = [[c for i in canvas[r * w:(r + 1) * w] for c in pal[i]]
                    for r in range(h)]
            out = "%s-%d.png" % (path[:-4], frames)
            png(out, w, h, rows)
            print("wrote", out)
        assert disposal in (0, 1), "unexpected disposal %d" % disposal
        transp = -1
        frames += 1
        total_cs += delay
    print("%s: %dx%d, %d colours, %d frames, %.2f s, %.2f fps on average, "
          "loops: %s" % (path, w, h, ncol, frames, total_cs / 100,
                         frames * 100 / total_cs if total_cs else 0, loop))


main()
