"""Fetch one image per class from the official Kvasir v2 zip using HTTP range requests (zip central directory)."""
import struct, sys, zlib, urllib.request, concurrent.futures as cf, os
URL = 'https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip'
TOTAL = 2489312085
tail = open(sys.argv[1], 'rb').read(); base = TOTAL - len(tail); out = sys.argv[2]
eocd = tail.rfind(b'PK\x05\x06')
n_entries, cd_size, cd_off = struct.unpack('<HII', tail[eocd + 10:eocd + 20])
pos = cd_off - base; entries = []
for _ in range(n_entries):
    (sig, _, _, _, method, _, _, crc, csize, usize, nlen, xlen, clen, _, _, _, lho) = struct.unpack('<IHHHHHHIIIHHHHHII', tail[pos:pos + 46])
    name = tail[pos + 46:pos + 46 + nlen].decode('utf8')
    entries.append((name, method, csize, lho, crc)); pos += 46 + nlen + xlen + clen
classes = sorted({e[0].split('/')[1] for e in entries if e[0].count('/') == 2 and e[0].endswith('.jpg')})
print(len(entries), 'entries;', len(classes), 'classes:', classes)
counts = {c: sum(1 for e in entries if e[0].split('/')[1:2] == [c] and e[0].endswith('.jpg')) for c in classes}
print('images per class:', counts)
pick = []
for c in classes:
    files = sorted(e for e in entries if e[0].split('/')[1:2] == [c] and e[0].endswith('.jpg'))
    pick.append(files[len(files) // 2])

def get(e):
    name, method, csize, lho, crc = e
    import subprocess
    data = subprocess.run(['curl', '-s', '--max-time', '180', '-r', f'{lho}-{lho + 30 + 1024 + csize}', URL],
                          capture_output=True, check=True).stdout
    nlen, xlen = struct.unpack('<HH', data[26:30])
    raw = data[30 + nlen + xlen:30 + nlen + xlen + csize]
    content = raw if method == 0 else zlib.decompress(raw, -15)
    assert zlib.crc32(content) == crc, 'CRC mismatch'
    path = os.path.join(out, name.split('/')[1] + '.jpg'); open(path, 'wb').write(content)
    return path
with cf.ThreadPoolExecutor(8) as ex:
    for p in ex.map(get, pick): print('ok', p)
