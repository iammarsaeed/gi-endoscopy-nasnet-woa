"""Parallel ranged download via curl subprocesses (resumes each chunk until complete)."""
import os, subprocess, sys, time, threading
url, total, out, n = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
os.makedirs(out + '.parts', exist_ok=True)
chunk = (total + n - 1) // n
ranges = [(i * chunk, min(total, (i + 1) * chunk) - 1) for i in range(n)]
def worker(i, a, b):
    p = os.path.join(out + '.parts', f'{i:03d}')
    while True:
        have = os.path.getsize(p) if os.path.exists(p) else 0
        if a + have > b: return
        with open(p, 'ab') as f:
            subprocess.run(['curl', '-s', '-L', '--max-time', '600', '-r', f'{a + have}-{b}', url], stdout=f)
ts = [threading.Thread(target=worker, args=(i, a, b)) for i, (a, b) in enumerate(ranges)]
[t.start() for t in ts]; t0 = time.time()
while any(t.is_alive() for t in ts):
    time.sleep(20)
    got = sum(os.path.getsize(os.path.join(out + '.parts', f)) for f in os.listdir(out + '.parts'))
    print(f'{time.time()-t0:5.0f}s {got/1048576:7.0f}/{total/1048576:.0f} MB {got/(time.time()-t0)/1048576:.2f} MB/s', flush=True)
with open(out, 'wb') as fo:
    for i in range(n):
        fo.write(open(os.path.join(out + '.parts', f'{i:03d}'), 'rb').read())
print('DONE', os.path.getsize(out) == total, flush=True)
