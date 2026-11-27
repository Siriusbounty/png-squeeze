// png-squeeze-01 entry: lossless PNG recompression for 1280x720 screenshots.
//
// Pipeline per image: decode the stored RGBA PNG, analyse it (opaque / grey /
// <=256 colours / single transparent key), build the candidate forms (palette
// by count and by luminance, grey / grey+alpha / RGB / RGBA, with tRNS key),
// screen each form x filter choice (fixed filters, min-sum, fewest-pairs,
// per-row brute force with zlib trials) with zlib level 6, then write the
// winner with a zopfli-style optimal-parsing DEFLATE encoder written here
// from scratch (LZ77 hash chains, shortest-path optimal parsing over a
// cached match frontier, length-limited Huffman, dynamic/fixed/stored block
// choice). Everything is deterministic: fixed effort budgets, no clock,
// no randomness, no environment reads.
//
// MIT License. See LICENSE. Palette/filter ideas follow the kit's own
// scale.mjs screen (make.fun, MIT); the DEFLATE encoder is original work.

import { deflateSync, inflateSync } from 'node:zlib';
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// ---------------------------------------------------------------- PNG codec

const SIGNATURE = Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10);
const CHANNELS = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

const CRC_TABLE = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(bytes, start, end) {
  let c = -1;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

function adler32(bytes) {
  let a = 1, b = 0;
  for (let i = 0; i < bytes.length;) {
    const end = Math.min(bytes.length, i + 5552);
    for (; i < end; i++) { a += bytes[i]; b += a; }
    a %= 65521; b %= 65521;
  }
  return ((b << 16) | a) >>> 0;
}

function chunkOf(type, data) {
  const out = new Uint8Array(12 + data.length);
  const v = new DataView(out.buffer);
  v.setUint32(0, data.length);
  out[4] = type.charCodeAt(0); out[5] = type.charCodeAt(1);
  out[6] = type.charCodeAt(2); out[7] = type.charCodeAt(3);
  out.set(data, 8);
  v.setUint32(8 + data.length, crc32(out, 4, 8 + data.length));
  return out;
}

function pngOf(chunks) {
  let size = SIGNATURE.length;
  for (const c of chunks) size += c.length;
  const out = new Uint8Array(size);
  out.set(SIGNATURE, 0);
  let at = SIGNATURE.length;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  return out;
}

function ihdrOf(width, height, colourType) {
  const d = new Uint8Array(13);
  const v = new DataView(d.buffer);
  v.setUint32(0, width); v.setUint32(4, height);
  d[8] = 8; d[9] = colourType;
  return d;
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = p > a ? p - a : a - p;
  const pb = p > b ? p - b : b - p;
  const pc = p > c ? p - c : c - p;
  if (pa <= pb && pa <= pc) return a;
  return pb <= pc ? b : c;
}

function filterRow(type, cur, prev, bpp, out) {
  const n = cur.length;
  if (type === 0) { out.set(cur); return; }
  if (type === 1) {
    for (let i = 0; i < n; i++) out[i] = cur[i] - (i >= bpp ? cur[i - bpp] : 0);
  } else if (type === 2) {
    if (prev === null) { out.set(cur); return; }
    for (let i = 0; i < n; i++) out[i] = cur[i] - prev[i];
  } else if (type === 3) {
    for (let i = 0; i < n; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0;
      const b = prev === null ? 0 : prev[i];
      out[i] = cur[i] - ((a + b) >> 1);
    }
  } else {
    for (let i = 0; i < n; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0;
      const b = prev === null ? 0 : prev[i];
      const c = prev !== null && i >= bpp ? prev[i - bpp] : 0;
      out[i] = cur[i] - paeth(a, b, c);
    }
  }
}

function signedSum(row) {
  let s = 0;
  for (let i = 0; i < row.length; i++) { const v = row[i]; s += v < 128 ? v : 256 - v; }
  return s;
}

/** Decode any bit-depth-8, non-interlaced PNG of colour type 0/2/3/4/6 to RGBA. */
function decodePng(bytes) {
  if (bytes.length < 8 || SIGNATURE.some((b, i) => bytes[i] !== b)) throw new Error('bad PNG signature');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let at = 8, header = null, palette = null, trns = null;
  const idat = [];
  let ended = false;
  while (at < bytes.length) {
    if (ended) throw new Error('bytes after IEND');
    if (at + 12 > bytes.length) throw new Error('short chunk');
    const len = view.getUint32(at);
    if (at + 12 + len > bytes.length) throw new Error('chunk overruns file');
    const type = String.fromCharCode(bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]);
    if (crc32(bytes, at + 4, at + 8 + len) !== view.getUint32(at + 8 + len)) throw new Error(type + ' CRC');
    const data = bytes.subarray(at + 8, at + 8 + len);
    at += 12 + len;
    if (type === 'IHDR') {
      const dv = new DataView(data.buffer, data.byteOffset, 13);
      header = { width: dv.getUint32(0), height: dv.getUint32(4), depth: data[8], colourType: data[9], interlace: data[12] };
      if (header.depth !== 8 || !Object.hasOwn(CHANNELS, header.colourType) || header.interlace !== 0) throw new Error('unsupported IHDR');
    } else if (type === 'PLTE') palette = data;
    else if (type === 'tRNS') trns = data;
    else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') ended = true;
    else throw new Error('unexpected chunk ' + type);
  }
  if (!header || !ended || idat.length === 0) throw new Error('incomplete PNG');
  const { width, height, colourType } = header;
  const bpp = CHANNELS[colourType];
  const stride = width * bpp;
  const expected = height * (stride + 1);
  const raw = inflateSync(idat.length === 1 ? idat[0] : Buffer.concat(idat), { maxOutputLength: expected + 1 });
  if (raw.length !== expected) throw new Error('bad inflate length');
  const samples = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const ft = raw[y * (stride + 1)];
    if (ft > 4) throw new Error('bad filter ' + ft);
    const src = y * (stride + 1) + 1, o = y * stride, p = o - stride;
    if (ft === 0) samples.set(raw.subarray(src, src + stride), o);
    else if (ft === 1) for (let i = 0; i < stride; i++) samples[o + i] = raw[src + i] + (i >= bpp ? samples[o + i - bpp] : 0);
    else if (ft === 2) for (let i = 0; i < stride; i++) samples[o + i] = raw[src + i] + (y > 0 ? samples[p + i] : 0);
    else if (ft === 3) for (let i = 0; i < stride; i++) samples[o + i] = raw[src + i] + (((i >= bpp ? samples[o + i - bpp] : 0) + (y > 0 ? samples[p + i] : 0)) >> 1);
    else for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? samples[o + i - bpp] : 0;
      const b = y > 0 ? samples[p + i] : 0;
      const c = y > 0 && i >= bpp ? samples[p + i - bpp] : 0;
      samples[o + i] = raw[src + i] + paeth(a, b, c);
    }
  }
  const n = width * height;
  const rgba = new Uint8Array(n * 4);
  if (colourType === 6) rgba.set(samples);
  else if (colourType === 2) {
    const key = trns === null ? -1 : (trns[1] << 16) | (trns[3] << 8) | trns[5];
    for (let i = 0, s = 0, d = 0; i < n; i++, s += 3, d += 4) {
      rgba[d] = samples[s]; rgba[d + 1] = samples[s + 1]; rgba[d + 2] = samples[s + 2];
      rgba[d + 3] = ((samples[s] << 16) | (samples[s + 1] << 8) | samples[s + 2]) === key ? 0 : 255;
    }
  } else if (colourType === 0) {
    const key = trns === null ? -1 : trns[1];
    for (let i = 0, d = 0; i < n; i++, d += 4) {
      const g = samples[i];
      rgba[d] = rgba[d + 1] = rgba[d + 2] = g;
      rgba[d + 3] = g === key ? 0 : 255;
    }
  } else if (colourType === 4) {
    for (let i = 0, s = 0, d = 0; i < n; i++, s += 2, d += 4) {
      rgba[d] = rgba[d + 1] = rgba[d + 2] = samples[s];
      rgba[d + 3] = samples[s + 1];
    }
  } else {
    const entries = palette.length / 3;
    const lut = new Uint8Array(entries * 4);
    for (let e = 0; e < entries; e++) {
      lut[e * 4] = palette[e * 3]; lut[e * 4 + 1] = palette[e * 3 + 1]; lut[e * 4 + 2] = palette[e * 3 + 2];
      lut[e * 4 + 3] = trns !== null && e < trns.length ? trns[e] : 255;
    }
    for (let i = 0, d = 0; i < n; i++, d += 4) {
      const e = samples[i];
      if (e >= entries) throw new Error('palette index out of range');
      rgba[d] = lut[e * 4]; rgba[d + 1] = lut[e * 4 + 1]; rgba[d + 2] = lut[e * 4 + 2]; rgba[d + 3] = lut[e * 4 + 3];
    }
  }
  return { width, height, rgba };
}

// ---------------------------------------------------------------- analysis

function wordsOf(rgba) {
  const at = rgba.byteOffset % 4 === 0 ? rgba : rgba.slice();
  return new Uint32Array(at.buffer, at.byteOffset, at.length >>> 2);
}

/** opaque / grey / colours (<=256, else null). */
function analyse(rgba) {
  const n = rgba.length >>> 2;
  const u32 = wordsOf(rgba);
  let opaque = true, grey = true;
  const counts = new Map();
  let many = false;
  for (let i = 0, d = 0; i < n; i++, d += 4) {
    if (rgba[d + 3] !== 255) opaque = false;
    if (rgba[d] !== rgba[d + 1] || rgba[d] !== rgba[d + 2]) grey = false;
    if (!many) {
      const c = u32[i];
      const k = counts.get(c);
      if (k === undefined) {
        if (counts.size === 256) many = true;
        else counts.set(c, 1);
      } else counts.set(c, k + 1);
    }
  }
  const colours = many ? null : [...counts.keys()].map((c) => {
    const b = new Uint8Array(Uint32Array.of(c).buffer);
    return { rgba: [b[0], b[1], b[2], b[3]], word: c, count: counts.get(c) };
  });
  return { opaque, grey, colours };
}

/** The one fully transparent value the image may key by tRNS, else null. */
function transparentKey(rgba, grey) {
  let key = -1;
  for (let d = 0; d < rgba.length; d += 4) {
    const a = rgba[d + 3];
    if (a === 255) continue;
    if (a !== 0) return null;
    const v = grey ? rgba[d] : (rgba[d] << 16) | (rgba[d + 1] << 8) | rgba[d + 2];
    if (key === -1) key = v;
    else if (key !== v) return null;
  }
  if (key === -1) return null;
  for (let d = 0; d < rgba.length; d += 4) {
    if (rgba[d + 3] !== 255) continue;
    const v = grey ? rgba[d] : (rgba[d] << 16) | (rgba[d + 1] << 8) | rgba[d + 2];
    if (v === key) return null;
  }
  return key;
}

function samplesOf(rgba, ct) {
  const n = rgba.length >>> 2;
  if (ct === 6) return rgba;
  const ch = ct === 0 ? 1 : ct === 4 ? 2 : 3;
  const out = new Uint8Array(n * ch);
  for (let i = 0, d = 0, s = 0; i < n; i++, d += 4, s += ch) {
    if (ct === 0) out[s] = rgba[d];
    else if (ct === 4) { out[s] = rgba[d]; out[s + 1] = rgba[d + 3]; }
    else { out[s] = rgba[d]; out[s + 1] = rgba[d + 1]; out[s + 2] = rgba[d + 2]; }
  }
  return out;
}

function paletteOrder(colours, order) {
  const lum = (c) => 299 * c.rgba[0] + 587 * c.rgba[1] + 114 * c.rgba[2];
  return [...colours].sort((x, y) => {
    const ta = x.rgba[3] === 255 ? 1 : 0;
    const tb = y.rgba[3] === 255 ? 1 : 0;
    if (ta !== tb) return ta - tb;
    if (ta === 0 && x.rgba[3] !== y.rgba[3]) return x.rgba[3] - y.rgba[3];
    if (order === 'luminance') return lum(x) - lum(y) || x.word - y.word;
    return y.count - x.count || x.word - y.word;
  });
}

function paletteForm(rgba, colours, order) {
  const entries = paletteOrder(colours, order);
  const index = new Map(entries.map((c, i) => [c.word, i]));
  const plte = new Uint8Array(entries.length * 3);
  let transparent = 0;
  entries.forEach((c, i) => {
    plte[i * 3] = c.rgba[0]; plte[i * 3 + 1] = c.rgba[1]; plte[i * 3 + 2] = c.rgba[2];
    if (c.rgba[3] !== 255) transparent = i + 1;
  });
  const trns = transparent === 0 ? null : Uint8Array.from(entries.slice(0, transparent), (c) => c.rgba[3]);
  const n = rgba.length >>> 2;
  const u32 = wordsOf(rgba);
  const indices = new Uint8Array(n);
  for (let i = 0; i < n; i++) indices[i] = index.get(u32[i]);
  return { plte, trns, indices };
}

function formsOf(rgba) {
  const { opaque, grey, colours } = analyse(rgba);
  const forms = [];
  if (colours !== null) {
    for (const [order, filters] of [['count', [0]], ['luminance', [0, 'minsum', 'brute']]]) {
      const p = paletteForm(rgba, colours, order);
      forms.push({ colourType: 3, bpp: 1, samples: p.indices, plte: p.plte, trns: p.trns, filters });
    }
    if (!grey) return forms;
  }
  const key = opaque ? null : transparentKey(rgba, grey);
  let ct, trns = null;
  if (grey) {
    ct = opaque || key !== null ? 0 : 4;
    if (key !== null) trns = Uint8Array.of(0, key);
  } else {
    ct = opaque || key !== null ? 2 : 6;
    if (key !== null) trns = Uint8Array.of(0, key >>> 16, 0, (key >>> 8) & 0xff, 0, key & 0xff);
  }
  const bpp = { 0: 1, 2: 3, 4: 2, 6: 4 }[ct];
  forms.push({ colourType: ct, bpp, samples: samplesOf(rgba, ct), plte: null, trns, filters: [0, 1, 2, 4, 'minsum', 'pairs', 'brute'] });
  return forms;
}

// ---------------------------------------------------------------- filtering

const SCREEN = Object.freeze({ level: 6, memLevel: 8 });
const BRUTE_CONTEXT = 32768;
const BRUTE_LEVEL = 4;

function pairsOf(row, seen, stamp) {
  let n = 0;
  for (let i = 1; i < row.length; i++) {
    const pair = (row[i - 1] << 8) | row[i];
    if (seen[pair] !== stamp) { seen[pair] = stamp; n++; }
  }
  return n;
}

/** Filtered data for `samples` by `choice` (fixed type, 'minsum', 'pairs', 'brute'). */
function filtered(samples, width, height, bpp, choice, raw = new Uint8Array(height * (width * bpp + 1))) {
  const stride = width * bpp;
  const trial = Array.from({ length: 5 }, () => new Uint8Array(stride));
  const seen = new Uint32Array(65536);
  let stamp = 0;
  for (let y = 0; y < height; y++) {
    const cur = samples.subarray(y * stride, (y + 1) * stride);
    const prev = y === 0 ? null : samples.subarray((y - 1) * stride, y * stride);
    const at = y * (stride + 1);
    if (typeof choice === 'number') {
      raw[at] = choice;
      filterRow(choice, cur, prev, bpp, raw.subarray(at + 1, at + 1 + stride));
      continue;
    }
    const before = raw.subarray(Math.max(0, at - BRUTE_CONTEXT), at);
    let type = 0, best = Infinity;
    for (let t = 0; t < 5; t++) {
      filterRow(t, cur, prev, bpp, trial[t]);
      let cost;
      if (choice === 'minsum') cost = signedSum(trial[t]);
      else if (choice === 'pairs') cost = pairsOf(trial[t], seen, ++stamp);
      else {
        const row = new Uint8Array(stride + 1);
        row[0] = t; row.set(trial[t], 1);
        cost = deflateSync(row, before.length === 0
          ? { level: BRUTE_LEVEL, memLevel: 9 }
          : { level: BRUTE_LEVEL, memLevel: 9, dictionary: before }).length;
      }
      if (cost < best) { best = cost; type = t; }
    }
    raw[at] = type;
    raw.set(trial[type], at + 1);
  }
  return raw;
}

const chunkBytes = (form) =>
  (form.plte === null ? 0 : 12 + form.plte.length) + (form.trns === null ? 0 : 12 + form.trns.length);

/** Screen every form x filter choice with zlib level 6; return the winner. */
function screen(rgba, width, height) {
  let best = null, spare = null;
  let brute = null; // best 'brute' candidate, for the tie-break below
  for (const form of formsOf(rgba)) {
    const size = height * (width * form.bpp + 1);
    const chunks = chunkBytes(form);
    for (const choice of form.filters) {
      const data = filtered(form.samples, width, height, form.bpp, choice,
        spare !== null && spare.length === size ? spare : new Uint8Array(size));
      spare = null;
      const zlib = deflateSync(data, SCREEN);
      const bytes = zlib.length + chunks;
      if (choice === 'brute' && (brute === null || bytes < brute.bytes)) {
        brute = { form, choice, bytes };
      }
      if (best === null || bytes < best.bytes) {
        if (best !== null) spare = best.data;
        best = { form, choice, data, bytes };
      } else spare = data;
    }
  }
  // Tie-break: per-row brute force is judged by zlib trials, which underrate
  // it under a stronger encoder; when the screen winner is the 'pairs'
  // heuristic and brute is within 0.5% of it, brute usually wins once
  // optimally parsed, so take it. (Restricted to 'pairs' winners: against a
  // fixed filter like 0 the proxy gap does not predict the same way.)
  if (brute !== null && best.choice === 'pairs' && brute.bytes <= best.bytes * 1.005) {
    const form = brute.form;
    const size = height * (width * form.bpp + 1);
    const data = filtered(form.samples, width, height, form.bpp, 'brute', new Uint8Array(size));
    return { form, choice: 'brute', data, bytes: brute.bytes };
  }
  return best;
}

// ---------------------------------------------------------------- DEFLATE
// Original zopfli-style optimal-parsing encoder: LZ77 hash chains, a
// cached per-position match frontier, shortest-path optimal parsing under a
// cost model from the parse's own symbol frequencies (iterated), and
// length-limited Huffman codes, with dynamic / fixed / stored block choice.

const LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const MAX_MATCH = 258;
const MIN_MATCH = 3;
const WINDOW = 32768;

const CHAIN_LIMIT = 256;   // hash-chain candidates per position (new matches only;
                           // long matches ride the carried frontier)
const NICE_LEN = 258;      // stop the chain walk at a match this long
const GOOD_LEN = 32;       // past this, quarter the chain limit
// Effort by filtered size (fixed, data-dependent, never clock-based): small
// images get a longer chain walk with no good-match cut-off; large ones a
// moderate walk, so the whole set stays well inside the wall budget.
const SMALL_MAX = 1200000; // filtered bytes at or below which an image is small
const SMALL_CHAIN = 512;
const SMALL_GOOD = 1000000; // good-match cut-off disabled for small images
const BIG_CHAIN = 256;
const BIG_GOOD = 32;
const PARSE_ITERS = 3;     // optimal-parse passes per block (first: fixed costs)
const BLOCK_SIZE = 1048576; // input bytes per DEFLATE block (1 MiB: Huffman tables
                           // adapt to local statistics without losing much match context)

function lenCodeOf(len) {
  let lo = 0, hi = 28;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (LEN_BASE[m] <= len) lo = m; else hi = m - 1; }
  return lo;
}
function distCodeOf(d) {
  let lo = 0, hi = 29;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (DIST_BASE[m] <= d) lo = m; else hi = m - 1; }
  return lo;
}

class BitWriter {
  constructor(cap) { this.buf = new Uint8Array(cap); this.pos = 0; this.bitbuf = 0; this.bitcnt = 0; }
  grow(need) {
    if (this.pos + need + 8 > this.buf.length) {
      const nb = new Uint8Array(Math.max(this.buf.length * 2, this.pos + need + 1024));
      nb.set(this.buf.subarray(0, this.pos));
      this.buf = nb;
    }
  }
  bits(v, n) {
    this.bitbuf |= (v & ((1 << n) - 1)) << this.bitcnt;
    this.bitcnt += n;
    while (this.bitcnt >= 8) {
      this.grow(1);
      this.buf[this.pos++] = this.bitbuf & 0xff;
      this.bitbuf >>>= 8;
      this.bitcnt -= 8;
    }
  }
  align() {
    if (this.bitcnt > 0) { this.grow(1); this.buf[this.pos++] = this.bitbuf & 0xff; this.bitbuf = 0; this.bitcnt = 0; }
  }
  bytes(b) { this.align(); this.grow(b.length); this.buf.set(b, this.pos); this.pos += b.length; }
  result() { this.align(); return this.buf.subarray(0, this.pos); }
}

/** Length-limited Huffman code lengths from symbol frequencies (max 15). */
function huffmanLengths(freq, maxLen) {
  const n = freq.length;
  const lengths = new Uint8Array(n);
  let count = 0, single = -1;
  for (let i = 0; i < n; i++) if (freq[i] > 0) { count++; single = i; }
  if (count === 0) { lengths[0] = 1; return lengths; }
  if (count === 1) {
    lengths[single] = 1;
    for (let i = 0; i < n; i++) if (i !== single) { lengths[i] = 1; break; }
    return lengths;
  }
  const nf = [], ns = [], nl = [], nr = [];
  const heap = [];
  const push = (node) => {
    heap.push(node);
    let i = heap.length - 1;
    while (i > 0) { const p = (i - 1) >> 1; if (nf[heap[p]] <= nf[heap[i]]) break; const t = heap[p]; heap[p] = heap[i]; heap[i] = t; i = p; }
  };
  const pop = () => {
    const top = heap[0], last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1, r = l + 1;
        let m = i;
        if (l < heap.length && nf[heap[l]] < nf[heap[m]]) m = l;
        if (r < heap.length && nf[heap[r]] < nf[heap[m]]) m = r;
        if (m === i) break;
        const t = heap[m]; heap[m] = heap[i]; heap[i] = t; i = m;
      }
    }
    return top;
  };
  for (let i = 0; i < n; i++) if (freq[i] > 0) { nf.push(freq[i]); ns.push(i); nl.push(-1); nr.push(-1); push(nf.length - 1); }
  while (heap.length > 1) {
    const a = pop(), b = pop();
    nf.push(nf[a] + nf[b]); ns.push(-1); nl.push(a); nr.push(b);
    push(nf.length - 1);
  }
  const stack = [[heap[0], 0]];
  while (stack.length > 0) {
    const top = stack.pop(), node = top[0], d = top[1];
    if (ns[node] >= 0) lengths[ns[node]] = d;
    else { stack.push([nl[node], d + 1]); stack.push([nr[node], d + 1]); }
  }
  // Clamp to maxLen, then repair the Kraft sum with exact integer arithmetic
  // (units of 2^-maxLen): lengthen the longest codes until the sum is at
  // most 1, then shorten the longest codes back to exactly 1. A Huffman code
  // must be complete (Kraft sum exactly 1) or the decoder rejects it.
  let excess = 0;
  for (let i = 0; i < n; i++) {
    if (lengths[i] > maxLen) lengths[i] = maxLen;
    if (lengths[i] > 0) excess += 1 << (maxLen - lengths[i]);
  }
  excess -= 1 << maxLen;
  let guard = 0;
  while (excess > 0 && guard++ < 10000) {
    // Lengthen the longest lengthenable code (rarest first on a tie).
    let best = -1, bestLen = -1;
    for (let i = 0; i < n; i++) {
      if (freq[i] > 0 && lengths[i] < maxLen && lengths[i] > bestLen) { best = i; bestLen = lengths[i]; }
    }
    if (best < 0) break;
    // Among codes of that length take the rarest one.
    for (let i = 0; i < n; i++) {
      if (freq[i] > 0 && lengths[i] === bestLen && freq[i] < freq[best]) best = i;
    }
    excess -= 1 << (maxLen - bestLen - 1);
    lengths[best]++;
  }
  guard = 0;
  while (excess < 0 && guard++ < 10000) {
    // Shorten the longest code (most frequent on a tie) to add Kraft back.
    let best = -1, bestLen = -1;
    for (let i = 0; i < n; i++) {
      if (freq[i] > 0 && lengths[i] > 1 && lengths[i] > bestLen) { best = i; bestLen = lengths[i]; }
    }
    if (best < 0) break;
    for (let i = 0; i < n; i++) {
      if (freq[i] > 0 && lengths[i] === bestLen && freq[i] > freq[best]) best = i;
    }
    excess += 1 << (maxLen - bestLen);
    lengths[best]--;
  }
  return lengths;
}

/** Canonical Huffman codes for code lengths. */
function canonicalCodes(lengths) {
  const n = lengths.length;
  const codes = new Int32Array(n);
  const blCount = new Int32Array(16);
  for (let i = 0; i < n; i++) if (lengths[i] > 0) blCount[lengths[i]]++;
  const nextCode = new Int32Array(16);
  let code = 0;
  for (let b = 1; b <= 15; b++) { code = (code + blCount[b - 1]) << 1; nextCode[b] = code; }
  for (let i = 0; i < n; i++) if (lengths[i] > 0) codes[i] = nextCode[lengths[i]]++;
  return codes;
}

/** Huffman codes with their bits reversed, so BitWriter.bits emits them MSB-first. */
function reverseCodes(codes, lengths) {
  const rev = new Int32Array(codes.length);
  for (let i = 0; i < codes.length; i++) {
    const l = lengths[i];
    if (l === 0) continue;
    let v = codes[i], r = 0;
    for (let b = 0; b < l; b++) { r = (r << 1) | (v & 1); v >>>= 1; }
    rev[i] = r;
  }
  return rev;
}

const FIXED_LIT = (() => {
  const l = new Uint8Array(288);
  for (let i = 0; i < 144; i++) l[i] = 8;
  for (let i = 144; i < 256; i++) l[i] = 9;
  for (let i = 256; i < 280; i++) l[i] = 7;
  for (let i = 280; i < 288; i++) l[i] = 8;
  return l;
})();
const FIXED_DIST = new Uint8Array(32).fill(5);
const CL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

// ---------------------------------------------------------------- LZ77

const HASH_BITS = 15;
const HASH_SIZE = 1 << HASH_BITS;
const HASH_MASK = HASH_SIZE - 1;

function hash3(data, i) {
  const v = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
  return ((v * 2654435761) >>> 17) & HASH_MASK;
}

function matchLen(data, a, b, maxLen) {
  let len = 0;
  while (len + 4 <= maxLen) {
    if (data[a + len] !== data[b + len]) return len;
    if (data[a + len + 1] !== data[b + len + 1]) return len + 1;
    if (data[a + len + 2] !== data[b + len + 2]) return len + 2;
    if (data[a + len + 3] !== data[b + len + 3]) return len + 3;
    len += 4;
  }
  while (len < maxLen && data[a + len] === data[b + len]) len++;
  return len;
}

/**
 * Per-position match frontier for data[start, end): for each position, the
 * Pareto-optimal (length, distance) pairs — for each distinct distance, the
 * longest match with that distance. Candidates come off the hash chain in
 * increasing distance order, so the first candidate reaching a length gives
 * that length's smallest distance.
 * Returns { off: Int32Array(n+1), lens: number[], dists: number[] }.
 */
/**
 * Per-position match frontier for data[start, end): for each position, the
 * Pareto-optimal (length, distance) pairs — for each distinct distance, the
 * longest match with that distance.
 *
 * A match at one position continues at the next with length one less at the
 * same distance, so the frontier is carried forward (extended to its true
 * length by comparing the bytes past it) and the hash-chain walk only has to
 * find *new* matches — the most recent few candidates, which hold the small
 * distances. That keeps a thorough search cheap.
 * Returns { off: Int32Array(n+1), lens: number[], dists: number[] }.
 */
function buildPairs(data, start, end, chainLimit, niceLen, goodLen) {
  const n = end - start;
  const head = new Int32Array(HASH_SIZE).fill(-1);
  const prev = new Int32Array(n);
  const off = new Int32Array(n + 1);
  const plens = [], pdists = [];
  const sublen = new Int32Array(MAX_MATCH + 1);
  let clen = [], cdist = []; // frontier carried from the previous position
  for (let pos = 0; pos < n; pos++) {
    const i = start + pos;
    const maxPossible = Math.min(MAX_MATCH, n - pos);
    // 1. Continue the frontier: each pair loses one byte of length, then is
    // extended to its true length by comparing the bytes past it.
    const pairs = [];
    for (let k = 0; k < clen.length; k++) {
      let L = clen[k] - 1;
      const d = cdist[k];
      if (L < MIN_MATCH) continue;
      while (L < maxPossible && data[i + L] === data[i - d + L]) L++;
      pairs.push([L, d]);
    }
    // 2. Hash-chain walk for new matches (small limit: the frontier above
    // already holds the long ones; the walk finds the near, new ones).
    if (pos + MIN_MATCH <= n) {
      const h = hash3(data, i);
      let cur = head[h];
      let limit = chainLimit;
      let count = 0;
      let maxLen = 0;
      while (cur >= 0 && count < limit) {
        const dist = pos - cur;
        if (dist > WINDOW) break;
        count++;
        const ci = start + cur;
        if (maxLen >= maxPossible) break;
        if (maxLen >= MIN_MATCH && data[ci + maxLen] !== data[i + maxLen]) { cur = prev[cur]; continue; }
        const len = matchLen(data, i, ci, maxPossible);
        if (len >= MIN_MATCH && len > maxLen) {
          pairs.push([len, dist]);
          maxLen = len;
          if (len >= niceLen) break;
          if (len >= goodLen) limit = chainLimit >> 2;
        }
        cur = prev[cur];
      }
      prev[pos] = head[h];
      head[h] = pos;
    }
    // 3. Merge: smallest distance wins for each length.
    pairs.sort((a, b) => a[1] - b[1] || b[0] - a[0]);
    let setUpTo = 2, maxLen = 0;
    for (let k = 0; k < pairs.length; k++) {
      const L = pairs[k][0], d = pairs[k][1];
      if (L > setUpTo) {
        for (let l = setUpTo + 1; l <= L; l++) sublen[l] = d;
        setUpTo = L;
      }
      if (L > maxLen) maxLen = L;
    }
    // 4. Extract the Pareto pairs; they are the next position's frontier.
    off[pos] = plens.length;
    clen = []; cdist = [];
    if (maxLen >= MIN_MATCH) {
      let L = 3, d = sublen[3];
      while (L <= maxLen) {
        let L2 = L;
        while (L2 + 1 <= maxLen && sublen[L2 + 1] === d) L2++;
        plens.push(L2); pdists.push(d);
        clen.push(L2); cdist.push(d);
        L = L2 + 1;
        if (L <= maxLen) d = sublen[L];
      }
      for (let l = 3; l <= maxLen; l++) sublen[l] = 0;
    }
  }
  off[n] = plens.length;
  return { off, lens: plens, dists: pdists };
}

// ---------------------------------------------------------------- optimal parse

function costTables(litLen, distLen) {
  // Symbols the previous parse never used have length 0; price them at the
  // maximum instead of free, or the parse fills with phantom-free literals.
  const litCost = new Float64Array(256);
  for (let i = 0; i < 256; i++) litCost[i] = litLen[i] || 15;
  const lenCost = new Float64Array(MAX_MATCH + 1);
  for (let L = 3; L <= MAX_MATCH; L++) {
    const c = lenCodeOf(L);
    lenCost[L] = (litLen[257 + c] || 15) + LEN_EXTRA[c];
  }
  const distCost = new Float64Array(WINDOW + 1);
  for (let d = 1; d <= WINDOW; d++) {
    const c = distCodeOf(d);
    distCost[d] = (distLen[c] || 15) + DIST_EXTRA[c];
  }
  return { litCost, lenCost, distCost };
}

/** Shortest-path parse of data[start, end) under the given bit costs. */
function optimalParse(data, start, end, pairs, litCost, lenCost, distCost) {
  const n = end - start;
  const costs = new Float64Array(n + 1);
  costs.fill(Infinity);
  costs[0] = 0;
  const fromLen = new Int32Array(n + 1);
  const fromDist = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) {
    const c = costs[i];
    if (c === Infinity) continue;
    const nc = c + litCost[data[start + i]];
    if (nc < costs[i + 1]) { costs[i + 1] = nc; fromLen[i + 1] = 0; }
    const p0 = pairs.off[i], p1 = pairs.off[i + 1];
    for (let p = p0; p < p1; p++) {
      const L = pairs.lens[p], d = pairs.dists[p];
      const mc = c + lenCost[L] + distCost[d];
      const j = i + L;
      if (mc < costs[j]) { costs[j] = mc; fromLen[j] = L; fromDist[j] = d; }
    }
  }
  const isMatch = [], a = [], b = [];
  let j = n;
  while (j > 0) {
    const L = fromLen[j];
    if (L === 0) { isMatch.push(0); a.push(data[start + j - 1]); b.push(0); j -= 1; }
    else { isMatch.push(1); a.push(L); b.push(fromDist[j]); j -= L; }
  }
  isMatch.reverse(); a.reverse(); b.reverse();
  return { isMatch, a, b };
}

function tokenFreqs(tokens, litFreq, distFreq) {
  litFreq.fill(0); distFreq.fill(0);
  for (let k = 0; k < tokens.isMatch.length; k++) {
    if (tokens.isMatch[k] === 0) litFreq[tokens.a[k]]++;
    else { litFreq[257 + lenCodeOf(tokens.a[k])]++; distFreq[distCodeOf(tokens.b[k])]++; }
  }
  litFreq[256]++;
}

function tokenBits(tokens, litLen, distLen) {
  let bits = 0;
  for (let k = 0; k < tokens.isMatch.length; k++) {
    if (tokens.isMatch[k] === 0) bits += litLen[tokens.a[k]];
    else {
      const lc = lenCodeOf(tokens.a[k]);
      bits += litLen[257 + lc] + LEN_EXTRA[lc];
      const dc = distCodeOf(tokens.b[k]);
      bits += distLen[dc] + DIST_EXTRA[dc];
    }
  }
  return bits;
}

// ---------------------------------------------------------------- block encode

/** RLE of the litlen+dist code lengths, per RFC 1951 section 3.2.7. */
function rleLengths(lens) {
  const rle = [];
  let i = 0;
  const total = lens.length;
  while (i < total) {
    const v = lens[i];
    if (v === 0) {
      let run = 1;
      while (i + run < total && lens[i + run] === 0) run++;
      while (run >= 11) { const take = Math.min(run, 138); rle.push({ sym: 18, extra: take - 11, bits: 7 }); run -= take; i += take; }
      if (run >= 3) { rle.push({ sym: 17, extra: run - 3, bits: 3 }); i += run; }
      else { for (let k = 0; k < run; k++) rle.push({ sym: 0, extra: 0, bits: 0 }); i += run; }
    } else {
      rle.push({ sym: v, extra: 0, bits: 0 });
      i++;
      let run = 0;
      while (i + run < total && lens[i + run] === v) run++;
      while (run >= 3) { const take = Math.min(run, 6); rle.push({ sym: 16, extra: take - 3, bits: 2 }); run -= take; i += take; }
      for (let k = 0; k < run; k++) { rle.push({ sym: v, extra: 0, bits: 0 }); i++; }
    }
  }
  return rle;
}

function writeTokens(bw, tokens, litRev, litLen, distRev, distLen) {
  for (let k = 0; k < tokens.isMatch.length; k++) {
    if (tokens.isMatch[k] === 0) {
      const b = tokens.a[k];
      bw.bits(litRev[b], litLen[b]);
    } else {
      const L = tokens.a[k], D = tokens.b[k];
      const lc = lenCodeOf(L);
      bw.bits(litRev[257 + lc], litLen[257 + lc]);
      if (LEN_EXTRA[lc]) bw.bits(L - LEN_BASE[lc], LEN_EXTRA[lc]);
      const dc = distCodeOf(D);
      bw.bits(distRev[dc], distLen[dc]);
      if (DIST_EXTRA[dc]) bw.bits(D - DIST_BASE[dc], DIST_EXTRA[dc]);
    }
  }
  bw.bits(litRev[256], litLen[256]);
}

function writeDynamic(bw, tokens, litLen, distLen, isFinal) {
  bw.bits(isFinal ? 1 : 0, 1);
  bw.bits(2, 2);
  let hlit = 257;
  for (let i = 287; i >= 257; i--) if (litLen[i] > 0) { hlit = i + 1; break; }
  let hdist = 1;
  for (let i = 31; i >= 1; i--) if (distLen[i] > 0) { hdist = i + 1; break; }
  const lens = new Uint8Array(hlit + hdist);
  lens.set(litLen.subarray(0, hlit), 0);
  lens.set(distLen.subarray(0, hdist), hlit);
  const rle = rleLengths(lens);
  const clf = new Int32Array(19);
  for (const s of rle) clf[s.sym]++;
  const clLen = huffmanLengths(clf, 7);
  let hclen = 4;
  for (let i = 18; i >= 4; i--) if (clLen[CL_ORDER[i]] > 0) { hclen = i + 1; break; }
  bw.bits(hlit - 257, 5);
  bw.bits(hdist - 1, 5);
  bw.bits(hclen - 4, 4);
  for (let i = 0; i < hclen; i++) bw.bits(clLen[CL_ORDER[i]], 3);
  const clRev = reverseCodes(canonicalCodes(clLen), clLen);
  for (const s of rle) {
    bw.bits(clRev[s.sym], clLen[s.sym]);
    if (s.bits) bw.bits(s.extra, s.bits);
  }
  writeTokens(bw, tokens, reverseCodes(canonicalCodes(litLen), litLen), litLen,
    reverseCodes(canonicalCodes(distLen), distLen), distLen);
}

function writeFixed(bw, tokens, isFinal) {
  bw.bits(isFinal ? 1 : 0, 1);
  bw.bits(1, 2);
  writeTokens(bw, tokens, reverseCodes(canonicalCodes(FIXED_LIT), FIXED_LIT), FIXED_LIT,
    reverseCodes(canonicalCodes(FIXED_DIST), FIXED_DIST), FIXED_DIST);
}

function writeStored(bw, data, start, end, isFinal) {
  let pos = start;
  for (;;) {
    const chunkLen = Math.min(65535, end - pos);
    const last = pos + chunkLen >= end;
    bw.bits(isFinal && last ? 1 : 0, 1);
    bw.bits(0, 2);
    bw.align();
    bw.bytes(Uint8Array.of(chunkLen & 0xff, (chunkLen >> 8) & 0xff, (~chunkLen) & 0xff, ((~chunkLen) >> 8) & 0xff));
    bw.bytes(data.subarray(pos, pos + chunkLen));
    pos += chunkLen;
    if (last) break;
  }
}

/** Bits a stored encoding of data[start, end) adds after the current bit position. */
function storedBits(bitcnt, start, end) {
  let bits = 0, pos = start, first = true;
  while (pos < end) {
    const chunkLen = Math.min(65535, end - pos);
    if (first) {
      bits += 3;
      bits += (8 - ((bitcnt + 3) % 8)) % 8;
      first = false;
    } else bits += 3 + 5;
    bits += 32 + chunkLen * 8;
    pos += chunkLen;
  }
  return bits;
}

function writeBlock(bw, data, start, end, isFinal, big) {
  const n = end - start;
  if (n === 0) {
    bw.bits(isFinal ? 1 : 0, 1);
    bw.bits(1, 2);
    const litRev = reverseCodes(canonicalCodes(FIXED_LIT), FIXED_LIT);
    bw.bits(litRev[256], 7);
    return;
  }
  // Effort by image size (fixed, data-dependent, never clock-based): small
  // images get a thorough chain walk with no good-match cut-off; large ones
  // a moderate walk, so the whole set stays well inside the wall budget.
  const pairs = buildPairs(data, start, end, big ? BIG_CHAIN : SMALL_CHAIN, NICE_LEN, big ? BIG_GOOD : SMALL_GOOD);
  let litLen = FIXED_LIT, distLen = FIXED_DIST;
  let tokens = null;
  for (let iter = 0; iter < PARSE_ITERS; iter++) {
    const { litCost, lenCost, distCost } = costTables(litLen, distLen);
    tokens = optimalParse(data, start, end, pairs, litCost, lenCost, distCost);
    const lf = new Int32Array(288), df = new Int32Array(32);
    tokenFreqs(tokens, lf, df);
    litLen = huffmanLengths(lf, 15);
    distLen = huffmanLengths(df, 15);
  }
  // Size the three block forms without emitting, then emit only the winner.
  const dynBits = 3 + dynamicHeaderBits(litLen, distLen) + tokenBits(tokens, litLen, distLen) + litLen[256];
  const fixBits = 3 + tokenBits(tokens, FIXED_LIT, FIXED_DIST) + 7;
  const stoBits = storedBits(bw.bitcnt, start, end);
  if (stoBits <= dynBits && stoBits <= fixBits) writeStored(bw, data, start, end, isFinal);
  else if (fixBits <= dynBits) writeFixed(bw, tokens, isFinal);
  else writeDynamic(bw, tokens, litLen, distLen, isFinal);
}

/** Bits of a dynamic block's header (after the 3-bit block header). */
function dynamicHeaderBits(litLen, distLen) {
  let hlit = 257;
  for (let i = 287; i >= 257; i--) if (litLen[i] > 0) { hlit = i + 1; break; }
  let hdist = 1;
  for (let i = 31; i >= 1; i--) if (distLen[i] > 0) { hdist = i + 1; break; }
  const lens = new Uint8Array(hlit + hdist);
  lens.set(litLen.subarray(0, hlit), 0);
  lens.set(distLen.subarray(0, hdist), hlit);
  const rle = rleLengths(lens);
  const clf = new Int32Array(19);
  for (const s of rle) clf[s.sym]++;
  const clLen = huffmanLengths(clf, 7);
  let hclen = 4;
  for (let i = 18; i >= 4; i--) if (clLen[CL_ORDER[i]] > 0) { hclen = i + 1; break; }
  let bits = 5 + 5 + 4 + 3 * hclen;
  for (const s of rle) bits += clLen[s.sym] + s.bits;
  return bits;
}

/** A whole zlib stream of `data` (Uint8Array) with the optimal-parsing encoder. */
function deflateData(data) {
  const out = new BitWriter(data.length + 65536);
  out.bytes(Uint8Array.of(0x78, 0x9c));
  const big = data.length > SMALL_MAX;
  const nb = Math.max(1, Math.ceil(data.length / BLOCK_SIZE));
  for (let b = 0; b < nb; b++) {
    const start = b * BLOCK_SIZE, end = Math.min(data.length, start + BLOCK_SIZE);
    writeBlock(out, data, start, end, b === nb - 1, big);
  }
  const ad = adler32(data);
  out.bytes(Uint8Array.of((ad >>> 24) & 0xff, (ad >>> 16) & 0xff, (ad >>> 8) & 0xff, ad & 0xff));
  return out.result();
}

// ---------------------------------------------------------------- entry

function pngOfForm(width, height, form, idat) {
  const chunks = [chunkOf('IHDR', ihdrOf(width, height, form.colourType))];
  if (form.plte !== null) chunks.push(chunkOf('PLTE', form.plte));
  if (form.trns !== null) chunks.push(chunkOf('tRNS', form.trns));
  chunks.push(chunkOf('IDAT', idat));
  chunks.push(chunkOf('IEND', new Uint8Array(0)));
  return pngOf(chunks);
}

function processImage(rgba, width, height) {
  const best = screen(rgba, width, height);
  const idat = deflateData(best.data);
  return pngOfForm(width, height, best.form, idat);
}

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
for (const id of readdirSync(I).sort()) {
  try {
    const image = decodePng(new Uint8Array(readFileSync(join(I, id, 'image.png'))));
    const out = processImage(image.rgba, image.width, image.height);
    mkdirSync(join(O, id), { recursive: true });
    writeFileSync(join(O, id, 'image.png'), out);
    console.error(id + ': ' + out.length + ' bytes');
  } catch (err) {
    console.error(id + ': FAILED ' + String(err && err.message ? err.message : err));
  }
}
