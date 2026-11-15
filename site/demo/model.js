// The TextCNN, chunking, pooling and attribution from legal_risk_classifier,
// ported line for line so the browser gives the same answers as Python.
// Files come from `legal-risk-export-web`. No dependencies.

const TOKEN = /[a-z0-9][a-z0-9.\-\/]*/g; // vocab.tokenize
const WORD = /\S+/g;                      // chunking.word_offsets
const EPS = 1e-6;                         // pooling._EPS
const PAD = 0, UNK = 1;

function halfToFloat(bits) {
  const s = bits & 0x8000 ? -1 : 1, e = (bits >> 10) & 0x1f, m = bits & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

export function tokenize(text) {
  return text.toLowerCase().match(TOKEN) || [];
}

export function loadModel(config, vocabList, buffer) {
  const halves = new Uint16Array(buffer);
  const weights = {};
  let offset = 0;
  for (const { name, shape } of config.tensors) {
    const size = shape.reduce((a, b) => a * b, 1);
    const out = new Float32Array(size);
    for (let i = 0; i < size; i++) out[i] = halfToFloat(halves[offset + i]);
    weights[name] = out;
    offset += size;
  }
  const stoi = new Map(vocabList.map((token, i) => [token, i]));
  return { config, weights, stoi, cache: new Map() };
}

function encode(model, text, maxTokens) {
  const ids = tokenize(text).slice(0, maxTokens).map(t => model.stoi.get(t) ?? UNK);
  while (ids.length < maxTokens) ids.push(PAD);
  return ids;
}

// A convolution is a sum over its k offsets of (filter weights . embedding),
// so each token's contribution at each offset is computed once and cached.
function projection(model, id) {
  let p = model.cache.get(id);
  if (p) return p;
  const { embedding_dim: D, num_filters: F, kernel_sizes } = model.config;
  const emb = model.weights.embedding.subarray(id * D, (id + 1) * D);
  p = new Float32Array(kernel_sizes.reduce((a, k) => a + k, 0) * F);
  let base = 0;
  kernel_sizes.forEach((k, c) => {
    const w = model.weights[`conv${c}.weight`]; // (F, D, k)
    for (let j = 0; j < k; j++) {
      for (let f = 0; f < F; f++) {
        let sum = 0;
        const row = f * D * k;
        for (let d = 0; d < D; d++) sum += w[row + d * k + j] * emb[d];
        p[base + j * F + f] = sum;
      }
    }
    base += k * F;
  });
  model.cache.set(id, p);
  return p;
}

// One window: pooled filter activations, their positions, and six probabilities.
function forward(model, ids) {
  const { num_filters: F, kernel_sizes, labels } = model.config;
  const proj = ids.map(id => projection(model, id));
  const values = [], positions = [];
  let base = 0;
  kernel_sizes.forEach((k, c) => {
    const bias = model.weights[`conv${c}.bias`];
    for (let f = 0; f < F; f++) {
      let best = -Infinity, at = 0;
      for (let t = 0; t + k <= ids.length; t++) {
        let a = bias[f];
        for (let j = 0; j < k; j++) a += proj[t + j][base + j * F + f];
        a = Math.max(a, 0);
        if (a > best) { best = a; at = t; }
      }
      values.push(best); positions.push(at);
    }
    base += k * F;
  });
  const W = model.weights["classifier.weight"], b = model.weights["classifier.bias"];
  const probs = labels.map((_, l) => {
    let z = b[l];
    for (let i = 0; i < values.length; i++) z += W[l * values.length + i] * values[i];
    return 1 / (1 + Math.exp(-z));
  });
  return { probs, values, positions };
}

export function chunk(text, window, overlap) {
  const offsets = [...text.matchAll(WORD)].map(m => [m.index, m.index + m[0].length]);
  const chunks = [];
  for (let start = 0; start < offsets.length; start += window - overlap) {
    const slice = offsets.slice(start, start + window);
    const [from, to] = [slice[0][0], slice[slice.length - 1][1]];
    chunks.push({ index: chunks.length, start: from, end: to, text: text.slice(from, to) });
    if (start + window >= offsets.length) break;
  }
  return chunks;
}

function pool(column, method) {
  if (method === "max") return Math.max(...column);
  let logSum = 0; // noisy-OR: 1 - prod(1 - p)
  for (const p of column) logSum += Math.log1p(-Math.min(Math.max(p, EPS), 1 - EPS));
  return 1 - Math.exp(logSum);
}

export function predict(model, text) {
  const { chunking, labels, pooling, contract_thresholds } = model.config;
  const chunks = chunk(text, chunking.window, chunking.overlap);
  const probs = chunks.map(c => forward(model, encode(model, c.text, chunking.max_tokens)).probs);
  const scores = labels.map((_, l) => (chunks.length ? pool(probs.map(p => p[l]), pooling) : 0));
  return {
    chunks, probs, scores,
    predicted: labels.map((_, l) => scores[l] >= contract_thresholds[l]),
  };
}

// attribution.explain: the n-grams whose filters pushed a label's logit up most.
export function explain(model, text, label, topN = 6, maxTokens = 400) {
  const words = tokenize(text).slice(0, maxTokens);
  if (!words.length) return [];
  const { values, positions } = forward(model, encode(model, text, maxTokens));
  const { num_filters: F, kernel_sizes, labels } = model.config;
  const W = model.weights["classifier.weight"], row = labels.indexOf(label) * values.length;
  const phrases = [];
  kernel_sizes.forEach((width, c) => {
    const ranked = [];
    for (let f = 0; f < F; f++) {
      const i = c * F + f;
      ranked.push({ i, contribution: values[i] * W[row + i] });
    }
    ranked.sort((a, b) => b.contribution - a.contribution);
    for (const { i, contribution } of ranked.slice(0, topN)) {
      if (contribution <= 0) break;
      const start = positions[i];
      if (start >= words.length) continue; // peaked inside padding
      phrases.push({ text: words.slice(start, Math.min(start + width, words.length)).join(" "), contribution });
    }
  });
  const best = new Map();
  for (const p of phrases) if (!best.has(p.text) || p.contribution > best.get(p.text).contribution) best.set(p.text, p);
  return [...best.values()].sort((a, b) => b.contribution - a.contribution).slice(0, topN);
}
