// const fs = require("fs");
import * as FileSystem from 'expo-file-system';
import { Buffer } from 'buffer';
const softmax = (x, size)=> {
    //find max value 
    let maxVal = x[0];
    for (let i = 1; i < size; i++) {
      if (x[i] > maxVal) {
        maxVal = x[i]; 
      }
    }
    // let maxVal = Math.max(...x);
    
  
    // exp and sum
    let sum = 0;
    for (let i = 0; i < size; i++) {
      x[i] = Math.exp(x[i] - maxVal);
      sum += x[i];
    }
  
    // normalize 
    for (let i = 0; i < size; i++) {
      x[i] /= sum;
    }
  
};

const randomU32 = (state) => {
    state = BigInt(state);
  
    state ^= state >> BigInt(12);
    state ^= state << BigInt(25);
    state ^= state >> BigInt(27);
    return Number((state * BigInt("0x2545F4914F6CDD1D")) >> BigInt(32));
  }
  
const randomF32 = (state) => {
    let u32 = randomU32(state);
  
    return (u32 >> 8) / 16777216.0;
};

const matmul = (xout, x, w, n, d) => {

    for (let i = 0; i < d; i++) {
      let val = 0;
      for (let j = 0; j < n; j++) {
        val += w[i*n + j] * x[j];
      }
      xout[i] = val; 
    }
}

const compare = (a, b) => {
    if (a.prob > b.prob) return -1;
    if (a.prob < b.prob) return 1;
    return 0;
}
  
const compareTokens = (a, b) => {
    if (a.str > b.str) return -1;
    if (a.str < b.str) return 1;
    return 0;
}

const rmsnorm = (o, x, weight, size) => {

    // calculate sum of squares  
    let ss = 0.0;
    for (let j = 0; j < size; j++) {
      ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5; 
    ss = 1.0 / Math.sqrt(ss);
  
    // normalize and scale
    for (let j = 0; j < size; j++) {
      o[j] = weight[j] * (ss * x[j]);
    }
}

const memoryMapWeights = (weights, config, data, sharedWeights,offset) => {

    let headSize = config.dim / config.nHeads;
  
    weights.tokenEmbeddingTable = new Float32Array(data.buffer, offset, config.vocabSize * config.dim);
    offset += weights.tokenEmbeddingTable.byteLength;
  
    weights.rmsAttWeight = new Float32Array(data.buffer, offset, config.nLayers * config.dim);
    offset += weights.rmsAttWeight.byteLength;
  
    weights.wq = new Float32Array(data.buffer, offset, config.nLayers * config.dim * config.nHeads * headSize);
    offset += weights.wq.byteLength;
  
    weights.wk = new Float32Array(data.buffer, offset, config.nLayers * config.dim * config.nKVHeads * headSize);
    offset += weights.wk.byteLength;
  
    weights.wv = new Float32Array(data.buffer, offset, config.nLayers * config.dim * config.nKVHeads * headSize);
    offset += weights.wv.byteLength;
  
    weights.wo = new Float32Array(data.buffer, offset, config.nLayers * config.nHeads * headSize * config.dim);
    offset += weights.wo.byteLength;
  
    weights.rmsFFNWeight = new Float32Array(data.buffer, offset, config.nLayers * config.dim);
    offset += weights.rmsFFNWeight.byteLength;
  
    weights.w1 = new Float32Array(data.buffer, offset, config.nLayers * config.dim * config.hiddenDim);
    offset += weights.w1.byteLength;
  
    weights.w2 = new Float32Array(data.buffer, offset, config.nLayers * config.hiddenDim * config.dim );
    offset += weights.w2.byteLength;
  
    weights.w3 = new Float32Array(data.buffer, offset, config.nLayers * config.dim * config.hiddenDim);
    offset += weights.w3.byteLength;
  
    weights.rmsFinalWeight = new Float32Array(data.buffer, offset, config.dim);
    offset += weights.rmsFinalWeight.byteLength;
  
    offset += config.seqLen * headSize / 2;
    offset += config.seqLen * headSize / 2;
  
    if (!sharedWeights) {
      weights.wcls = new Float32Array(data.buffer, offset, config.vocabSize * config.dim); 
    } else {
      weights.wcls = weights.tokenEmbeddingTable; 
    } 
}

const mallocRunState = (state,config) => {

    let kvDim = (config.dim * config.nKVHeads) / config.nHeads;
  
    state.x = new Float32Array(config.dim);
    state.xb = new Float32Array(config.dim);
    state.xb2 = new Float32Array(config.dim);
    
    state.hb = new Float32Array(config.hiddenDim);
    state.hb2 = new Float32Array(config.hiddenDim);
  
    state.q = new Float32Array(config.dim);
    state.k = new Float32Array(kvDim);
    state.v = new Float32Array(kvDim);
  
    state.att = new Float32Array(config.nHeads * config.seqLen);
  
    state.logits = new Float32Array(config.vocabSize);
  
    state.keyCache = new Float32Array(config.nLayers * config.seqLen * kvDim);
    state.valueCache = new Float32Array(config.nLayers * config.seqLen * kvDim);
  
}

const buildTokenizer = async(tokenizer, tokenizerPath, vocabSize) => {
  tokenizer.vocabSize = vocabSize;

  tokenizer.vocab = [];
  tokenizer.vocabScores = [];
  tokenizer.bytePieces = Array(512).fill("");

  for (let i = 0; i < 256; i++) {
    tokenizer.bytePieces[i * 2] = String.fromCharCode(i);
    tokenizer.bytePieces[i * 2 + 1] = "";
  }

  // read in file
  // let data = await fs.promises.readFile(tokenizerPath);
  let data = await FileSystem.readAsStringAsync(tokenizerPath, { encoding: 'base64' });
  data = Buffer.from(data, 'base64');


  // parse header
  let view = new DataView(data.buffer);
  tokenizer.maxTokenLength = view.getInt32(0, true);

  // read vocab
  let offset = 4;
  for (let i = 0; i < vocabSize; i++) {
    tokenizer.vocabScores[i] = view.getFloat32(offset, true);
    offset += 4;

    let len = view.getInt32(offset, true);
    offset += 4;

    let str = "";
    for (let j = 0; j < len; j++) {
      str += String.fromCharCode(data[offset++]);
    }
    tokenizer.vocab[i] = str;
  }
}


const buildSampler = (vocabSize, temperature, topp, rngSeed) => {
    const probIndex = Array(vocabSize);
  
    for(let i=0;i<vocabSize;i++) {
      probIndex[i] = {index:0,prob:0.0};
    }
  
    return {
      vocabSize,
      temperature,
      topp, 
      rngState: rngSeed,
      probIndex: probIndex
    };
}

const sampleTopp = (probabilities, topp, probIndex, coin) => {
    let n0 = 0;
  
    let cutoff = (1 - topp) / (probabilities.length - 1);
  
    for (let i = 0; i < probabilities.length; i++) {
      if (probabilities[i] >= cutoff) {
        probIndex[n0].index = i;
        probIndex[n0].prob = probabilities[i];
        n0++;
      }
    }
    probIndex.sort(compare);
  
    let cumulativeProb = 0;
    let lastIdx = n0 - 1;
  
    for (let i = 0; i < n0; i++) {
      cumulativeProb += probIndex[i].prob;
      if (cumulativeProb > topp) {
        lastIdx = i;
        break;
      }
    }
  
    let r = coin * cumulativeProb;
    let cdf = 0;
  
    for (let i = 0; i <= lastIdx; i++) {
      cdf += probIndex[i].prob;
      if (r < cdf) return probIndex[i].index;
    }
  
    return probIndex[lastIdx].index;
}

const sampleMult = (probabilities, size,coin) => {
    let cdf = 0.0;
  
    for (let i = 0; i < size; i++) {
      cdf += probabilities[i];
      if (coin < cdf) return i;
    }
  
    return probabilities.length - 1;
}

const sampleArgmax = (probabilities, vocabSize) => {
    let maxIdx = 0;
    let maxP = probabilities[0];
  
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxP) {
        maxIdx = i;
        maxP = probabilities[i];
      }
    }
  
    return maxIdx;
}

const sample = (sampler, logits) => {
    let next;
  
    if (sampler.temperature == 0) {
      // argmax sampling
      next = sampleArgmax(logits,sampler.vocabSize);
      // console.log(5,next);
      // process.exit(0);
  
  
    } else {
      
      // apply temperature
      // for (let i = 0; i < logits.length; i++) {
      for (let i = 0; i < sampler.vocabSize; i++) {
        logits[i] /= sampler.temperature; 
      }
  
      softmax(logits, sampler.vocabSize);
  
  // console.log(46,'sampler.rngState',sampler.rngState);
      // sampling
      let coin = Number(randomF32(sampler.rngState));
  
      // console.log(50,'coin',coin);
  
      if (sampler.topp <= 0 || sampler.topp >= 1) {
        next = sampleMult(logits, sampler.vocabSize,coin);
      } else {
        next = sampleTopp(logits, sampler.topp, sampler.probIndex, coin); 
      }
  
    }
    return next;
}

const readCheckpoint = async(checkpointPath) => {

  const SIZE_OF_FLOAT = 4;
  const SIZE_OF_CONFIG = SIZE_OF_FLOAT*7;

  // let data = await fs.promises.readFile(checkpointPath);
  let data = await FileSystem.readAsStringAsync(checkpointPath, { encoding: 'base64' });
  data = Buffer.from(data, 'base64');

  let view = new DataView(data.buffer);
  let config = {};
  config.dim = view.getInt32(SIZE_OF_FLOAT*0, true);
  config.hiddenDim = view.getInt32(SIZE_OF_FLOAT*1, true);
  config.nLayers = view.getInt32(SIZE_OF_FLOAT*2, true);
  config.nHeads = view.getInt32(SIZE_OF_FLOAT*3, true);
  config.nKVHeads = view.getInt32(SIZE_OF_FLOAT*4, true);
  config.vocabSize = Math.abs(view.getInt32(SIZE_OF_FLOAT*5, true));
  config.seqLen = view.getInt32(SIZE_OF_FLOAT*6, true);
  let offset = SIZE_OF_CONFIG;


  const sharedWeights = config.vocabSize > 0 ? 1 : 0;

  let weights = {};
  
  memoryMapWeights(weights, config, data, sharedWeights, offset);

  return {
    config: config,
    weights: weights,
  };
}

const buildTransformer = async(t, checkpointPath) => {
    // In JS, objects are passed by reference, so you can directly modify properties of the object.
    const res = await readCheckpoint(checkpointPath);
    t.config = res.config;
    t.weights = res.weights;
    t.state = {};
    mallocRunState(t.state, t.config);
}

const chat = async(transformer, tokenizer, sampler, userPrompt, sysPrompt, steps) => {

    let bufferSize = 1024;
    let userBuffer = Buffer.alloc(bufferSize);
    let sysBuffer = Buffer.alloc(bufferSize);
  
    let renderedPrompt = '';
    let promptTokens = [];
  
    let token, next, pos = 0;
    let userTurn = true;
  
    while(pos < steps) {
  
      if (userTurn) {
  
        // get system prompt
        if (pos === 0 && sysPrompt) {
          renderedPrompt += `[SYS] ${sysPrompt}\n\n`;
        }
  
        // get user prompt  
        process.stdout.write('User: ');
        let n = await process.stdin.read(userBuffer);
        let userInput = userBuffer.slice(0, n).toString();
        renderedPrompt += `[USER] ${userInput} [/USER]`;
  
        // encode prompt  
        encode(tokenizer, renderedPrompt, 1, 0, promptTokens);
        let userIdx = 0;
        userTurn = false;
  
        console.log('\nAssistant: ');
  
      }
  
      // get token
      if (userIdx < promptTokens.length) {
        token = promptTokens[userIdx++]; 
      } else {
        token = next;
      }
  
      // finish assistant turn on EOS
      if (token === 2) {
        userTurn = true;
        continue;
      }
  
      // forward pass
      let logits = forward(transformer, token, pos);
      next = sample(sampler, logits);
  
      // print assistant response
      if (userIdx >= promptTokens.length && next !== 2) {
        let piece = decode(tokenizer, token, next);
        process.stdout.write(piece);
      }
  
      pos++;
  
    }
  
    console.log();
  
}

const decode = (t, prevToken, token) => {

    let piece = t.vocab[token];
  
    // handle whitespace after BOS 
    if (prevToken === 1 && piece[0] === ' ') {
      piece = piece.substring(1); 
    }
  
    // handle raw byte tokens
    let matches = piece.match(/^<0x(\d{2})>$/);
    if (matches) {
      let byte = parseInt(matches[1], 16);
      piece = t.bytePieces[String.fromCharCode(byte)];
    }
  
    return piece;
}

const encode = (t, text, bos, eos, tokens) => {
    if (!t.sortedVocab) {
      t.sortedVocab = [];
      for (let i = 0; i < t.vocabSize; i++) {
        t.sortedVocab.push({ str: t.vocab[i], id: i });
      }
      t.sortedVocab = t.sortedVocab.sort(compareTokens);
    }
  
    let nTokens = 0;
  
    if (bos) {
      tokens[nTokens] = 1;
      nTokens++;
    }
  
    if (text) {
      let dummyPrefix = t.sortedVocab[0].id;
      tokens[nTokens] = dummyPrefix;
      nTokens++;
    }
  
    for (let i = 0; i < text.length; i++) {
      let char = text.charCodeAt(i);
  
      if ((char & 0xc0) != 0x80) {
        let piece = String.fromCharCode(char);
  
        let id = t.sortedVocab.findIndex(item => item.str === piece);
        if (id != -1) {
          tokens[nTokens] = id;
          nTokens++;
        } else {
          tokens[nTokens] = char + 3;
          nTokens++;
        }
      }
    }
  
    while (true) {
      let bestScore = -Infinity;
      let bestId = -1;
      let bestIdx = -1;
  
      for (let i = 0; i < nTokens - 1; i++) {
        let pair = t.vocab[tokens[i]] + t.vocab[tokens[i + 1]];
        let id = t.sortedVocab.findIndex(item => item.str === pair);
        
        if (id != -1 && t.vocabScores[id] > bestScore) {
          bestScore = t.vocabScores[id];
          bestId = id;
          bestIdx = i;
        }
      }
  
      if (bestIdx == -1) break;
  
      tokens[bestIdx] = bestId;
  
      for (let i = bestIdx + 1; i < nTokens - 1; i++) {
        tokens[i] = tokens[i + 1];
      }
      nTokens--;
    }
  
    if (eos) {
      tokens[nTokens] = 2;
      nTokens++;
    }
    return nTokens;
}

const forward = (transformer, token, pos) => {

    // a few convenience variables
    const p = transformer.config; 
    const w = transformer.weights;
    const s = transformer.state;
    let x = s.x;
    const dim = p.dim;
    const kvDim = (p.dim * p.nKVHeads) / p.nHeads;
    const kvMul = p.nHeads / p.nKVHeads;
    const hiddenDim = p.hiddenDim;  
    const headSize = dim / p.nHeads;
  
    // copy the token embedding into x
    x.set(w.tokenEmbeddingTable.subarray(token * dim, (token + 1) * dim));
  
    // forward all the layers
    for (let l = 0; l < p.nLayers; l++) {
  
      // attention rmsnorm
      rmsnorm(s.xb, x, w.rmsAttWeight.subarray(l * dim, (l + 1) * dim), dim);
   
    
      // qkv matmuls for this position
      matmul(s.q, s.xb, w.wq.subarray(l * dim * dim, (l + 1) * dim * dim), dim, dim);
      matmul(s.k, s.xb, w.wk.subarray(l * dim * kvDim, (l + 1) * dim * kvDim), dim, kvDim);
      matmul(s.v, s.xb, w.wv.subarray(l * dim * kvDim, (l + 1) * dim * kvDim), dim, kvDim);
  
      // RoPE relative positional encoding
      for (let i = 0; i < dim; i += 2) {
        const headDim = i % headSize;
        const freq = 1 / Math.pow(10000, headDim / headSize);    
        const val = pos * freq;
        const fcr = Math.cos(val);
        const fci = Math.sin(val);
        const rotn = i < kvDim ? 2 : 1; 
        for (let v = 0; v < rotn; v++) {
          let vec = v === 0 ? s.q : s.k;
          const v0 = vec[i];
          const v1 = vec[i+1];
          vec[i] = v0 * fcr - v1 * fci;
          vec[i+1] = v0 * fci + v1 * fcr;
        }
      }
  
  
      // save k/v to cache
      const loff = l * p.seqLen * kvDim;
  
      const keyCacheRow = s.keyCache.subarray(loff + pos * kvDim, loff + (pos + 1) * kvDim);
      const valueCacheRow = s.valueCache.subarray(loff + pos * kvDim, loff + (pos + 1) * kvDim);
      keyCacheRow.set(s.k);
      valueCacheRow.set(s.v);
  
      // multihead attention
      for (let h = 0; h < p.nHeads; h++) {
        // get q vector for this head  
        const q = s.q.subarray(h * headSize, (h + 1) * headSize);
        // att scores for this head
        const att = s.att.subarray(h * p.seqLen, (h + 1) * p.seqLen);
        
        for (let t = 0; t <= pos; t++) {
          // get k vector
          const k = s.keyCache.subarray(loff + t * kvDim + Math.floor(h / kvMul) * headSize, loff + t * kvDim + Math.floor(h / kvMul) * headSize + headSize);
  
          // att score
          let score = 0;
          for (let i = 0; i < headSize; i++) {
            score += q[i] * k[i];
          }
          score /= Math.sqrt(headSize);
          // save to att buffer  
          att[t] = score; 
        }
        // softmax att weights  
        softmax(att, pos + 1);
  
        // weighted sum of v into xb
        const xb = s.xb.subarray(h * headSize, (h + 1) * headSize);
        xb.fill(0);
        for (let t = 0; t <= pos; t++) {
          const v = s.valueCache.subarray(loff + t * kvDim + Math.floor(h / kvMul) * headSize, loff + t * kvDim + Math.floor(h / kvMul) * headSize + headSize);
          const a = att[t];
          for (let i = 0; i < headSize; i++) {
            xb[i] += a * v[i];
          }
        }
      }
  
      // final matmul to get att output 
      matmul(s.xb2, s.xb, w.wo.subarray(l * dim * dim, (l + 1) * dim * dim), dim, dim);
  
      // residual connection 
      for (let i = 0; i < dim; i++) {
        x[i] += s.xb2[i]; 
      }
  
      // ffn rmsnorm
      rmsnorm(s.xb, x, w.rmsFFNWeight.subarray(l * dim, (l + 1) * dim), dim);
  
      // w1 and w3 matmuls
      matmul(s.hb, s.xb, w.w1.subarray(l * dim * hiddenDim, (l + 1) * dim * hiddenDim), dim, hiddenDim);
      matmul(s.hb2, s.xb, w.w3.subarray(l * dim * hiddenDim, (l + 1) * dim * hiddenDim), dim, hiddenDim);
  
      // SwiGLU non-linearity
      for (let i = 0; i < hiddenDim; i++) {
        let val = s.hb[i];
        val *= 1 / (1 + Math.exp(-val)); 
        val *= s.hb2[i];
        s.hb[i] = val;
      }
  
      // final matmul for ffn output
      matmul(s.xb, s.hb, w.w2.subarray(l * hiddenDim * dim, (l + 1) * hiddenDim * dim), hiddenDim, dim);
  
      // residual connection
      for (let i = 0; i < dim; i++) {
        x[i] += s.xb[i];
      }
    }
  
    // final rmsnorm
    rmsnorm(x, x, w.rmsFinalWeight, dim);
  
    // classifier into logits
    matmul(s.logits, x, w.wcls, p.dim, p.vocabSize);
  
    return s.logits;
}

const generate = (transformer, tokenizer, sampler, prompt, steps, streamText) => {
    // encode prompt
    let promptTokens = [];
    const numPromptTokens = encode(tokenizer, prompt, 1, 0, promptTokens);
  
    if (numPromptTokens < 1) {
      console.error("something is wrong, expected at least 1 prompt token\n");
      // process.exit(-1);
    }
  
    let token = promptTokens[0];
    let pos = 0;
  

    while (pos < steps) {
      let logits = forward(transformer, token, pos);
  
      let next;
      if (pos < numPromptTokens - 1) {
        // if we are still processing the input prompt, force the next prompt token
        next = promptTokens[pos + 1];
      } else {
        // sample next token
        next = sample(sampler, logits);
      }
      pos++;
  
      // finish on EOS
      if (next === 1) break;
  
      // print token
      let piece = decode(tokenizer, token, next);
      // process.stdout.write(piece);
      // console.log(piece);
      streamText(piece);

      token = next;
  
      // start timer after first iter
      if (pos === 1) {
        startTime = Date.now();
      }
    }
  
    // print stats
    if (pos > 1) {
      let endTime = Date.now();
      let seconds = (endTime - startTime) / 1000;
      // console.log(`Tokens per second: ${(pos - 1) / seconds}`);
      streamText(`\n\nTokens per second: ${(pos - 1) / seconds}`);
    }
}

export {
    softmax,
    randomU32,
    randomF32,
    matmul,
    compare,
    compareTokens,
    rmsnorm,
    memoryMapWeights,
    mallocRunState,
    buildTokenizer,
    buildSampler,
    sampleTopp,
    sampleMult,
    sampleArgmax,
    sample,
    readCheckpoint,
    buildTransformer,
    chat,
    decode,
    encode,
    forward,
    generate,
};