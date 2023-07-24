#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>

// ----------------------------------------------------------------------------
// Transformer and RunState structs

struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

struct TransformerWeights {
    std::vector<float> token_embedding_table;
    std::vector<float> rms_att_weight;
    std::vector<float> rms_ffn_weight;
    std::vector<float> wq;
    std::vector<float> wk;
    std::vector<float> wv;
    std::vector<float> wo;
    std::vector<float> w1;
    std::vector<float> w2;
    std::vector<float> w3;
    std::vector<float> rms_final_weight;
    std::vector<float> freq_cis_real;
    std::vector<float> freq_cis_imag;
};

struct RunState {
    std::vector<float> x;
    std::vector<float> xb;
    std::vector<float> xb2;
    std::vector<float> hb;
    std::vector<float> hb2;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> att;
    std::vector<float> logits;
    std::vector<float> key_cache;
    std::vector<float> value_cache;
};

void malloc_run_state(RunState& s, const Config& p) {
    s.x.resize(p.dim);
    s.xb.resize(p.dim);
    s.xb2.resize(p.dim);
    s.hb.resize(p.hidden_dim);
    s.hb2.resize(p.hidden_dim);
    s.q.resize(p.dim);
    s.k.resize(p.dim);
    s.v.resize(p.dim);
    s.att.resize(p.seq_len);
    s.logits.resize(p.vocab_size);
    s.key_cache.resize(p.n_layers * p.seq_len * p.dim);
    s.value_cache.resize(p.n_layers * p.seq_len * p.dim);
}

void malloc_weights(TransformerWeights& w, const Config& p) {
    w.token_embedding_table.resize(p.vocab_size * p.dim);
    w.rms_att_weight.resize(p.n_layers * p.dim);
    w.rms_ffn_weight.resize(p.n_layers * p.dim);
    w.wq.resize(p.n_layers * p.dim * p.dim);
    w.wk.resize(p.n_layers * p.dim * p.dim);
    w.wv.resize(p.n_layers * p.dim * p.dim);
    w.wo.resize(p.n_layers * p.dim * p.dim);
    w.w1.resize(p.n_layers * p.hidden_dim * p.dim);
    w.w2.resize(p.n_layers * p.dim * p.hidden_dim);
    w.w3.resize(p.n_layers * p.hidden_dim * p.dim);
    w.rms_final_weight.resize(p.dim);
    w.freq_cis_real.resize(p.seq_len * p.dim / 2);
    w.freq_cis_imag.resize(p.seq_len * p.dim / 2);
}

void free_run_state(RunState& s) {
    // vectors will automatically be deallocated when going out of scope
}

void free_weights(TransformerWeights& w) {
    // vectors will automatically be deallocated when going out of scope
}

// ----------------------------------------------------------------------------
// Helper functions

void rmsnorm(std::vector<float>& o, const std::vector<float>& x, const std::vector<float>& weight) {
    float ss = 0.0f;
    for (float val : x) {
        ss += val * val;
    }
    ss /= static_cast<float>(x.size());
    ss += 1e-5f;
    ss = 1.0f / std::sqrt(ss);
    for (size_t i = 0; i < x.size(); i++) {
        o[i] = weight[i] * (ss * x[i]);
    }
}

void softmax(std::vector<float>& x) {
    float max_val = x[0];
    for (float val : x) {
        max_val = std::max(max_val, val);
    }
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (size_t i = 0; i < x.size(); i++) {
        x[i] /= sum;
    }
}

void matmul(std::vector<float>& xout, const std::vector<float>& x, const std::vector<float>& w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

int sample(const std::vector<float>& probabilities) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
    return dist(gen);
}

// ----------------------------------------------------------------------------
// Transformer function

void transformer(int token, int pos, const Config& p, RunState& s, TransformerWeights& w) {
    std::vector<float>& x = s.x;
    int dim = p.dim;
    int hidden_dim = p.hidden_dim;
    int head_size = dim / p.n_heads;

    // Copy the token embedding into x
    std::copy(w.token_embedding_table.begin() + token * dim,
              w.token_embedding_table.begin() + (token + 1) * dim,
              x.begin());

    // Pluck out the "pos" row of freq_cis_real and freq_cis_imag
    std::copy(w.freq_cis_real.begin() + pos * head_size / 2,
              w.freq_cis_real.begin() + (pos + 1) * head_size / 2,
              s.freq_cis_real_row.begin());
    std::copy(w.freq_cis_imag.begin() + pos * head_size / 2,
              w.freq_cis_imag.begin() + (pos + 1) * head_size / 2,
              s.freq_cis_imag_row.begin());

    // Forward all the layers
    for (int l = 0; l < p.n_layers; l++) {

        // Attention rmsnorm
        rmsnorm(s.xb, x, w.rms_att_weight);

        // QKV matmuls for this position
        matmul(s.q, s.xb, w.wq, dim, dim);
        matmul(s.k, s.xb, w.wk, dim, dim);
        matmul(s.v, s.xb, w.wv, dim, dim);

        // Apply RoPE rotation to the q and k vectors for each head
        for (int h = 0; h < p.n_heads; h++) {
            // Get the q and k vectors for this head
            float* q = s.q.data() + h * head_size;
            float* k = s.k.data() + h * head_size;
            // Rotate q and k by the freq_cis_real and freq_cis_imag
            for (int i = 0; i < head_size; i += 2) {
                float q0 = q[i];
                float q1 = q[i + 1];
                float k0 = k[i];
                float k1 = k[i + 1];
                float fcr = s.freq_cis_real_row[i / 2];
                float fci = s.freq_cis_imag_row[i / 2];
                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // Save key,value at this time step (pos) to our kv cache
        int loff = l * p.seq_len * dim; // kv cache layer offset for convenience
        float* key_cache_row = s.key_cache.data() + loff + pos * dim;
        float* value_cache_row = s.value_cache.data() + loff + pos * dim;
        std::copy(s.k.begin(), s.k.end(), key_cache_row);
        std::copy(s.v.begin(), s.v.end(), value_cache_row);

        // Multihead attention. Iterate over all heads
        for (int h = 0; h < p.n_heads; h++) {
            // Get the query vector for this head
            float* q = s.q.data() + h * head_size;
            // Iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // Get the key vector for this head and at this timestep
                float* k = s.key_cache.data() + loff + t * dim + h * head_size;
                // Calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= std::sqrtf(head_size);
                // Save the score to the attention buffer
                s.att[t] = score;
            }

            // Softmax the scores to get attention weights, from 0..pos inclusively
            softmax(s.att);

            // Weighted sum of the values, store back into xb
            for (int i = 0; i < head_size; i++) {
                float val = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    val += s.att[t] * s.value_cache[loff + t * dim + h * head_size + i]; // Note bad locality
                }
                s.xb[h * head_size + i] = val;
            }
        }

        // Final matmul to get the output of the attention
        matmul(s.xb2, s.xb, w.wo, dim, dim);

        // Residual connection back into x
        for (size_t i = 0; i < x.size(); i++) {
            x[i] += s.xb2[i];
        }

        // FFN rmsnorm
        rmsnorm(s.xb, x, w.rms_ffn_weight);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // First calculate self.w1(x) and self.w3(x)
        matmul(s.hb, s.xb, w.w1, dim, hidden_dim);
        matmul(s.hb2, s.xb, w.w3, dim, hidden_dim);

        // F.silu; silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        for (size_t i = 0; i < s.hb.size(); i++) {
            s.hb[i] = s.hb[i] * (1.0f / (1.0f + std::expf(-s.hb[i])));
        }

        // Elementwise multiply with w3(x)
        for (size_t i = 0; i < s.hb.size(); i++) {
            s.hb[i] = s.hb[i] * s.hb2[i];
        }

        // Final matmul to get the output of the ffn
        matmul(s.xb, s.hb, w.w2, hidden_dim, dim);

        // Residual connection
        for (size_t i = 0; i < x.size(); i++) {
            x[i] += s.xb[i];
        }
    }

    // Final rmsnorm
    rmsnorm(x, x, w.rms_final_weight);

    // Classifier into logits
    matmul(s.logits, x, w.token_embedding_table, p.dim, p.vocab_size);
}

int main(int argc, char* argv[]) {
    std::cout << std::unitbuf; // Disable stdout buffering for immediate output

    // Poor man's C argparse
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <checkpoint_file> [temperature] [seed]\n";
        return 1;
    }

    std::string checkpoint = argv[1];
    float temperature = (argc >= 3) ? std::atof(argv[2]) : 0.9f;
    unsigned int seed = (argc >= 4) ? std::atoi(argv[3]) : static_cast<unsigned int>(std::time(nullptr));

    // Read in the config header
    Config config;
    std::ifstream file(checkpoint, std::ios::binary);
    if (!file) {
        std::cerr << "Unable to open file: " << checkpoint << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(&config), sizeof(Config));

    // Create and initialize the Transformer
    TransformerWeights weights;
    malloc_weights(weights, config);
    checkpoint_init_weights(weights, config, file);
    file.close();

    // Create and initialize the application RunState
    RunState state;
    malloc_run_state(state, config);

    // The current position we are in
    int next;
    int token = 1; // 1 = BOS token in Llama-2 sentencepiece
    int pos = 0;

    while (pos < config.seq_len) {
        // Forward the transformer to get logits for the next token
        transformer(token, pos, config, state, weights);

        // Sample the next token
        if (temperature == 0.0f) {
            // Greedy argmax sampling
            next = argmax(state.logits, config.vocab_size);
        } else {
            // Apply the temperature to the logits
            for (float& val : state.logits) {
                val /= temperature;
            }
            // Apply softmax to the logits to get the probabilities for the next token
            softmax(state.logits);

            // We now want to sample from this distribution to get the next token
            next = sample(state.logits);
        }

        std::cout << next << std::endl;

        // Advance forward
        token = next;
        pos++;
    }

    free_run_state(state);
    free_weights(weights);
    return 0;
}
