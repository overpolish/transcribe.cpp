// arch/sortformer/weights.cpp - Sortformer hparam KV reader + weight
// catalog for the Sortformer-specific tensors (encoder_proj + transformer
// + diar head). The conformer weights are loaded separately (parakeet
// reuse); see model.cpp::load.

#include "weights.h"

#include "ggml.h"
#include "gguf.h"
#include "transcribe-log.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace transcribe::sortformer {

namespace {

transcribe_status kv_u32(const gguf_context * g, const char * key, int32_t & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = static_cast<int32_t>(gguf_get_val_u32(g, id));
    return TRANSCRIBE_OK;
}

transcribe_status kv_f32(const gguf_context * g, const char * key, float & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = gguf_get_val_f32(g, id);
    return TRANSCRIBE_OK;
}

transcribe_status kv_str(const gguf_context * g, const char * key, std::string & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = gguf_get_val_str(g, id);
    return TRANSCRIBE_OK;
}

}  // namespace

// Nemotron 3 Diarization (Sortformer v3) as NVIDIA publishes it, converted
// by NeMo-Speech.cpp (nvidia/Nemotron-3-Diarization, *.q8_0.gguf): its own
// `sortformer.*` keys rather than `stt.*`. Its `sortformer.streaming.*` keys
// are training provenance only, so the shipped operating point is the model
// card's offline-style "very high latency" row.
static transcribe_status read_v3_hparams(const gguf_context * g, SortformerHParams & hp) {
    std::string text;
    if (const transcribe_status st = kv_str(g, "sortformer.version", text); st != TRANSCRIBE_OK) {
        return st;
    }
    const auto flag = [&](const char * key) {
        const int64_t id = gguf_find_key(g, key);
        return id >= 0 && gguf_get_val_bool(g, id);
    };
    if (text != "v3" || !flag("sortformer.high_resolution") || !flag("sortformer.learnable_silence") ||
        !flag("sortformer.encoder.pre_block_norm") || flag("sortformer.encoder.qkv_bias") ||
        flag("sortformer.encoder.qk_norm")) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: unsupported NeMo-Speech model (version %s)", text.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    hp.version = 3;

#define RD_U32(key, field)                                                          \
    if (const transcribe_status st = kv_u32(g, key, hp.field); st != TRANSCRIBE_OK) \
    return st
#define RD_F32(key, field)                                                          \
    if (const transcribe_status st = kv_f32(g, key, hp.field); st != TRANSCRIBE_OK) \
    return st

    RD_U32("sortformer.num_speakers", max_speakers);
    RD_U32("sortformer.encoder.n_layers", enc_n_layers);
    RD_U32("sortformer.encoder.d_model", enc_d_model);
    RD_U32("sortformer.encoder.n_heads", enc_n_heads);
    RD_U32("sortformer.encoder.d_ff", enc_d_ff);
    RD_U32("sortformer.encoder.subsampling_factor", enc_subsampling_factor);
    RD_U32("sortformer.encoder.feat_in", enc_feat_in);
    RD_U32("sortformer.encoder.pos_emb_max_len", enc_pos_emb_max_len);
    RD_F32("sortformer.encoder.rope_base", enc_rope_base);
    RD_F32("sortformer.encoder.rotary_fraction", enc_rotary_fraction);
    RD_U32("sortformer.transformer.hidden_size", tf_d_model);
    RD_U32("sortformer.upsample_factor", upsample_factor);
    RD_U32("sortformer.scoring.spkcache_sil_frames_per_spk", stream_sil_frames_per_spk);

    float window_size = 0.0f, window_stride = 0.0f;
    if (const transcribe_status st = kv_f32(g, "sortformer.preprocessor.window_size", window_size);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = kv_f32(g, "sortformer.preprocessor.window_stride", window_stride);
        st != TRANSCRIBE_OK) {
        return st;
    }
    RD_U32("sortformer.preprocessor.sample_rate", fe_sample_rate);
    RD_U32("sortformer.preprocessor.n_fft", fe_n_fft);
    RD_U32("sortformer.preprocessor.features", fe_num_mels);
    RD_F32("sortformer.preprocessor.dither", fe_dither);
    RD_F32("sortformer.preprocessor.preemph", fe_pre_emphasis);
#undef RD_U32
#undef RD_F32
    hp.fe_win_length = static_cast<int32_t>(std::lround(window_size * static_cast<float>(hp.fe_sample_rate)));
    hp.fe_hop_length = static_cast<int32_t>(std::lround(window_stride * static_cast<float>(hp.fe_sample_rate)));
    hp.fe_window     = "hann";
    hp.fe_normalize  = "none";  // NeMo "NA"
    // The upsampler restores the mel grid: one output frame per mel frame.
    hp.frame_hop = hp.fe_hop_length;

    hp.stream_chunk_len              = 340;
    hp.stream_chunk_right_context    = 40;
    hp.stream_fifo_len               = 40;
    hp.stream_spkcache_update_period = 300;
    hp.stream_spkcache_len           = 264;

    const int32_t head_dim = hp.enc_n_heads > 0 ? hp.enc_d_model / hp.enc_n_heads : 0;
    const int32_t n_rot    = static_cast<int32_t>(static_cast<float>(head_dim) * hp.enc_rotary_fraction);
    if (head_dim == 0 || hp.enc_d_model % hp.enc_n_heads != 0 || n_rot <= 0 || n_rot % 2 != 0 ||
        hp.upsample_factor != hp.enc_subsampling_factor || hp.fe_num_mels != hp.enc_feat_in) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: unsupported v3 geometry");
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

transcribe_status read_sortformer_hparams(const gguf_context * g, SortformerHParams & hp) {
    if (g == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (gguf_find_key(g, "stt.sortformer.max_speakers") < 0 && gguf_find_key(g, "sortformer.version") >= 0) {
        return read_v3_hparams(g, hp);
    }

#define RD_U32(key, field)                                                          \
    if (const transcribe_status st = kv_u32(g, key, hp.field); st != TRANSCRIBE_OK) \
    return st
#define RD_F32(key, field)                                                          \
    if (const transcribe_status st = kv_f32(g, key, hp.field); st != TRANSCRIBE_OK) \
    return st
#define RD_STR(key, field)                                                          \
    if (const transcribe_status st = kv_str(g, key, hp.field); st != TRANSCRIBE_OK) \
    return st

    RD_U32("stt.sortformer.max_speakers", max_speakers);
    RD_U32("stt.sortformer.frame_hop", frame_hop);

    RD_U32("stt.sortformer.encoder.n_layers", enc_n_layers);
    RD_U32("stt.sortformer.encoder.d_model", enc_d_model);
    RD_U32("stt.sortformer.encoder.n_heads", enc_n_heads);
    RD_U32("stt.sortformer.encoder.d_ff", enc_d_ff);
    RD_U32("stt.sortformer.encoder.conv_kernel", enc_conv_kernel);
    RD_U32("stt.sortformer.encoder.subsampling_factor", enc_subsampling_factor);
    RD_U32("stt.sortformer.encoder.subsampling_channels", enc_subsampling_channels);
    RD_U32("stt.sortformer.encoder.feat_in", enc_feat_in);
    RD_U32("stt.sortformer.encoder.pos_emb_max_len", enc_pos_emb_max_len);
    RD_STR("stt.sortformer.encoder.conv_norm_type", enc_conv_norm_type);

    RD_U32("stt.sortformer.transformer.n_layers", tf_n_layers);
    RD_U32("stt.sortformer.transformer.d_model", tf_d_model);
    RD_U32("stt.sortformer.transformer.n_heads", tf_n_heads);
    RD_U32("stt.sortformer.transformer.d_ff", tf_d_ff);
    RD_STR("stt.sortformer.transformer.activation", tf_activation);
    {
        const int64_t id = gguf_find_key(g, "stt.sortformer.transformer.pre_ln");
        hp.tf_pre_ln     = (id >= 0) && gguf_get_val_bool(g, id);
    }

    RD_U32("stt.frontend.num_mels", fe_num_mels);
    RD_U32("stt.frontend.sample_rate", fe_sample_rate);
    RD_U32("stt.frontend.n_fft", fe_n_fft);
    RD_U32("stt.frontend.win_length", fe_win_length);
    RD_U32("stt.frontend.hop_length", fe_hop_length);
    RD_STR("stt.frontend.window", fe_window);
    RD_STR("stt.frontend.normalize", fe_normalize);
    RD_F32("stt.frontend.dither", fe_dither);
    RD_F32("stt.frontend.pre_emphasis", fe_pre_emphasis);

    RD_U32("stt.sortformer.stream.chunk_len", stream_chunk_len);
    RD_U32("stt.sortformer.stream.spkcache_len", stream_spkcache_len);
    RD_U32("stt.sortformer.stream.fifo_len", stream_fifo_len);
    RD_U32("stt.sortformer.stream.spkcache_update_period", stream_spkcache_update_period);

#undef RD_U32
#undef RD_F32
#undef RD_STR

    if (hp.tf_n_heads == 0 || hp.tf_d_model % hp.tf_n_heads != 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: tf_d_model %d not divisible by tf_n_heads %d", hp.tf_d_model,
                hp.tf_n_heads);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (hp.tf_pre_ln) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: pre_ln=true transformer not supported (post-LN only)");
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

namespace {

// Look up a tensor by name and validate its ne against the expected dims
// (fast-to-slow; pass -1 to skip a dim). Returns null + logs on failure.
ggml_tensor * get_checked(ggml_context * ctx, const char * name, int64_t ne0, int64_t ne1) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (t == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: missing tensor %s", name);
        return nullptr;
    }
    if ((ne0 >= 0 && t->ne[0] != ne0) || (ne1 >= 0 && t->ne[1] != ne1)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "sortformer: tensor %s shape mismatch: have [%lld,%lld] want [%lld,%lld]",
                name, (long long) t->ne[0], (long long) t->ne[1], (long long) ne0, (long long) ne1);
        return nullptr;
    }
    return t;
}

// v3 tensors under NeMo-Speech.cpp's names (see read_v3_hparams).
transcribe_status build_v3_weights(ggml_context * ctx, const SortformerHParams & hp, SortformerWeights & w) {
    const int64_t d     = hp.tf_d_model;
    const int64_t ed    = hp.enc_d_model;
    const int64_t edff  = hp.enc_d_ff;
    const int64_t spk   = hp.max_speakers;
    const int64_t stack = static_cast<int64_t>(hp.enc_feat_in) * hp.enc_subsampling_factor;
    char          name[96];

#define GET(dst, nm, e0, e1)                           \
    do {                                               \
        (dst) = get_checked(ctx, (nm), (e0), (e1));    \
        if ((dst) == nullptr)                          \
            return TRANSCRIBE_ERR_GGUF;                \
    } while (0)

    GET(w.stack_proj_w, "encoder.pre_encode.proj.weight", stack, ed);
    GET(w.norm_embed_w, "encoder.embed_norm.weight", ed, -1);
    GET(w.norm_embed_b, "encoder.embed_norm.bias", ed, -1);
    GET(w.final_norm_w, "encoder.final_norm.weight", ed, -1);
    GET(w.final_norm_b, "encoder.final_norm.bias", ed, -1);
    GET(w.enc_proj_w, "encoder_proj.weight", ed, d);
    GET(w.enc_proj_b, "encoder_proj.bias", d, -1);
    GET(w.upsample_w, "subpixel_upsample.weight", 3, d);
    GET(w.upsample_b, "subpixel_upsample.bias", d * hp.upsample_factor, -1);
    GET(w.fc1_w, "head.first_hidden_to_hidden.weight", d, d);
    GET(w.fc1_b, "head.first_hidden_to_hidden.bias", d, -1);
    GET(w.single_spk_head_w, "head.single_hidden_to_spks.weight", d, spk);
    GET(w.single_spk_head_b, "head.single_hidden_to_spks.bias", spk, -1);
    GET(w.silence_emb, "learnable_sil_emb", ed, -1);

    w.rope_blocks.resize(static_cast<size_t>(hp.enc_n_layers));
    for (int i = 0; i < hp.enc_n_layers; ++i) {
        SortformerRopeBlock & b = w.rope_blocks[static_cast<size_t>(i)];
#define GETB(dst, suffix, e0, e1)                                              \
    do {                                                                       \
        std::snprintf(name, sizeof(name), "encoder.layers.%d.%s", i, suffix);  \
        GET(dst, name, e0, e1);                                                \
    } while (0)
        GETB(b.norm1_w, "norm1.weight", ed, -1);
        GETB(b.norm1_b, "norm1.bias", ed, -1);
        GETB(b.attn_qkv_w, "attn.w_qkv.weight", ed, 3 * ed);
        GETB(b.attn_o_w, "attn.out_proj.weight", ed, ed);
        GETB(b.attn_o_b, "attn.out_proj.bias", ed, -1);
        GETB(b.norm2_w, "norm2.weight", ed, -1);
        GETB(b.norm2_b, "norm2.bias", ed, -1);
        GETB(b.ff_in_w, "ffn.net.0.weight", ed, edff);
        GETB(b.ff_in_b, "ffn.net.0.bias", edff, -1);
        GETB(b.ff_out_w, "ffn.net.3.weight", edff, ed);
        GETB(b.ff_out_b, "ffn.net.3.bias", ed, -1);
#undef GETB
    }
#undef GET
    return TRANSCRIBE_OK;
}

}  // namespace

transcribe_status build_sortformer_weights(ggml_context *            ctx,
                                           const SortformerHParams & hp,
                                           SortformerWeights &       w,
                                           const char *              tensor_prefix) {
    if (hp.is_v3()) {
        return build_v3_weights(ctx, hp, w);
    }
    const int64_t d   = hp.tf_d_model;
    const int64_t dff = hp.tf_d_ff;
    const int64_t ed  = hp.enc_d_model;
    const int64_t spk = hp.max_speakers;
    const char *  pfx = (tensor_prefix != nullptr) ? tensor_prefix : "";
    char          name[160];

#define GET(dst, nm, e0, e1)                                  \
    do {                                                      \
        std::snprintf(name, sizeof(name), "%s%s", pfx, (nm)); \
        (dst) = get_checked(ctx, name, (e0), (e1));           \
        if ((dst) == nullptr)                                 \
            return TRANSCRIBE_ERR_GGUF;                       \
    } while (0)

    GET(w.enc_proj_w, "diar.encoder_proj.weight", ed, d);
    GET(w.enc_proj_b, "diar.encoder_proj.bias", d, -1);

    w.tf_blocks.resize(static_cast<size_t>(hp.tf_n_layers));
    for (int i = 0; i < hp.tf_n_layers; ++i) {
        SortformerTfBlock & b = w.tf_blocks[static_cast<size_t>(i)];
        char                blk[96];
#define GETB(dst, suffix, e0, e1)                                      \
    do {                                                               \
        std::snprintf(blk, sizeof(blk), "tf.blocks.%d.%s", i, suffix); \
        GET(dst, blk, e0, e1);                                         \
    } while (0)
        GETB(b.norm1_w, "norm_1.weight", d, -1);
        GETB(b.norm1_b, "norm_1.bias", d, -1);
        GETB(b.attn_q_w, "attn.q.weight", d, d);
        GETB(b.attn_q_b, "attn.q.bias", d, -1);
        GETB(b.attn_k_w, "attn.k.weight", d, d);
        GETB(b.attn_k_b, "attn.k.bias", d, -1);
        GETB(b.attn_v_w, "attn.v.weight", d, d);
        GETB(b.attn_v_b, "attn.v.bias", d, -1);
        GETB(b.attn_o_w, "attn.out.weight", d, d);
        GETB(b.attn_o_b, "attn.out.bias", d, -1);
        GETB(b.norm2_w, "norm_2.weight", d, -1);
        GETB(b.norm2_b, "norm_2.bias", d, -1);
        GETB(b.ff_in_w, "ff.in.weight", d, dff);
        GETB(b.ff_in_b, "ff.in.bias", dff, -1);
        GETB(b.ff_out_w, "ff.out.weight", dff, d);
        GETB(b.ff_out_b, "ff.out.bias", d, -1);
#undef GETB
    }

    GET(w.fc1_w, "diar.fc1.weight", d, d);
    GET(w.fc1_b, "diar.fc1.bias", d, -1);
    GET(w.single_spk_head_w, "diar.single_spk_head.weight", d, spk);
    GET(w.single_spk_head_b, "diar.single_spk_head.bias", spk, -1);
    GET(w.spk_head_w, "diar.spk_head.weight", 2 * d, spk);
    GET(w.spk_head_b, "diar.spk_head.bias", spk, -1);

#undef GET
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::sortformer
