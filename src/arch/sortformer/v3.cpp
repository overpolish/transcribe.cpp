// arch/sortformer/v3.cpp - the high-resolution Sortformer v3 graphs
// (Nemotron 3 Diarization). v3 keeps v2's arrival-order speaker cache and
// FIFO (stream.cpp) but swaps the networks around them:
//
//   Graph A: mel [M, n_mels] -> stack 8 frames -> Linear 1024 -> 512
//            (the per-frame embeddings the cache stores, no norm applied).
//   Graph B: [spkcache | fifo | chunk] embeddings -> LayerNorm -> 31 pre-LN
//            RoPE Transformer blocks -> LayerNorm -> Linear 512 -> 192 ->
//            Conv1d 192 -> 8*192 (k3 p1) -> sub-pixel to the 10 ms grid ->
//            relu -> fc1 -> relu -> speaker head -> sigmoid.
//
// Ported from NVIDIA NeMo-Speech.cpp (Apache-2.0; src/asr/diar/
// sortformer_model.cpp, src/asr/encoder/rope_transformer.cpp), the parity
// oracle for this family's v3 path.

#include "ggml.h"
#include "sortformer.h"

#include <cmath>

namespace transcribe::sortformer {

namespace {

ggml_tensor * norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), w), b);
}

ggml_tensor * dense(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    return b != nullptr ? ggml_add(ctx, y, b) : y;
}

// One pre-LN RoPE block over x = [d, T]: self-attention with rotary
// positions (NeoX, rotate-half) on q and k, then a GELU (erf) FFN.
ggml_tensor * rope_block(ggml_context *              ctx,
                         const SortformerHParams &   hp,
                         const SortformerRopeBlock & b,
                         ggml_tensor *               x,
                         ggml_tensor *               positions,
                         int                         T) {
    const int d        = hp.enc_d_model;
    const int n_head   = hp.enc_n_heads;
    const int head_dim = d / n_head;
    const int n_rot    = static_cast<int>(static_cast<float>(head_dim) * hp.enc_rotary_fraction);

    // One projection, read as three [hd, H, T] views: q, then k, then v.
    ggml_tensor * xn      = norm(ctx, x, b.norm1_w, b.norm1_b);
    ggml_tensor * qkv     = ggml_mul_mat(ctx, b.attn_qkv_w, xn);  // [3d, T]
    const auto    section = [&](int i) {
        return ggml_view_3d(ctx, qkv, head_dim, n_head, T, static_cast<size_t>(head_dim) * qkv->nb[0], qkv->nb[1],
                               static_cast<size_t>(i) * d * qkv->nb[0]);
    };
    ggml_tensor * q = section(0);
    ggml_tensor * k = section(1);
    ggml_tensor * v = section(2);

    q = ggml_rope_ext(ctx, q, positions, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, hp.enc_pos_emb_max_len, hp.enc_rope_base,
                      1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, hp.enc_pos_emb_max_len, hp.enc_rope_base,
                      1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // [hd, H, T] -> [hd, T, H]
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);  // [T_k, T_q, H]
    kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);

    ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));     // [T, hd, H]
    ggml_tensor * att = ggml_mul_mat(ctx, v_t, kq);                           // [hd, T_q, H]
    att               = ggml_cont(ctx, ggml_permute(ctx, att, 0, 2, 1, 3));  // [hd, H, T]
    att               = ggml_reshape_2d(ctx, att, d, T);

    x = ggml_add(ctx, x, dense(ctx, b.attn_o_w, att, b.attn_o_b));

    ggml_tensor * f = norm(ctx, x, b.norm2_w, b.norm2_b);
    f               = ggml_gelu_erf(ctx, dense(ctx, b.ff_in_w, f, b.ff_in_b));
    f               = dense(ctx, b.ff_out_w, f, b.ff_out_b);
    return ggml_add(ctx, x, f);
}

}  // namespace

V3StackBuild build_v3_stack_graph(ggml_context * ctx, const SortformerHParams & hp, const SortformerWeights & w, int M) {
    V3StackBuild b{};
    const int     n_mels = hp.fe_num_mels;
    const int     sub    = hp.enc_subsampling_factor;
    ggml_tensor * mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, n_mels);
    ggml_set_name(mel_in, "chunk.mel.in");
    ggml_set_input(mel_in);

    // [M, n_mels] -> [n_mels, M], zero-padded to whole groups of `sub`
    // frames, then each group's frames laid end to end: [n_mels*sub, M/sub].
    ggml_tensor * frames = ggml_cont(ctx, ggml_transpose(ctx, mel_in));
    const int     pad    = (sub - M % sub) % sub;
    if (pad > 0) {
        frames = ggml_pad(ctx, frames, 0, pad, 0, 0);
    }
    ggml_tensor * stacked = ggml_reshape_2d(ctx, frames, static_cast<int64_t>(n_mels) * sub, (M + pad) / sub);
    ggml_tensor * out     = ggml_mul_mat(ctx, w.stack_proj_w, stacked);  // [enc_d_model, T_diar]

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1024, false);
    ggml_build_forward_expand(graph, out);
    b.graph  = graph;
    b.mel_in = mel_in;
    b.out    = out;
    return b;
}

V3InferBuild build_v3_infer_graph(ggml_context *            ctx,
                                  const SortformerHParams & hp,
                                  const SortformerWeights & w,
                                  int                       T_concat) {
    V3InferBuild  b{};
    const int     ed        = hp.enc_d_model;
    const int     hd        = hp.tf_d_model;
    const int     up        = hp.upsample_factor;
    ggml_tensor * concat_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ed, T_concat);
    ggml_set_name(concat_in, "stream.concat.in");
    ggml_set_input(concat_in);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_concat);
    ggml_set_name(positions, "stream.positions.in");
    ggml_set_input(positions);

    ggml_tensor * x = norm(ctx, concat_in, w.norm_embed_w, w.norm_embed_b);
    for (const SortformerRopeBlock & blk : w.rope_blocks) {
        x = rope_block(ctx, hp, blk, x, positions, T_concat);
    }
    x = norm(ctx, x, w.final_norm_w, w.final_norm_b);

    ggml_tensor * proj = dense(ctx, w.enc_proj_w, x, w.enc_proj_b);  // [hd, T]

    // Conv1d over time wants [T, C]; its output [T, up*hd] goes back to
    // channels-first, and each frame's up*hd channels become `up` frames
    // of hd: channel u*hd + c is sub-frame u's channel c.
    ggml_tensor * time_major = ggml_cont(ctx, ggml_transpose(ctx, proj));
    ggml_tensor * conv       = ggml_conv_1d(ctx, w.upsample_w, time_major, 1, 1, 1);
    conv                     = ggml_add(ctx, conv, ggml_reshape_2d(ctx, w.upsample_b, 1, hd * up));
    ggml_tensor * fine = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_transpose(ctx, conv)), hd, T_concat * up);

    ggml_tensor * h     = ggml_relu(ctx, fine);
    h                   = ggml_relu(ctx, dense(ctx, w.fc1_w, h, w.fc1_b));
    ggml_tensor * preds = ggml_sigmoid(ctx, dense(ctx, w.single_spk_head_w, h, w.single_spk_head_b));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, preds);
    b.graph     = graph;
    b.concat_in = concat_in;
    b.positions = positions;
    b.preds     = preds;  // [n_spk, T_concat * up]
    return b;
}

}  // namespace transcribe::sortformer
