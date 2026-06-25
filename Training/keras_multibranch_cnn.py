"""
Build the Codon-Aware Multi-Branch CNN (mRNAStabilityNet) as a Keras
functional model and generate:
  - figures/keras_multibranch_cnn.png   (plot_model diagram)
  - figures/keras_multibranch_cnn.svg
  - figures/keras_multibranch_cnn.pdf
  - figures/multibranch_cnn_summary.png (text summary rendered as image)

The architecture exactly mirrors the PyTorch mRNAStabilityNet:
  5'UTR branch  : Emb(6,16) -> 3x DilatedConv1D -> TriplePool -> 192d
  CDS branch    : Emb(66,16) -> 3x DilatedConv1D(k=5) -> TriplePool -> 192d
  3'UTR branch  : Emb(6,16) -> 3x DilatedConv1D -> TriplePool -> 192d
  Feature branch: 75-dim -> passthrough
  Head          : 651 -> 128 -> 64 -> 1
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# ══════════════════════════════════════════════════════════════════════════
#  Custom layers (to match PyTorch model exactly)
# ══════════════════════════════════════════════════════════════════════════

class ChannelLayerNorm(layers.Layer):
    """LayerNorm over channels for (B, L, C) tensors — matches PyTorch version."""
    def build(self, input_shape):
        self.ln = layers.LayerNormalization(axis=-1)
        self.ln.build(input_shape)

    def call(self, x):
        return self.ln(x)

    def get_config(self):
        return super().get_config()


class TriplePool(layers.Layer):
    """Masked max + mean + attention pooling -> concat 3C dims.

    Combines three pooling strategies in one layer so that plot_model
    shows a single clean node instead of 3 separate branches.
    """
    def __init__(self, dim, **kw):
        super().__init__(**kw)
        self.score_dense = layers.Dense(1)

    def call(self, inputs):
        x, mask = inputs                         # x: (B,L,C), mask: (B,L)
        m = tf.cast(tf.expand_dims(mask, -1), x.dtype)
        has_token = tf.cast(tf.reduce_any(mask, axis=1, keepdims=True), x.dtype)

        # mean pool
        p_mean = tf.reduce_sum(x * m, axis=1) / tf.maximum(
            tf.reduce_sum(m, axis=1), 1.0)

        # max pool
        neg_inf = tf.constant(-1e9, dtype=x.dtype)
        p_max = tf.reduce_max(x * m + (1.0 - m) * neg_inf, axis=1) * has_token

        # attention pool
        s = tf.squeeze(self.score_dense(x), -1)
        s = tf.where(mask, s, tf.constant(-1e9, dtype=s.dtype))
        w = tf.expand_dims(tf.nn.softmax(s, axis=1), -1)
        p_attn = tf.reduce_sum(x * w, axis=1) * has_token

        return tf.concat([p_max, p_mean, p_attn], axis=-1)

    def compute_output_shape(self, input_shape):
        x_shape = input_shape[0]
        return (x_shape[0], x_shape[-1] * 3)


# ══════════════════════════════════════════════════════════════════════════
#  Branch builder
# ══════════════════════════════════════════════════════════════════════════

class ComputeMask(layers.Layer):
    """Compute padding mask: True where token_id != 0."""
    def call(self, x):
        return tf.not_equal(x, 0)


def build_conv_branch(token_input, vocab_size, emb_dim, channels,
                      kernel_size, dropout, name_prefix):
    """One ConvBranch: Embedding -> 3x DilatedConv1D -> TriplePool -> 192d."""

    # Auto-compute mask from token IDs (PAD=0)
    mask = ComputeMask(name=f"{name_prefix}_padding_mask")(token_input)

    # Embedding
    x = layers.Embedding(vocab_size, emb_dim, mask_zero=False,
                         name=f"{name_prefix}_embedding")(token_input)

    # 3x Conv blocks with increasing dilation
    for i, ch in enumerate(channels):
        d = 2 ** i
        x = layers.Conv1D(ch, kernel_size, padding="same", dilation_rate=d,
                          name=f"{name_prefix}_conv1d_{i}")(x)
        x = ChannelLayerNorm(name=f"{name_prefix}_layernorm_{i}")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu_{i}")(x)
        x = layers.Dropout(dropout, name=f"{name_prefix}_dropout_{i}")(x)

    # Triple pooling (max + mean + attn) -> 64*3 = 192d
    pooled = TriplePool(channels[-1],
                        name=f"{name_prefix}_triple_pool")([x, mask])
    return pooled


# ══════════════════════════════════════════════════════════════════════════
#  Full model
# ══════════════════════════════════════════════════════════════════════════

def build_model(utr5_len=512, cds_len=700, utr3_len=1024, n_extra=75,
                emb_dim=16, channels=(32, 64, 64), dropout=0.3,
                head_hidden=128):

    # ── Inputs (4 inputs only; masks computed internally) ───────────────
    utr5_ids = layers.Input(shape=(utr5_len,), dtype="int32", name="utr5_token_ids")
    cds_ids  = layers.Input(shape=(cds_len,),  dtype="int32", name="cds_token_ids")
    utr3_ids = layers.Input(shape=(utr3_len,), dtype="int32", name="utr3_token_ids")
    extra    = layers.Input(shape=(n_extra,),  dtype="float32", name="extra_features_75d")

    # ── 5'UTR branch (nt vocab=6, kernel=7) ──────────────────────────────
    z5 = build_conv_branch(utr5_ids, vocab_size=6, emb_dim=emb_dim,
                           channels=channels, kernel_size=7, dropout=dropout,
                           name_prefix="utr5")

    # ── CDS branch (codon vocab=66, kernel=5) ───────────────────────────
    zc = build_conv_branch(cds_ids, vocab_size=66, emb_dim=emb_dim,
                           channels=channels, kernel_size=5, dropout=dropout,
                           name_prefix="cds")

    # ── 3'UTR branch (nt vocab=6, kernel=7) ──────────────────────────────
    z3 = build_conv_branch(utr3_ids, vocab_size=6, emb_dim=emb_dim,
                           channels=channels, kernel_size=7, dropout=dropout,
                           name_prefix="utr3")

    # ── Concatenate all branches ─────────────────────────────────────────
    merged = layers.Concatenate(name="concatenate_all_branches")(
        [z5, zc, z3, extra])    # 192 + 192 + 192 + 75 = 651

    # ── Classification head ──────────────────────────────────────────────
    h = layers.Dense(head_hidden, name="dense_1")(merged)
    h = layers.LayerNormalization(name="head_layernorm_1")(h)
    h = layers.Activation("relu", name="head_relu_1")(h)
    h = layers.Dropout(dropout, name="head_dropout_1")(h)

    h = layers.Dense(head_hidden // 2, name="dense_2")(h)
    h = layers.LayerNormalization(name="head_layernorm_2")(h)
    h = layers.Activation("relu", name="head_relu_2")(h)
    h = layers.Dropout(dropout, name="head_dropout_2")(h)

    out = layers.Dense(1, name="dense_out")(h)

    model = keras.Model(
        inputs=[utr5_ids, cds_ids, utr3_ids, extra],
        outputs=out,
        name="mRNAStabilityNet_MultiBranchCNN",
    )
    return model


# ══════════════════════════════════════════════════════════════════════════
#  Summary text -> PNG
# ══════════════════════════════════════════════════════════════════════════

def render_summary_png(model, out_path):
    import io, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buf = io.StringIO()
    model.summary(print_fn=lambda s: buf.write(s + "\n"), line_length=100,
                  expand_nested=False, show_trainable=False)
    txt = buf.getvalue()

    header = (
        "Codon-Aware Multi-Branch CNN  —  mRNAStabilityNet  (~221K params)\n"
        "(replicated from kevincho417/mrna-stability-prediction-website)\n\n"
        "Input:  5'UTR [B,512] vocab=6  |  CDS [B,700] vocab=66  |"
        "  3'UTR [B,1024] vocab=6  |  Extra [B,75]\n"
        "Loss: BCEWithLogitsLoss  ·  Optim: AdamW  ·  Sched: CosineAnnealing\n"
        "─" * 100 + "\n\n"
    )
    full = header + txt

    lines = full.split("\n")
    n = len(lines)
    line_h = 0.28
    fig_h = max(n * line_h + 1.0, 8)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.01, 0.99, full, transform=ax.transAxes, va="top", ha="left",
            fontsize=10.5, fontfamily="monospace", color="#1a1a2e", linespacing=1.4)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    # save txt too
    txt_path = out_path.replace(".png", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full)
    return txt_path


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def custom_plot_model(model, out_path, title=None):
    """Generate a diagram with the same teal-green style as the existing
    ConvTransformer Keras diagram (keras_convtransformer.png).

    Uses pydot to build HTML-table nodes that show:
      - layer name (bold)
      - input / output labels with shapes
      - consistent teal colour scheme
    """
    import pydot

    graph = pydot.Dot(graph_type="digraph", rankdir="TB")
    graph.set_graph_defaults(
        bgcolor="white", fontname="Arial", fontsize="10",
        nodesep="0.5", ranksep="0.6", dpi="150",
    )
    graph.set_node_defaults(shape="none", fontname="Arial")
    graph.set_edge_defaults(color="#666666", arrowsize="0.7")

    # Colour palette matching the existing ConvTransformer diagram
    TEAL     = "#2CA5A8"  # main layer colour
    BLUE     = "#4A90D9"  # 5'UTR branch
    GREEN    = "#2EAD6B"  # CDS branch
    PURPLE   = "#7B68EE"  # 3'UTR branch
    ORANGE   = "#E8913A"  # feature / special layers
    RED      = "#D94452"  # head layers
    POOL_CLR = "#5DADE2"  # pooling
    CAT_CLR  = "#8B5CF6"  # concatenate
    OUT_CLR  = "#1ABC9C"  # output

    # map layer name prefix -> colour
    def layer_color(name):
        if name.startswith("utr5"):   return BLUE
        if name.startswith("cds"):    return GREEN
        if name.startswith("utr3"):   return PURPLE
        if name.startswith("extra"):  return ORANGE
        if "concatenate" in name:     return CAT_CLR
        if "dense" in name or "head" in name: return RED
        if "pool" in name:            return POOL_CLR
        return TEAL

    # ── Title ────────────────────────────────────────────────────────────
    if title:
        graph.add_node(pydot.Node(
            "title", shape="plaintext",
            label=f'<<B><FONT POINT-SIZE="14">{title}</FONT></B>>',
        ))

    # ── Build nodes ──────────────────────────────────────────────────────
    layer_ids = {}  # layer.name -> node_id
    for layer in model.layers:
        lid = f"node_{id(layer)}"
        layer_ids[layer.name] = lid

        # get input/output shapes
        try:
            in_shape = str(layer.input_shape) if hasattr(layer, 'input_shape') else ""
        except (AttributeError, RuntimeError):
            in_shape = ""
        try:
            out_shape = str(layer.output_shape) if hasattr(layer, 'output_shape') else ""
        except (AttributeError, RuntimeError):
            out_shape = ""

        # clean up shape lists
        if isinstance(in_shape, list):
            in_shape = str(in_shape)
        if isinstance(out_shape, list):
            out_shape = str(out_shape)

        color = layer_color(layer.name)

        # Build HTML-table label matching existing ConvTransformer style
        html = f'''<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4"
               BGCOLOR="{color}">
          <TR><TD COLSPAN="2"><B><FONT COLOR="white" POINT-SIZE="9">{layer.name}</FONT></B>
              <BR/><FONT COLOR="#E0E0E0" POINT-SIZE="7">({layer.__class__.__name__})</FONT></TD></TR>
          <TR>
            <TD><FONT COLOR="white" POINT-SIZE="7">input</FONT><BR/>
                <FONT COLOR="#E0E0E0" POINT-SIZE="7">{in_shape}</FONT></TD>
            <TD><FONT COLOR="white" POINT-SIZE="7">output</FONT><BR/>
                <FONT COLOR="#E0E0E0" POINT-SIZE="7">{out_shape}</FONT></TD>
          </TR>
        </TABLE>>'''
        graph.add_node(pydot.Node(lid, label=html))

    # ── Build edges from model connectivity ──────────────────────────────
    for layer in model.layers:
        lid = layer_ids[layer.name]

        # get inbound layers
        inbound = []
        if hasattr(layer, '_inbound_nodes'):
            for node in layer._inbound_nodes:
                if hasattr(node, 'inbound_layers'):
                    parents = node.inbound_layers
                    if not isinstance(parents, (list, tuple)):
                        parents = [parents]
                    inbound.extend(parents)
                elif hasattr(node, 'input_tensors'):
                    # Keras 3.x
                    tensors = node.input_tensors
                    if not isinstance(tensors, (list, tuple)):
                        tensors = [tensors]
                    for t in tensors:
                        if hasattr(t, '_keras_history'):
                            inbound.append(t._keras_history[0])

        for parent in inbound:
            pid = layer_ids.get(parent.name)
            if pid:
                # colour edges by branch
                ecolor = "#666666"
                if parent.name.startswith("utr5") or layer.name.startswith("utr5"):
                    ecolor = BLUE
                elif parent.name.startswith("cds") or layer.name.startswith("cds"):
                    ecolor = GREEN
                elif parent.name.startswith("utr3") or layer.name.startswith("utr3"):
                    ecolor = PURPLE
                elif parent.name.startswith("extra"):
                    ecolor = ORANGE
                graph.add_edge(pydot.Edge(pid, lid, color=ecolor))

    # ── Title edge ───────────────────────────────────────────────────────
    if title:
        first_inputs = [l for l in model.layers if isinstance(l, layers.InputLayer)]
        if first_inputs:
            graph.add_edge(pydot.Edge("title", layer_ids[first_inputs[0].name],
                                       style="invis"))

    # ── Save ─────────────────────────────────────────────────────────────
    for ext in ("png", "svg", "pdf"):
        p = out_path.replace(".png", f".{ext}")
        graph.write(p, format=ext)
        print(f"  custom diagram -> {p}")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)

    model = build_model()

    # 1) print summary
    model.summary(line_length=110, expand_nested=True)

    # 2) custom styled diagram (matching ConvTransformer style)
    title = "Multi-Branch CNN (mRNAStabilityNet) &mdash; {:,} params".format(
        model.count_params())
    custom_plot_model(model,
                      os.path.join(out_dir, "keras_multibranch_cnn.png"),
                      title=title)

    # 3) summary as PNG
    sum_path = os.path.join(out_dir, "multibranch_cnn_summary.png")
    txt_path = render_summary_png(model, sum_path)
    print(f"  summary PNG -> {sum_path}")
    print(f"  summary TXT -> {txt_path}")

    print(f"\nDone! Total params: {model.count_params():,}")


if __name__ == "__main__":
    main()
