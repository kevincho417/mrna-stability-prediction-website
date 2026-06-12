from graphviz import Digraph
def add_layer(g, nid, name, ltype, ish, osh, fill="#FFFFFF"):
    label = f"""<
    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4" BGCOLOR="{fill}">
        <TR><TD><B>{name}</B></TD></TR>
        <TR><TD>{ltype}</TD></TR>
        <TR><TD ALIGN="LEFT">input: {ish}</TD></TR>
        <TR><TD ALIGN="LEFT">output: {osh}</TD></TR>
    </TABLE>
    >"""
    g.node(nid, label=label, shape="plain")
def branch(parent, cname, title, pre, L, vocab_lbl, kernel):
    # channel progression for lightweight model: emb16 -> 32 -> 64 -> 64
    s = lambda c: f"(None, {L}, {c})"
    convs = [(16,32),(32,64),(64,64)]
    with parent.subgraph(name=f"cluster_{cname}") as c:
        c.attr(label=title, color="#666666", style="rounded", penwidth="1.2",
               fontname="Times New Roman", fontsize="18")
        add_layer(c, f"{pre}_in", pre, "InputLayer", f"[(None, {L})]", f"[(None, {L})]", "#F8F8F8")
        add_layer(c, f"{pre}_emb", f"{pre}_embedding", "Embedding", f"(None, {L})", s(16), "#FCFCFC")
        c.edge(f"{pre}_in", f"{pre}_emb")
        prev = f"{pre}_emb"
        for i,(ci,co) in enumerate(convs):
            cid=f"{pre}_conv{i}"; nid=f"{pre}_ln{i}"; rid=f"{pre}_relu{i}"; did=f"{pre}_drop{i}"
            add_layer(c, cid, f"conv1d k={kernel}, dil {2**i}", "Conv1D", s(ci), s(co))
            add_layer(c, nid, f"layer_norm", "LayerNorm", s(co), s(co))
            add_layer(c, rid, "relu", "ReLU", s(co), s(co))
            add_layer(c, did, "dropout", "Dropout", s(co), s(co))
            c.edge(prev, cid); c.edge(cid, nid); c.edge(nid, rid); c.edge(rid, did)
            prev = did
        add_layer(c, f"{pre}_mask", f"{pre}_mask", "Mask (token!=0)", f"(None, {L})", f"(None, {L})", "#F3E9F7")
        add_layer(c, f"{pre}_pool", f"{pre}_masked_pool", "MaskedPool\n(max+mean+attn, empty=0)",
                  f"[{s(64)}, (None, {L})]", "(None, 192)", "#DDEBF7")
        c.edge(prev, f"{pre}_pool")
        c.edge(f"{pre}_in", f"{pre}_mask")
        c.edge(f"{pre}_mask", f"{pre}_pool", style="dashed", label="mask")
        c.body.append(f'{{rank=same; {pre}_drop2; {pre}_mask;}}')
    return f"{pre}_pool"

dot = Digraph("Lite_Model", format="png", engine="dot")
dot.attr(rankdir="TB", splines="ortho", compound="true", newrank="true",
         nodesep="0.45", ranksep="0.7", bgcolor="white", pad="0.2")
dot.attr("node", fontname="Times New Roman", fontsize="12")
dot.attr("edge", color="black", arrowsize="0.7", penwidth="1.0")

branch(dot, "utr5", "5'UTR Branch  (nucleotide, kernel=7)", "utr5", 512, "6", 7)
branch(dot, "utr3", "3'UTR Branch  (nucleotide, kernel=7)", "utr3", 1024, "6", 7)
branch(dot, "cds",  "CDS Branch  (codon-level, kernel=5)",  "cds", 700, "66", 5)

add_layer(dot, "extra_in", "extra", "InputLayer", "[(None, 75)]", "[(None, 75)]", "#E8F5E9")
dot.body.append("{rank=same; utr5_pool; utr3_pool; cds_pool; extra_in;}")

add_layer(dot, "concat", "concatenate", "Concatenate",
          "[(None,192), (None,192), (None,192), (None,75)]", "(None, 651)", "#FFF3CD")
add_layer(dot, "d0", "dense", "Linear", "(None, 651)", "(None, 128)")
add_layer(dot, "ln0", "layer_norm", "LayerNorm", "(None, 128)", "(None, 128)")
add_layer(dot, "r0", "relu", "ReLU", "(None, 128)", "(None, 128)")
add_layer(dot, "dr0", "dropout", "Dropout", "(None, 128)", "(None, 128)")
add_layer(dot, "d1", "dense_1", "Linear", "(None, 128)", "(None, 64)")
add_layer(dot, "ln1", "layer_norm_1", "LayerNorm", "(None, 64)", "(None, 64)")
add_layer(dot, "r1", "relu_1", "ReLU", "(None, 64)", "(None, 64)")
add_layer(dot, "dr1", "dropout_1", "Dropout", "(None, 64)", "(None, 64)")
add_layer(dot, "out", "output", "Linear  (logit, then sigmoid)", "(None, 64)", "(None, 1)", "#D6EFD6")

for sp in ["utr5_pool","utr3_pool","cds_pool","extra_in"]:
    dot.edge(sp, "concat")
for a,b in [("concat","d0"),("d0","ln0"),("ln0","r0"),("r0","dr0"),("dr0","d1"),
            ("d1","ln1"),("ln1","r1"),("r1","dr1"),("dr1","out")]:
    dot.edge(a,b)

dot.render("full_model_lite", cleanup=True)
dot.format="svg"; dot.render("full_model_lite", cleanup=True)
print("done")
