# Literature Review: mRNA Stability / Degradation Prediction

本專案目標是由 `5'UTR + CDS + 3'UTR` 序列預測 mRNA stability / degradation label。既有文獻大致可分成三類：手工序列特徵與傳統機器學習、end-to-end 深度學習，以及 RNA foundation / language model。

| 類型 | 文獻 | 任務與資料 | 輸入表示 | 方法 | 主要結果 | 對本專案的啟示 |
|---|---|---|---|---|---|---|
| MPRA / k-mer baseline | Rabani et al., 2017, *Cell* / UTR-seq | zebrafish 3'UTR reporter library，約 90,000 個 110 nt 片段，多時間點 mRNA level | 3'UTR sequence；傳統模型使用 3-7 nt k-mer composition | regression / sequence element analysis | 找到 AU-rich、PUM、miR-430 等 degradation-associated elements；k-mer composition 可捕捉 regulatory motifs | k-mer 對 stability 很有訊號；小資料時，k-mer feature 或 k-mer token 是合理 baseline |
| MPRA / functional annotation | Zhao et al., 2014, *Nature Biotechnology* / fast-UTR | >2,000 human genes 的 3'UTR 片段，測 mRNA abundance、stability、protein output | 3'UTR fragments | massively parallel reporter assay + motif/variant analysis | 發現 87 個 novel cis-regulatory elements，並量化 motif variant effect | 3'UTR motif 對 stability 重要；模型解釋可聚焦 ARE、miRNA、RBP motif |
| Deep learning on 3'UTR dynamics | Benegas et al., 2022, *Bioinformatics* / DeepUTR | UTR-seq 90,000 個 110 nt 3'UTR sequences，預測 8 個後續時間點 mRNA levels / degradation dynamics | raw 3'UTR sequence + initial mRNA level | CNN / RNN / multi-task models，Integrated Gradients 解釋 | DNN 優於傳統 linear degradation rate methods；CNN 較易解釋，RNN 可捕捉 positional effect | end-to-end sequence model 可學 motif 與位置效果；但資料量需足夠 |
| Full-length sequence DL | Agarwal & Kelley et al., 2022, *Genome Biology* / Saluki | human/mouse endogenous mRNA half-life | full mRNA sequence up to 12,288 nt + coding frame + splice-site annotation | hybrid 1D CNN + GRU；multi-species heads | human half-life Pearson r 約 0.77，mouse 約 0.73；可預測 variant effect | full-length mRNA 有用；CNN+RNN/attention 需處理 variable length 與長序列 |
| Crowdsourced RNA degradation DL | Wayment-Steele et al., 2022, *Nature Machine Intelligence* / OpenVaccine | 6,043 個 102-130 nt RNA constructs，單核苷酸解析度 degradation / SHAPE labels | sequence + predicted structure / loop features | Kaggle ensemble，常見 GRU/LSTM/CNN/GNN 等 | winning model 約 41% nucleotide-level predictions within experimental error；可泛化到 504-1,588 nt mRNA | ensemble 與 structure-aware features 很有幫助；短序列訓練可泛化但要小心 domain shift |
| Conv + self-attention | He et al., 2023, *Briefings in Bioinformatics* / RNAdegformer | OpenVaccine nucleotide-level degradation + in vitro half-life | nucleotide embedding + biophysical features，例如 BPP / distance / structure | convolution + self-attention + supervised / unsupervised / semi-supervised learning | 優於 OpenVaccine top solution；對較長 mRNA half-life 有較好 correlation；attention map 可解釋 | 本專案 ConvTransformer 方向合理；Conv 可學 local motifs，attention 可學 global dependencies |
| Biophysical + ML | Cetnar et al., 2024, *Nature Communications* | >50,000 synthetic bacterial mRNAs，half-life 約 20 sec 到 20 min | sequence design variables + biophysical model outputs | biophysical modeling + ML kinetic decay model | 高準確度與 generalizability；量化 translation rate、secondary structure、G-quadruplex、RppH 等因素 | 若能加入 secondary structure / translation rate features，可能提升模型 |
| RNA foundation model | Zhou et al., 2025, *Genome Biology* / LAMAR | RNA regulation tasks；3'UTR half-life prediction | pretrained Transformer representation | pretrain on ~15M mammalian/viral genome+transcriptome sequences，fine-tune downstream | LAMAR-DR 在 3'UTR half-life 預測達 MSE 0.176、Spearman 0.647，優於 RNA-FM/RNAErnie | 若資料小，pretrained RNA LM 可能比從零訓練 Transformer 更有效 |
| Long RNA language model | Li et al., 2025, *Genome Biology* / HydraRNA | full-length RNA downstream tasks，含 mRNA half-life / translation | full-length RNA sequence | hybrid Hydra SSM + multi-head attention，masked language pretraining | half-life prediction Pearson R² 約 0.334；指出 CDS 對 half-life variance contribution 最大 | full-length modeling 應保留 CDS；長序列模型需要 linear-time / downsampling strategy |

## Summary for This Project

1. k-mer feature / k-mer token 有文獻基礎，因為許多 mRNA stability 訊號來自短 motif，例如 ARE、PUM、miRNA target、poly-U/poly-A-like elements。
2. Conv + Transformer 是合理架構：Conv 負責 local motif extraction，Transformer 捕捉長距離 dependency。
3. 小資料集下，從零訓練 Transformer 不一定勝過 feature-based MLP。文獻中表現最好的 end-to-end / foundation model 通常依賴大量 MPRA 資料或大規模 pretraining。
4. Ensemble 是實用方向。OpenVaccine 類型任務中，ensemble 常能提升泛化；本專案中 MLP + Transformer ensemble 的 CV auROC 也從 0.7936 提升到約 0.804。
5. 若要再提升，可加入 RNA secondary structure / base-pairing probability / minimum free energy / codon optimality features，或使用 pretrained RNA language model。

## References

- Rabani et al. Massively parallel reporter assay of 3'UTR sequences identifies in vivo rules for mRNA degradation. *Cell*, 2017.
- Zhao et al. Massively parallel functional annotation of 3' untranslated regions. *Nature Biotechnology*, 2014.
- Benegas et al. Computational modeling of mRNA degradation dynamics using deep neural networks. *Bioinformatics*, 2022.
- Agarwal & Kelley et al. The genetic and biochemical determinants of mRNA degradation rates in mammals. *Genome Biology*, 2022.
- Wayment-Steele et al. Deep learning models for predicting RNA degradation via dual crowdsourcing. *Nature Machine Intelligence*, 2022.
- He et al. RNAdegformer: accurate prediction of mRNA degradation at nucleotide resolution with deep learning. *Briefings in Bioinformatics*, 2023.
- Cetnar et al. Predicting synthetic mRNA stability using massively parallel kinetic measurements, biophysical modeling, and machine learning. *Nature Communications*, 2024.
- Zhou et al. A foundation language model to decipher diverse regulation of RNAs. *Genome Biology*, 2025.
- Li et al. HydraRNA: a hybrid architecture based full-length RNA language model. *Genome Biology*, 2025.
