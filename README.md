# Domain-Specific Legal Document Embedder

## Objective

This project builds and evaluates a domain-specific embedder for legal PDF documents. The pipeline extracts paragraph-level text, trains/adapts embedding models, clusters the resulting vectors with traditional methods, and compares domain behavior against `all-MiniLM-L6-v2`.

The goal is to understand whether domain training improves legal-document similarity and cluster behavior, especially when legal text contains repeated structure, citations, boilerplate, and document-type-specific phrasing.

This directly maps to the assignment brief: (1) a domain-specific embedder is trained on the legal corpus rather than reused off-the-shelf; (2) the resulting embeddings are clustered using traditional methods (KMeans, Agglomerative); (3) performance is benchmarked against all-MiniLM-L6-v2 as the baseline using both similarity checks (cosine intra/inter-cluster gap, nearest-neighbor inspection) and clustering-separation metrics (silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI); and (4) embedding dimensionality is kept intentionally small (100-d Word2Vec, standard 384-d MiniLM) since the brief specifies approach matters more than embedding length.

## Dataset

The corpus contains 5 legal PDFs segmented into 4,067 paragraph-level records.

| Source file                                 |    Document type | Paragraphs |
| ------------------------------------------- | ---------------: | ---------: |
| `20240809133545974_23-980_petbrief.pdf`     |   petition brief |      1,403 |
| `22-1165_10n2.pdf`                          |     slip opinion |        318 |
| `23-980bsacus_facebookvamalgamatedbank.pdf` |     amicus brief |      1,061 |
| `USCOURTS-ilnd-1_04-cv-00397-5.pdf`         | district opinion |        499 |
| `comp26286.pdf`                             |       SEC filing |        786 |

Each paragraph keeps its source file and document-type label. The labels are not used for unsupervised clustering, but they are used afterward to compute ARI and NMI.

## Approach

1. `src/01_preprocessing.py` extracts text from PDFs with `pdfplumber`, performs legal-aware cleaning, and writes `data/processed/paragraphs.jsonl`.
2. `src/02_train_domain_embedder.py` trains a 100-dimensional Word2Vec skip-gram model on the legal corpus.
3. `src/03_generate_embeddings.py` creates Word2Vec domain embeddings and plain MiniLM baseline embeddings for the same paragraph records.
4. `src/04_clustering.py` applies KMeans and Agglomerative clustering. KMeans labels are used for the reported metrics.
5. `src/05_evaluation_metrics.py` computes clustering, similarity, and document-type alignment metrics.
6. `src/06_visualizations.py` generates PCA plots, t-SNE comparison, silhouette plots, metric bars, document-type distribution, and cosine similarity heatmaps.

Additional experiment scripts evaluate MiniLM fine-tuning and fine-tuning duration:

- `src/main_minilm_comparison.py`: plain MiniLM vs. fine-tuned MiniLM
- `src/continue_minilm_to_50.py`: continuation fine-tuning from 5 to 50 epochs
- `src/07_epoch_trend.py`: trend plot for the epoch sweep

## Metrics And Visualizations

The evaluation uses both internal clustering metrics and semantic alignment checks:

| Category            | Checks                                                                    |
| ------------------- | ------------------------------------------------------------------------- |
| Cluster geometry    | Silhouette, Davies-Bouldin, Calinski-Harabasz                             |
| Label alignment     | Adjusted Rand Index, Normalized Mutual Information                        |
| Similarity behavior | Intra/inter-cluster cosine similarity, nearest-neighbor checks            |
| Visual inspection   | PCA cluster maps, t-SNE comparison, silhouette plots, similarity heatmaps |

- Silhouette score (–1 to 1): how well-separated clusters are geometrically; higher means tighter, more distinct clusters.
- Davies-Bouldin index: average similarity between each cluster and its closest neighbor; lower is better separation.
- Calinski-Harabasz score: ratio of between-cluster to within-cluster dispersion; higher means more distinct clusters.
- ARI / NMI: how well the unsupervised clusters agree with the known document-type labels; 0 = random, 1 = perfect match. This is the "did it actually learn something meaningful" check, as opposed to silhouette which only checks geometry.
- Intra/inter cosine gap: difference between average similarity of same-cluster pairs vs. different-cluster pairs; a bigger gap means the embedding space itself separates topics well.

## Results

### Run 1: Word2Vec Domain Embedder vs. Plain MiniLM

| Metric                        | Word2Vec | Plain MiniLM | Better result |
| ----------------------------- | -------: | -----------: | ------------- |
| Silhouette score              |    0.417 |        0.067 | Word2Vec      |
| Davies-Bouldin index          |    1.732 |        5.226 | Word2Vec      |
| Calinski-Harabasz score       |    793.7 |        147.9 | Word2Vec      |
| Intra/inter similarity gap    |    0.226 |        0.058 | Word2Vec      |
| Adjusted Rand Index           |    0.006 |        0.075 | MiniLM        |
| Normalized Mutual Information |    0.006 |        0.077 | MiniLM        |

Word2Vec creates much tighter geometric clusters and a stronger cosine separation gap. However, those clusters do not align well with known document types. Plain MiniLM has weaker cluster geometry but slightly better document-type recovery.

### Run 2: Plain MiniLM vs. Fine-Tuned MiniLM

| Metric                        | Plain MiniLM | Fine-tuned MiniLM |
| ----------------------------- | -----------: | ----------------: |
| Silhouette score              |        0.070 |             0.098 |
| Davies-Bouldin index          |        4.177 |             4.312 |
| Calinski-Harabasz score       |        115.4 |             148.2 |
| Adjusted Rand Index           |        0.062 |             0.401 |
| Normalized Mutual Information |        0.087 |             0.522 |
| Intra/inter similarity gap    |        0.100 |             0.151 |

Fine-tuned MiniLM gives the strongest semantic result. ARI and NMI improve sharply, showing that the adapted embedder better recovers legal document-type structure.

### Run 3: Fine-Tuning Duration

| Epochs | Silhouette | Davies-Bouldin | Calinski-Harabasz |   ARI |   NMI | Separation gap |
| -----: | ---------: | -------------: | ----------------: | ----: | ----: | -------------: |
|      5 |      0.146 |           3.41 |             380.2 | 0.321 | 0.429 |          0.146 |
|     10 |      0.159 |           3.86 |             397.6 | 0.372 | 0.489 |          0.233 |
|     20 |      0.128 |           4.27 |             241.5 | 0.390 | 0.512 |          0.219 |
|     35 |      0.108 |           4.28 |             173.3 | 0.393 | 0.516 |          0.172 |
|     50 |      0.098 |           4.31 |             148.2 | 0.401 | 0.522 |          0.151 |

Label-alignment metrics improve and then plateau, while internal geometry metrics peak earlier and then degrade. The epoch-50 checkpoint is selected because it gives the strongest ARI/NMI, even though epoch 10 has cleaner geometric separation.

## Findings

Fine-tuned MiniLM is the strongest model for this corpus because it improves recovery of real legal document-type structure. Word2Vec remains useful as a transparent domain-trained baseline: it captures repeated legal phrasing very well, but that strength also causes it to group boilerplate across document types.

The SEC filing is the most clearly recovered category. Court briefs and court opinions remain more entangled because they share dense legal formulae, citations, procedural language, and repeated case-specific terms. This explains why silhouette and ARI/NMI point in different directions: silhouette rewards neat cluster geometry, while ARI/NMI reward agreement with known semantic categories.

## Diagnostic Behavior

Cross-tabulation of fine-tuned MiniLM clusters against document types shows asymmetric recovery:

| Document type    | Dominant cluster | Purity | Interpretation                                            |
| ---------------- | ---------------: | -----: | --------------------------------------------------------- |
| SEC filing       |        Cluster 4 |  75.2% | Distinct regulatory/enforcement vocabulary separates well |
| District opinion |        Cluster 0 |  64.3% | Court-opinion language separates partially                |
| Slip opinion     |        Cluster 3 |  67.0% | Court language overlaps with briefs                       |
| Petition brief   |        Cluster 1 |  48.0% | Advocacy-style text overlaps with amicus brief            |
| Amicus brief     | Clusters 1 and 3 |    N/A | Most entangled with petition brief and court language     |

t-SNE plots are useful for local visual inspection, but the quantitative claims rely on silhouette, Davies-Bouldin, ARI, NMI, and cosine similarity checks. t-SNE can make local neighborhoods look cleaner than the full high-dimensional geometry actually is.

## Challenges And Limitations

- The corpus is small: 5 PDFs and 4,067 paragraph units.
- Paragraph-level splitting increases sample count, but many legal paragraphs share boilerplate language.
- KMeans selected `k=2` by silhouette even though there are 5 document types, so ARI/NMI should be interpreted as partial structure recovery.
- Same-document positive pairs are a practical fine-tuning heuristic; manually curated positives and hard negatives would likely improve semantic separation.
- Word2Vec and MiniLM use different embedding dimensions, so geometry metrics are not a pure architecture comparison.

## Code Structure

```text
src/
├── 01_preprocessing.py
├── 02_train_domain_embedder.py
├── 02b_finetune_minilm.py
├── 03_generate_embeddings.py
├── 03b_generate_minilm_comparison.py
├── 04_clustering.py
├── 05_evaluation_metrics.py
├── 06_visualizations.py
├── 07_epoch_trend.py
├── config.py
├── continue_minilm_to_50.py
├── io_utils.py
├── main.py
└── main_minilm_comparison.py
```

## Output Artifacts

```text
outputs/
├── 1/  # Word2Vec domain embedder vs plain MiniLM
│   ├── figures/
│   ├── models/domain_word2vec.model
│   ├── cluster_results.json
│   ├── embedding_metadata.json
│   └── metrics.json
├── 2/  # plain MiniLM vs fine-tuned MiniLM
│   ├── figures/
│   ├── models/finetuned_minilm/
│   ├── cluster_results.json
│   ├── embedding_metadata.json
│   └── metrics.json
└── 3/  # fine-tuning duration ablation
    ├── checkpoints/
    ├── figures/
    ├── metrics_log.csv
    └── metrics_log.json
```

## How To Reproduce

The experiments have already been run, and the results above are based on the saved artifacts. To reproduce the pipeline from scratch:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
cd "/home/neel/Desktop/# Domain-Specific-Legal-Document-Embedder"
python src/main.py
```

Optional extended runs:

```bash
python src/main_minilm_comparison.py
python src/continue_minilm_to_50.py
python src/07_epoch_trend.py
```

## Resources Referred

| Method or model                                                         | Reference                                                                            |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `all-MiniLM-L6-v2`                                                      | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2                        |
| Word2Vec skip-gram                                                      | Mikolov et al., 2013, https://arxiv.org/abs/1301.3781                                |
| Silhouette score                                                        | Rousseeuw, 1987, https://doi.org/10.1016/0377-0427(87)90125-7                        |
| t-SNE                                                                   | van der Maaten and Hinton, 2008, https://www.jmlr.org/papers/v9/vandermaaten08a.html |
| KMeans, Agglomerative, Davies-Bouldin, Calinski-Harabasz, ARI, NMI, PCA | scikit-learn implementations                                                         |

Environment notes:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
pip install --upgrade pyopenssl cryptography
pip install "huggingface-hub>=0.19.3,<1.0"
```
