# Resources Dump

Raw references, libraries, and documentation used in this project.

---

## Python Libraries

| Library | Version (conda base) | Purpose | Docs |
|---|---|---|---|
| pdfplumber | 0.11.5 | PDF text extraction | https://github.com/jsvine/pdfplumber |
| gensim | 4.1.2 | Word2Vec training | https://radimrehurek.com/gensim/models/word2vec.html |
| scikit-learn | 1.0.2 | Clustering, metrics, PCA, t-SNE | https://scikit-learn.org/stable/ |
| sentence-transformers | 2.6.1 | Baseline MiniLM embeddings | https://www.sbert.net/ |
| matplotlib | 3.7.4 | Plotting | https://matplotlib.org/ |
| seaborn | 0.11.2 | Heatmaps, color palettes | https://seaborn.pydata.org/ |
| numpy | (conda base) | Array operations | https://numpy.org/doc/ |
| torch | 2.1.0 | Backend for sentence-transformers | https://pytorch.org/ |
| transformers | 4.36.2 | HuggingFace model loading | https://huggingface.co/docs/transformers/ |
| nltk | 3.8.1 | (available, not heavily used) | https://www.nltk.org/ |

## Pretrained Models

| Model | Source | Usage |
|---|---|---|
| `all-MiniLM-L6-v2` | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Baseline sentence embedder (384D) |

## Algorithms & Methods

| Method | Reference |
|---|---|
| Word2Vec (Skip-gram) | Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013) — https://arxiv.org/abs/1301.3781 |
| KMeans clustering | Lloyd (1982), sklearn implementation |
| Agglomerative clustering | sklearn `AgglomerativeClustering` with cosine affinity |
| Silhouette score | Rousseeuw (1987) — https://doi.org/10.1016/0377-0427(87)90125-7 |
| Davies-Bouldin index | Davies & Bouldin (1979) |
| Calinski-Harabasz score | Caliński & Harabasz (1974) |
| Adjusted Rand Index | Hubert & Arabie (1985) |
| Normalized Mutual Information | Strehl & Ghosh (2002) |
| PCA | Pearson (1901), sklearn `PCA` |
| t-SNE | van der Maaten & Hinton (2008) — https://www.jmlr.org/papers/v9/vandermaaten08a.html |

## Sample Data (Legal PDFs)

| File | Description | Source type |
|---|---|---|
| `20240809133545974_23-980_petbrief.pdf` | Facebook v. Amalgamated Bank — Petitioners' brief, SCOTUS No. 23-980 | Supreme Court |
| `22-1165_10n2.pdf` | Slip opinion / syllabus, October Term 2023 | Supreme Court |
| `23-980bsacus_facebookvamalgamatedbank.pdf` | U.S. amicus brief supporting respondents, SCOTUS No. 23-980 | Supreme Court |
| `USCOURTS-ilnd-1_04-cv-00397-5.pdf` | Memorandum opinion, N.D. Illinois | District Court |
| `comp26286.pdf` | SEC enforcement document, D.N.J. | Regulatory / SEC |

## Project Plan References

- Original assignment spec: `question.md`
- Implementation plan: `plan.md` (Word2Vec/FastText approach, legal PDF adjustments, ARI/NMI evaluation)

## Environment Notes

- **Conda environment:** `base` at `/home/neel/anaconda3`
- **Fixes applied during setup:**
  - `pip install --upgrade pyopenssl cryptography` (gensim/smart_open import error)
  - `pip install "huggingface-hub>=0.19.3,<1.0"` (transformers compatibility)
- **Run command:**
  ```bash
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate base
  python src/main.py
  ```

## Output Artifacts

```
outputs/
├── figures/
│   ├── 01_pca_domain_clusters.png
│   ├── 02_pca_baseline_clusters.png
│   ├── 03_tsne_comparison.png
│   ├── 04_silhouette_domain.png
│   ├── 05_metrics_comparison.png
│   ├── 06_doc_type_distribution.png
│   ├── 07_similarity_heatmap_domain.png
│   └── 08_similarity_heatmap_baseline.png
├── models/
│   └── domain_word2vec.model
├── cluster_results.json
├── embedding_metadata.json
└── metrics.json

data/processed/
├── paragraphs.jsonl
└── preprocessing_stats.json
```
