# Summary — Domain-Specific Legal Embedder

## Objective

Build and evaluate a domain-specific embedder trained on legal PDF documents, compared against the general-purpose baseline `all-MiniLM-L6-v2`, using clustering quality and similarity metrics.

## Corpus

| Document | Type | Paragraphs |
|---|---|---|
| `20240809133545974_23-980_petbrief.pdf` | Supreme Court petition brief | 1,403 |
| `22-1165_10n2.pdf` | Slip opinion / syllabus | 318 |
| `23-980bsacus_facebookvamalgamatedbank.pdf` | Amicus brief | 1,061 |
| `USCOURTS-ilnd-1_04-cv-00397-5.pdf` | District court opinion | 499 |
| `comp26286.pdf` | SEC filing | 786 |

**Total:** 5 PDFs → 4,067 paragraph-level units, ~61 chars avg, 7,100 unique tokens.

## Approach

1. **PDF extraction** via `pdfplumber`, with legal-aware cleaning (preserve citations, defined terms, domain vocabulary).
2. **Segmentation** at paragraph level with fallback chunking (~600 chars) when PDF layout lacks blank-line breaks.
3. **Domain embedder:** Word2Vec skip-gram (100D, 20 epochs) trained on the legal corpus; document-unit vectors via mean pooling of in-vocabulary word vectors.
4. **Baseline:** `all-MiniLM-L6-v2` (384D), no fine-tuning.
5. **Clustering:** KMeans (k=2, selected by silhouette sweep) and Agglomerative clustering on L2-normalized embeddings.
6. **Evaluation:** Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI/NMI vs document-type labels, intra/inter-cluster cosine similarity, nearest-neighbor checks.

## Key Results

| Metric | Domain (Word2Vec) | Baseline (MiniLM) | Winner |
|---|---|---|---|
| Silhouette score ↑ | **0.417** | 0.067 | Domain |
| Davies-Bouldin index ↓ | **1.73** | 5.23 | Domain |
| Calinski-Harabasz ↑ | **794** | 148 | Domain |
| Intra/inter similarity gap ↑ | **0.226** | 0.058 | Domain |
| ARI vs doc type ↑ | 0.006 | **0.075** | Baseline |
| NMI vs doc type ↑ | 0.006 | **0.077** | Baseline |

### Interpretation

**Domain embedder wins on unsupervised cluster separation.** The Word2Vec model produces much tighter, better-separated clusters (silhouette 0.42 vs 0.07). Nearest-neighbor checks confirm it captures legal boilerplate and repeated phrasing well — e.g., "interstate commerce or of the mails" paragraphs from different documents cluster together with >0.97 cosine similarity.

**Baseline wins on document-type recovery.** MiniLM's higher ARI/NMI (though still low in absolute terms) suggests its general semantic understanding partially distinguishes document categories (petition brief vs SEC filing), whereas the domain model overfits to shared legal formulae that appear across all document types.

**Why the split?** Legal text is highly formulaic. Word2Vec trained on this corpus learns domain-specific n-grams and standard phrases ("Risk Factors", "Counsel of Record", securities Act references) that repeat across document types. This improves within-cluster cohesion but blurs boundaries between document categories. MiniLM captures broader topical semantics (court opinion vs regulatory complaint) at the cost of weaker overall cluster geometry on this short, repetitive text.

## Challenges

- **PDF paragraph detection:** Initial extraction yielded only 2 paragraphs (missing `\n\n` breaks). Fixed with line-based and sentence-chunk fallback.
- **Small document count:** Only 5 source PDFs; paragraph-level units inflate sample size but many chunks share identical legal language.
- **Optimal k=2:** Silhouette sweep selected k=2, not k=5 (document types), limiting ARI/NMI usefulness.
- **Environment:** Required conda `base` with pyOpenSSL and huggingface-hub version fixes for gensim and sentence-transformers.

## Model Behavior

- Domain Word2Vec (vocab: 3,114 words, 100D) excels at finding near-duplicate legal phrasing across documents.
- Baseline MiniLM produces more dispersed embeddings with lower absolute similarities, but better weak alignment with document-type labels.
- For a production legal search/RAG system, a hybrid approach (domain Word2Vec + MiniLM concatenation, or fine-tuned legal BERT) would likely outperform either alone.

## Outputs

- **Code:** `src/01_preprocessing.py` through `src/06_visualizations.py`, orchestrated by `src/main.py`
- **Figures:** `outputs/figures/` (8 plots: PCA clusters, t-SNE comparison, silhouette, metrics bar chart, doc-type distribution, similarity heatmaps)
- **Metrics:** `outputs/metrics.json`
- **Model:** `outputs/models/domain_word2vec.model`

## Run Command

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate base
cd "/home/neel/Desktop/Custom Embedding"
python src/main.py
```
