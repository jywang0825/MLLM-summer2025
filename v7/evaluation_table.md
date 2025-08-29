# Ego4D NLQ Uniform Captions Evaluation Results

## Summary Statistics

| Metric | Score | Standard Deviation | Range |
|--------|-------|-------------------|-------|
| **BLEU-1** | 0.1588 | ±0.0738 | 0.0000-0.3750 |
| **BLEU-2** | 0.0475 | ±0.0453 | 0.0000-0.2554 |
| **BLEU-3** | 0.0244 | ±0.0280 | 0.0000-0.1841 |
| **BLEU-4** | 0.0139 | ±0.0144 | 0.0000-0.1258 |
| **METEOR** | 0.1608 | ±0.0702 | 0.0000-0.4408 |
| **ROUGE1_F1** | 0.1926 | ±0.0825 | 0.0000-0.5128 |
| **ROUGE2_F1** | 0.0272 | ±0.0415 | 0.0000-0.2162 |
| **ROUGEL_F1** | 0.1415 | ±0.0570 | 0.0000-0.3590 |
| **Semantic Similarity** | 0.0000 | ±0.0000 | 0.0000-0.0000 |
| **CLAIRE Score** | 0.0000 | ±0.0000 | 0.0000-0.0000 |

## Dataset Information

- **Total Videos Evaluated**: 298
- **Model**: Uniform frame sampling with InternVL2.5 captioning
- **Evaluation Date**: July 9, 2025

## Performance Analysis

### Strong Points:
- **BLEU-1 (0.1588)**: Good word-level overlap
- **METEOR (0.1608)**: Reasonable semantic similarity
- **ROUGE-1 F1 (0.1926)**: Decent recall of important words

### Areas for Improvement:
- **BLEU-3/4**: Low scores indicate limited phrase-level matching
- **ROUGE-2**: Suggests room for improvement in bigram overlap
- **Semantic Similarity**: Not computed (requires additional setup)

### Overall Assessment:
The uniform captioning approach shows moderate performance with room for improvement in higher-order language modeling and semantic understanding. 