# Data Files

All sequences are in standard FASTA format.
Counts verified against manuscript Table 1 and Table 2.

| File                   | Sequences | Description                              |
|------------------------|-----------|------------------------------------------|
| pos_predenoise.fasta   | 1,631     | IEDB positives before denoising          |
| neg_predenoise.fasta   | 3,113     | IEDB negatives before denoising          |
| pos_cleaned.fasta      | 1,198     | Positives after conflict removal         |
| neg_cleaned.fasta      | 3,277     | Negatives after 3-source reconstruction  |
| train_pos.fasta        | 958       | Training positives (50% CD-HIT split)    |
| train_neg.fasta        | 2,621     | Training negatives (50% CD-HIT split)    |
| test_pos.fasta         | 240       | Independent test positives               |
| test_neg.fasta         | 656       | Independent test negatives               |
| synthetic_train.fasta  | 618       | ProtGPT2 generated AIP candidates        |

## Notes
- All partitioning performed cluster-wise at 50% CD-HIT identity
- Synthetic sequences filtered at 40% CD-HIT against training positives
- Negative sources: IEDB failures (68.2%), SATPdb (8.4%), APD6 (23.4%)
- 432 conflicting sequences removed prior to partitioning
