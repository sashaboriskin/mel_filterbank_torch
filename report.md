# Исследование влияния параметров log-mel фильтрбанков и grouped convolutions на классификацию речевых команд

## 2. План экспериментов

| # | Переменная | Значения | Цель |
|---|-----------|----------|------|
| 1 | n_mels | 20, 40, 80 | Влияние частотного разрешения на качество |
| 2 | groups (n_mels=80) | 1, 2, 4, 8, 16 | Снижение FLOPs/params за счёт grouped convolutions |
| 3 | groups + shuffle (n_mels=80) | 2, 4, 8, 16 | Компенсация потери межгрупповой информации через channel shuffle |

## 3. Exp1: baseline with n_mels

### Results

| n_mels | Test Accuracy (%) | Params | FLOPs |
|--------|-------------------|--------|-------|
| 20     |    0.9964         | 22,818 | 2,311,808      |
| 40     |    0.9964         | 26,658 | 3,087,488      |
| 80     |    0.9951         | 34,338 | 4,638,848      |

![train/loss для разных n_mels](img/exp1_train_loss.jpg)

![val/accuracy для разных n_mels](img/exp1_val_accuracy.jpg)


### Conclusion

All three configurations achieve near-identical test accuracy (99.5–99.6%), indicating that for a simple yes/no classification task the frequency resolution beyond 20 mel bands provides no measurable benefit. However, n_mels=80 was selected as the baseline for subsequent experiments, as it provides the widest parameter and FLOPs budget — making the effects of grouped convolutions and channel shuffle more pronounced and easier to analyze.

## 4. Exp 2: grouped convolutions

### Results

| Groups | Test Accuracy (%) | Params | FLOPs | Δ FLOPs vs g=1 (%) |
|--------|-------------------|--------|-------|---------------------|
| 1      | 0.9951 | 34,338 | 4,638,848 | —      |
| 2      | 0.9818 | 17,442 | 2,319,488 | −50.0% |
| 4      | 0.9782 |  8,994 | 1,159,808 | −75.0% |
| 8      | 0.9745 |  4,770 |   579,968 | −87.5% |
| 16     | 0.9684 |  2,658 |   290,048 | −93.7% |

![train/loss для разных groups](img/exp2_train_loss.jpg)

![val/accuracy для разных groups](img/exp2_val_accuracy.jpg)

### Conclusion

Grouped convolutions provide a nearly linear trade-off between computational cost and accuracy. Doubling the number of groups halves the FLOPs while reducing accuracy by roughly 0.4–0.7 percentage points per step. Even at groups=16, where FLOPs are reduced by 93.7% and parameter count drops from 34K to just 2.6K, the model still retains 96.8% accuracy — only 2.7% below the baseline. This confirms that for a binary classification task the standard convolution is heavily over-parameterized.

## 5. Exp 3: эффект channel shuffle

### Results

<!-- Таблица: сравнение n_mels_80_groups_N vs n_mels_80_groups_N_shuffle -->

| Groups | Accuracy без shuffle (%) | Accuracy с shuffle (%) | Δ Accuracy |
|--------|--------------------------|------------------------|------------|
| 2      | 0.9818 | 0.9867 | +0.49% |
| 4      | 0.9782 | 0.9903 | +1.21% |
| 8      | 0.9745 | 0.9806 | +0.61% |
| 16     | 0.9684 | 0.9709 | +0.25% |


<!-- Скриншот: TensorBoard → val/accuracy, выбрать пару с наибольшим эффектом shuffle, например groups=8 и groups=8_shuffle -->

![train/loss для разных groups](img/exp3_train_loss.jpg)

![val/accuracy для разных groups](img/exp3_val_accuracy.jpg)

### Conclusion

Channel shuffle consistently improves accuracy across all group sizes, confirming that inter-group information exchange is beneficial. The largest gain is observed at groups=4 (+1.21%), where shuffle brings accuracy to 99.03% — nearly matching the ungrouped baseline (99.51%) while using only 25% of its FLOPs. For groups=2 and groups=8 the improvement is moderate (+0.49% and +0.61% respectively). At groups=16 the effect is smallest (+0.25%), suggesting that with very aggressive grouping the representational bottleneck becomes too severe for shuffle alone to compensate.

## 6. Общее сравнение

### Accuracy vs Efficiency

<!-- Scatter plot: по оси X — FLOPs (или Params), по оси Y — Test Accuracy. Каждая точка — один ран. Подписать точки именами конфигураций. Построить в matplotlib. -->

![Trade-off accuracy vs FLOPs](figures/tradeoff.png)

### Лучшие конфигурации

| Criterion | Configuration | Test Accuracy (%) | FLOPs |
|-----------|--------------|-------------------|-------|
| Best accuracy   | n_mels=80, groups=1           | 99.51 | 4,638,848 |
| Best trade-off  | n_mels=80, groups=4, shuffle  | 99.03 | 1,159,808 |
| Min FLOPs       | n_mels=80, groups=16, shuffle | 97.09 |   290,048 |

## 7. Conclusion

For a binary yes/no speech command classification task, increasing mel frequency resolution beyond 20 bands yields no accuracy gain — all three n_mels settings achieve ~99.5%. Grouped convolutions effectively reduce computational cost: each doubling of groups halves FLOPs at the expense of only ~0.5% accuracy. Channel shuffle partially recovers this loss by restoring cross-group feature interaction, with the sweet spot at groups=4 + shuffle (99.03% accuracy at 75% FLOPs reduction). For deployment on resource-constrained devices, the groups=4 + shuffle configuration offers the best accuracy-efficiency trade-off, while groups=16 + shuffle provides a viable option when extreme compression (16x FLOPs reduction) is required with only a 2.4% accuracy penalty.
