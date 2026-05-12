# Baseline Results

## Summary: Flower vs iMF (Best Epoch Avg SR)

| Benchmark      | Flower | iMF   |
|----------------|--------|-------|
| LIBERO-Spatial | 0.980  | 0.976 | # Both run with 50 n_eval. Flower retrained with EMA off.
| LIBERO-Object  | 0.988  | 0.992 | # Both run with 50 n_eval. Flower retrained with EMA off.
| LIBERO-Goal    | 0.972  | 0.950 | # Both run with 50 n_eval. Flower retrained with EMA off.
| LIBERO-10      | 0.904  | 0.916 | # Both run with 50 n_eval. Flower retrained with EMA off.
| LIBERO-90      | -      | -     | # Run with 50 n_eval

---

## iMF LIBERO-Object (Job 40461499)

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.992  |
| 29    | 0.970  |
| 39    | 0.980  |
| 49    | 0.965  |
| 59    | 0.982  |
| 69    | 0.972  |
| 79    | 0.966  |
| 89    | 0.968  |
| 99    | 0.984  |
| 109   | 0.990  |
| 119   | 0.980  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Pick up alphabet soup      | 0.981 | 0.979 | 1.000 | 0.979 | 0.981 | 1.000 | 1.000 | 0.962 | 1.000 | 1.000  | 1.000  |
| 2. Pick up cream cheese       | 0.981 | 0.939 | 0.960 | 0.856 | 0.962 | 0.902 | 0.918 | 0.880 | 0.979 | 0.960  | 0.921  |
| 3. Pick up salad dressing     | 1.000 | 0.958 | 1.000 | 0.979 | 1.000 | 0.981 | 1.000 | 0.960 | 1.000 | 1.000  | 0.981  |
| 4. Pick up BBQ sauce          | 0.979 | 0.979 | 0.979 | 0.939 | 0.958 | 0.981 | 1.000 | 0.960 | 1.000 | 1.000  | 0.979  |
| 5. Pick up ketchup            | 1.000 | 1.000 | 0.981 | 1.000 | 0.979 | 0.979 | 0.939 | 1.000 | 0.938 | 0.979  | 1.000  |
| 6. Pick up tomato sauce       | 1.000 | 1.000 | 1.000 | 0.981 | 0.979 | 0.920 | 0.881 | 0.941 | 0.960 | 0.981  | 0.958  |
| 7. Pick up butter             | 1.000 | 1.000 | 0.981 | 1.000 | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000  | 1.000  |
| 8. Pick up milk               | 1.000 | 0.859 | 0.941 | 0.941 | 1.000 | 0.979 | 0.941 | 0.979 | 0.979 | 1.000  | 0.960  |
| 9. Pick up chocolate pudding  | 1.000 | 1.000 | 0.962 | 1.000 | 1.000 | 0.979 | 0.981 | 1.000 | 1.000 | 1.000  | 1.000  |
| 10. Pick up orange juice      | 0.981 | 0.981 | 1.000 | 0.979 | 0.981 | 1.000 | 1.000 | 1.000 | 0.981 | 0.979  | 1.000  |
| **Average**                   | **0.992** | **0.970** | **0.980** | **0.965** | **0.982** | **0.972** | **0.966** | **0.968** | **0.984** | **0.990** | **0.980** |

---

## iMF LIBERO-Goal (Job 40461391)

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.928  |
| 29    | 0.934  |
| 39    | 0.937  |
| 49    | 0.926  |
| 59    | 0.938  |
| 69    | 0.945  |
| 79    | 0.950  |
| 89    | 0.946  |
| 99    | 0.946  |
| 109   | 0.950  |
| 119   | 0.950  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Open middle drawer of cabinet    | 0.958 | 0.958 | 0.979 | 0.941 | 0.981 | 0.958 | 0.958 | 0.901 | 0.938 | 0.939  | 0.918  |
| 2. Put bowl on stove                | 1.000 | 0.981 | 1.000 | 0.981 | 1.000 | 0.981 | 0.981 | 1.000 | 0.981 | 1.000  | 0.981  |
| 3. Put wine bottle on top of cabinet| 0.880 | 0.962 | 0.962 | 0.981 | 0.880 | 0.904 | 1.000 | 0.941 | 0.881 | 0.960  | 1.000  |
| 4. Open top drawer & put bowl inside| 0.760 | 0.737 | 0.705 | 0.720 | 0.742 | 0.801 | 0.779 | 0.880 | 0.821 | 0.857  | 0.761  |
| 5. Put bowl on top of cabinet       | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.981 | 0.979 | 1.000 | 1.000 | 1.000  | 1.000  |
| 6. Push plate to front of stove     | 0.962 | 0.962 | 0.960 | 0.962 | 0.941 | 1.000 | 0.960 | 0.920 | 0.960 | 0.981  | 0.981  |
| 7. Put cream cheese in bowl         | 0.838 | 0.763 | 0.861 | 0.763 | 0.881 | 0.840 | 0.857 | 0.881 | 0.901 | 0.840  | 0.878  |
| 8. Turn on stove                    | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000  | 1.000  |
| 9. Put bowl on plate                | 1.000 | 1.000 | 0.960 | 0.958 | 0.979 | 1.000 | 0.981 | 0.979 | 1.000 | 1.000  | 1.000  |
| 10. Put wine bottle on rack         | 0.880 | 0.981 | 0.939 | 0.960 | 0.979 | 0.981 | 1.000 | 0.958 | 0.981 | 0.918  | 0.979  |
| **Average**                         | **0.928** | **0.934** | **0.937** | **0.926** | **0.938** | **0.945** | **0.950** | **0.946** | **0.946** | **0.950** | **0.950** |

---

## iMF LIBERO-10 (Job 40461556)

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.881  |
| 29    | 0.811  |
| 39    | 0.857  |
| 49    | 0.866  |
| 59    | 0.877  |
| 69    | 0.873  |
| 79    | 0.886  |
| 89    | 0.884  |
| 99    | 0.892  |
| 109   | 0.887  |
| 119   | 0.916  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Put alphabet soup & tomato sauce in basket (LR2)       | 0.979 | 0.918 | 0.881 | 0.878 | 0.917 | 0.960 | 0.958 | 0.918 | 0.938 | 0.958  | 0.960  |
| 2. Put cream cheese & butter in basket (LR2)              | 0.958 | 0.939 | 0.939 | 0.939 | 0.920 | 0.899 | 0.899 | 0.880 | 0.979 | 0.897  | 0.981  |
| 3. Turn on stove & put moka pot (K3)                      | 1.000 | 0.979 | 0.857 | 0.958 | 0.939 | 0.939 | 0.962 | 0.981 | 0.981 | 1.000  | 0.979  |
| 4. Put black bowl in drawer & close (K4)                  | 1.000 | 1.000 | 0.958 | 0.941 | 0.920 | 0.958 | 0.958 | 0.939 | 0.941 | 1.000  | 0.979  |
| 5. Put white mug on left plate & yellow mug on right (LR5)| 0.838 | 0.761 | 0.780 | 0.763 | 0.880 | 0.760 | 0.899 | 0.875 | 0.838 | 0.918  | 0.881  |
| 6. Pick up book & place in caddy (S1)                     | 0.918 | 0.878 | 0.897 | 0.897 | 0.877 | 0.856 | 0.918 | 0.918 | 0.897 | 0.877  | 0.897  |
| 7. Put white mug on plate & chocolate pudding right (LR6) | 0.699 | 0.678 | 0.782 | 0.843 | 0.840 | 0.801 | 0.838 | 0.780 | 0.643 | 0.840  | 0.861  |
| 8. Put alphabet soup & cream cheese in basket (LR1)       | 1.000 | 0.979 | 1.000 | 0.981 | 0.958 | 0.899 | 0.942 | 0.941 | 0.981 | 0.962  | 1.000  |
| 9. Put both moka pots on stove (K8)                       | 0.641 | 0.361 | 0.643 | 0.599 | 0.660 | 0.761 | 0.599 | 0.764 | 0.744 | 0.583  | 0.723  |
| 10. Put yellow & white mug in microwave & close (K6)      | 0.779 | 0.620 | 0.835 | 0.861 | 0.861 | 0.899 | 0.881 | 0.841 | 0.981 | 0.837  | 0.897  |
| **Average**                                                | **0.881** | **0.811** | **0.857** | **0.866** | **0.877** | **0.873** | **0.886** | **0.884** | **0.892** | **0.887** | **0.916** |

---

## iMF LIBERO-Spatial (Job 40461397)

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.956  |
| 29    | 0.932  |
| 39    | 0.962  |
| 49    | 0.946  |
| 59    | 0.968  |
| 69    | 0.974  |
| 79    | 0.961  |
| 89    | 0.958  |
| 99    | 0.976  |
| 109   | 0.966  |
| 119   | 0.966  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Bowl between plate & ramekin    | 0.938 | 0.981 | 1.000 | 0.958 | 0.958 | 0.958 | 0.878 | 0.917 | 0.918 | 0.918  | 0.918  |
| 2. Bowl next to ramekin            | 0.981 | 0.979 | 1.000 | 1.000 | 1.000 | 0.981 | 1.000 | 1.000 | 1.000 | 0.981  | 0.962  |
| 3. Bowl from table center          | 0.979 | 0.981 | 0.979 | 1.000 | 1.000 | 0.979 | 0.979 | 0.962 | 0.981 | 0.979  | 0.979  |
| 4. Bowl on cookie box              | 0.960 | 0.920 | 1.000 | 0.941 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000  | 1.000  |
| 5. Bowl in top drawer              | 1.000 | 0.939 | 0.979 | 1.000 | 0.962 | 0.960 | 0.960 | 0.958 | 0.981 | 0.979  | 0.979  |
| 6. Bowl on ramekin                 | 0.801 | 0.683 | 0.861 | 0.798 | 0.921 | 0.899 | 0.835 | 0.941 | 0.960 | 0.920  | 0.938  |
| 7. Bowl next to cookie box         | 1.000 | 1.000 | 1.000 | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.981  | 1.000  |
| 8. Bowl on stove                   | 0.960 | 1.000 | 1.000 | 0.981 | 0.938 | 1.000 | 1.000 | 0.960 | 1.000 | 1.000  | 1.000  |
| 9. Bowl next to plate              | 0.981 | 1.000 | 0.921 | 0.942 | 0.981 | 0.960 | 0.979 | 0.960 | 1.000 | 1.000  | 0.979  |
| 10. Bowl on wooden cabinet         | 0.960 | 0.841 | 0.881 | 0.862 | 0.918 | 1.000 | 0.979 | 0.880 | 0.918 | 0.897  | 0.901  |
| **Average**                        | **0.956** | **0.932** | **0.962** | **0.946** | **0.968** | **0.974** | **0.961** | **0.958** | **0.976** | **0.966** | **0.966** |

---

## iMF LIBERO-Spatial (Job 39829339)

*Properly trained on libero_spatial data, 10 n_eval. Job still running as of epoch 96; last evaluation at epoch 89.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.929  |
| 29    | 0.963  |
| 39    | 0.892  |
| 49    | 0.883  |
| 59    | 0.963  |
| 69    | 0.900  |
| 79    | 0.963  |
| 89    | 0.975  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1. Bowl between plate & ramekin    | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2. Bowl next to ramekin            | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 |
| 3. Bowl from table center          | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 |
| 4. Bowl on cookie box              | 0.917 | 0.917 | 0.917 | 0.875 | 1.000 | 0.917 | 1.000 | 1.000 |
| 5. Bowl in top drawer              | 0.917 | 0.917 | 0.792 | 0.917 | 1.000 | 0.917 | 1.000 | 1.000 |
| 6. Bowl on ramekin                 | 0.917 | 0.792 | 0.625 | 0.500 | 0.708 | 0.750 | 0.792 | 0.833 |
| 7. Bowl next to cookie box         | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8. Bowl on stove                   | 1.000 | 1.000 | 1.000 | 0.875 | 1.000 | 0.792 | 1.000 | 1.000 |
| 9. Bowl next to plate              | 0.917 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 0.917 | 1.000 |
| 10. Bowl on wooden cabinet         | 0.625 | 1.000 | 0.708 | 0.792 | 0.917 | 0.750 | 1.000 | 0.917 |
| **Average**                        | **0.929** | **0.963** | **0.892** | **0.883** | **0.963** | **0.900** | **0.963** | **0.975** |

---

## iMF LIBERO-10 — Ablation: heads-swap, ratio 0.5 (Job 40670763)

*Ablation of iMF LIBERO-10 baseline (Job 40461556) with heads-swap enabled and swap ratio 0.5. 50 n_eval. Job killed during epoch 108. Last evaluation at epoch 99.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.853  |
| 29    | 0.833  |
| 39    | 0.843  |
| 49    | 0.842  |
| 59    | 0.855  |
| 69    | 0.872  |
| 79    | 0.899  |
| 89    | 0.870  |
| 99    | 0.879  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1. Put alphabet soup & tomato sauce in basket (LR2)       | 0.939 | 0.899 | 0.920 | 0.918 | 0.938 | 0.981 | 0.960 | 0.917 | 0.917 |
| 2. Put cream cheese & butter in basket (LR2)              | 0.958 | 1.000 | 0.960 | 0.801 | 0.918 | 0.840 | 0.920 | 0.880 | 0.920 |
| 3. Turn on stove & put moka pot (K3)                      | 0.859 | 0.921 | 0.881 | 0.843 | 0.920 | 0.920 | 0.877 | 0.960 | 0.920 |
| 4. Put black bowl in drawer & close (K4)                  | 0.942 | 0.958 | 0.960 | 0.960 | 0.942 | 0.960 | 0.979 | 0.958 | 0.960 |
| 5. Put white mug on left plate & yellow mug on right (LR5)| 0.776 | 0.798 | 0.819 | 0.615 | 0.782 | 0.899 | 0.880 | 0.798 | 0.838 |
| 6. Pick up book & place in caddy (S1)                     | 0.918 | 0.758 | 0.878 | 0.897 | 0.877 | 0.856 | 0.918 | 0.880 | 0.897 |
| 7. Put white mug on plate & chocolate pudding right (LR6) | 0.859 | 0.779 | 0.596 | 0.760 | 0.861 | 0.859 | 0.859 | 0.779 | 0.744 |
| 8. Put alphabet soup & cream cheese in basket (LR1)       | 0.979 | 0.958 | 0.958 | 1.000 | 0.920 | 0.878 | 0.960 | 0.981 | 0.958 |
| 9. Put both moka pots on stove (K8)                       | 0.558 | 0.500 | 0.681 | 0.761 | 0.615 | 0.667 | 0.777 | 0.704 | 0.800 |
| 10. Put yellow & white mug in microwave & close (K6)      | 0.742 | 0.761 | 0.779 | 0.859 | 0.782 | 0.857 | 0.859 | 0.841 | 0.841 |
| **Average**                                                | **0.853** | **0.833** | **0.843** | **0.842** | **0.855** | **0.872** | **0.899** | **0.870** | **0.879** |

---

## iMF LIBERO-10 — Ablation: heads-swap (Job 40670769)

*Ablation of iMF LIBERO-10 baseline (Job 40461556) with heads-swap enabled (default swap ratio). 50 n_eval. Last evaluation at epoch 99.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.856  |
| 29    | 0.878  |
| 39    | 0.872  |
| 49    | 0.876  |
| 59    | 0.866  |
| 69    | 0.874  |
| 79    | 0.840  |
| 89    | 0.872  |
| 99    | 0.886  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1. Put alphabet soup & tomato sauce in basket (LR2)       | 0.883 | 0.960 | 0.981 | 0.877 | 0.979 | 0.899 | 0.960 | 0.958 | 0.920 |
| 2. Put cream cheese & butter in basket (LR2)              | 0.942 | 0.941 | 0.960 | 0.904 | 0.920 | 0.981 | 0.921 | 0.979 | 1.000 |
| 3. Turn on stove & put moka pot (K3)                      | 0.960 | 0.920 | 0.918 | 0.979 | 0.920 | 1.000 | 0.918 | 0.981 | 0.979 |
| 4. Put black bowl in drawer & close (K4)                  | 0.958 | 0.899 | 0.878 | 0.962 | 0.962 | 0.960 | 0.960 | 0.921 | 0.981 |
| 5. Put white mug on left plate & yellow mug on right (LR5)| 0.859 | 0.822 | 0.761 | 0.819 | 0.752 | 0.861 | 0.861 | 0.862 | 0.880 |
| 6. Pick up book & place in caddy (S1)                     | 0.880 | 0.856 | 0.878 | 0.877 | 0.897 | 0.897 | 0.878 | 0.877 | 0.897 |
| 7. Put white mug on plate & chocolate pudding right (LR6) | 0.841 | 0.896 | 0.763 | 0.763 | 0.857 | 0.737 | 0.720 | 0.841 | 0.598 |
| 8. Put alphabet soup & cream cheese in basket (LR1)       | 0.981 | 0.981 | 0.960 | 0.881 | 0.899 | 0.920 | 0.960 | 0.862 | 0.941 |
| 9. Put both moka pots on stove (K8)                       | 0.561 | 0.601 | 0.763 | 0.739 | 0.614 | 0.700 | 0.559 | 0.599 | 0.822 |
| 10. Put yellow & white mug in microwave & close (K6)      | 0.697 | 0.901 | 0.861 | 0.962 | 0.857 | 0.780 | 0.663 | 0.837 | 0.841 |
| **Average**                                                | **0.856** | **0.878** | **0.872** | **0.876** | **0.866** | **0.874** | **0.840** | **0.872** | **0.886** |

---

## iMF LIBERO-10 — Ablation: ratio 0.5 (Job 40670765)

*Ablation of iMF LIBERO-10 baseline (Job 40461556) with swap ratio 0.5 (heads-swap disabled). 50 n_eval. Job killed during epoch 112. Last evaluation at epoch 109.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.895  |
| 29    | 0.838  |
| 39    | 0.875  |
| 49    | 0.874  |
| 59    | 0.875  |
| 69    | 0.876  |
| 79    | 0.878  |
| 89    | 0.886  |
| 99    | 0.909  |
| 109   | 0.891  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| 1. Put alphabet soup & tomato sauce in basket (LR2)       | 0.939 | 0.857 | 0.897 | 0.917 | 0.897 | 0.958 | 0.979 | 0.960 | 0.979 | 0.918  |
| 2. Put cream cheese & butter in basket (LR2)              | 0.979 | 0.782 | 0.859 | 0.918 | 0.960 | 0.917 | 0.958 | 0.960 | 0.941 | 0.960  |
| 3. Turn on stove & put moka pot (K3)                      | 0.981 | 0.921 | 0.918 | 0.960 | 0.960 | 0.941 | 0.958 | 0.979 | 0.958 | 0.981  |
| 4. Put black bowl in drawer & close (K4)                  | 0.958 | 0.941 | 0.981 | 1.000 | 0.941 | 0.979 | 0.941 | 0.979 | 0.962 | 1.000  |
| 5. Put white mug on left plate & yellow mug on right (LR5)| 0.941 | 0.822 | 0.800 | 0.803 | 0.821 | 0.840 | 0.817 | 0.883 | 0.838 | 0.806  |
| 6. Pick up book & place in caddy (S1)                     | 0.899 | 0.856 | 0.897 | 0.897 | 0.897 | 0.878 | 0.859 | 0.897 | 0.918 | 0.918  |
| 7. Put white mug on plate & chocolate pudding right (LR6) | 0.841 | 0.700 | 0.819 | 0.686 | 0.713 | 0.801 | 0.821 | 0.779 | 0.896 | 0.857  |
| 8. Put alphabet soup & cream cheese in basket (LR1)       | 0.939 | 1.000 | 0.962 | 0.941 | 0.941 | 0.902 | 0.960 | 1.000 | 0.979 | 0.962  |
| 9. Put both moka pots on stove (K8)                       | 0.660 | 0.763 | 0.694 | 0.699 | 0.720 | 0.726 | 0.702 | 0.577 | 0.764 | 0.681  |
| 10. Put yellow & white mug in microwave & close (K6)      | 0.816 | 0.740 | 0.920 | 0.920 | 0.897 | 0.821 | 0.780 | 0.841 | 0.859 | 0.822  |
| **Average**                                                | **0.895** | **0.838** | **0.875** | **0.874** | **0.875** | **0.876** | **0.878** | **0.886** | **0.909** | **0.891** |

---

## iMF LIBERO-10 — From-scratch ablation: default (Job 40945548)

*From-scratch (no pretrained checkpoint) iMF on LIBERO-10 with config defaults: `model.ratio=0.25`, `model.imf_head_depth=8`. 50 n_eval. Last evaluation at epoch 119.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.859  |
| 29    | 0.878  |
| 39    | 0.888  |
| 49    | 0.904  |
| 59    | 0.869  |
| 69    | 0.879  |
| 79    | 0.872  |
| 89    | 0.892  |
| 99    | 0.892  |
| 109   | 0.874  |
| 119   | 0.889  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Put alphabet soup & tomato sauce in basket (LR2)       | 0.960 | 0.960 | 0.960 | 0.942 | 0.859 | 0.897 | 0.939 | 0.960 | 0.939 | 0.941  | 0.899  |
| 2. Put cream cheese & butter in basket (LR2)              | 0.981 | 0.920 | 0.981 | 0.979 | 0.958 | 0.958 | 0.899 | 0.918 | 0.918 | 0.979  | 0.939  |
| 3. Turn on stove & put moka pot (K3)                      | 0.979 | 0.981 | 0.960 | 0.938 | 0.958 | 0.920 | 0.896 | 0.859 | 1.000 | 0.880  | 0.958  |
| 4. Put black bowl in drawer & close (K4)                  | 0.979 | 0.939 | 0.960 | 1.000 | 0.962 | 0.981 | 1.000 | 1.000 | 0.939 | 1.000  | 0.981  |
| 5. Put white mug on left plate & yellow mug on right (LR5)| 0.317 | 0.824 | 0.819 | 0.881 | 0.861 | 0.838 | 0.819 | 0.857 | 0.780 | 0.864  | 0.920  |
| 6. Pick up book & place in caddy (S1)                     | 0.918 | 0.840 | 0.899 | 0.918 | 0.816 | 0.877 | 0.897 | 0.878 | 0.899 | 0.857  | 0.897  |
| 7. Put white mug on plate & chocolate pudding right (LR6) | 0.740 | 0.697 | 0.861 | 0.819 | 0.774 | 0.897 | 0.835 | 0.878 | 0.941 | 0.941  | 0.798  |
| 8. Put alphabet soup & cream cheese in basket (LR1)       | 1.000 | 1.000 | 0.981 | 0.941 | 0.942 | 0.921 | 0.958 | 0.979 | 0.941 | 0.902  | 0.979  |
| 9. Put both moka pots on stove (K8)                       | 0.760 | 0.739 | 0.561 | 0.776 | 0.700 | 0.622 | 0.737 | 0.777 | 0.659 | 0.559  | 0.679  |
| 10. Put yellow & white mug in microwave & close (K6)      | 0.960 | 0.880 | 0.901 | 0.841 | 0.861 | 0.881 | 0.742 | 0.816 | 0.899 | 0.820  | 0.840  |
| **Average**                                                | **0.859** | **0.878** | **0.888** | **0.904** | **0.869** | **0.879** | **0.872** | **0.892** | **0.892** | **0.874** | **0.889** |

---

## iMF LIBERO-10 — From-scratch ablation: heads (Job 40945552)

*From-scratch (no pretrained checkpoint) iMF on LIBERO-10 with `model.imf_head_depth=12` (heads-swap, default ratio 0.25). 50 n_eval. Last evaluation at epoch 119.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.833  |
| 29    | 0.841  |
| 39    | 0.894  |
| 49    | 0.856  |
| 59    | 0.892  |
| 69    | 0.864  |
| 79    | 0.884  |
| 89    | 0.879  |
| 99    | 0.881  |
| 109   | 0.890  |
| 119   | 0.877  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Put alphabet soup & tomato sauce in basket (LR2)       | 0.942 | 0.920 | 0.917 | 0.958 | 0.918 | 0.960 | 0.920 | 0.958 | 0.960 | 0.920  | 0.960  |
| 2. Put cream cheese & butter in basket (LR2)              | 0.939 | 0.958 | 0.920 | 0.883 | 0.960 | 0.902 | 0.921 | 0.960 | 0.918 | 0.960  | 0.921  |
| 3. Turn on stove & put moka pot (K3)                      | 0.939 | 0.939 | 1.000 | 0.960 | 0.981 | 0.939 | 0.857 | 0.899 | 0.939 | 0.962  | 0.979  |
| 4. Put black bowl in drawer & close (K4)                  | 0.958 | 0.941 | 0.942 | 0.941 | 0.939 | 0.938 | 0.960 | 0.960 | 0.981 | 0.920  | 0.979  |
| 5. Put white mug on left plate & yellow mug on right (LR5)| 0.659 | 0.881 | 0.901 | 0.859 | 0.862 | 0.901 | 0.939 | 0.881 | 0.840 | 0.960  | 0.901  |
| 6. Pick up book & place in caddy (S1)                     | 0.897 | 0.897 | 0.899 | 0.857 | 0.918 | 0.840 | 0.918 | 0.899 | 0.880 | 0.897  | 0.878  |
| 7. Put white mug on plate & chocolate pudding right (LR6) | 0.756 | 0.897 | 0.939 | 0.796 | 0.901 | 0.817 | 0.941 | 0.837 | 0.857 | 0.800  | 0.918  |
| 8. Put alphabet soup & cream cheese in basket (LR1)       | 0.979 | 0.981 | 0.962 | 0.941 | 0.981 | 0.883 | 0.923 | 0.901 | 0.901 | 0.921  | 0.901  |
| 9. Put both moka pots on stove (K8)                       | 0.697 | 0.482 | 0.761 | 0.700 | 0.696 | 0.683 | 0.720 | 0.740 | 0.777 | 0.700  | 0.662  |
| 10. Put yellow & white mug in microwave & close (K6)      | 0.563 | 0.516 | 0.699 | 0.660 | 0.764 | 0.782 | 0.739 | 0.760 | 0.761 | 0.861  | 0.675  |
| **Average**                                                | **0.833** | **0.841** | **0.894** | **0.856** | **0.892** | **0.864** | **0.884** | **0.879** | **0.881** | **0.890** | **0.877** |

---

## iMF LIBERO-10 — From-scratch ablation: ratio (Job 40945549)

*From-scratch (no pretrained checkpoint) iMF on LIBERO-10 with `model.ratio=0.5` (default `imf_head_depth=8`). 50 n_eval. Last evaluation at epoch 119.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.867  |
| 29    | 0.838  |
| 39    | 0.853  |
| 49    | 0.842  |
| 59    | 0.870  |
| 69    | 0.829  |
| 79    | 0.877  |
| 89    | 0.912  |
| 99    | 0.901  |
| 109   | 0.933  |
| 119   | 0.897  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Put alphabet soup & tomato sauce in basket (LR2)       | 1.000 | 0.920 | 0.960 | 0.838 | 0.899 | 0.841 | 0.920 | 0.960 | 0.958 | 0.958  | 0.981  |
| 2. Put cream cheese & butter in basket (LR2)              | 0.960 | 0.881 | 0.921 | 0.941 | 0.979 | 0.899 | 0.960 | 0.981 | 1.000 | 1.000  | 0.981  |
| 3. Turn on stove & put moka pot (K3)                      | 0.960 | 0.897 | 0.899 | 0.877 | 0.824 | 0.763 | 0.857 | 0.981 | 0.899 | 1.000  | 0.817  |
| 4. Put black bowl in drawer & close (K4)                  | 0.859 | 1.000 | 0.960 | 0.981 | 0.923 | 1.000 | 0.979 | 0.960 | 0.981 | 1.000  | 0.960  |
| 5. Put white mug on left plate & yellow mug on right (LR5)| 0.756 | 0.779 | 0.897 | 0.821 | 0.901 | 0.801 | 0.859 | 0.859 | 0.880 | 0.841  | 0.859  |
| 6. Pick up book & place in caddy (S1)                     | 0.819 | 0.878 | 0.837 | 0.897 | 0.838 | 0.897 | 0.897 | 0.878 | 0.918 | 0.877  | 0.918  |
| 7. Put white mug on plate & chocolate pudding right (LR6) | 0.840 | 0.681 | 0.718 | 0.700 | 0.740 | 0.678 | 0.796 | 0.899 | 0.838 | 0.880  | 0.776  |
| 8. Put alphabet soup & cream cheese in basket (LR1)       | 0.979 | 0.981 | 0.918 | 0.981 | 0.979 | 0.981 | 0.979 | 0.962 | 0.921 | 0.981  | 0.942  |
| 9. Put both moka pots on stove (K8)                       | 0.796 | 0.598 | 0.620 | 0.684 | 0.716 | 0.736 | 0.660 | 0.840 | 0.817 | 0.897  | 0.878  |
| 10. Put yellow & white mug in microwave & close (K6)      | 0.700 | 0.763 | 0.795 | 0.700 | 0.899 | 0.696 | 0.859 | 0.803 | 0.800 | 0.899  | 0.857  |
| **Average**                                                | **0.867** | **0.838** | **0.853** | **0.842** | **0.870** | **0.829** | **0.877** | **0.912** | **0.901** | **0.933** | **0.897** |

---

## iMF LIBERO-10 — From-scratch ablation: both (Job 40945553)

*From-scratch (no pretrained checkpoint) iMF on LIBERO-10 with `model.ratio=0.5` and `model.imf_head_depth=12`. 50 n_eval. Last evaluation at epoch 119.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.898  |
| 29    | 0.866  |
| 39    | 0.905  |
| 49    | 0.889  |
| 59    | 0.861  |
| 69    | 0.884  |
| 79    | 0.896  |
| 89    | 0.885  |
| 99    | 0.903  |
| 109   | 0.916  |
| 119   | 0.899  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Put alphabet soup & tomato sauce in basket (LR2)       | 1.000 | 0.938 | 1.000 | 0.938 | 0.918 | 0.939 | 0.958 | 0.941 | 1.000 | 1.000  | 1.000  |
| 2. Put cream cheese & butter in basket (LR2)              | 0.979 | 1.000 | 0.918 | 0.938 | 0.938 | 0.896 | 0.960 | 0.920 | 0.899 | 1.000  | 0.918  |
| 3. Turn on stove & put moka pot (K3)                      | 0.981 | 0.981 | 1.000 | 0.979 | 0.981 | 1.000 | 0.939 | 0.939 | 0.958 | 1.000  | 0.917  |
| 4. Put black bowl in drawer & close (K4)                  | 0.981 | 0.920 | 0.960 | 1.000 | 0.941 | 0.920 | 0.918 | 0.899 | 0.979 | 0.958  | 0.979  |
| 5. Put white mug on left plate & yellow mug on right (LR5)| 0.901 | 0.901 | 0.880 | 0.939 | 0.899 | 0.899 | 0.941 | 0.939 | 0.920 | 0.902  | 0.962  |
| 6. Pick up book & place in caddy (S1)                     | 0.856 | 0.857 | 0.899 | 0.857 | 0.777 | 0.897 | 0.857 | 0.897 | 0.918 | 0.856  | 0.878  |
| 7. Put white mug on plate & chocolate pudding right (LR6) | 0.841 | 0.798 | 0.877 | 0.798 | 0.958 | 0.901 | 0.819 | 0.822 | 0.861 | 0.901  | 0.901  |
| 8. Put alphabet soup & cream cheese in basket (LR1)       | 1.000 | 0.981 | 1.000 | 0.941 | 0.941 | 0.941 | 0.942 | 0.920 | 0.958 | 1.000  | 0.958  |
| 9. Put both moka pots on stove (K8)                       | 0.579 | 0.458 | 0.699 | 0.679 | 0.421 | 0.582 | 0.721 | 0.676 | 0.758 | 0.638  | 0.639  |
| 10. Put yellow & white mug in microwave & close (K6)      | 0.861 | 0.822 | 0.822 | 0.822 | 0.840 | 0.862 | 0.899 | 0.901 | 0.779 | 0.904  | 0.840  |
| **Average**                                                | **0.898** | **0.866** | **0.905** | **0.889** | **0.861** | **0.884** | **0.896** | **0.885** | **0.903** | **0.916** | **0.899** |

---

## Flower LIBERO-10 — Ablation: 1 NFE (Job 41312388)

*Ablation of Flower LIBERO-10 baseline (Job 41312374) with sampling reduced to 1 NFE (number of function evaluations). 50 n_eval. Retrained with EMA off (callbacks.ema.evaluate_ema_weights_instead=False). Last evaluation at epoch 119.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.004  |
| 29    | 0.180  |
| 39    | 0.350  |
| 49    | 0.109  |
| 59    | 0.514  |
| 69    | 0.408  |
| 79    | 0.675  |
| 89    | 0.845  |
| 99    | 0.617  |
| 109   | 0.884  |
| 119   | 0.846  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Put alphabet soup & tomato sauce in basket (LR2)       | 0.000 | 0.000 | 0.220 | 0.000 | 0.542 | 0.061 | 0.720 | 0.918 | 0.518 | 0.857  | 0.939  |
| 2. Put cream cheese & butter in basket (LR2)              | 0.000 | 0.160 | 0.279 | 0.321 | 0.718 | 0.563 | 0.779 | 0.921 | 0.800 | 0.962  | 0.962  |
| 3. Turn on stove & put moka pot (K3)                      | 0.000 | 0.359 | 0.821 | 0.396 | 0.338 | 0.721 | 0.979 | 0.921 | 0.679 | 0.979  | 1.000  |
| 4. Put black bowl in drawer & close (K4)                  | 0.000 | 0.223 | 0.503 | 0.099 | 0.881 | 0.141 | 0.838 | 1.000 | 0.904 | 1.000  | 0.941  |
| 5. Put white mug on left plate & yellow mug on right (LR5)| 0.000 | 0.418 | 0.381 | 0.059 | 0.421 | 0.583 | 0.558 | 0.843 | 0.700 | 0.939  | 0.859  |
| 6. Pick up book & place in caddy (S1)                     | 0.021 | 0.199 | 0.237 | 0.040 | 0.500 | 0.551 | 0.817 | 0.639 | 0.261 | 0.877  | 0.817  |
| 7. Put white mug on plate & chocolate pudding right (LR6) | 0.000 | 0.021 | 0.301 | 0.000 | 0.404 | 0.120 | 0.340 | 0.902 | 0.215 | 0.704  | 0.599  |
| 8. Put alphabet soup & cream cheese in basket (LR1)       | 0.000 | 0.240 | 0.662 | 0.019 | 0.540 | 0.457 | 0.628 | 0.921 | 0.859 | 0.921  | 0.920  |
| 9. Put both moka pots on stove (K8)                       | 0.000 | 0.000 | 0.040 | 0.000 | 0.296 | 0.244 | 0.399 | 0.679 | 0.441 | 0.598  | 0.540  |
| 10. Put yellow & white mug in microwave & close (K6)      | 0.019 | 0.181 | 0.059 | 0.160 | 0.498 | 0.638 | 0.696 | 0.700 | 0.798 | 1.000  | 0.883  |
| **Average**                                                | **0.004** | **0.180** | **0.350** | **0.109** | **0.514** | **0.408** | **0.675** | **0.845** | **0.617** | **0.884** | **0.846** |

---

## Flower LIBERO-10 — Ablation: 2 NFE (Job 41312389)

*Ablation of Flower LIBERO-10 baseline (Job 41312374) with sampling reduced to 2 NFE. 50 n_eval. Retrained with EMA off (callbacks.ema.evaluate_ema_weights_instead=False). Last evaluation at epoch 119.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.310  |
| 29    | 0.391  |
| 39    | 0.686  |
| 49    | 0.737  |
| 59    | 0.713  |
| 69    | 0.729  |
| 79    | 0.724  |
| 89    | 0.868  |
| 99    | 0.855  |
| 109   | 0.878  |
| 119   | 0.900  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Put alphabet soup & tomato sauce in basket (LR2)       | 0.558 | 0.220 | 0.500 | 0.897 | 0.841 | 0.518 | 0.460 | 0.859 | 0.899 | 0.917  | 0.921  |
| 2. Put cream cheese & butter in basket (LR2)              | 0.058 | 0.881 | 0.901 | 0.880 | 0.596 | 0.939 | 0.960 | 0.979 | 0.901 | 0.883  | 0.941  |
| 3. Turn on stove & put moka pot (K3)                      | 0.476 | 0.157 | 0.739 | 0.942 | 0.960 | 0.764 | 0.718 | 1.000 | 0.880 | 0.979  | 0.960  |
| 4. Put black bowl in drawer & close (K4)                  | 0.340 | 0.295 | 0.979 | 0.880 | 0.880 | 0.897 | 0.960 | 1.000 | 1.000 | 0.960  | 1.000  |
| 5. Put white mug on left plate & yellow mug on right (LR5)| 0.179 | 0.522 | 0.718 | 0.859 | 0.736 | 0.579 | 0.861 | 0.941 | 1.000 | 0.960  | 0.979  |
| 6. Pick up book & place in caddy (S1)                     | 0.798 | 0.160 | 0.694 | 0.779 | 0.817 | 0.679 | 0.756 | 0.918 | 0.878 | 0.918  | 0.899  |
| 7. Put white mug on plate & chocolate pudding right (LR6) | 0.000 | 0.596 | 0.362 | 0.337 | 0.660 | 0.777 | 0.780 | 0.361 | 0.558 | 0.704  | 0.739  |
| 8. Put alphabet soup & cream cheese in basket (LR1)       | 0.319 | 0.861 | 0.804 | 0.960 | 0.838 | 0.861 | 0.704 | 0.921 | 0.962 | 0.941  | 0.865  |
| 9. Put both moka pots on stove (K8)                       | 0.061 | 0.000 | 0.322 | 0.202 | 0.301 | 0.458 | 0.361 | 0.803 | 0.601 | 0.639  | 0.819  |
| 10. Put yellow & white mug in microwave & close (K6)      | 0.316 | 0.218 | 0.838 | 0.636 | 0.498 | 0.819 | 0.679 | 0.899 | 0.877 | 0.878  | 0.877  |
| **Average**                                                | **0.310** | **0.391** | **0.686** | **0.737** | **0.713** | **0.729** | **0.724** | **0.868** | **0.855** | **0.878** | **0.900** |

---

## Flower LIBERO-10 — Ablation: 3 NFE (Job 41312390)

*Ablation of Flower LIBERO-10 baseline (Job 41312374) with sampling reduced to 3 NFE. 50 n_eval. Retrained with EMA off (callbacks.ema.evaluate_ema_weights_instead=False). Last evaluation at epoch 119.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.572  |
| 29    | 0.719  |
| 39    | 0.591  |
| 49    | 0.569  |
| 59    | 0.772  |
| 69    | 0.810  |
| 79    | 0.753  |
| 89    | 0.834  |
| 99    | 0.861  |
| 109   | 0.867  |
| 119   | 0.872  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Put alphabet soup & tomato sauce in basket (LR2)       | 0.841 | 0.671 | 0.337 | 0.502 | 0.420 | 0.862 | 0.537 | 0.796 | 0.816 | 0.880  | 0.941  |
| 2. Put cream cheese & butter in basket (LR2)              | 0.237 | 0.958 | 0.800 | 0.676 | 0.901 | 0.878 | 0.958 | 0.859 | 0.958 | 0.880  | 0.902  |
| 3. Turn on stove & put moka pot (K3)                      | 0.817 | 0.696 | 0.901 | 0.760 | 0.962 | 1.000 | 0.918 | 1.000 | 0.939 | 0.918  | 0.941  |
| 4. Put black bowl in drawer & close (K4)                  | 0.599 | 0.901 | 0.843 | 0.941 | 0.901 | 1.000 | 0.981 | 1.000 | 0.979 | 1.000  | 0.979  |
| 5. Put white mug on left plate & yellow mug on right (LR5)| 0.458 | 0.880 | 0.737 | 0.160 | 0.683 | 0.782 | 0.662 | 0.897 | 0.962 | 0.857  | 0.941  |
| 6. Pick up book & place in caddy (S1)                     | 0.800 | 0.780 | 0.401 | 0.777 | 0.700 | 0.718 | 0.659 | 0.877 | 0.857 | 0.918  | 0.838  |
| 7. Put white mug on plate & chocolate pudding right (LR6) | 0.141 | 0.601 | 0.244 | 0.298 | 0.861 | 0.413 | 0.441 | 0.702 | 0.720 | 0.798  | 0.702  |
| 8. Put alphabet soup & cream cheese in basket (LR1)       | 0.800 | 0.841 | 0.821 | 0.880 | 0.958 | 0.822 | 0.979 | 0.979 | 0.958 | 0.962  | 0.979  |
| 9. Put both moka pots on stove (K8)                       | 0.468 | 0.357 | 0.197 | 0.361 | 0.519 | 0.740 | 0.558 | 0.357 | 0.585 | 0.619  | 0.620  |
| 10. Put yellow & white mug in microwave & close (K6)      | 0.558 | 0.502 | 0.635 | 0.340 | 0.819 | 0.880 | 0.841 | 0.877 | 0.840 | 0.840  | 0.881  |
| **Average**                                                | **0.572** | **0.719** | **0.591** | **0.569** | **0.772** | **0.810** | **0.753** | **0.834** | **0.861** | **0.867** | **0.872** |

---

## Flower LIBERO-Object (Job 41312370)

*Flower (RF baseline) finetuning on LIBERO-Object. 50 n_eval. Retrained with EMA off (callbacks.ema.evaluate_ema_weights_instead=False). Last evaluation at epoch 119.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.952  |
| 29    | 0.922  |
| 39    | 0.976  |
| 49    | 0.975  |
| 59    | 0.971  |
| 69    | 0.972  |
| 79    | 0.968  |
| 89    | 0.976  |
| 99    | 0.988  |
| 109   | 0.976  |
| 119   | 0.976  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Pick up alphabet soup      | 0.981 | 0.960 | 0.979 | 0.962 | 1.000 | 1.000 | 0.979 | 1.000 | 1.000 | 1.000  | 1.000  |
| 2. Pick up cream cheese       | 0.979 | 0.639 | 1.000 | 0.960 | 0.960 | 0.941 | 0.962 | 0.881 | 0.981 | 0.962  | 0.941  |
| 3. Pick up salad dressing     | 1.000 | 0.902 | 1.000 | 0.962 | 0.921 | 0.960 | 0.981 | 1.000 | 0.918 | 1.000  | 1.000  |
| 4. Pick up BBQ sauce          | 0.981 | 1.000 | 0.843 | 0.981 | 1.000 | 0.942 | 0.938 | 1.000 | 0.981 | 1.000  | 0.979  |
| 5. Pick up ketchup            | 0.897 | 0.800 | 1.000 | 0.981 | 0.942 | 0.981 | 0.902 | 0.960 | 1.000 | 1.000  | 0.960  |
| 6. Pick up tomato sauce       | 0.780 | 1.000 | 0.979 | 0.921 | 0.960 | 1.000 | 0.960 | 1.000 | 1.000 | 0.960  | 0.958  |
| 7. Pick up butter             | 1.000 | 1.000 | 0.962 | 1.000 | 1.000 | 1.000 | 1.000 | 0.981 | 1.000 | 1.000  | 1.000  |
| 8. Pick up milk               | 0.941 | 0.962 | 1.000 | 0.981 | 0.962 | 0.920 | 0.981 | 0.962 | 1.000 | 0.861  | 0.939  |
| 9. Pick up chocolate pudding  | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000  | 1.000  |
| 10. Pick up orange juice      | 0.979 | 0.962 | 1.000 | 1.000 | 0.960 | 0.981 | 0.979 | 0.979 | 1.000 | 0.979  | 0.981  |
| **Average**                   | **0.952** | **0.922** | **0.976** | **0.975** | **0.971** | **0.972** | **0.968** | **0.976** | **0.988** | **0.976** | **0.976** |

---

## Flower LIBERO-10 (Job 41312374)

*Flower (RF baseline) finetuning on LIBERO-10. 50 n_eval. Retrained with EMA off (callbacks.ema.evaluate_ema_weights_instead=False). Last evaluation at epoch 119.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.681  |
| 29    | 0.711  |
| 39    | 0.707  |
| 49    | 0.792  |
| 59    | 0.840  |
| 69    | 0.714  |
| 79    | 0.895  |
| 89    | 0.849  |
| 99    | 0.850  |
| 109   | 0.904  |
| 119   | 0.855  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Put alphabet soup & tomato sauce in basket (LR2)       | 0.819 | 0.878 | 0.676 | 0.718 | 0.880 | 0.801 | 0.857 | 0.901 | 0.713 | 0.960  | 0.897  |
| 2. Put cream cheese & butter in basket (LR2)              | 0.918 | 0.678 | 0.660 | 0.962 | 0.881 | 0.920 | 0.938 | 1.000 | 0.960 | 0.920  | 0.918  |
| 3. Turn on stove & put moka pot (K3)                      | 0.981 | 0.840 | 0.939 | 1.000 | 0.979 | 0.981 | 0.979 | 0.981 | 0.979 | 0.981  | 0.981  |
| 4. Put black bowl in drawer & close (K4)                  | 0.758 | 0.979 | 0.939 | 0.979 | 1.000 | 0.780 | 0.962 | 0.918 | 1.000 | 0.939  | 1.000  |
| 5. Put white mug on left plate & yellow mug on right (LR5)| 0.280 | 0.623 | 0.660 | 0.780 | 0.819 | 0.535 | 0.918 | 0.920 | 0.958 | 0.979  | 0.942  |
| 6. Pick up book & place in caddy (S1)                     | 0.756 | 0.596 | 0.513 | 0.819 | 0.878 | 0.521 | 0.838 | 0.819 | 0.878 | 0.878  | 0.838  |
| 7. Put white mug on plate & chocolate pudding right (LR6) | 0.442 | 0.877 | 0.362 | 0.760 | 0.779 | 0.579 | 0.821 | 0.840 | 0.657 | 0.801  | 0.659  |
| 8. Put alphabet soup & cream cheese in basket (LR1)       | 0.760 | 0.622 | 0.960 | 0.801 | 0.883 | 0.881 | 0.962 | 0.819 | 0.880 | 0.902  | 0.962  |
| 9. Put both moka pots on stove (K8)                       | 0.417 | 0.340 | 0.678 | 0.559 | 0.603 | 0.357 | 0.800 | 0.441 | 0.577 | 0.758  | 0.540  |
| 10. Put yellow & white mug in microwave & close (K6)      | 0.675 | 0.679 | 0.679 | 0.540 | 0.694 | 0.780 | 0.880 | 0.856 | 0.899 | 0.917  | 0.816  |
| **Average**                                                | **0.681** | **0.711** | **0.707** | **0.792** | **0.840** | **0.714** | **0.895** | **0.849** | **0.850** | **0.904** | **0.855** |

---

## Flower LIBERO-Spatial (Job 41312369)

*Flower (RF baseline) finetuning on LIBERO-Spatial. 50 n_eval. Retrained with EMA off (callbacks.ema.evaluate_ema_weights_instead=False). Last evaluation at epoch 119.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.924  |
| 29    | 0.924  |
| 39    | 0.946  |
| 49    | 0.926  |
| 59    | 0.956  |
| 69    | 0.968  |
| 79    | 0.930  |
| 89    | 0.966  |
| 99    | 0.968  |
| 109   | 0.980  |
| 119   | 0.966  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Bowl between plate & ramekin    | 0.958 | 0.981 | 0.981 | 0.979 | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000  | 1.000  |
| 2. Bowl next to ramekin            | 1.000 | 0.979 | 0.958 | 0.803 | 0.979 | 1.000 | 1.000 | 0.981 | 0.979 | 1.000  | 0.979  |
| 3. Bowl from table center          | 0.941 | 0.901 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.981 | 1.000 | 0.979  | 0.979  |
| 4. Bowl on cookie box              | 0.939 | 0.962 | 0.938 | 0.941 | 0.939 | 0.981 | 0.960 | 1.000 | 0.962 | 1.000  | 1.000  |
| 5. Bowl in top drawer              | 0.840 | 0.921 | 0.861 | 0.941 | 0.920 | 0.962 | 0.583 | 0.979 | 0.939 | 0.979  | 0.921  |
| 6. Bowl on ramekin                 | 0.920 | 0.756 | 0.881 | 0.880 | 0.938 | 0.881 | 0.902 | 0.960 | 0.941 | 0.962  | 0.942  |
| 7. Bowl next to cookie box         | 0.960 | 0.979 | 1.000 | 1.000 | 1.000 | 0.981 | 1.000 | 1.000 | 1.000 | 0.979  | 0.979  |
| 8. Bowl on stove                   | 1.000 | 0.981 | 1.000 | 0.958 | 0.979 | 1.000 | 0.979 | 0.958 | 0.960 | 0.981  | 0.981  |
| 9. Bowl next to plate              | 0.920 | 0.902 | 0.902 | 0.819 | 0.920 | 0.941 | 0.897 | 0.920 | 0.962 | 0.941  | 0.941  |
| 10. Bowl on wooden cabinet         | 0.758 | 0.881 | 0.941 | 0.942 | 0.902 | 0.938 | 0.979 | 0.881 | 0.941 | 0.979  | 0.941  |
| **Average**                        | **0.924** | **0.924** | **0.946** | **0.926** | **0.956** | **0.968** | **0.930** | **0.966** | **0.968** | **0.980** | **0.966** |

---

## Flower LIBERO-Goal (Job 41312371)

*Flower (RF baseline) finetuning on LIBERO-Goal. 50 n_eval. Retrained with EMA off (callbacks.ema.evaluate_ema_weights_instead=False). Last evaluation at epoch 119.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.918  |
| 29    | 0.880  |
| 39    | 0.921  |
| 49    | 0.912  |
| 59    | 0.936  |
| 69    | 0.919  |
| 79    | 0.942  |
| 89    | 0.937  |
| 99    | 0.960  |
| 109   | 0.972  |
| 119   | 0.958  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 | Ep 99 | Ep 109 | Ep 119 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| 1. Open middle drawer of cabinet    | 1.000 | 1.000 | 0.920 | 1.000 | 1.000 | 0.979 | 0.960 | 0.939 | 0.960 | 0.979  | 0.981  |
| 2. Put bowl on stove                | 0.962 | 0.981 | 1.000 | 0.979 | 0.979 | 0.979 | 0.960 | 0.960 | 1.000 | 1.000  | 1.000  |
| 3. Put wine bottle on top of cabinet| 0.878 | 0.941 | 0.979 | 0.920 | 0.881 | 0.981 | 0.939 | 0.921 | 0.981 | 0.941  | 0.979  |
| 4. Open top drawer & put bowl inside| 0.840 | 0.438 | 0.683 | 0.606 | 0.763 | 0.744 | 0.782 | 0.801 | 0.881 | 0.941  | 0.798  |
| 5. Put bowl on top of cabinet       | 1.000 | 0.979 | 1.000 | 1.000 | 0.938 | 0.981 | 1.000 | 1.000 | 0.960 | 0.979  | 1.000  |
| 6. Push plate to front of stove     | 1.000 | 0.861 | 0.920 | 0.897 | 0.981 | 0.721 | 0.960 | 0.960 | 0.981 | 0.979  | 0.920  |
| 7. Put cream cheese in bowl         | 0.841 | 0.760 | 0.902 | 0.821 | 0.920 | 0.901 | 0.979 | 0.864 | 0.921 | 0.981  | 0.960  |
| 8. Turn on stove                    | 0.979 | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000  | 1.000  |
| 9. Put bowl on plate                | 0.962 | 0.920 | 0.880 | 0.981 | 0.962 | 0.941 | 0.880 | 0.941 | 0.960 | 0.941  | 0.962  |
| 10. Put wine bottle on rack         | 0.718 | 0.938 | 0.923 | 0.920 | 0.941 | 0.960 | 0.960 | 0.979 | 0.960 | 0.981  | 0.979  |
| **Average**                         | **0.918** | **0.880** | **0.921** | **0.912** | **0.936** | **0.919** | **0.942** | **0.937** | **0.960** | **0.972** | **0.958** |

---

## iMF LIBERO-Spatial (Job 39763601)

*Trained on libero_10 data, but evaluated on libero_spatial tasks (default `libero_benchmark: libero_spatial` in config was not overridden). 10 n_eval. Job killed during epoch 96. Last evaluation at epoch 89.*

### Average Success Rate per Epoch

| Epoch | Avg SR |
|-------|--------|
| 19    | 0.917  |
| 29    | 0.929  |
| 39    | 0.913  |
| 49    | 0.954  |
| 59    | 0.942  |
| 69    | 0.892  |
| 79    | 0.938  |
| 89    | 0.992  |

### Per-Task Success Rate

| Task | Ep 19 | Ep 29 | Ep 39 | Ep 49 | Ep 59 | Ep 69 | Ep 79 | Ep 89 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1. Bowl between plate & ramekin    | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 | 0.833 | 0.917 | 1.000 |
| 2. Bowl next to ramekin            | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3. Bowl from table center          | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4. Bowl on cookie box              | 0.833 | 0.792 | 0.875 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5. Bowl in top drawer              | 1.000 | 0.917 | 0.792 | 1.000 | 1.000 | 0.792 | 0.917 | 1.000 |
| 6. Bowl on ramekin                 | 0.708 | 0.792 | 0.667 | 0.833 | 0.625 | 0.792 | 0.792 | 1.000 |
| 7. Bowl next to cookie box         | 1.000 | 0.917 | 1.000 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8. Bowl on stove                   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 9. Bowl next to plate              | 0.917 | 1.000 | 0.917 | 1.000 | 0.917 | 1.000 | 1.000 | 1.000 |
| 10. Bowl on wooden cabinet         | 0.875 | 0.875 | 0.875 | 1.000 | 0.875 | 0.500 | 0.750 | 0.917 |
| **Average**                        | **0.917** | **0.929** | **0.913** | **0.954** | **0.942** | **0.892** | **0.938** | **0.992** |
