# Batch 05 Integration Project — Geometry & Representation Lab

รวม Chapters 13–15:

```text
high-dimensional data
      ↓
StandardScaler
      ├────────────→ SVM classification
      │
      └→ PCA
          ├────────→ SVM classification
          ├────────→ K-Means clustering
          └────────→ DBSCAN clustering
```

## Goal

ตอบคำถาม 3 แบบโดยไม่สับสนกัน:

1. **Supervised:** margin classifier แยก labels ได้ดีแค่ไหน?
2. **Unsupervised:** geometry มี cluster structure แบบใด?
3. **Representation:** PCA ลด dimensions แล้วเก็บข้อมูลที่มีประโยชน์แค่ไหน?

## Run

```bash
source .venv/bin/activate
python -m pip install -r requirements-batch05.txt

python integration-project-batch05/src/geometry_lab.py \
  --output-dir reports/batch05 \
  --seed 42
```

## Outputs

JSON report:

- raw feature count
- PCA feature count
- PCA explained variance
- SVM validation/test metrics
- K-Means silhouette
- K-Means ARI diagnostic
- DBSCAN cluster/noise counts

## Critical interpretation rule

ARI ใช้ true labels เป็น **diagnostic หลัง clustering** เท่านั้นใน synthetic lab นี้

ห้ามใช้ labels เพื่อ tune clustering แล้วอ้างว่าเป็น unsupervised discovery

## Required Extensions

1. RBF SVC
2. PCA k sweep
3. K-Means K sweep
4. DBSCAN eps sweep
5. t-SNE visualization
6. optional UMAP visualization
7. repeated seeds
8. runtime comparison
9. neighborhood preservation metric
10. real-world dataset

## Mastery Questions

- PCA fit จาก train หรือ full data?
- ทำไม SVM ต้อง scale แต่ tree ไม่จำเป็นแบบเดียวกัน?
- silhouette สูงแปลว่ามี real-world classes จริงไหม?
- PCA 95% variance แปลว่าเก็บ 95% class informationไหม?
- t-SNE/UMAP islands พิสูจน์ cluster จริงหรือไม่?
