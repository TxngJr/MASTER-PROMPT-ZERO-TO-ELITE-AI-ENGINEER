# Mini Project — Training Data Lake Pipeline

Build a pipeline that:
1. ingests raw event JSON
2. validates a contract
3. deduplicates by event ID/version
4. separates late events using a watermark
5. writes date/region partitions
6. compacts small files
7. creates a training snapshot manifest
8. records lineage and transform version
9. performs a point-in-time leakage audit
10. emits quality metrics for MLOps monitoring

Optional: write/read actual Parquet with PyArrow.
