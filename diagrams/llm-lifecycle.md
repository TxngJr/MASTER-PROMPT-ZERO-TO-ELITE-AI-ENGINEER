# LLM Lifecycle

~~~mermaid
flowchart TD
    A[Raw data] --> B[Cleaning / dedup / split]
    B --> C[Tokenizer trained on train split]
    C --> D[Causal packing]
    D --> E[Random-init decoder]
    E --> F[Pretraining]
    F --> G[Validation / checkpoint]
    G --> H[SFT]
    H --> I[Preference optimization]
    I --> J[Evaluation / safety]
    J --> K[Quantization / inference optimization]
    K --> L[Serving / deployment]
    L --> M[Monitoring / research iteration]
~~~
