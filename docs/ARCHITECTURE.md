# DermLens architecture

DermLens has two runtime layers:

- The React/Vite client in `frontend/web` provides the image-analysis, metrics, and methodology views.
- The FastAPI service in `backend` exposes retrieval and evaluation endpoints and reads locally generated research artifacts.

The research pipeline in `src` prepares lesion-grouped splits, creates image embeddings, builds the FAISS index, retrieves nearest examples, and applies metadata-aware and rarity-aware ranking. The browser communicates with the API through the Vite `/api` proxy during development.

The dataset and generated artifacts are intentionally kept outside version control. A fresh checkout therefore contains the application code and reproducible scripts, but requires a local HAM10000 setup before analysis endpoints can return results.
