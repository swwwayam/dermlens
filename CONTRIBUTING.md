# Contributing to DermLens

DermLens is a research prototype. Contributions should keep the research and clinical-safety boundaries clear.

## Before opening a change

1. Create a focused branch from `main`.
2. Do not commit HAM10000 images, embeddings, FAISS indexes, uploads, or generated reports.
3. Run the metric tests:

   ```powershell
   python -m unittest discover -s backend/evaluation/tests -p "test_*.py"
   ```

4. Build the React frontend:

   ```powershell
   cd frontend/web
   npm ci
   npm run build
   ```

Keep pull requests small, describe how the change was tested, and call out any changes to evaluation methodology.
