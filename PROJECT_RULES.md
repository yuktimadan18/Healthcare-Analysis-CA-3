# Project Rules: CDC BRFSS 2024 Healthcare Analytics

## Project objective
Build an explainable, reproducible healthcare analytics framework using the CDC BRFSS 2024 combined landline and cellular survey dataset.

## Development rules
1. Inspect the existing repository before creating or modifying files.
2. Do not recreate the virtual environment or reinstall dependencies without explicit approval.
3. Use the project interpreter: .venv\Scripts\python.exe.
4. Do not change the established folder structure without approval.
5. Do not generate fabricated metrics, results, citations, or dataset facts.
6. Do not select the predictive target until the actual dataset and official codebook have been inspected.
7. Preserve the original ZIP archive in data/raw.
8. Never commit the raw dataset or other large/private data files to Git.
9. Keep data ingestion, preprocessing, analysis, modeling, evaluation, and explainability modular.
10. Make all preprocessing and model evaluation reproducible.
11. Prevent data leakage and use a justified validation strategy.
12. Treat BRFSS as survey data. Investigate relevant CDC documentation on weighting and survey design before making population-level claims.
13. Distinguish association from causation. Do not present predictions as clinical diagnoses.
14. Document important design decisions and explain the purpose of every notebook and script.
15. Run appropriate tests and report failures honestly.

## Required workflow
Milestone 1: Verify project structure, interpreter, dependencies, dataset archive, and documentation.
Milestone 2: Inspect the official CDC codebook and dataset metadata.
Milestone 3: Perform data quality assessment and exploratory data analysis.
Milestone 4: Conduct statistical analysis and knowledge discovery.
Milestone 5: Select and justify an appropriate prediction target based on evidence.
Milestone 6: Train and compare a baseline model and an improved model where justified.
Milestone 7: Evaluate results and add explainability.
Milestone 8: Develop the interactive dashboard.
Milestone 9: Complete documentation, report, reproducibility checks, and GitHub preparation.

## Agent behavior
Work on one milestone at a time. Before implementation, explain the proposed files and actions. Do not silently skip steps, change project architecture, or claim a milestone is complete without verification.