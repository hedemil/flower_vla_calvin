# Company Thesis Repository – Workflow Guide

This repository is a **container repo** used to pin exact versions of each student’s thesis code
via **git submodules**.  
All actual development happens in the **personal thesis repositories**.

---

## Repository structure

```text
company-repo/
├── emil/        # Emil Hed thesis code (git submodule)
├── khalifa/     # Khalifa thesis code (git submodule?)
├── dataset/     # Shared datasets (NOT versioned)
├── scripts/     # Shared helper / cluster scripts (optional)
└── README.md
```

## In company repo (technically inside personal repo)
```bash
cd emil
git checkout -b feature/my-feature

# make changes
git add .
git commit -m "Describe change"
git push -u origin feature/my-feature
```

# Update company repo
```bash
cd ..
git status
# shows: modified: emil (new commits)

git add emil
git commit -m "Update Emil submodule to latest main"
git push
```
