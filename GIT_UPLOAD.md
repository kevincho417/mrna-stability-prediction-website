# Git Upload Guide

This project is prepared for Git sharing. The `.gitignore` excludes the course dataset, local dependency folders, cache files, server logs, generated CodeIgniter app folders, and extra experiment outputs.

## 1. Initialize And Commit

```bash
git init
git add .
git commit -m "Initial mRNA stability prediction project"
```

## 2. Push To GitHub

Create an empty repository on GitHub first, then run:

```bash
git branch -M main
git remote add origin https://github.com/<your-account>/<your-repo>.git
git push -u origin main
```

If the remote already exists:

```bash
git remote set-url origin https://github.com/<your-account>/<your-repo>.git
git push -u origin main
```

## 3. Large File Notes

The final PyTorch checkpoint files are below GitHub's single-file size limit, but they still make the repository larger. If you prefer a lightweight public repository, move model checkpoints to GitHub Releases or cloud storage and keep only the training scripts and result summaries in Git.

## 4. Privacy / Permission Check

The course dataset is already excluded:

```gitignore
Dataset/
```

Before making the repository public, also confirm that the project PDFs can be redistributed. If not, add these paths to `.gitignore` before pushing:

```gitignore
*.pdf
*.png
```
