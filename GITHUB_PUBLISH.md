# Publishing to GitHub

1. Create a new empty **public or private** repository named `radboud-breast-imaging` on your GitHub account; do **not** initialize with a README.
2. From this extracted directory, run:

```bash
git init
git add .
git commit -m "Initial public release: synthetic breast imaging research pilot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/radboud-breast-imaging.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username. GitHub requires a working authentication setup (e.g. `gh auth login` or SSH).

Suggested GitHub description:
> CPU-friendly synthetic limited-angle breast-imaging reconstruction and lesion detection research pilot; 16 exploratory experiments with validation and ablations.

Suggested topics: `medical-imaging`, `tomography`, `cone-beam-ct`, `breast-imaging`, `image-reconstruction`, `python`, `synthetic-data`

Before making the repository public, review the scripts and experiment reports, confirm results reproduce on your machine, and remove any content you do not want published.
