# GitHub Commit Guide

Follow these steps to commit the project to GitHub for the first time:

1. Initialize the git repository (if not already done):
```bash
cd /Users/denizakdemir/Dropbox/dakdemirGithub/GitHubProjects/multistate_nn
git init
```

2. First make sure the GitHub repository exists at: https://github.com/denizakdemir/multistate_nn
   Create it if it doesn't exist yet.

3. Add all files:
```bash
git add .
```

4. Make the first commit:
```bash
git commit -m "Initial commit of MultiStateNN: Neural network models for multistate processes"
```

5. Add the remote repository:
```bash
git remote add origin git@github.com:denizakdemir/multistate_nn.git
```

6. Push to the remote repository:
```bash
git push -u origin main
```

Note: If your default branch is 'master' instead of 'main', replace 'main' with 'master' in the command above, or run this command first to rename your branch:
```bash
git branch -M main
```

Your project is now ready on GitHub\!

Key features in this commit:
- A complete implementation of neural network-based multistate models
- Both deterministic and Bayesian (with Pyro) model variants
- Simulation functionality for patient trajectories
- Cumulative incidence function (CIF) calculation and visualization
- AIDS case study example with synthetic data
- Comprehensive documentation

After push, you may want to enable GitHub Pages to host the documentation.
