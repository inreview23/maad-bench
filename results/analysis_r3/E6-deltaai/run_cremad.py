"""CREMA-D AST/PANNs rotations on DeltaAI: same call as e6_repr.run_rotation on Spark."""
import sys
import rotate_repr
backbone, jobs = sys.argv[1], int(sys.argv[2])
rotate_repr.rotate_one("cremad", backbone, pca_dim=64, settings=("primary", "secondary"), jobs=jobs, collect=("primary",))
print("done", backbone, flush=True)
