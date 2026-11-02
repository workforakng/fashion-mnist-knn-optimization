import sys, torch
sys.path.insert(0, sys.argv[1])
from sklearn.model_selection import train_test_split
from src.data import load_splits
from src.classical import fit_pca
from src.train_loo import train_softknn
torch.set_num_threads(8)
xa, ya, xv, yv, _, _ = load_splits(val_size=6000, seed=42)
xa, _, ya, _ = train_test_split(xa, ya, train_size=12000, stratify=ya, random_state=42)
pca = fit_pca(xa, 64)
for tau in (0.5, 0.1):
  for ex in (False, True):
    cfg = dict(embed_dim=64, k=7, tau=tau, lr=1e-3, optimizer="adam", epochs=6, batch_size=256,
               patience=6, augment=None, augmented_bank_copies=0, exclude_self=ex,
               pca_components=pca.components_.copy(), seed=42)
    m, h, bv = train_softknn(xa, ya, xv, yv, cfg, device="cpu", verbose=False)
    print(f"tau={tau} exclude_self={ex}: best_val={bv:.4f} final_train_acc={h[-1]['train_acc']:.4f} final_loss={h[-1]['train_loss']:.4f}")
