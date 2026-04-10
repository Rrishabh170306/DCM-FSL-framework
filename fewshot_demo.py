import argparse
import csv
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image

import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        roc_curve,
        auc,
        precision_recall_curve,
    )
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import label_binarize
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CSVImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, images_root: str, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_indices = defaultdict(list)

        rows = []
        labels = set()
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_id = row.get("id")
                label = row.get("label")
                if file_id is None or label is None:
                    continue
                rows.append((file_id.strip(), label.strip()))
                labels.add(label.strip())

        self.class_names = sorted(labels)
        label_to_idx = {name: i for i, name in enumerate(self.class_names)}

        missing = 0
        for file_id, label in rows:
            _, ext = os.path.splitext(file_id)
            filename = file_id if ext else f"{file_id}.png"
            path = os.path.join(images_root, filename)
            if not os.path.isfile(path):
                missing += 1
                continue
            label_idx = label_to_idx[label]
            self.class_to_indices[label_idx].append(len(self.samples))
            self.samples.append((path, label_idx))

        if missing:
            print(f"Warning: {missing} images listed in CSV were not found on disk.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def resolve_images_root(dataset_dir: str) -> str:
    nested = os.path.join(dataset_dir, "train", "train")
    if os.path.isdir(nested):
        return nested
    first_level = os.path.join(dataset_dir, "train")
    if os.path.isdir(first_level):
        return first_level
    return dataset_dir


def load_data(dataset_dir: str, image_size: int = 84):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    csv_path = os.path.join(dataset_dir, "trainLabels.csv")
    if os.path.isfile(csv_path):
        images_root = resolve_images_root(dataset_dir)
        dataset = CSVImageDataset(csv_path, images_root, transform=transform)
        if len(dataset) == 0:
            raise RuntimeError("CSV dataset loaded zero samples. Check paths and file names.")
        return dataset, dataset.class_to_indices, dataset.class_names

    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        class_to_indices[label].append(idx)

    return dataset, class_to_indices, dataset.classes


def split_indices(class_to_indices, train_ratio: float = 0.8, seed: int = 42):
    rng = random.Random(seed)
    train_map = {}
    eval_map = {}

    for cls, idxs in class_to_indices.items():
        idxs_copy = idxs[:]
        rng.shuffle(idxs_copy)
        split = max(1, int(len(idxs_copy) * train_ratio))
        train_map[cls] = idxs_copy[:split]
        eval_map[cls] = idxs_copy[split:]

    return train_map, eval_map


def select_eval_classes(class_to_indices, n_way: int, k_shot: int, q_query: int, seed: int):
    eligible = [c for c, idxs in class_to_indices.items() if len(idxs) >= k_shot + q_query]
    if len(eligible) < n_way:
        raise ValueError(
            f"Not enough classes with at least {k_shot + q_query} samples. "
            f"Found {len(eligible)} eligible classes; need {n_way}."
        )
    rng = random.Random(seed)
    return rng.sample(eligible, n_way)


def create_episode(
    dataset,
    class_to_indices,
    n_way: int,
    k_shot: int,
    q_query: int,
    device,
    class_pool=None,
    label_map=None,
):
    if class_pool is None:
        eligible = [c for c, idxs in class_to_indices.items() if len(idxs) >= k_shot + q_query]
        if len(eligible) < n_way:
            raise ValueError(
                f"Not enough classes with at least {k_shot + q_query} samples. "
                f"Found {len(eligible)} eligible classes; need {n_way}."
            )
        sampled_classes = random.sample(eligible, n_way)
    else:
        sampled_classes = list(class_pool)
        if len(sampled_classes) < n_way:
            raise ValueError(
                f"Not enough classes in class_pool. Found {len(sampled_classes)}; need {n_way}."
            )
        if len(sampled_classes) != n_way:
            sampled_classes = random.sample(sampled_classes, n_way)

    if label_map is None:
        label_map = {cls: i for i, cls in enumerate(sampled_classes)}

    support_images, support_labels = [], []
    query_images, query_labels = [], []

    for cls in sampled_classes:
        new_label = label_map[cls]
        idxs = random.sample(class_to_indices[cls], k_shot + q_query)
        support_idxs = idxs[:k_shot]
        query_idxs = idxs[k_shot:]

        for idx in support_idxs:
            img, _ = dataset[idx]
            support_images.append(img)
            support_labels.append(new_label)

        for idx in query_idxs:
            img, _ = dataset[idx]
            query_images.append(img)
            query_labels.append(new_label)

    support_x = torch.stack(support_images).to(device)
    query_x = torch.stack(query_images).to(device)
    support_y = torch.tensor(support_labels, dtype=torch.long, device=device)
    query_y = torch.tensor(query_labels, dtype=torch.long, device=device)

    return support_x, support_y, query_x, query_y


class ConvNet(nn.Module):
    def __init__(self, emb_dim: int = 64, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention

        self.encoder = nn.Sequential(
            self._conv_block(3, 32),
            self._conv_block(32, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, emb_dim)

        if self.use_attention:
            self.attn = nn.Linear(emb_dim, emb_dim, bias=False)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)

        if self.use_attention:
            weights = torch.softmax(self.attn(x), dim=-1)
            x = x * weights

        return F.normalize(x, dim=-1)


def compute_prototypes(embeddings, labels, n_way: int):
    prototypes = []
    for c in range(n_way):
        class_emb = embeddings[labels == c]
        prototypes.append(class_emb.mean(dim=0))
    return torch.stack(prototypes)


def prototypical_logits(query_emb, prototypes):
    return -torch.cdist(query_emb, prototypes, p=2)


def prototypical_loss(query_emb, query_labels, prototypes):
    logits = prototypical_logits(query_emb, prototypes)
    loss = F.cross_entropy(logits, query_labels)
    preds = torch.argmax(logits, dim=1)
    return loss, preds, logits


def supervised_contrastive_loss(embeddings, labels, temperature: float = 0.1):
    emb = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(emb, emb.T) / temperature

    n = labels.size(0)
    mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(n, device=labels.device, dtype=torch.bool)

    sim = sim.masked_fill(self_mask, -1e9)
    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

    pos_mask = mask & (~self_mask)
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / pos_count

    return -mean_log_prob_pos.mean()


def train(
    model,
    dataset,
    class_to_indices,
    n_way: int,
    k_shot: int,
    q_query: int,
    episodes: int,
    device,
    use_dc: bool,
    dc_std: float,
    use_contrastive: bool,
    contrastive_weight: float,
    lr: float,
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    accs = []
    f1s = [] if SKLEARN_AVAILABLE else None

    for ep in range(1, episodes + 1):
        support_x, support_y, query_x, query_y = create_episode(
            dataset, class_to_indices, n_way, k_shot, q_query, device
        )

        support_emb = model(support_x)
        query_emb = model(query_x)

        if use_dc:
            support_emb = support_emb + torch.randn_like(support_emb) * dc_std

        prototypes = compute_prototypes(support_emb, support_y, n_way)
        proto_loss, preds, _ = prototypical_loss(query_emb, query_y, prototypes)

        loss = proto_loss
        if use_contrastive:
            all_emb = torch.cat([support_emb, query_emb], dim=0)
            all_labels = torch.cat([support_y, query_y], dim=0)
            c_loss = supervised_contrastive_loss(all_emb, all_labels)
            loss = loss + contrastive_weight * c_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (preds == query_y).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)

        if SKLEARN_AVAILABLE:
            _, _, f1, _ = precision_recall_fscore_support(
                query_y.cpu().numpy(), preds.cpu().numpy(), average="macro", zero_division=0
            )
            f1s.append(f1)

        print(f"Episode {ep:03d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    return losses, accs, f1s


def evaluate(
    model,
    dataset,
    class_to_indices,
    n_way: int,
    k_shot: int,
    q_query: int,
    episodes: int,
    device,
    use_dc: bool,
    dc_std: float,
    class_pool=None,
    label_map=None,
):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    correct_dists = []
    incorrect_dists = []

    with torch.no_grad():
        for _ in range(episodes):
            support_x, support_y, query_x, query_y = create_episode(
                dataset,
                class_to_indices,
                n_way,
                k_shot,
                q_query,
                device,
                class_pool=class_pool,
                label_map=label_map,
            )

            support_emb = model(support_x)
            query_emb = model(query_x)

            if use_dc:
                support_emb = support_emb + torch.randn_like(support_emb) * dc_std

            prototypes = compute_prototypes(support_emb, support_y, n_way)
            _, preds, logits = prototypical_loss(query_emb, query_y, prototypes)
            probs = torch.softmax(logits, dim=1)

            dists = -logits
            idx = torch.arange(dists.size(0), device=dists.device)
            correct = dists[idx, query_y]
            mask = torch.ones_like(dists, dtype=torch.bool)
            mask[idx, query_y] = False
            incorrect = dists[mask]

            all_preds.append(preds.cpu().numpy())
            all_labels.append(query_y.cpu().numpy())
            all_scores.append(probs.cpu().numpy())
            correct_dists.append(correct.cpu().numpy())
            incorrect_dists.append(incorrect.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_score = np.vstack(all_scores) if all_scores else None
    correct_dists = np.concatenate(correct_dists) if correct_dists else None
    incorrect_dists = np.concatenate(incorrect_dists) if incorrect_dists else None

    if SKLEARN_AVAILABLE:
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
    else:
        acc = (y_true == y_pred).mean()
        precision = recall = f1 = 0.0
        cm = None

    return acc, precision, recall, f1, cm, y_true, y_score, correct_dists, incorrect_dists


def plot_curves(losses, accs, out_dir: str, f1s=None):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss vs Episodes")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_vs_episodes.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(accs, label="Accuracy")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Episodes")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_episodes.png"), dpi=150)
    plt.close()

    if f1s:
        plt.figure(figsize=(6, 4))
        plt.plot(f1s, label="F1-score")
        plt.xlabel("Episode")
        plt.ylabel("F1-score")
        plt.title("F1-score vs Episodes")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "f1_vs_episodes.png"), dpi=150)
        plt.close()


def plot_confusion_matrix(cm, out_dir: str):
    if cm is None:
        return

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def plot_roc_pr(y_true, y_score, out_dir: str, label_names=None):
    if not SKLEARN_AVAILABLE or y_score is None:
        print("Skipping ROC/PR curves: scikit-learn not available or scores missing.")
        return

    n_classes = y_score.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        n_classes = 2

    fpr = {}
    tpr = {}
    roc_auc = {}
    precision = {}
    recall = {}
    pr_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_score.ravel()
    )
    pr_auc["micro"] = auc(recall["micro"], precision["micro"])

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    for i in range(n_classes):
        name = label_names[i] if label_names and i < len(label_names) else f"Class {i}"
        plt.plot(fpr[i], tpr[i], label=f"{name} (AUC={roc_auc[i]:.2f})")
    plt.plot(fpr["micro"], tpr["micro"], "k--", label=f"micro (AUC={roc_auc['micro']:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    for i in range(n_classes):
        name = label_names[i] if label_names and i < len(label_names) else f"Class {i}"
        plt.plot(recall[i], precision[i], label=f"{name} (AUC={pr_auc[i]:.2f})")
    plt.plot(
        recall["micro"],
        precision["micro"],
        "k--",
        label=f"micro (AUC={pr_auc['micro']:.2f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_recall_curve.png"), dpi=150)
    plt.close()


def plot_distribution_overlap(correct_dists, incorrect_dists, out_dir: str):
    if correct_dists is None or incorrect_dists is None:
        return

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(correct_dists, bins=30, alpha=0.7, label="Correct class")
    plt.hist(incorrect_dists, bins=30, alpha=0.7, label="Incorrect classes")
    plt.xlabel("Prototype Distance")
    plt.ylabel("Count")
    plt.title("Distribution Overlap")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "distribution_overlap.png"), dpi=150)
    plt.close()


def visualize_tsne(
    model,
    dataset,
    class_to_indices,
    n_way: int,
    k_shot: int,
    q_query: int,
    device,
    out_dir: str,
    max_points: int = 200,
    class_pool=None,
    label_map=None,
    label_names=None,
):
    if not SKLEARN_AVAILABLE:
        print("Skipping t-SNE: scikit-learn not available.")
        return

    model.eval()
    support_x, support_y, query_x, query_y = create_episode(
        dataset,
        class_to_indices,
        n_way,
        k_shot,
        q_query,
        device,
        class_pool=class_pool,
        label_map=label_map,
    )

    with torch.no_grad():
        support_emb = model(support_x)
        query_emb = model(query_x)

    emb = torch.cat([support_emb, query_emb], dim=0).cpu().numpy()
    labels = torch.cat([support_y, query_y], dim=0).cpu().numpy()

    if emb.shape[0] > max_points:
        idx = np.random.choice(emb.shape[0], max_points, replace=False)
        emb = emb[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=10)
    emb_2d = tsne.fit_transform(emb)

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    for c in np.unique(labels):
        pts = emb_2d[labels == c]
        name = label_names[int(c)] if label_names and int(c) < len(label_names) else f"Class {c}"
        plt.scatter(pts[:, 0], pts[:, 1], s=12, label=name)
    plt.legend(loc="best", fontsize=8)
    plt.title("t-SNE Embeddings")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tsne_embeddings.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Few-Shot Learning Demo (Prototypical Networks)")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1, choices=[1, 5])
    parser.add_argument("--q_query", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--use_dc", action="store_true")
    parser.add_argument("--dc_std", type=float, default=0.05)
    parser.add_argument("--use_contrastive", action="store_true")
    parser.add_argument("--contrastive_weight", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--eval_random_classes",
        action="store_true",
        help="Sample new classes each eval episode (disables ROC/PR).",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset, class_to_indices, class_names = load_data(args.dataset_dir)
    train_map, eval_map = split_indices(class_to_indices, train_ratio=0.8, seed=args.seed)

    # Ensure query count is feasible for training and evaluation
    min_train = min(len(v) for v in train_map.values())
    min_eval = min(len(v) for v in eval_map.values()) if all(len(v) > 0 for v in eval_map.values()) else 0
    if min_train < args.k_shot + args.q_query:
        args.q_query = max(1, min_train - args.k_shot)
        print(f"Adjusted q_query to {args.q_query} for training feasibility")
    if min_eval and min_eval < args.k_shot + args.q_query:
        args.q_query = max(1, min_eval - args.k_shot)
        print(f"Adjusted q_query to {args.q_query} for eval feasibility")

    model = ConvNet(emb_dim=args.emb_dim, use_attention=args.use_attention).to(device)

    losses, accs, f1s = train(
        model,
        dataset,
        train_map,
        args.n_way,
        args.k_shot,
        args.q_query,
        args.episodes,
        device,
        args.use_dc,
        args.dc_std,
        args.use_contrastive,
        args.contrastive_weight,
        args.lr,
    )

    eval_classes = None
    label_map = None
    label_names = None
    use_fixed_eval = not args.eval_random_classes
    if use_fixed_eval:
        eval_classes = select_eval_classes(
            eval_map, args.n_way, args.k_shot, args.q_query, args.seed
        )
        label_map = {cls: i for i, cls in enumerate(eval_classes)}
        if class_names:
            label_names = [class_names[cls] for cls in eval_classes]

    acc, precision, recall, f1, cm, y_true, y_score, correct_dists, incorrect_dists = evaluate(
        model,
        dataset,
        eval_map,
        args.n_way,
        args.k_shot,
        args.q_query,
        args.eval_episodes,
        device,
        args.use_dc,
        args.dc_std,
        class_pool=eval_classes,
        label_map=label_map,
    )

    print("\nFinal Evaluation Metrics")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    plot_curves(losses, accs, args.output_dir, f1s=f1s)
    if use_fixed_eval:
        plot_confusion_matrix(cm, args.output_dir)
        plot_roc_pr(y_true, y_score, args.output_dir, label_names=label_names)
    else:
        print("Skipping confusion matrix and ROC/PR for random eval classes.")
    plot_distribution_overlap(correct_dists, incorrect_dists, args.output_dir)
    visualize_tsne(
        model,
        dataset,
        eval_map,
        args.n_way,
        args.k_shot,
        args.q_query,
        device,
        args.output_dir,
        class_pool=eval_classes if use_fixed_eval else None,
        label_map=label_map if use_fixed_eval else None,
        label_names=label_names if use_fixed_eval else None,
    )


if __name__ == "__main__":
    main()
