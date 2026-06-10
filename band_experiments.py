"""
Band-combination experiment harness.

Define a list of experiments (each a band_spec + setup steps), run them
sequentially with run_band_experiments(...). Each experiment produces:

  {output_dir}/{experiment_name}/
      best_model.pt          — best checkpoint
      summary.json           — full results (metrics, thresholds, permutation)
      channel_stats.json     — normalization stats
      training_log.json      — per-epoch loss/mIoU

After all experiments, a summary CSV is written/updated at:
  {output_dir}/experiments_summary.csv

Experiments are skipped if their summary.json already exists, so you can
interrupt and resume safely. Set force=True to re-run.
"""
import os, json, time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import marsh_utils as mu


# ─────────────────────────────────────────────────────────────────────────────
# Run one experiment end-to-end
# ─────────────────────────────────────────────────────────────────────────────
def run_band_experiment(
    experiment_name,
    band_spec,
    base_paths,
    train_patches,
    val_patches,
    test_patches,
    config_class,
    output_dir_root,
    setup_steps=None,
    device='cuda',
    force=False,
):
    exp_dir = os.path.join(output_dir_root, experiment_name)
    summary_path = os.path.join(exp_dir, 'summary.json')
    
    if not force and os.path.exists(summary_path):
        print(f"[{experiment_name}] already complete, loading cached summary")
        with open(summary_path) as f:
            return json.load(f)
    
    os.makedirs(exp_dir, exist_ok=True)
    print(f"\n{'='*70}\n[{experiment_name}]\n  bands: {[b[0] for b in band_spec]}\n{'='*70}")
    t_start = time.time()
    
    # ── 1. Override Config and ensure derived bands exist ──
    config_class.BAND_SPEC = band_spec
    paths = dict(base_paths)
    for step_name, step_kwargs in (setup_steps or []):
        getattr(mu, step_name)(paths, **step_kwargs)
    
    # ── 2. Patches are pre-built — skipping ──
    print(f"  patches (reused): train={len(train_patches)}, val={len(val_patches)}, test={len(test_patches)}")
    
    # ── 3. Normalization stats (band-specific) ──
    stats_path = os.path.join(exp_dir, 'channel_stats.json')
    channel_means, channel_stds = mu.compute_channel_stats(train_patches, stats_path)

    # ── 4. Datasets & loaders ──
    train_ds = mu.MarshSegmentationDataset(train_patches, augmentation=mu.get_augmentations('train', channel_means, channel_stds))
    val_ds   = mu.MarshSegmentationDataset(val_patches,   augmentation=mu.get_augmentations('val',   channel_means, channel_stds))
    test_ds  = mu.MarshSegmentationDataset(test_patches,  augmentation=mu.get_augmentations('test',  channel_means, channel_stds))

    bs = config_class.BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=2)

    # ── 5. Model, loss, optimizer, scheduler ──
    num_classes = len(config_class.CLASS_NAMES)
    model = smp.Unet(
        encoder_name=config_class.ENCODER,
        encoder_weights=config_class.ENCODER_WEIGHTS,
        in_channels=len(band_spec),
        classes=num_classes,
    ).to(device)
    criterion = mu.CombinedLoss(num_classes=num_classes, ignore_index=config_class.IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_class.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config_class.EPOCHS)
    metric    = mu.IoUMetric(num_classes=num_classes, ignore_index=config_class.IGNORE_INDEX)
    ckpt_path = os.path.join(exp_dir, 'best_model.pt')

    # ── 6. Train ──
    best_iou = mu.train(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        num_epochs=config_class.EPOCHS, num_classes=config_class.N_CLASSES, ignore_index= config_class.IGNORE_INDEX,
        ckpt_path=ckpt_path, device=device, class_names=config_class.CLASS_NAMES,
    )

    # ── 7. Reload best, run evaluation ──
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])

    val_pc  = mu.evaluate_precision_coverage(model, val_loader,  num_classes=num_classes,
                                              ignore_index=config_class.IGNORE_INDEX, device=device)
    thresholds = mu.pick_thresholds(val_pc, config_class.CLASSES_OF_INTEREST, target_precision=0.9)
    test_pc = mu.evaluate_precision_coverage(model, test_loader, num_classes=num_classes,
                                              ignore_index=config_class.IGNORE_INDEX, device=device)
    perm = mu.channel_permutation_importance_per_class(
        model, val_loader, num_classes=num_classes,
        ignore_index=config_class.IGNORE_INDEX, n_repeats=3, device=device,
        class_names=config_class.CLASS_NAMES,
        band_names=[b[0] for b in band_spec],
    )

    # ── 8. Save summary ──
    elapsed_min = (time.time() - t_start) / 60.0
    summary = {
        'experiment_name':   experiment_name,
        'band_spec':         band_spec,
        'n_channels':        len(band_spec),
        'band_names':        [b[0] for b in band_spec],
        'best_val_miou':     float(best_iou),
        'best_epoch':        int(best_ckpt.get('epoch', -1)),
        'iou_per_class':     [float(x) for x in best_ckpt.get('iou_per_class', [])],
        'class_names':       list(config_class.CLASS_NAMES.values())
                              if hasattr(config_class.CLASS_NAMES, 'values')
                              else list(config_class.CLASS_NAMES),
        'thresholds':        {int(k): float(v) for k, v in thresholds.items()},
        'channel_means':     channel_means.tolist(),
        'channel_stds':      channel_stds.tolist(),
        'permutation_importance': {
            'baseline_iou': perm['baseline_iou'].tolist(),
            'drops_mean':   perm['drops_mean'].tolist(),
            'drops_std':    perm['drops_std'].tolist(),
        },
        'training_time_min': elapsed_min,
        'timestamp':         datetime.now().isoformat(),
        'config': {k: v for k, v in vars(config_class).items()
                   if not k.startswith('_') and not callable(v)},
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  → wrote {summary_path}  ({elapsed_min:.1f} min)")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Run many experiments, update CSV
# ─────────────────────────────────────────────────────────────────────────────
def run_band_experiments(experiments, base_paths, polygons_gdf, config_class,
                         output_dir_root, device='cuda', force=False):
    os.makedirs(output_dir_root, exist_ok=True)
    
    # ── Build patches ONCE — splits don't depend on BAND_SPEC ──
    # Patches are spatial windows tagged with train/val/test; they don't
    # care which bands the model uses, only the spatial layout.
    print(f"Building patches (one-time, shared across all experiments)...")
    
    # Set BAND_SPEC to a minimal one for the patch build — this is just to
    # satisfy whatever the build function expects. Any spec works since
    # patches are spatial.
    config_class.BAND_SPEC = experiments[0]['band_spec']
    
    # Run setup steps from the FIRST experiment so any derived bands the
    # patch builder might check are present.
    paths = dict(base_paths)
    for step_name, step_kwargs in (experiments[0].get('setup') or []):
        getattr(mu, step_name)(paths, **step_kwargs)
    
    patches = list(mu.build_patches_with_splits_multi(
        paths=paths, polygons_gdf=polygons_gdf, cfg=config_class,
    ))
    train_patches = [p for p in patches if p.get('split') == 'train']
    val_patches   = [p for p in patches if p.get('split') == 'val']
    test_patches  = [p for p in patches if p.get('split') == 'test']
    print(f"  Patches: train={len(train_patches)}, val={len(val_patches)}, test={len(test_patches)}\n")
    
    # ── Run each experiment with pre-built patches ──
    summaries = []
    for exp in experiments:
        s = run_band_experiment(
            experiment_name=exp['name'],
            band_spec=exp['band_spec'],
            base_paths=base_paths,
            train_patches=train_patches,
            val_patches=val_patches,
            test_patches=test_patches,
            config_class=config_class,
            output_dir_root=output_dir_root,
            setup_steps=exp.get('setup'),
            device=device,
            force=force,
        )
        summaries.append(s)
    
    # ... existing CSV building ...

    # Build comparison CSV
    rows = []
    for s in summaries:
        row = {
            'name':          s['experiment_name'],
            'n_bands':       s['n_channels'],
            'bands':         ', '.join(s['band_names']),
            'best_val_miou': s['best_val_miou'],
            'best_epoch':    s['best_epoch'],
            'train_min':     s['training_time_min'],
        }
        for c, name in enumerate(s['class_names']):
            iou = s['iou_per_class'][c] if c < len(s['iou_per_class']) else float('nan')
            row[f'iou_{name}'] = iou
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(output_dir_root, exist_ok=True)
    csv_path = os.path.join(output_dir_root, 'experiments_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*70}\nWrote summary CSV → {csv_path}\n{'='*70}")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Example experiments — edit / extend as needed
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLE_EXPERIMENTS = [
    {
        'name': '01_baseline_pan_ndvi_ndre',
        'band_spec': [
            ('pan_orthomosaic', 1),
            ('ndvi', 1),
            ('ndre', 1),
        ],
        'setup': [],   # ndvi/ndre handled by ensure_indices outside the harness
    },
    {
        'name': '02_pan_ndvi_savi',          # replace NDRE with SAVI (less redundant)
        'band_spec': [
            ('pan_orthomosaic', 1),
            ('ndvi', 1),
            ('savi', 1),
        ],
        'setup': [
            ('ensure_savi', {'ms_key': 'pansharp_ms'}),
        ],
    },
    {
        'name': '03_pan_ndvi_tpi_small',     # add geomorphology
        'band_spec': [
            ('pan_orthomosaic', 1),
            ('ndvi', 1),
            ('tpi_small', 1),
        ],
        'setup': [
            ('ensure_tpi', {'dem_key': 'dem_high_res',
                            'neighborhood_m': 0.3, 'out_key': 'tpi_small'}),
        ],
    },
    {
        'name': '04_pan_ndvi_tpi_small_large',  # multi-scale TPI
        'band_spec': [
            ('pan_orthomosaic', 1),
            ('ndvi', 1),
            ('tpi_small', 1),
            ('tpi_large', 1),
        ],
        'setup': [
            ('ensure_tpi', {'dem_key': 'dem_high_res',
                            'neighborhood_m': 0.3, 'out_key': 'tpi_small'}),
            ('ensure_tpi', {'dem_key': 'dem_high_res',
                            'neighborhood_m': 2.0, 'out_key': 'tpi_large'}),
        ],
    },
    {
        'name': '05_pan_savi_tpi_small_dist',   # physically-distinct signals
        'band_spec': [
            ('pan_orthomosaic',    1),
            ('savi',               1),
            ('tpi_small',          1),
            ('dist_to_channel',    1),
        ],
        'setup': [
            ('ensure_savi', {'ms_key': 'pansharp_ms'}),
            ('ensure_tpi',  {'dem_key': 'dem_high_res',
                             'neighborhood_m': 0.3, 'out_key': 'tpi_small'}),
            ('ensure_ndwi', {'ms_key': 'pansharp_ms'}),
            ('ensure_channel_mask_from_ndwi', {'ndwi_key': 'ndwi'}),
            ('ensure_distance_to_channel', {'channel_mask_key': 'channel_mask'}),
        ],
    },
]

import glob
import pandas as pd


def load_band_experiment_results(experiments_dir):
    """Load every summary.json in experiments_dir into structured DataFrames.

    Returns a dict with three DataFrames:
      'summary': indexed by experiment_name; high-level metrics
                 columns: n_bands, bands, best_val_miou, best_epoch, train_min
      'iou':     MultiIndex (experiment, class); column 'val_iou'
      'perm':    MultiIndex (experiment, band, class);
                 columns: drop_mean, drop_std, baseline_iou
    """
    rows_summary, rows_iou, rows_perm = [], [], []
    for fp in sorted(glob.glob(os.path.join(experiments_dir, '*/summary.json'))):
        with open(fp) as f:
            s = json.load(f)
        name = s['experiment_name']

        rows_summary.append({
            'experiment':    name,
            'n_bands':       s['n_channels'],
            'bands':         ', '.join(s['band_names']),
            'best_val_miou': s['best_val_miou'],
            'best_epoch':    s['best_epoch'],
            'train_min':     s['training_time_min'],
        })

        for c, class_name in enumerate(s['class_names']):
            iou = s['iou_per_class'][c] if c < len(s['iou_per_class']) else float('nan')
            rows_iou.append({'experiment': name, 'class': class_name, 'val_iou': iou})

        perm = s['permutation_importance']
        for ch, band in enumerate(s['band_names']):
            for c, class_name in enumerate(s['class_names']):
                rows_perm.append({
                    'experiment':   name,
                    'band':         band,
                    'class':        class_name,
                    'drop_mean':    perm['drops_mean'][ch][c],
                    'drop_std':     perm['drops_std'][ch][c],
                    'baseline_iou': perm['baseline_iou'][c],
                })

    return {
        'summary': pd.DataFrame(rows_summary).set_index('experiment'),
        'iou':     pd.DataFrame(rows_iou).set_index(['experiment', 'class']),
        'perm':    pd.DataFrame(rows_perm).set_index(['experiment', 'band', 'class']),
    }


# ─── Common views — pick whichever question you're asking ───
def view_summary(results):
    """One row per experiment, sorted by val mIoU. Quick comparison."""
    return results['summary'].sort_values('best_val_miou', ascending=False)


def view_iou_matrix(results):
    """experiment × class matrix of val IoU. Catches 'this combo killed class X'."""
    return results['iou']['val_iou'].unstack('class')


def view_perm_for_experiment(results, experiment_name):
    """For one experiment, band × class matrix of permutation drops.
    Answers: 'in THIS combo, which band carries which class?'"""
    return (results['perm']
            .xs(experiment_name, level='experiment')['drop_mean']
            .unstack('class'))


def view_perm_for_class(results, class_name):
    """For one class, experiment × band matrix of permutation drops.
    Answers: 'which band is doing the work for crab_platform, across combos?'"""
    return (results['perm']
            .xs(class_name, level='class')['drop_mean']
            .unstack('band'))


def view_perm_for_band(results, band_name):
    """For one band, experiment × class matrix of permutation drops.
    Answers: 'does NDRE earn its place, and for which classes?'"""
    return (results['perm']
            .xs(band_name, level='band')['drop_mean']
            .unstack('class'))

def view_perm_table(results, experiment_name, baseline=True):
    """Per-class permutation drops for one experiment, formatted mean±std.

    Returns a string-valued DataFrame (rows = bands, columns = classes) that
    prints exactly like the live channel_permutation_importance_per_class output.
    Set baseline=True to also print baseline per-class IoU above the table.
    """
    perm   = results['perm'].xs(experiment_name, level='experiment')
    means  = perm['drop_mean'].unstack('class')
    stds   = perm['drop_std'].unstack('class')
    # preserve column order from BAND_SPEC ordering in the JSON
    means  = means.reindex(perm.index.get_level_values('class').unique(), axis=1)
    stds   = stds.reindex(means.columns, axis=1)

    if baseline:
        base = (results['perm']
                .xs(experiment_name, level='experiment')['baseline_iou']
                .groupby('class').first()
                .reindex(means.columns))
        print(f"Baseline per-class IoU for {experiment_name}:")
        for cls, v in base.items():
            print(f"  {cls:20s} {v:.4f}")
        print()

    print(f"Per-class permutation drops (mean ± std):")
    formatted = pd.DataFrame(
        {col: [f"{means.loc[idx, col]:+.3f}±{stds.loc[idx, col]:.3f}"
               for idx in means.index]
         for col in means.columns},
        index=means.index,
    )
    return formatted

import matplotlib.pyplot as plt
import numpy as np

def show_perm_grid(results, n_cols=2, figsize_per_cell=(6, 4)):
    names = list(results['summary'].index)
    n = len(names)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(figsize_per_cell[0]*n_cols, figsize_per_cell[1]*n_rows))
    axes = np.atleast_2d(axes).ravel()
    
    for i, name in enumerate(names):
        df = (results['perm'].xs(name, level='experiment')['drop_mean']
              .unstack('class'))
        im = axes[i].imshow(df.values, cmap='RdYlGn', vmin=-0.1, vmax=0.7, aspect='auto')
        axes[i].set_xticks(range(len(df.columns)))
        axes[i].set_xticklabels(df.columns, rotation=45, ha='right')
        axes[i].set_yticks(range(len(df.index)))
        axes[i].set_yticklabels(df.index)
        axes[i].set_title(name)
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                axes[i].text(c, r, f"{df.values[r,c]:+.2f}",
                            ha='center', va='center',
                            fontsize=8, color='black')
    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    fig.tight_layout()
    return fig
