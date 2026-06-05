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
    polygons_gdf,
    config_class,
    output_dir_root,
    setup_steps=None,
    device='cuda',
    force=False,
):
    """
    Train one model with the given BAND_SPEC and record results.

    Args:
        experiment_name: directory-safe name (e.g. 'pan_ndvi_savi')
        band_spec: list of (raster_key, band_num) tuples
        base_paths: starting paths dict (will be copied + augmented)
        polygons_gdf: labels GeoDataFrame
        config_class: the Config class — its BAND_SPEC gets overridden
        output_dir_root: parent directory for all experiments
        setup_steps: list of (ensure_function_name, kwargs) tuples to run
                     before building patches, e.g.
                     [('ensure_slope', {'dem_key': 'dem_high_res',
                                        'smooth_sigma_m': 0.05})]
        force: re-run even if summary.json exists

    Returns: the summary dict (also written to summary.json)
    """
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
        ensure_fn = getattr(mu, step_name)
        ensure_fn(paths, **step_kwargs)

    # ── 2. Build patches (with new BAND_SPEC) ──
    patches = list(mu.build_patches_with_splits_multi(
        paths=paths,
        polygons_gdf=polygons_gdf,
        cfg=config_class,
    ))
    train_patches = [p for p in patches if p.get('split') == 'train']
    val_patches   = [p for p in patches if p.get('split') == 'val']
    test_patches  = [p for p in patches if p.get('split') == 'test']
    print(f"  patches: train={len(train_patches)}, val={len(val_patches)}, test={len(test_patches)}")

    # ── 3. Normalization stats ──
    stats_path = os.path.join(exp_dir, 'channel_stats.json')
    channel_means, channel_stds = mu.compute_channel_stats(
        train_patches, stats_path
    )

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
    thresholds = mu.pick_thresholds(val_pc, target_precision=0.9)
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
def run_band_experiments(
    experiments, base_paths, polygons_gdf, config_class,
    output_dir_root, device='cuda', force=False,
):
    """
    Run a list of experiments and write a summary CSV.

    experiments: list of dicts, each with keys:
        'name'        — experiment name (becomes subdirectory)
        'band_spec'   — list of (raster_key, band_num) tuples
        'setup'       — optional list of (ensure_fn_name, kwargs) to run first
    """
    os.makedirs(output_dir_root, exist_ok=True)
    summaries = []
    for exp in experiments:
        s = run_band_experiment(
            experiment_name=exp['name'],
            band_spec=exp['band_spec'],
            base_paths=base_paths,
            polygons_gdf=polygons_gdf,
            config_class=config_class,
            output_dir_root=output_dir_root,
            setup_steps=exp.get('setup'),
            device=device,
            force=force,
        )
        summaries.append(s)

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
