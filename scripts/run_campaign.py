"""Validate or submit an explicitly authorized deployed campaign; defaults to local dry-run."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rl.campaign import RUNS, polarity_configs, validate_authorization, validate_main_jobs, validate_training_jobs, digest
from rl.checkpointing import atomic_json


def preparation_manifest(campaign_id):
    return {'schema': 1, 'phase': 'main', 'campaign_id': campaign_id, 'launch': False,
            'authorization': None, 'runs': list(RUNS), 'k': None, 'budget_usd': None,
            'deadline_utc_epoch': None, 'credit_expiry_utc_epoch': None, 'verified_credit_usd': None,
            'preflight_spent_usd': None, 'initial_verified_credit_usd': None, 'max_gpu_jobs': 3,
            'configs': polarity_configs(campaign_id),
            'waves': [['original-evaluation'], list(RUNS[:3]), list(RUNS[3:]),
                      [r+'-evaluation' for r in RUNS[:3]], [r+'-evaluation' for r in RUNS[3:]]],
            'jobs': [], 'source_hashes': {}, 'dataset_manifests': {},
            'remote_checks_pending': ['vllm request seeds and adapter synchronization',
                                      'full original greedy accuracy 45–50%', 'remote W&B authentication',
                                      'Volume commit/readback', 'controller survival after client disconnect',
                                      'measured cost and verified credit/expiry']}


def preflight_manifest(campaign_id):
    return {'schema': 1, 'phase': 'preflight', 'campaign_id': campaign_id+'-preflight', 'launch': False,
            'authorization': None, 'budget_usd': 2.5, 'elapsed_cap_seconds': 5400, 'max_gpu_jobs': 1,
            'deadline_utc_epoch': None, 'credit_expiry_utc_epoch': None, 'verified_credit_usd': None,
            'pilots': [{'arm': arm, 'seed': 0, 'updates': 20, 'diagnostic_updates': [20],
                        'forced_resume_after': 10 if arm == 'full' else None} for arm in ('full','positive','negative')],
            'sampled_benchmarks': {'targets': ['original','full-pilot'], 'questions': 64, 'k': 64},
            'symbolic_benchmark_rows': 100, 'source_hashes': {}, 'jobs': []}


def dry_run(manifest):
    try:
        validate_authorization(manifest)
        refusal = None
    except (ValueError, TypeError) as exc:
        refusal = str(exc)
    return {'mode': 'dry-run', 'remote_calls': 0, 'launch_allowed': refusal is None,
            'refusal': refusal, 'manifest_hash': digest(manifest), 'waves': manifest.get('waves', [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest')
    parser.add_argument('--prepare', help='write unapproved manifests into this output directory')
    parser.add_argument('--campaign-id', default='polarity-study')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        if args.launch:
            parser.error('--prepare cannot launch')
        output = Path(args.prepare)
        manifest = preparation_manifest(args.campaign_id)
        atomic_json(output/'campaign-manifest.json', manifest)
        atomic_json(output/'preflight-manifest.json', preflight_manifest(args.campaign_id))
    elif args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
    else:
        parser.error('--manifest or --prepare is required')
    if not args.launch:
        print(json.dumps(dry_run(manifest), indent=2))
        return
    validate_authorization(manifest)
    if not manifest.get('jobs') or not manifest.get('source_hashes') or not manifest.get('dataset_manifests'):
        raise ValueError('unresolved measured jobs, code hashes or dataset manifests')
    if manifest["phase"] == "main":
        validate_main_jobs(manifest)
    if manifest["phase"] == "training_only":
        validate_training_jobs(manifest)
    import modal
    controller = modal.Function.from_name(manifest.get('deployment_name', 'lm-rl-polarity'), 'campaign_controller')
    call = controller.spawn(manifest)
    print(json.dumps({'controller_call_id': call.object_id, 'campaign_id': manifest['campaign_id']}))


if __name__ == '__main__':
    main()
