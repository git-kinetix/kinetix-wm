#!/usr/bin/env python3
"""Create manifest of SDP10 renderings with animation_radius > 200cm.

Scans DynamoDB, finds matching renderings, resolves video S3 keys,
uploads manifest JSON to S3 for workers to consume.

Also uploads the ViT-B/16 checkpoint to S3 if not already there.
"""

import argparse
import json
import os
import tempfile

import boto3

BUCKET = "kinetix-rd-storage"
MANIFEST_KEY = "lewm/true_motion_manifest.json"
CHECKPOINT_KEY = "lewm/vitb16_pretrained.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-radius", type=float, default=200)
    parser.add_argument("--profile", default="rd-ireland")
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name="eu-west-1")
    dynamodb = session.resource("dynamodb", region_name="eu-west-1")
    s3 = session.client("s3", region_name="eu-west-1")

    # Step 1: Get qualifying animation_ids
    anim_table = dynamodb.Table("Dev_Animations")
    qualifying_aids = set()
    scan_kw = {}
    print(f"Scanning Dev_Animations for radius > {args.min_radius}...")
    while True:
        resp = anim_table.scan(**scan_kw)
        for item in resp.get("Items", []):
            r = item.get("animation_radius")
            if r is not None:
                try:
                    if float(r) > args.min_radius:
                        qualifying_aids.add(item["animation_id"])
                except (ValueError, TypeError):
                    pass
        if "LastEvaluatedKey" not in resp:
            break
        scan_kw["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    print(f"  {len(qualifying_aids)} qualifying animations")

    # Step 2: Get renderings
    prod_table = dynamodb.Table("Prod_Renderings")
    items = []
    scan_kw = {}
    print("Scanning Prod_Renderings...")
    while True:
        resp = prod_table.scan(**scan_kw)
        for item in resp.get("Items", []):
            if (item.get("animation_id") in qualifying_aids
                    and item.get("features_uri") and item.get("frames_uri")):
                rid = item["rendering_id"]
                # Resolve video key
                video_prefix = f"preprocessed_datasets/SDP_10_static_camera/videos/{rid}/"
                items.append({
                    "rendering_id": rid,
                    "animation_id": item["animation_id"],
                    "features_uri": item["features_uri"],
                    "video_prefix": video_prefix,
                    "video_key": None,  # resolved below
                    "bucket": BUCKET,
                })
        if "LastEvaluatedKey" not in resp:
            break
        scan_kw["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    print(f"  {len(items)} renderings found")

    # Step 3: Resolve video keys (batch check S3)
    print("Resolving video S3 keys...")
    resolved = []
    for i, item in enumerate(items):
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=item["video_prefix"], MaxKeys=3)
        mp4s = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".mp4")]
        if mp4s:
            item["video_key"] = mp4s[0]
            resolved.append(item)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(items)} checked, {len(resolved)} with videos")
    print(f"  {len(resolved)} renderings with videos")

    # Step 4: Upload manifest
    manifest = {"items": resolved, "total": len(resolved), "min_radius": args.min_radius}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(manifest, f)
        tmp_path = f.name

    print(f"Uploading manifest ({len(resolved)} items) to s3://{BUCKET}/{MANIFEST_KEY}")
    s3.upload_file(tmp_path, BUCKET, MANIFEST_KEY)
    os.remove(tmp_path)

    # Step 5: Upload ViT-B/16 checkpoint if needed
    try:
        s3.head_object(Bucket=BUCKET, Key=CHECKPOINT_KEY)
        print(f"Checkpoint already on S3: s3://{BUCKET}/{CHECKPOINT_KEY}")
    except Exception:
        local_ckpt = os.path.expanduser("~/.stable_worldmodel/vjepa_checkpoints/vitb16_pretrained.pt")
        if os.path.exists(local_ckpt):
            print(f"Uploading checkpoint to s3://{BUCKET}/{CHECKPOINT_KEY}...")
            s3.upload_file(local_ckpt, BUCKET, CHECKPOINT_KEY)
        else:
            print("WARNING: No local checkpoint found. Workers will need it on S3.")

    print(f"\nManifest: s3://{BUCKET}/{MANIFEST_KEY}")
    print(f"Checkpoint: s3://{BUCKET}/{CHECKPOINT_KEY}")
    print(f"\nReady to launch workers. Example:")
    print(f"  sky jobs launch sky/encode_batch.yaml --num-nodes 4 --name lewm-encode")


if __name__ == "__main__":
    main()
