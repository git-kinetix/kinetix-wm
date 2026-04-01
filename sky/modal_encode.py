"""
Modal app: ViT-B/16 embedding extraction + body pose deltas for 2722 SDP10 videos.
Fans out across up to 20 GPU containers using Modal .map().
Based on ~/Repos/parallel-processing/methods/modal/app.py pattern.
"""

import modal
import os
import json

# ── Modal image: bake ViT-B/16 weights ──────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "timm==1.0.15",
        "boto3",
        "botocore",
        "Pillow",
        "numpy",
        "av",
        "h5py",
    )
    .run_commands(
        # Pre-download ViT-B/16 weights into the image
        'python -c "'
        "import timm; "
        "m = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0); "
        'print(f\\"embed_dim={m.embed_dim}, params={sum(p.numel() for p in m.parameters())}\\")"'
    )
)

# ── AWS credentials ─────────────────────────────────────────────
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

if not AWS_ACCESS_KEY_ID:
    try:
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(os.path.expanduser("~/.aws/credentials"))
        for profile in ["rd-ireland", "default"]:
            if profile in cfg:
                AWS_ACCESS_KEY_ID = cfg[profile].get("aws_access_key_id", "")
                AWS_SECRET_ACCESS_KEY = cfg[profile].get("aws_secret_access_key", "")
                if AWS_ACCESS_KEY_ID:
                    break
    except Exception:
        pass

aws_secret = modal.Secret.from_dict({
    "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
    "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
    "AWS_DEFAULT_REGION": "eu-west-1",
})

app = modal.App("lewm-encode")

BUCKET = "kinetix-rd-storage"
OUTPUT_PREFIX = "lewm/embeddings_modal/"


# ── GPU worker ──────────────────────────────────────────────────
@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[aws_secret],
    max_containers=20,
    scaledown_window=30,
    retries=2,
)
class Encoder:
    @modal.enter()
    def setup(self):
        import torch
        import timm
        import time

        self._t0 = time.time()
        self.device = "cuda"

        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        # Load ViT-B/16 (cached in image)
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.model = self.model.to(self.device).eval().half()
        self.model = torch.compile(self.model)
        self.embed_dim = self.model.embed_dim  # 768

        # ImageNet normalization tensors
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1).half()
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1).half()

        # Warmup torch.compile
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=self.device, dtype=torch.float16)
            _ = self.model(dummy)

        # S3 client
        import boto3
        from botocore.config import Config
        self.s3 = boto3.client("s3", region_name="eu-west-1", config=Config(
            max_pool_connections=64, connect_timeout=5, read_timeout=10,
            tcp_keepalive=True, retries={"max_attempts": 3, "mode": "adaptive"},
        ))

        print(f"[setup] ViT-B/16 loaded+compiled in {time.time()-self._t0:.1f}s")

    @modal.method()
    def process_rendering(self, rendering_json: str) -> str:
        import time
        import io
        import tempfile
        import numpy as np
        import torch
        import av
        from PIL import Image

        rec = json.loads(rendering_json)
        rid = rec["rendering_id"]
        video_key = rec["video_key"]
        npz_uri = rec["features_uri"]

        result = {
            "rendering_id": rid,
            "download_seconds": 0,
            "encode_seconds": 0,
            "upload_seconds": 0,
            "total_seconds": 0,
            "n_frames": 0,
            "success": False,
            "error": None,
        }

        t_start = time.time()

        try:
            # ── Download + decode video ────────────────────────
            t_dl = time.time()
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                self.s3.download_file(BUCKET, video_key, tmp.name)
                container = av.open(tmp.name)
                frames = []
                for frame in container.decode(video=0):
                    img = frame.to_image().convert("RGB").resize((224, 224), Image.BILINEAR)
                    frames.append(np.asarray(img))
                container.close()

            if not frames:
                raise ValueError("No frames decoded")
            frames_np = np.stack(frames)

            # ── Download NPZ for body poses ────────────────────
            parts = npz_uri.replace("s3://", "").split("/", 1)
            npz_bucket, npz_key = parts[0], parts[1]
            with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
                self.s3.download_file(npz_bucket, npz_key, tmp.name)
                npz = np.load(tmp.name, allow_pickle=True)

            poses = npz.get("poses_beta0_world")
            trans = npz.get("trans_beta0_world")
            if poses is None or trans is None:
                raise ValueError("No poses_beta0_world or trans_beta0_world")
            if poses.ndim == 4:
                poses = poses[0]
            if trans.ndim == 3:
                trans = trans[0]

            # Compute 6D root delta poses
            root_rot = poses[:, 0, :]
            d_trans = np.diff(trans, axis=0)
            d_rot = np.diff(root_rot, axis=0)
            deltas = np.concatenate([d_trans, d_rot], axis=1).astype(np.float32)
            last = np.full((1, 6), np.nan, dtype=np.float32)
            actions = np.concatenate([deltas, last], axis=0)

            # Align
            n = min(len(frames_np), len(actions))
            frames_np = frames_np[:n]
            actions = actions[:n]

            result["download_seconds"] = time.time() - t_dl

            # ── Encode frames ──────────────────────────────────
            t_enc = time.time()
            batch_size = 128
            all_embs = []

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                for i in range(0, n, batch_size):
                    batch = frames_np[i:i+batch_size]
                    t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device, dtype=torch.float16) / 255.0
                    t = (t - self.mean) / self.std
                    emb = self.model(t)  # (B, 768)
                    all_embs.append(emb.float().cpu().numpy())

            embeddings = np.concatenate(all_embs, axis=0)  # (n, 768)
            result["encode_seconds"] = time.time() - t_enc

            # ── Upload to S3 ───────────────────────────────────
            t_up = time.time()
            buf = io.BytesIO()
            np.savez_compressed(buf, embedding=embeddings, action=actions)
            buf.seek(0)
            out_key = f"{OUTPUT_PREFIX}{rid}.npz"
            self.s3.put_object(Bucket=BUCKET, Key=out_key, Body=buf.getvalue())
            result["upload_seconds"] = time.time() - t_up

            result["n_frames"] = n
            result["success"] = True

        except Exception as e:
            result["error"] = str(e)

        result["total_seconds"] = time.time() - t_start
        return json.dumps(result)


# ── Local entrypoint ────────────────────────────────────────────
@app.local_entrypoint()
def main():
    import time
    import boto3

    # Load manifest from S3
    s3 = boto3.client("s3", region_name="eu-west-1")
    print("Downloading manifest...")
    resp = s3.get_object(Bucket=BUCKET, Key="lewm/true_motion_manifest.json")
    manifest = json.loads(resp["Body"].read())
    items = manifest["items"]
    print(f"Manifest: {len(items)} renderings")

    # Fan out
    rendering_jsons = [json.dumps(item) for item in items]

    t_start = time.time()
    encoder = Encoder()
    results_raw = list(encoder.process_rendering.map(
        rendering_jsons,
        order_outputs=False,
    ))
    wall_clock = time.time() - t_start

    # Parse results
    results = [json.loads(r) for r in results_raw]
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    total_dl = sum(r["download_seconds"] for r in successes)
    total_enc = sum(r["encode_seconds"] for r in successes)
    total_up = sum(r["upload_seconds"] for r in successes)
    total_frames = sum(r["n_frames"] for r in successes)

    # Cost: T4 ~ $0.59/hr on Modal
    total_gpu_sec = sum(r["total_seconds"] for r in successes)
    cost = total_gpu_sec * 0.59 / 3600

    print(f"\n{'='*60}")
    print(f"RESULTS ({len(items)} renderings, up to 20 containers)")
    print(f"{'='*60}")
    print(f"Wall clock:        {wall_clock:.1f}s ({wall_clock/60:.1f} min)")
    print(f"Renderings done:   {len(successes)}/{len(items)}")
    print(f"Frames processed:  {total_frames:,}")
    print(f"Throughput:        {total_frames/wall_clock:.0f} fps")
    print(f"Avg download:      {total_dl/len(successes):.2f}s/rendering")
    print(f"Avg encode:        {total_enc/len(successes):.2f}s/rendering")
    print(f"Avg upload:        {total_up/len(successes):.2f}s/rendering")
    print(f"Compute ratio:     {total_enc/(total_dl+1e-8):.2f} (>1 = compute-bound)")
    print(f"Estimated cost:    ${cost:.2f}")
    print(f"Failures:          {len(failures)}")

    if failures:
        for f in failures[:5]:
            print(f"  FAIL: {f['rendering_id'][:40]} — {f['error']}")

    # Save results summary
    with open("/tmp/encode_results.json", "w") as f:
        json.dump({"successes": len(successes), "failures": len(failures),
                    "total_frames": total_frames, "wall_clock": wall_clock,
                    "cost": cost, "results": results}, f)
    print(f"\nResults saved to /tmp/encode_results.json")
    print(f"Embeddings at: s3://{BUCKET}/{OUTPUT_PREFIX}")
