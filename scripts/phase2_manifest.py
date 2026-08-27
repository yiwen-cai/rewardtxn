#!/usr/bin/env python3
"""
Phase 2B: StepManifest + StepToken (持久化面, sidecar 模式)

不改 slime 源码: 在 megatron checkpoint 目录内写 sidecar (step_token.json),
使 checkpoint 目录成为"持久化提交单元":
  SAVE_DIR/<iter>/step_token.json  = {token, iter, weights_hash, prev_token,
                                      prev_iter, n_manifest, ts}

token = RTX-ST-<iter>-<weights_hash[:16]>  (权重绑定 -> 幂等识别基础:
                                         同权重同 token, 不同权重不同 token)

模式:
  watch <save_dir>        轮询 latest_checkpointed_iteration.txt, 新 iter 生成 sidecar
  audit <save_dir>        读取全部 sidecar -> 已提交步列表 (协议层恢复识别, Reconciler 输入)
  token <save_dir> [iter] 打印指定/最新 iter 的 token

weights_hash 方法: 对 checkpoint 内文件清单 (relpath:size:mtime_ns) 排序后 SHA256,
+ latest_checkpointed_iteration 内容。快速 (清单级) 且能检测文件增删/大小/时间变化;
内容级哈希 (采样) 由 2C Reconciler 校验时可选启用。
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

TOKEN_PREFIX = "RTX-ST"
SIDECAR = "step_token.json"


def manifest_dir(save_dir: Path) -> Path:
    """sidecar 权威存储: SAVE_DIR 上级的 manifests/ (宿主可写);
    checkpoint 目录本身由容器 root 所有 (无 sudo 不可写),
    token 通过 weights_hash (内容采样) 与 checkpoint 强绑定。"""
    return Path(os.environ.get("RTX_MANIFEST_DIR", str(save_dir.parent / "manifests")))


def weights_hash(ckpt_dir: Path) -> str:
    lines = []
    latest = ckpt_dir / "latest_checkpointed_iteration.txt"
    if latest.exists():
        lines.append(f"latest={latest.read_text().strip()}")
    for p in sorted(ckpt_dir.rglob("*")):
        if p.is_file() and p.name != SIDECAR:
            st = p.stat()
            rel = str(p.relative_to(ckpt_dir))
            # 内容采样哈希 (头部+尾部 4KB): 检测同 size 内容替换, 且对 mtime 迁移鲁棒
            sample = hashlib.sha256()
            with open(p, "rb") as f:
                head = f.read(4096)
                sample.update(head)
                if st.st_size > 8192:
                    f.seek(-4096, 2)
                    sample.update(f.read(4096))
            lines.append(f"{rel}:{st.st_size}:{sample.hexdigest()}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def make_token(iter_no: int, wh: str) -> str:
    return f"{TOKEN_PREFIX}-{iter_no}-{wh[:16]}"


def _iter_dirname(p: Path):
    """兼容 megatron 命名 iter_0000003 与裸数字 3"""
    n = p.name
    if n.isdigit():
        return int(n)
    if n.startswith("iter_") and n[5:].isdigit():
        return int(n[5:])
    return None


def manifest_paths(save_dir: Path):
    dirs = [p for p in save_dir.iterdir() if p.is_dir() and _iter_dirname(p) is not None]
    return sorted(p for p in dirs if sidecar_file(save_dir, _iter_dirname(p)).exists()), sorted(dirs)


def iter_no(p: Path) -> int:
    return _iter_dirname(p)


def audit(save_dir: Path) -> dict:
    """读取全部 sidecar -> 已提交步 (含 token/哈希/链)"""
    entries = []
    for p in sorted(save_dir.iterdir()):
        it = _iter_dirname(p)
        if it is None:
            continue
        sc = sidecar_file(save_dir, it)
        if sc.exists():
            entries.append(json.loads(sc.read_text()))
        else:
            entries.append({"iter": it, "status": "NO_TOKEN"})
    entries.sort(key=lambda e: e["iter"])
    committed = [e for e in entries if e.get("status") != "NO_TOKEN"]
    return {"committed_steps": committed, "committed_iters": [e["iter"] for e in committed],
            "missing_token_iters": [e["iter"] for e in entries if e.get("status") == "NO_TOKEN"]}


def sidecar_file(save_dir: Path, iter_no_i: int) -> Path:
    return manifest_dir(save_dir) / f"step_token_{iter_no_i}.json"


def write_sidecar(save_dir: Path, iter_no_i: int, prev=None):
    # 目录可能命名为 iter_0000003 或 3
    cand = [p for p in save_dir.iterdir() if p.is_dir() and _iter_dirname(p) == iter_no_i]
    ckpt = cand[0] if cand else save_dir / str(iter_no_i)
    wh = weights_hash(ckpt)
    token = make_token(iter_no_i, wh)
    rec = {
        "token": token, "iter": iter_no_i, "weights_hash": wh,
        "prev_token": prev["token"] if prev else None,
        "prev_iter": prev["iter"] if prev else None,
        "n_manifest": 0,  # 2C 扩展: GroupManifest 引用数
        "ts": time.time(),
    }
    (sidecar_file(save_dir, iter_no_i)).parent.mkdir(parents=True, exist_ok=True)
    (sidecar_file(save_dir, iter_no_i)).write_text(json.dumps(rec, indent=2))
    return rec


def watch(save_dir: Path, poll_sec: float = 5.0, max_iters: int = 1000):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    last_iter = -1
    print(f"[manifest] watching {save_dir}", flush=True)
    while True:
        latest = save_dir / "latest_checkpointed_iteration.txt"
        if latest.exists():
            try:
                cur = int(latest.read_text().strip())
            except Exception:
                cur = last_iter
            if cur != last_iter and cur >= 0:
                # 可能中间 iter 未观察 (watch 中途启动): 对每个有目录无 sidecar 的 iter 补写
                ckpts, _ = manifest_paths(save_dir)
                all_iters = sorted(
                    _iter_dirname(p) for p in save_dir.iterdir()
                    if p.is_dir() and _iter_dirname(p) is not None
                )
                for it in all_iters:
                    if it <= last_iter or it > cur:
                        continue
                    sc = sidecar_file(save_dir, it)
                    if not sc.exists():
                        prev = audit(save_dir)["committed_steps"]
                        prev = prev[-1] if prev else None
                        rec = write_sidecar(save_dir, it, prev)
                        print(f"[manifest] iter {it}: token={rec['token']} prev={rec['prev_iter']}", flush=True)
                last_iter = cur
        time.sleep(poll_sec)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "watch"
    save_dir = sys.argv[2] if len(sys.argv) > 2 else None
    if not save_dir:
        print("usage: phase2_manifest.py watch|audit|token <save_dir> [iter]")
        sys.exit(1)
    save_dir = Path(save_dir)
    if mode == "watch":
        watch(save_dir)
    elif mode == "audit":
        res = audit(save_dir)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif mode == "token":
        it = sys.argv[3] if len(sys.argv) > 3 else None
        if it is None:
            res = audit(save_dir)
            it = res["committed_iters"][-1] if res["committed_iters"] else None
            if it is None:
                print("no committed step"); sys.exit(1)
        rec = json.loads(sidecar_file(Path(save_dir), int(it)).read_text())
        print(rec["token"])


if __name__ == "__main__":
    main()
