import os, torch, torch.distributed as dist
print(f"[nccl_alltest] rank={os.environ.get('RANK')} world={os.environ.get('WORLD_SIZE')} "
      f"master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}", flush=True)
dist.init_process_group(backend="nccl", init_method="env://")
r = dist.get_rank()
torch.cuda.set_device(0)
t = torch.ones(4 * 1024 * 1024, device="cuda:0") * (r + 1)
dist.all_reduce(t)
torch.cuda.synchronize()
print(f"[nccl_alltest] RANK {r} ALLREDUCE_OK sum0={t[0].item()} expect=3.0", flush=True)
dist.destroy_process_group()
