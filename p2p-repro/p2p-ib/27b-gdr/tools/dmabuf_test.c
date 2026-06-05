#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <infiniband/verbs.h>

/* NCCL's DMA_BUF capability probe: call ibv_reg_dmabuf_mr with a bad fd (-1).
 *   EBADF      -> driver/kernel SUPPORT dmabuf MR (got far enough to reject the fd)
 *   EOPNOTSUPP -> capability MISSING (mlx5_ib/ib_uverbs/userspace doesn't implement it)  */
int main(void) {
    int num = 0;
    struct ibv_device **list = ibv_get_device_list(&num);
    if (!list || num == 0) { printf("no IB devices\n"); return 1; }
    struct ibv_context *ctx = NULL;
    const char *want = "roceP3p1s0";
    for (int i = 0; i < num; i++)
        if (strcmp(ibv_get_device_name(list[i]), want) == 0) { ctx = ibv_open_device(list[i]); break; }
    if (!ctx) { printf("cannot open %s\n", want); return 1; }
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    if (!pd) { printf("alloc_pd failed\n"); return 1; }

    errno = 0;
    struct ibv_mr *mr = ibv_reg_dmabuf_mr(pd, 0, 4096, 0, -1, IBV_ACCESS_LOCAL_WRITE);
    int e = errno;
    printf("ibv_reg_dmabuf_mr(fd=-1) -> mr=%p errno=%d (%s)\n", (void*)mr, e, strerror(e));
    if (mr)                    printf("VERDICT: UNEXPECTED success\n");
    else if (e == EBADF)       printf("VERDICT: DMABUF-MR SUPPORTED (EBADF) -> blocker is downstream (pci_p2pdma/BAR)\n");
    else if (e == EOPNOTSUPP)  printf("VERDICT: DMABUF-MR NOT SUPPORTED (EOPNOTSUPP) -> mlx5_ib/ib_uverbs/userspace capability missing\n");
    else                       printf("VERDICT: other errno=%d\n", e);
    return 0;
}
