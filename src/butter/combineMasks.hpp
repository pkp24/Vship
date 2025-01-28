namespace butter{

__launch_bounds__(256)
__global__ void computeDiffmap_Kernel(float* mask_xyb0, float* mask_xyb1, float* mask_xyb2, float* mask_xyb_dc0, float* mask_xyb_dc1, float* mask_xyb_dc2, float* block_diff_dc0, float* block_diff_dc1, float* block_diff_dc2, float* block_diff_ac0, float* block_diff_ac1, float* block_diff_ac2, float* dst, int width){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;

    if (x >= width) return;

    float combined = block_diff_dc0[x]*mask_xyb_dc0[x] + block_diff_dc1[x]*mask_xyb_dc1[x] + block_diff_dc2[x]*mask_xyb_dc2[x];
    combined += block_diff_ac0[x]*mask_xyb0[x] + block_diff_ac1[x]*mask_xyb1[x] + block_diff_ac2[x]*mask_xyb2[x];

    if (x == 10000) printf("final result: %f with %f, %f, %f and %f, %f, %f.\n", sqrtf(combined), block_diff_dc0[x], block_diff_dc1[x], block_diff_dc2[x], mask_xyb_dc0[x], mask_xyb_dc1[x], mask_xyb_dc2[x]);
    dst[x] = sqrtf(combined);
}

void computeDiffmap(float* mask_xyb0, float* mask_xyb1, float* mask_xyb2, float* mask_xyb_dc0, float* mask_xyb_dc1, float* mask_xyb_dc2, float* block_diff_dc0, float* block_diff_dc1, float* block_diff_dc2, float* block_diff_ac0, float* block_diff_ac1, float* block_diff_ac2, float* dst, int width, hipStream_t stream){
    int th_x = std::min(256, width);
    int bl_x = (width-1)/th_x + 1;
    computeDiffmap_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(mask_xyb0, mask_xyb1, mask_xyb2, mask_xyb_dc0, mask_xyb_dc1, mask_xyb_dc2, block_diff_dc0, block_diff_dc1, block_diff_dc2, block_diff_ac0, block_diff_ac1, block_diff_ac2, dst, width);
    GPU_CHECK(hipGetLastError());
}

}