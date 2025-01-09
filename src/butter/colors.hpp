namespace butter{

__device__ float gamma(float v) {
  static const float kGamma = 0.372322653176;
  static const float limit = 37.8000499603;
  float bright = v - limit;
  if (bright >= 0) {
    static const float mul = 0.0950819040934;
    v -= bright * mul;
  }
  {
    static const float limit2 = 74.6154406429;
    float bright2 = v - limit2;
    if (bright2 >= 0) {
      static const float mul = 0.01;
      v -= bright2 * mul;
    }
  }
  {
    static const float limit2 = 82.8505938033;
    float bright2 = v - limit2;
    if (bright2 >= 0) {
      static const float mul = 0.0316722592629;
      v -= bright2 * mul;
    }
  }
  {
    static const float limit2 = 92.8505938033;
    float bright2 = v - limit2;
    if (bright2 >= 0) {
      static const float mul = 0.221249885752;
      v -= bright2 * mul;
    }
  }
  {
    static const float limit2 = 102.8505938033;
    float bright2 = v - limit2;
    if (bright2 >= 0) {
      static const float mul = 0.0402547853939;
      v -= bright2 * mul;
    }
  }
  {
    static const float limit2 = 112.8505938033;
    float bright2 = v - limit2;
    if (bright2 >= 0) {
      static const float mul = 0.021471798711500003;
      v -= bright2 * mul;
    }
  }
  static const float offset = 0.106544447664;
  static const float scale = 10.7950943969;
  float retval = scale * (offset + pow(v, kGamma));
  return retval;
}

__global__ void opsinDynamicsImage_kernel(float* src1, float* src2, float* src3, float* blurred1, float* blurred2, float* blurred3, int width, int height){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width*height) return;

    float3 sensitivity;
    float3 src = {src1[x], src2[x], src3[x]};
    float3 blurred = {blurred1[x], blurred2[x], blurred3[x]};
    opsin_absorbance(blurred);
    sensitivity.x = gamma(blurred.x) / blurred.x;
    sensitivity.y = gamma(blurred.y) / blurred.y;
    sensitivity.z = gamma(blurred.z) / blurred.z;
    opsin_absorbance(src);
    
}

void opsinDynamicsImage(Plane_d src[3], Plane_d temp[3], Plane_d temp2, float* gaussiankernel){
    //change src from linear SRGB to opsin dynamic XYB
    for (int i = 0; i < 3; i++){
        src[i].blur(temp[i], temp2, 1.2f, 0.0, gaussiankernel);
    }
}

}